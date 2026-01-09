import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


class FrequencyExpert1D(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # channel-wise conv: groups=channels, out_channels = channels
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels,
                               kernel_size=kernel_size, padding=padding, groups=channels, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels,
                               kernel_size=kernel_size, padding=padding, groups=channels, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x_pool):
        # x_pool: (B, C, F)
        y = self.conv1(x_pool)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.act2(y)
        return y  # (B, C, F)


class FrequencyGate(nn.Module):

    def __init__(self, channels, num_experts=3):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts

        self.mlp = nn.Sequential(
            nn.Linear(channels * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_experts)
        )

    def forward(self, Xf_mag):
        B, C, F, H, W = Xf_mag.shape
        freq_stats = Xf_mag.mean(dim=[3, 4])  # (B, C, F)
        gate_input = freq_stats.reshape(B, C * F)  # (B, C*F)(1,168)
        gate_logits = self.mlp(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=1)
        return gate_weights



# ------------------------------
# Frequency-Aware MoE (1D)
# ------------------------------
class FrequencyAwareMoE1D(nn.Module):
    """
    Frequency-Aware MoE that runs experts on pooled freq features.
    Input Xf_mag: (B, C, F, H, W)  (we use magnitude)
    Output Y_freq: (B, C, F, H, W)  (expanded from pooled expert outputs)
           gate_weights: (B, num_experts)
    """
    def __init__(self, channels, num_experts=3, kernel_size=3):
        super().__init__()
        self.channels = channels   # 24
        self.num_experts = num_experts

        # gate MLP: input (B, C) -> outputs num_experts logits per sample
        self.gate = FrequencyGate(channels,num_experts)

        # experts: each takes (B, C, F) and returns (B, C, F)
        self.experts = nn.ModuleList([FrequencyExpert1D(channels, kernel_size=kernel_size)
                                      for _ in range(num_experts)])

    def forward(self, Xf_mag):
        """
        Xf_mag: (B, C, F, H, W)
        """
        B, C, F, H, W = Xf_mag.shape

        gate_weights = self.gate(Xf_mag) # (B, num_experts)


        # 2) prepare expert inputs: pool spatial dims -> (B, C, F)
        Xf_pool = Xf_mag.mean(dim=[3,4])  # (B, C, F)

        # 3) expert outputs and weighted sum
        expert_outputs = []
        # each expert returns (B,C,F)
        for i, expert in enumerate(self.experts):
            out_i = expert(Xf_pool)  # (B, C, F)
            # expand gate weight to (B, C, F)
            gw = gate_weights[:, i].view(B, 1, 1)
            weighted = gw * out_i  # broadcast
            expert_outputs.append(weighted)

        Y_pool = sum(expert_outputs)  # (B, C, F)

        # 4) expand to (B, C, F, H, W) by unsqueeze + expand (cheap)
        Y_freq = Y_pool.unsqueeze(-1).unsqueeze(-1).expand(B, C, F, H, W)

        return Y_freq, gate_weights  # gate_weights (B, num_experts)


class TimeFrequencyMoE(nn.Module):
    def __init__(self, channels, num_experts=3, hidden_dim=64, kernel_size=3, attn_heads=4):
        super().__init__()
        
        self.channels = channels
        self.num_experts = num_experts
        self.H = 36
        self.W = 37

        self.time_branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True)
        )

        self.freq_moe = FrequencyAwareMoE1D(channels, num_experts=num_experts, kernel_size=kernel_size)

        self.attn_channels = channels  # attention embed_dim
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.attn_channels, 
                                                num_heads=attn_heads, batch_first=True)


        self.time_proj = nn.Linear(channels*self.H*self.W, self.attn_channels)
        self.freq_proj = nn.Linear(channels*self.H*self.W, self.attn_channels)


        self.fuse_layer = nn.Conv3d(2*self.attn_channels, channels, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"

        x_time = x.permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)  # (B*H*W, C, T)
        t_feat = self.time_branch(x_time)                         # (B*H*W, C, T)
        t_feat = t_feat.reshape(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # (B,T,C,H,W)


        x_t = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        Xf = torch.fft.rfft(x_t, dim=2, norm='ortho')
        Xf_mag = torch.abs(Xf)  
        freq_feat, gate_weights = self.freq_moe(Xf_mag)  # (B,C,F,H,W)


        time_seq = t_feat.reshape(B, T, C * H * W)
        B, C, F, H, W = freq_feat.shape
        freq_seq = freq_feat.permute(0, 2, 1, 3, 4).reshape(B, F, C * H * W)


        time_seq = self.time_proj(time_seq)      # (B, T, attn_channels)
        freq_seq = self.freq_proj(freq_seq)      # (B, F, attn_channels)



        attn_out, _ = self.cross_attn(time_seq, freq_seq, freq_seq)  # (B,T,attn_channels)

        attn_out = attn_out.unsqueeze(-1).unsqueeze(-1).expand(B, T, self.attn_channels, H, W)
        t_feat_exp = t_feat 

        if C != self.attn_channels:
            t_feat_exp = self.time_proj(t_feat.mean(dim=[3,4]).permute(0,1,2)).unsqueeze(-1).unsqueeze(-1).expand(B,T,self.attn_channels,H,W)


        x_fused = torch.cat([t_feat_exp, attn_out], dim=2)  # (B,T,2*attn_channels,H,W)
        x_out = self.fuse_layer(x_fused.permute(0,2,1,3,4))  # (B,C,T,H,W)
        x_out = x_out.permute(0,2,1,3,4)  # (B,T,C,H,W)

        return x_out, gate_weights

        