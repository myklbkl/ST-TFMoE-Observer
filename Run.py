import random
import torch.nn as nn
import torch
import pandas as pd
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ST_TFMoE_Observer import ST_TFMoE_Observer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from typing import Tuple, Any
import optuna
from optuna.samplers import TPESampler
from fvcore.nn import FlopCountAnalysis

os.environ["CUDA_VISIBLE_DEVICES"]="1"

seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


batch_size = 16      
time_step = 12       
predict_step = 12
in_shape = (12, 1, 71, 73)
N_S = 3                
en_de_c = 24   
N_h = 4                  
h_c = 24                 
N_T = 4           
T_c = 24                 
groups = 1                
num_epochs = 1


model = ST_TFMoE_Observer(in_shape=in_shape, N_S=N_S, en_de_c=en_de_c, N_h=N_h, h_c=h_c, N_T=N_T, T_c=T_c, incep_ker=list((3, 5, 7, 11)), groups=groups).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, verbose=True)

writer = SummaryWriter(f"./ST_TFMoE_Observer_epoch{num_epochs}")

def train_model(model, dataloader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    rmse_sum = 0.0
    train_bar_idx = tqdm(dataloader, desc=f'Training Epoch {epoch}/{num_epochs}') 
    for batch_idx, (input, target) in enumerate(train_bar_idx):
        optimizer.zero_grad()
        input, target = input.cuda(), target.cuda()
        output, *other_outputs = model(input) 
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        rmse = torch.sqrt(torch.mean((output - target) ** 2))
        rmse_sum += rmse.item()
        train_bar_idx.set_postfix({'Train Loss': loss.item(), 'RMSE': rmse.item()})

    avg_train_loss = train_loss / len(dataloader)
    avg_rmse = rmse_sum / len(dataloader)
    writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

    print(f"Train Epoch [{epoch}/{num_epochs}], Train Loss:{avg_train_loss:.8f}, RMSE:{avg_rmse:.6f}")
    return avg_train_loss, avg_rmse

def test_model(model, dataloader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    rmse_sum = 0.0
    test_bar_idx = tqdm(dataloader, desc=f'Testing Epoch {epoch}/{num_epochs}')
    with torch.no_grad():
        for input, target in test_bar_idx:
            input, target = input.cuda(), target.cuda()
            output, *other_outputs = model(input)  
            loss = criterion(output, target)
            test_loss += loss.item()
            rmse = torch.sqrt(torch.mean((output - target) ** 2))
            rmse_sum += rmse.item()
            test_bar_idx.set_postfix({'Test Loss1': loss.item(), 'RMSE1': rmse.item()})


    avg_test_loss = test_loss / len(dataloader)
    avg_rmse = rmse_sum / len(dataloader)
    writer.add_scalar("test/epoch_loss", avg_test_loss, epoch)

    print(f"Test Loss:{avg_test_loss:.8f}, RMSE: {avg_rmse:.6f}")
    return avg_test_loss, avg_rmse



train_data = TEC_Dataset(path='./Dataset/', train=True, time_step=time_step, predict_step=predict_step)
test_data = TEC_Dataset(path='./Dataset/', validation=True, time_step=time_step, predict_step=predict_step)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

best_loss = float('inf')
for epoch in range(1, num_epochs+1):
    train_loss, train_rmse = train_model(model, train_dataloader, criterion, optimizer, epoch)
    test_loss, test_rmse = test_model(model, test_dataloader, criterion, epoch)
    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr'] 
    writer.add_scalar("learning_rate", current_lr, epoch)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), f'./ST_TFMoE_Observer_epoch{num_epochs}.pth')
        print("Saved Best Model")

writer.close()
print(model)
print("Process completed!")