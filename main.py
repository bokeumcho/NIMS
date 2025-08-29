import os, time, json

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
import torch.nn.functional as F

from dataset import GK2ADataset
from model.model import SimVP
from model.module import pool_hw_only, pad_to_multiple
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # debugging stability
torch.backends.cudnn.deterministic = False # for parallelization stable

C, C_enc, C_hid = 1, 16, 64 
T = 6
T_interval = 10 # to 360
Ns, Nt = 4, 8
groups = 4
batch_size = 16
lr = 0.001
epochs = 100

# RR file path?
dataset = GK2ADataset(data_path='/home/work/team3/data_v2', reduce_size=2, T=T, T_interval=T_interval)
# torch.Size([16, 6, 1, 660, 750])
# 1299 * 1500 -> 660 * 750 

# Define split sizes (e.g., 70% train, 15% val, 15% test)
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # ensures all samples are used

print(train_size, val_size, test_size)

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= 16, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers= 8, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers= 8, shuffle=False, pin_memory=True)

model = SimVP(T, C, C_enc, C_hid, Ns, Nt, groups=groups).to(device)
#model = nn.DataParallel(model) # Parallelization
criterion = nn.MSELoss() # change to SSIM, + regularization term?
optimizer = SGD(model.parameters(), lr=lr)

best_val_loss = float("inf")
patience, patience_counter = 5, 0  # early stopping params

# reducer = nn.AvgPool2d(kernel_size=2)

for epoch in range(epochs):
    # -------- Training --------
    model.train()
    running_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        x = x.float().to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        x = pool_hw_only(x).contiguous()
        y = pool_hw_only(y).contiguous()
        x = pad_to_multiple(x, multiple=16).contiguous()
        y = pad_to_multiple(y, multiple=16).contiguous()

        # print('\n',x.shape, y.shape)

        with torch.amp.autocast('cuda'):
            pred = model(x)
            loss = criterion(pred, y)
            
            #### recursive
            # preds = model(x, y, tf_mode='scheduled', ss_ratio=0.5)  # [B, K*T, C, H, W]
            # # 타겟을 preds와 같은 순서로 펴기
            # targets = torch.stack([y[:, s:s+T] for s in range(K)], dim=1)  # [B, K, T, C, H, W]
            # targets = targets.reshape_as(preds)  # [B, K*T, C, H, W]
            # loss = F.l1_loss(preds, targets)  # 혹은 Charbonnier/SSIM 등

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    running_loss /= len(train_loader)

    # -------- Validation --------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            # x = x[..., :658, :750]
            # y = y[..., :658, :750]

            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            x = pool_hw_only(x).contiguous()
            y = pool_hw_only(y).contiguous()
            x = pad_to_multiple(x, multiple=16).contiguous()
            y = pad_to_multiple(y, multiple=16).contiguous()

            # print('\n',x.shape, y.shape)

            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: train_loss={running_loss:.4f}, val_loss={val_loss:.4f}")
    # -------- Saving every 5 epoch --------
    if (epoch + 1) % 5 == 0:
        save_path = f"model_simVP_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Saved checkpoint: {save_path}")

    # -------- Early Stopping --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_simVP.pt")  # save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# -------- Testing (after training finishes) --------
model.load_state_dict(torch.load("best_model_simVP.pt"))  # load best model
model.eval()
test_loss = 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x = x.float().to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        x = pool_hw_only(x).contiguous()
        y = pool_hw_only(y).contiguous()
        x = pad_to_multiple(x, multiple=16).contiguous()
        y = pad_to_multiple(y, multiple=16).contiguous()

        pred = model(x)
        loss = criterion(pred, y)
        test_loss += loss.item()
test_loss /= len(test_loader)

print(f"Final Test Loss: {test_loss:.4f}")
