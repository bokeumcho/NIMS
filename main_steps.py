import os, time, json
from args_parse import build_parser, validate_args, setup_device, save_config

# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, AdamW
import torch.nn.functional as F

from dataset_restricted_v2 import GK2ADataset, GK2ADataset_MultiSteps
from model.model_recursive import SimVP_AR_Decoder 
from model.model_noRec import SimVP_kT
from model.module import pool_hw_only, pad_to_multiple
from tqdm import tqdm
import numpy as np

parser = build_parser()
args = parser.parse_args()
args = validate_args(args)
device = setup_device(args)
if args.save_config:
    save_config(args)

# gpu_name = 1
# device = torch.device(f"cuda:{gpu_name}" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # debugging stability
torch.backends.cudnn.deterministic = False # for parallelization stable

# >>> ADDED: make matmul faster on Ampere+ (keeps numerics valid)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # pytorch>=2
except Exception:
    pass

C, C_enc, C_hid = args.C, args.C_enc, args.C_hid  # 4, 16, 64 
T = args.T # 6
T_interval = args.T_interval #10 
steps = args.steps # 2
model_name = args.model_name #'SimVP_AR_Decoder' # 'SimVP_kT'

Ns, Nt = args.Ns, args.Nt # 4, 8
groups = args.groups #4
batch_size = args.batch_size # 16
lr = args.lr # 0.001
epochs = args.epochs # 10

dataset = GK2ADataset_MultiSteps(data_path='/home/work/team3/data_v2', # '/home/work/team3/data_v2', #
                 reduce_size=1, T=T, T_interval=T_interval,
                 transform=None,
                 steps=steps) # steps: # of T-blocks to predict

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
# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= 16, shuffle=True, pin_memory=True)
# val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers= 8, shuffle=False, pin_memory=True)
# test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers= 8, shuffle=False, pin_memory=True)

# faster DataLoader settings
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True, pin_memory_device="cuda",
    persistent_workers=True, drop_last=True  # >>> CHANGED
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True, pin_memory_device="cuda",
    persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True, pin_memory_device="cuda",
    persistent_workers=True
)

if model_name == 'SimVP_AR_Decoder':
    model = SimVP_AR_Decoder(T, C, C_enc, C_hid, Ns, Nt, groups, horizon = T * steps).to(device)
elif model_name == 'SimVP_kT':
    model = SimVP_kT(T=T, k=steps, C_in = C, C_out = C, C_enc = C_enc, C_hid = C_hid,
                        Ns = Ns, Nt = Nt, groups=groups).to(device)
    
#model = nn.DataParallel(model) # Parallelization
criterion = nn.MSELoss() # change to SSIM, + regularization term?
# optimizer = SGD(model.parameters(), lr=lr)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)  # fused needs PyTorch>=2.2 + recent CUDA
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)

best_val_loss = float("inf")
patience, patience_counter = 5, 0  # early stopping params

# -------------------- AMP --------------------
use_amp = True
amp_dtype = torch.float16  
scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

# -------------------- scheduled sampling --------------------
ss_start = 1.0
decay_rate = 0.8  # closer to 1.0 = slower decay
global_step = 0

for epoch in range(epochs):
    # -------- Training --------
    model.train()
    running_loss = 0
    ss_ratio = ss_start * (decay_rate ** global_step)

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)

        x = pool_hw_only(x) #.contiguous()
        y = pool_hw_only(y) #.contiguous()
        x = pad_to_multiple(x, multiple=16) #.contiguous()
        y = pad_to_multiple(y, multiple=16) #.contiguous()

        # print('\n',x.shape, y.shape)
        optimizer.zero_grad(set_to_none=True)  # >>> CHANGED

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            if model_name == 'SimVP_kT':
                pred = model(x)
            elif model_name == 'SimVP_AR_Decoder':
                pred = model(x, detach_between_steps=True, tf_mode=True, teacher=y, ss_ratio=ss_ratio)
            loss = criterion(pred, y)
            
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        global_step += 1

        if global_step // 100 == 99: 
            torch.save(model.state_dict(), f"model_{model_name}_interm.pt")  # save best model

    running_loss /= len(train_loader)

    # -------- Validation --------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):

            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            x = pool_hw_only(x) #.contiguous()
            y = pool_hw_only(y) #.contiguous()
            x = pad_to_multiple(x, multiple=16).contiguous()
            y = pad_to_multiple(y, multiple=16).contiguous()

            # print('\n',x.shape, y.shape)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                if model_name == 'SimVP_kT':
                    pred = model(x)
                elif model_name == 'SimVP_AR_Decoder':
                    pred = model(x, detach_between_steps=True, tf_mode=False)
                loss = criterion(pred, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)

    scheduler.step()

    print(f"Epoch {epoch+1}: train_loss={running_loss:.4f}, val_loss={val_loss:.4f}")

    # -------- Early Stopping --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"best_model_{model_name}.pt")  # save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# -------- Testing (after training finishes) --------
model.load_state_dict(torch.load(f"best_model_{model_name}.pt"))  # load best model
model.eval()
test_loss = 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)

        x = pool_hw_only(x) #.contiguous()
        y = pool_hw_only(y) #.contiguous()
        x = pad_to_multiple(x, multiple=16).contiguous()
        y = pad_to_multiple(y, multiple=16).contiguous()

        if model_name == 'SimVP_kT':
            pred = model(x)
        elif model_name == 'SimVP_AR_Decoder':
            pred = model(x, detach_between_steps=True, tf_mode=False)
        
        loss = criterion(pred, y)
        test_loss += loss.item()
test_loss /= len(test_loader)

print(f"Final Test Loss: {test_loss:.4f}")
