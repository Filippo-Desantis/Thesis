import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import datetime


## UNET ARCHITECTURE ##
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if necessary due to odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConvPerVar(nn.Module):
    def __init__(self, in_channels, V, M):
        super().__init__()
        self.V = V
        self.M = M
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, M, kernel_size=1) for _ in range(V)
        ])

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        outputs = []
        for v in range(self.V):
            logits = self.convs[v](x)  # [B, M, H, W]
            outputs.append(logits)
        out = torch.stack(outputs, dim=1)  # [B, V, M, H, W]
        return out

# --- Full UNet for V variables, M models ---
class UNetModelClassifier(nn.Module):
    def __init__(self, V, M, base_channels=64):
        super().__init__()
        self.V = V
        self.M = M
        in_channels = V * M

        self.dropout = nn.Dropout2d(p=0.3)  

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)

        self.up1 = Up(base_channels*8 + base_channels*4, base_channels*4)
        self.up2 = Up(base_channels*4 + base_channels*2, base_channels*2)
        self.up3 = Up(base_channels*2 + base_channels, base_channels)

        self.outc = OutConvPerVar(base_channels, V, M)

    def forward(self, x):  # x: [B, V, M, H, W]
        B, V, M, H, W = x.shape
        x = x.reshape(B, V * M, H, W)  # [B, V*M, H, W]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
       
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.outc(x)  # [B, V, M, H, W]
        return out

## SLP ##

class SLP(nn.Module):
    def __init__(self, num_vars=4, num_models=4):
        super().__init__()
        self.num_models = num_models
        self.num_vars = num_vars
        self.linear = nn.Linear(in_features = num_models * num_vars, out_features = num_models * num_vars)

    def forward(self, x):
        B, V, M, H, W = x.shape
        # Reshape dividing features by classes 
        x = x.view(B * H * W, M * V) 
        x = self.linear(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_vars=4, num_models=4):
        super().__init__()
        self.num_models = num_models
        self.num_vars = num_vars
        
        self.fc1 = nn.Linear(num_models * num_vars, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_models * num_vars)

    def forward(self, x):
        B, V, M, H, W = x.shape
        x = x.view(B * H * W, V * M)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.out(x)  



def load_forecast_zarr(date,lon_min, lon_max,lat_min, lat_max):

    models = ["DLWP", "Pangu6", "SFNO", "FCN"]
    data = {}

    for model in models:
        path = "/scratch3/poliMI/Filippo/Earth2/" + model + f"/48h{model}/forecast_{date}.zarr"
        data[model] = xr.open_zarr(path)
        data[model]  = data[model].rename({"lon": "longitude", "lat": "latitude"}).sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    return data

def load_era5_grib(lon_min, lon_max,lat_min, lat_max):
    era5_grib_path = "/scratch3/poliMI/Filippo/ERA5/ERA5/era5_2024_merged.zarr" 
    era5_data = xr.open_dataset(era5_grib_path, engine="zarr", decode_timedelta=False).sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    era5_data = era5_data.rename({"u10": "u10m"})    
    datasets = {}

    variabili = ["t2m", "z", "t", "u10m"]
    for var in variabili:
        da = era5_data[var]
        if "isobaricInhPa" in da.dims:
            for lev in da.isobaricInhPa.values:
                if (var=="z" and lev==500) or (var=="t" and lev==850):
                    varname = f"{var}{int(lev)}"
                    datasets[varname] = da.sel(isobaricInhPa=lev)
        else:
            datasets[var] = da
    return datasets

def split_days_2024(n_train, n_val, n_test, seed=None):
    start_2024 = datetime.datetime(2024, 1, 1, 0) 
    end_2024 = datetime.datetime(2024, 12, 31, 0)
    all_days = pd.date_range(start_2024, end_2024, freq="D")
    all_days = all_days.to_pydatetime() 

    if seed is not None:
        np.random.seed(seed)

    all_days = np.array(all_days)
    np.random.shuffle(all_days)

    assert n_train + n_val + n_test <= len(all_days), "Too many days requested!"

    train_days = all_days[:n_train]
    val_days = all_days[n_train:n_train+n_val]
    test_days = all_days[n_train+n_val:n_train+n_val+n_test]

    return train_days, val_days, test_days

## ITALY COORDINATES ##

lon_min, lon_max = 5, 20
lat_min, lat_max = 35, 48


era5_grib = load_era5_grib(lon_min, lon_max,lat_min, lat_max)

import numpy as np
import datetime
from torch.utils.data import Dataset

class ForecastModelSelectionDataset(Dataset):
    def __init__(self, days, lon_min, lon_max, lat_min, lat_max):
        self.sampled_days = days
        self.num_days = len(days)
        self.n_forecast = 5
        self.variables = ["t2m", "z500", "t850", "u10m"]

        self.era5 = np.zeros((self.num_days * 4, len(self.variables), 53, 61))
        self.pred = np.zeros((self.num_days * 4, len(self.variables), 4, 53, 61)) ## number 4 stands for models

        self.datasets = era5_grib
        self.lon_min, self.lon_max = lon_min, lon_max
        self.lat_min, self.lat_max = lat_min, lat_max

        self.load_data()


    def load_data(self):
        counter = 0
        for start in self.sampled_days:
            date_str = start.strftime("%Y-%m-%d")

            ds_pred = load_forecast_zarr(
                date_str, self.lon_min, self.lon_max, self.lat_min, self.lat_max
            )

            for i in range(1, self.n_forecast):
                act = start + datetime.timedelta(hours=6 * i)

                for v_idx, v in enumerate(self.variables):
                    # ERA5
                    u = self.datasets[v].sel(time=act.strftime("%Y-%m-%d")).sel(time=act)
                    u["longitude"] = (u.longitude + 180) % 360 - 180
                    self.era5[counter, v_idx] = u.sortby(u.longitude)

                    # model predictions
                    for nmod, k in enumerate(ds_pred.keys()):
                        if (v=="u10m") and (k=="DLWP"):
                            continue
                        upred = ds_pred[k].isel(lead_time=i)[v][0]
                        upred["longitude"] = (upred.longitude + 180) % 360 - 180
                        self.pred[counter, v_idx, nmod] = upred.sortby(upred.longitude)
                
                counter += 1

    def __len__(self):
        return self.num_days * 4

    def __getitem__(self, idx):
        input_tensor = self.pred[idx]   # shape: (n_vars, n_modelli, lat, lon)
        target_tensor = self.era5[idx]  # shape: (n_vars, lat, lon)

        diff = abs(input_tensor - target_tensor[:, None, :, :])
       
        label_list = np.argmin(diff, axis=1)

        return (
            diff, # to comment if training
            input_tensor,  # X
            label_list    # y
        )

# split days
train_days, val_days, test_days = split_days_2024(n_train=300, n_val=33, n_test=33, seed=42)

from torch.optim.lr_scheduler import _LRScheduler

# Warmup scheduler (if necessary)
class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, target_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = min(1.0, step / self.warmup_steps)
        return [base_lr * scale for base_lr in self.target_lr]

if __name__ == '__main__':


    # dataset
    train_dataset = ForecastModelSelectionDataset(train_days, lon_min, lon_max, lat_min, lat_max)
    validation_dataset   = ForecastModelSelectionDataset(val_days, lon_min, lon_max, lat_min, lat_max)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) #per italia 15 batch
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True) #per italia 15 batch

    # Model
    criterion = nn.CrossEntropyLoss(label_smoothing=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNetModelClassifier(4, 4).to(device) ## (n_variables, n_models)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-3) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    loss_list = []
    val_loss_list = []
    i=0
    loss_min = np.inf
    print("init training")
    for epoch in range(1000):
        model.train()
        batch_idx = 0
        for inputs, labels in train_loader:

            batch_idx+=1
            inputs, labels = inputs.to("cuda"), labels.to("cuda")  
            outputs = model(inputs.to(torch.float))
            
            outputs = outputs.reshape(-1, 4) ## (-1, n_models)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)  # flatten for CrossEntropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_list.append(loss)
        i+=1

        if loss <loss_min:
            torch.save(model.state_dict(), 'unet.pt')
        print(i, "train loss", loss)
        
        if epoch % 100 == 0:
            plt.plot([l.cpu().detach().numpy() for l in loss_list])
            plt.savefig("loss_plot.png")

        model.eval()
        with torch.no_grad():
            for X, y in validation_loader:

                X, y = X.to("cuda"), y.to("cuda")  
                z = model(X.to(torch.float))
                
                z = z.reshape(-1, 4) ## (-1, n_models)
                y = y.view(-1)
                
                loss = criterion(z, y)  # flatten for CrossEntropy

                val_loss_list.append(loss)  
            if epoch % 100 == 0:
                plt.plot([l.cpu().detach().numpy() for l in val_loss_list])
                plt.savefig("val_loss_plot.png")
        
        scheduler.step(loss)
        print(i, "validation loss", loss)
