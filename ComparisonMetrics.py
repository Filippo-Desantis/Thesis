## Latitude weighted RMSE and ACC calculator, provides charts of the model predictions provided, we properly cited WeatherBench in the thesis which inspired this code ##

import xarray as xr
import numpy as np
import cfgrib
import pandas as pd
import zarr
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
import csv

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.latitude))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse

def compute_weighted_acc(da_fc, da_true, ds_c, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = ds_c.mean('time')
    fa = da_fc- clim
    a = da_true - clim

    weights_lat = np.cos(np.deg2rad(da_fc.latitude))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

def plot_g(ds,t):
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    lon =  ds["longitude"][:]
    lat =  ds["latitude"][:]

    im = ax.pcolormesh(
        lon,
        lat,
        ds,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
    )

    ax.set_title(f"{date_str} - Lead time: {6}hrs")
    ax.coastlines()
    ax.gridlines()


    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)


def plot_g_subplots(datasets, titles, cmap="Spectral_r"):
    n = len(datasets)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 5),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    if n == 1:
        axes = [axes]  

    for ax, da, title in zip(axes, datasets, titles):
        lon, lat = da["longitude"], da["latitude"]

        im = ax.pcolormesh(
            lon, lat, da,
            transform=ccrs.PlateCarree(),
            cmap=cmap
        )

        ax.set_title(title)
        ax.coastlines()
        ax.gridlines(draw_labels=False)

        # colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, pad=0.05)

    plt.tight_layout()
    plt.show()



base_zarr_path = "/scratch3/poliMI/Filippo/Earth2/SFNO/60hSFNO/forecast_{}.zarr"

era5_grib_path = "/scratch3/poliMI/Filippo/ERA5/ERA5/era5_2024_merged.zarr" # ground truth dataset
era5_data = xr.open_zarr(era5_grib_path, decode_timedelta=False)
era5_data = era5_data.rename({"u10": "u10m"})


n_forecast = 11
times = [6*i for i in range(1,n_forecast)]

var_list = ["u10m","z500","t2m","t850"]

#era5_data = era5_data.rename({"u10": "u10m"})

datasets = {}

for var in era5_data.data_vars:
    da = era5_data[var]
    if "isobaricInhPa" in da.dims:
        for lev in da.isobaricInhPa.values:
            varname = f"{var}{int(lev)}"
            datasets[varname] = da.sel(isobaricInhPa=lev)
    else:
        datasets[var] = da
    
models = ["FCN", "SFNO", "Pangu6", "DLWP"]       

list_clim =[]
#for iclim in range(1,9):  # use this in case you want to try different climatology periods how work
iclim = 8
for model in models:
    var_val = {} 
    base_zarr_path = f"/scratch3/poliMI/Filippo/Earth2/{model}/60h{model}/forecast_{{}}.zarr"
    for var in var_list:

        if var=="u10m" and model=="DLWP":
            continue

        var_val[var] = {}
        climatologia = datasets[var].isel(time=slice((79+93+94+89)*4, (79+93+94+89+11)*4)) ## climatology days' shifting


        avg_rmse  = np.zeros((n_forecast-1,1))
        avg_acc  = np.zeros((n_forecast-1,1))
        num_days = 11 ## days shifting

        times = [6*i for i in range(1,n_forecast)]
        for delta_day in range(num_days):
            ## starting date shifting
            start = datetime.datetime(2024,12,21,0) + datetime.timedelta(days=delta_day)

            date_str = start.strftime("%Y-%m-%d")

            ## here you find climatology calculation every 61 days, useful for the whole year ##
            # if delta_day % 61 == 0:
            #     climatologia = datasets[var].isel(time=slice(delta_day*4, (delta_day + 61)*4))
            # # print("========================================")

            zarr_path = base_zarr_path.format(date_str)
            ds_pred = xr.open_zarr(zarr_path)
            ds_pred = ds_pred.rename({"lon": "longitude", "lat": "latitude"})


            for i in range(1,n_forecast):
                var_f = var

                act = start + datetime.timedelta(hours=6*i)
                act.strftime("%Y-%d-%mT%H:%M:%S")

                u = datasets[var].sel(time=act.strftime("%Y-%m-%d")).sel(time=act)
                u["longitude"] = (u.longitude + 180) % 360 - 180
                u = u.sortby(u.longitude)

                upred = ds_pred.isel(lead_time=i)[var_f][0]  
                upred["longitude"] = (upred.longitude + 180) % 360 - 180
                upred = upred.sortby(upred.longitude)

                w_rmse = compute_weighted_rmse(upred, u, mean_dims=xr.ALL_DIMS).values
                w_acc = compute_weighted_acc(upred, u, climatologia).values
                avg_rmse[i-1] += w_rmse
                avg_acc[i-1] += w_acc
                
        var_val[var]["acc"] = avg_acc/num_days
        var_val[var]["rmse"] =  avg_rmse/num_days
        
    list_clim.append(var_val)



fig, axes = plt.subplots(
        4, 2, figsize=(20, 10)
    )
fig.tight_layout(pad=3.0)

for ic in range(4):
    for ip,var in enumerate(var_list):
        if ic==0 and var=="u10m":
            continue
        axes[ip, 0].plot(times,list_clim[ic][var]["rmse"], label = f"{models[ic]}")
        axes[ip, 0].set_title(f"RMSE: {var}")
        axes[ip, 1].plot(times,list_clim[ic][var]["acc"], label = f"{models[ic]}")
        axes[ip, 1].set_title(f"ACC: {var} ")
        axes[ip, 0].legend()
        axes[ip, 1].legend()

plt.savefig("Grafici_cap2/multimodello_61.png")

