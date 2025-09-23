import xarray as xr
import numpy as np
import cfgrib
import pandas as pd
import zarr
import os



def generate_quick_plot(io, date_str):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    try:
        variable = "z500"
        step = 4  # 24h lead time
        
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        
        im = ax.pcolormesh(
            io["lon"][:],
            io["lat"][:],
            io[variable][0, step],
            transform=ccrs.PlateCarree(),
            cmap="Spectral_r",
        )
        
        ax.set_title(f"Pangu6 {date_str} - Lead time: {6*step}hrs")
        ax.coastlines()
        ax.gridlines()

        # colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label(f"{variable} (m)")  # measure unit
        
        
        plot_path = f"mappe/Pangu6_z500_{date_str}.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Generated plot {plot_path}")
        
    except Exception as e:
        print(f"Plot failed for {date_str}: {str(e)}")

if __name__ == '__main__':
    base_zarr_path = "Pangu6/60hPangu6/forecast_{}.zarr"
    date_str = "2024-01-01"
    zarr_path = base_zarr_path.format(date_str)
    zarr.consolidate_metadata(zarr_path)

    if not os.path.exists(zarr_path):
        print(f"Zarr non trovato: {zarr_path}")

        
    ds_pred = xr.open_zarr(zarr_path)

    ds_pred = ds_pred.assign_coords(longitude=((ds_pred.lon + 180) % 360 - 180))
    ds_pred = ds_pred.sortby(ds_pred.lon)

    generate_quick_plot(ds_pred, date_str)
