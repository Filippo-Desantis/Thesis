from my_TESI_train import *

def grafico_errore(x_labels, lat_max, lat_min, lon_min, lon_max, giorno, stringa):

    import torch
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np

    lats = np.linspace(lat_max, lat_min, x_labels.shape[2])
    lons = np.linspace(lon_min, lon_max, x_labels.shape[3])

    variables = ["t2m","u10m"]
    tempi = [6]
    for lead_idx in range(len(tempi)):
        for var_idx, variable in enumerate(variables):  # loop on variables
            try:
                plt.figure(figsize=(8, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())

                im = ax.pcolormesh(
                    lons,
                    lats,
                    x_labels[lead_idx, var_idx, :, :].cpu(),
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r"
                )

                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax.coastlines(resolution="10m")
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

                ax.set_title(f"Labels {variable} - Lead time {tempi[lead_idx]} h - {giorno} - {stringa}")

                cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.08)
                cbar.set_label(f"Model")

                plot_path = f"{stringa}_italia_{variable}_lead{tempi[lead_idx]}.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close()
                print(f"Generated label plot {plot_path}")

            except Exception as e:
                print(f"Plot failed for lead {lead_idx}, variable {variable}: {str(e)}")


model = UNetModelClassifier(4, 4) ## (n_variables, n_models)
model.load_state_dict(torch.load("unet.pt", map_location="cuda"))
model.to("cuda")
model.eval()

# Inference

with torch.no_grad():

    test_dataset  = ForecastModelSelectionDataset(test_days, lon_min, lon_max, lat_min, lat_max)
    test_loader = DataLoader(test_dataset, batch_size=15, shuffle=True)

    i=0
    rmse_list = []
        
    ## Italy coordinates

    lon_min, lon_max = 5, 20
    lat_min, lat_max = 35, 48

    flag=True
    for test_error, test_inputs, test_labels in test_loader:
        test_error, test_inputs, test_labels = test_error.to("cuda"), test_inputs.to("cuda"), test_labels.to("cuda") 

        outputs = model(test_inputs.to(torch.float))
        probs = F.softmax(outputs, dim=2)
        preds = torch.argmax(probs, dim=2)

        if flag==True:
            grafico_errore(preds, lat_max, lat_min, lon_min, lon_max, test_days[0], "predicted")
            grafico_errore(test_labels, lat_max, lat_min, lon_min, lon_max, test_days[0], "true")
            flag=False
        
        correct = np.zeros(2)
        correct[0] = (test_labels[:,0,:,:] - preds[:,0,:,:] == 0).sum()
        correct[1] = (test_labels[:,1,:,:] - preds[:,1,:,:] == 0).sum()

        totale = 1 # total pixel counter
        for j in range(len(preds[:,0,:,:].shape)):
            totale *= preds[:,0,:,:].shape[j]

        accuracy = correct / totale
        print(accuracy)

        error = torch.gather(test_error, 2, preds.unsqueeze(2)).squeeze(2) 

        lats = np.arange(-90.0, 90.0 + 1e-12, 0.25)   # -> length 721
        assert lats.size == 721
        # index selection for Italy
        tol = 1e-8
        mask_ita = (lats >= lat_min - tol) & (lats <= lat_max + tol)
        lats_ita = lats[mask_ita]   # latitude vector for Italy

        weights_lat = np.cos(np.deg2rad(lats_ita))
        weights_lat /= weights_lat.mean()
        weights_lat = weights_lat[np.newaxis, :, np.newaxis]  
        weights_tensor = np.tile(weights_lat, (error.shape[0], 1, error.shape[3]))  
        rmse = []
        for idx in range(error.shape[1]):
            rmse.append(np.sqrt(((error[:,idx,:,:].cpu())**2 * weights_tensor).mean()))
        print("RMSE: ", rmse)

        for aa in range(test_error.shape[2]):
            print("RMSE modello ", aa, ":", np.sqrt(((test_error[:,0,aa,:,:].cpu())**2 * weights_tensor).mean()))
        i+=1
        print(i)
