######This is the code for getting anomaly trend for jjas
######Code is updated with 1 resolution IDW code data
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from matplotlib import gridspec

def load_and_preprocess_dataset(file_path):
    """
    Loads the NetCDF dataset and assigns datetime coordinates to the 'time' dimension.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xr.Dataset: Dataset with proper datetime coordinates.
    """
    ds = xr.open_dataset(file_path, decode_times=False)
    start_date = pd.Timestamp("1982-01-01")
    time = pd.date_range(start=start_date, periods=ds.dims["time"], freq="D")
    ds = ds.assign_coords(time=("time", time))
    return ds

def extract_jjas_and_apply_mask(ds):
    """
    Filters the dataset for JJAS months and applies land mask.

    Parameters:
        ds (xr.Dataset): Full dataset.

    Returns:
        xr.DataArray: JJAS subset of SST data with land mask applied.
        np.ndarray: Land mask array.
    """
    ds_jjas = ds.sel(time=ds["time"].dt.month.isin([6, 7, 8, 9]))
    land_mask = ~np.isnan(ds["sst"].isel(time=0))
    ds_jjas["sst"] = ds_jjas["sst"].where(land_mask)
    return ds_jjas, land_mask

def compute_anomalies(ds_jjas, land_mask, periods):
    """
    Computes SST anomalies for specified periods compared to the JJAS climatological mean.

    Parameters:
        ds_jjas (xr.Dataset): JJAS-only SST dataset.
        land_mask (np.ndarray): Mask indicating valid ocean grid points.
        periods (dict): Dictionary with period names and time slice ranges.

    Returns:
        dict: Dictionary containing SST anomalies for each period.
    """
    anomalies = {}
    climatological_mean_jjas = ds_jjas["sst"].mean(dim="time", skipna=True)
    
    for period_name, time_range in periods.items():
        period_data = ds_jjas.sel(time=time_range)
        period_mean = period_data["sst"].mean(dim="time", skipna=True)
        anomalies[period_name] = (period_mean - climatological_mean_jjas).where(land_mask)
    
    return anomalies

def plot_anomalies(anomalies, vmin=-0.6, vmax=0.6):
    """
    Plots SST anomalies for each period using Cartopy maps.

    Parameters:
        anomalies (dict): Dictionary containing SST anomalies per period.
        vmin (float): Minimum value for color normalization.
        vmax (float): Maximum value for color normalization.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.05])
    axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
    ]
    cax = fig.add_subplot(gs[2, :])  # Colorbar axis
    cmap = "coolwarm"
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for ax, (period_name, anomaly) in zip(axes, anomalies.items()):
        img = anomaly.plot(
            ax=ax,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        ax.set_title(f"SST Anomaly for {period_name} (JJAS)", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.set_label("SST Anomaly (°C)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig('sst_analysis_code/plots/anomaly_periodjjas.png')
    plt.show()

def main():
    """
    Main function to execute the SST anomaly trend analysis for JJAS and plot results.
    """
    file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'
    
    # Load and preprocess data
    ds = load_and_preprocess_dataset(file_path)
    
    # Filter JJAS and apply mask
    ds_jjas, land_mask = extract_jjas_and_apply_mask(ds)
    
    # Define decadal periods
    periods = {
        "D1 (1982-1992)": slice("1982-06-01", "1992-09-30"),
        "D2 (1993-2003)": slice("1993-06-01", "2003-09-30"),
        "D3 (2004-2013)": slice("2004-06-01", "2013-09-30"),
        "D4 (2014-2024)": slice("2014-06-01", "2024-09-30"),
    }

    # Compute anomalies
    anomalies = compute_anomalies(ds_jjas, land_mask, periods)

    # Plot anomalies
    plot_anomalies(anomalies)

# Run main function
if __name__ == "__main__":
    main()
