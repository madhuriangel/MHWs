
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from matplotlib import gridspec

def load_and_mask_dataset(file_path):
    """
    Loads the NetCDF SST dataset and applies a land mask based on NaN values in the SST variable.

    Parameters:
        file_path (str): Path to the NetCDF file.

    Returns:
        xr.Dataset: Dataset with time converted to datetime and land masked as NaN.
        np.ndarray: Land mask (True for ocean, False for land).
    """
    ds = xr.open_dataset(file_path, decode_times=False)
    start_date = pd.Timestamp("1982-01-01")
    time = pd.date_range(start=start_date, periods=ds.dims["time"], freq="D")
    ds = ds.assign_coords(time=("time", time))

    # Create land mask from first timestep (NaNs considered land)
    land_mask = ~np.isnan(ds["sst"].isel(time=0))
    ds["sst"] = ds["sst"].where(land_mask)
    
    return ds, land_mask

def define_time_periods():
    """
    Defines decadal time periods for SST anomaly analysis.

    Returns:
        dict: Dictionary mapping period labels to time slice ranges.
    """
    return {
        "D1 (1982-1992)": slice("1982-01-01", "1992-12-31"),
        "D2 (1993-2003)": slice("1993-01-01", "2003-12-31"),
        "D3 (2004-2013)": slice("2004-01-01", "2013-12-31"),
        "D4 (2014-2024)": slice("2014-01-01", "2024-12-31"),
    }

def calculate_anomalies(ds, land_mask, periods):
    """
    Calculates SST anomalies for defined time periods against the climatological mean.

    Parameters:
        ds (xr.Dataset): Dataset with SST and datetime coordinates.
        land_mask (np.ndarray): Mask for ocean regions.
        periods (dict): Dictionary of time slices by period name.

    Returns:
        dict: Dictionary of SST anomaly DataArrays for each period.
    """
    anomalies = {}
    climatological_mean = ds["sst"].mean(dim="time", skipna=True)

    for period_name, time_range in periods.items():
        period_data = ds.sel(time=time_range)
        period_mean = period_data["sst"].mean(dim="time", skipna=True)
        anomalies[period_name] = (period_mean - climatological_mean).where(land_mask)

    return anomalies

def plot_anomalies(anomalies, vmin=-0.6, vmax=0.6):
    """
    Plots SST anomaly maps for each time period.

    Parameters:
        anomalies (dict): SST anomalies for each time period.
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
        ax.set_title(f"SST Anomaly for {period_name}", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.set_label("SST Anomaly (°C)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig('sst_analysis_code/plots/allperiods_anomaly.png')
    plt.show()

def main():
    """
    Main function to run the SST anomaly analysis and plot decadal maps.
    """
    #file_path = 'Data_noaa_copernicus/noaa_avhrr/noaa_combined_1982_2024_1res.nc'
    file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'
    
    # Load dataset and create mask
    ds, land_mask = load_and_mask_dataset(file_path)
    
    # Define time periods
    periods = define_time_periods()

    # Calculate anomalies
    anomalies = calculate_anomalies(ds, land_mask, periods)

    # Plot the anomalies
    plot_anomalies(anomalies)

# Run the script
if __name__ == "__main__":
    main()
