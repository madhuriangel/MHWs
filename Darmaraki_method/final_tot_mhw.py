
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define font sizes
title_fontsize = 15
label_fontsize = 14
tick_fontsize = 12

# Load datasets
file_path1 = 'mhw_analysis_hobday/mhw_results/hobday1991_2024/mhw_stats_hobday34.nc'
file_path2 = 'mhw_analysis_alterhob/mhw_results/mhw_stats_alterhobday34.nc'

ds1 = xr.open_dataset(file_path1, decode_times=False)
ds2 = xr.open_dataset(file_path2, decode_times=False)

# Extract variables, replacing zeros with NaN for clarity
mhw_tot_events1 = ds1['mhw_total_events'].where(ds1['mhw_total_events'] != 0, np.nan)
mhw_tot_events2 = ds2['mhw_total_events'].where(ds2['mhw_total_events'] != 0, np.nan)

# Get latitude & longitude
lat = ds1['lat'].values
lon = ds1['lon'].values

# Ensure lat/lon are in ascending order
lat = np.sort(lat)
lon = np.sort(lon)

# Define min/max values for a common colorbar
vmin = np.nanmin([mhw_tot_events1.min(), mhw_tot_events2.min()])
vmax = np.nanmax([mhw_tot_events1.max(), mhw_tot_events2.max()])

# Create a figure with 2 subplots
fig, axes = plt.subplots(
    1, 2, figsize=(14, 8), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    constrained_layout=True
)

# Titles for the two plots
titles = [
    "Total Number of Marine Heatwaves (1982-2024)",
    "Total Number of Marine Heatwaves (1982-2024) (Alter)"
]

# Loop through datasets and plot on respective axes
img = None  # Will hold the last mappable object for colorbar
for ax, data, title in zip(axes, [mhw_tot_events1, mhw_tot_events2], titles):
    img = ax.pcolormesh(
        lon, lat, data, 
        cmap="cividis", vmin=vmin, vmax=vmax, 
        transform=ccrs.PlateCarree()
    )
    
    # Add map features
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
    
    # Set extent around region
    ax.set_extent([lon.min() - 0.1, lon.max() + 0.1, lat.min() - 0.1, lat.max() + 0.1], crs=ccrs.PlateCarree())
    
    # Remove lat/lon ticks or set custom tick parameters if required
    ax.set_xticks([])
    ax.set_yticks([])
    # If needed, customize tick label sizes as follows:
    # ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Set subplot title using the defined title_fontsize
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=10)

# Add a horizontal colorbar at the bottom using the defined label_fontsize and tick_fontsize
cbar = fig.colorbar(img, ax=axes, orientation='horizontal', fraction=0.05, pad=0.07)
cbar.set_label("Total Number of Marine Heatwaves (Events)", fontsize=label_fontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=tick_fontsize)

plt.show()
