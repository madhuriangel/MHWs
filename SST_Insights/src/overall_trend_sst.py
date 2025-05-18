import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from joblib import Parallel, delayed
import pymannkendall as mk
import pandas as pd

# File path for the NetCDF dataset
#file_path = 'Data_noaa_copernicus/noaa_avhrr/noaa_combined_1982_2024_1res.nc'
file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'

# Load the dataset
ds = xr.open_dataset(file_path, decode_times=False)

# Convert 'time' to a datetime index
start_date = "1982-01-01"
time = pd.date_range(start=start_date, periods=ds.sizes["time"], freq="D")
ds = ds.assign_coords(time=("time", time))

# Deseasonalize SST
monthly_climatology = ds["sst"].groupby("time.month").mean(dim="time", skipna=True)
deseasonalized_sst = ds["sst"].groupby("time.month") - monthly_climatology

# Vectorized Trend Calculation
def calculate_trend_vectorized(data, time):
    x = np.arange(len(time))
    slopes = np.full(data.shape[1:], np.nan)

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            y = data[:, i, j]
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) > 1:  # Ensure at least two valid points
                y_valid = y[valid_mask]
                x_valid = x[valid_mask]
                if np.ptp(y_valid) > 0:  # Check for non-constant values
                    slope, _, _, _, _ = linregress(x_valid, y_valid)
                    slopes[i, j] = slope * 365 * 10  # Convert to °C/decade
    return slopes


sst_data = deseasonalized_sst.values
trends = calculate_trend_vectorized(sst_data, time)

# Parallelized Mann-Kendall Test
def mann_kendall_test_parallel(data):
    def test_single_point(y):
        valid_data = y[~np.isnan(y)]
        if len(valid_data) > 1:
            result = mk.original_test(valid_data)
            return result.p
        return np.nan

    # Flatten grid for parallel processing
    flattened_data = data.reshape(data.shape[0], -1).T
    p_values = Parallel(n_jobs=-1)(delayed(test_single_point)(y) for y in flattened_data)
    return np.array(p_values).reshape(data.shape[1:])
    
significance = mann_kendall_test_parallel(sst_data)

# Mask insignificant trends (p > 0.05)
significant_trends = np.ma.masked_where(significance >= 0.05, trends)

# Dynamically calculate vmin and vmax
#vmin = np.nanmin(significant_trends)
#vmax = np.nanmax(significant_trends)

# Plotting
lon, lat = np.meshgrid(ds["lon"], ds["lat"])
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8))
c = ax.pcolormesh(lon, lat, significant_trends, cmap="coolwarm", vmin=0.1, vmax=0.5, transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.right_labels = False
gl.top_labels = False
ax.tick_params(axis='both', labelsize=15, width=3, labelcolor='black', length=6, direction='out', pad=8)

ax.set_title("SST Trends [1982–2024] (°C/decade)", fontsize=16, fontweight='bold', pad=10)

# Colorbar
cbar = plt.colorbar(c, ax=ax, orientation="vertical", label="°C/decade", fraction=0.04, pad=0.02)
cbar.set_label("°C/decade", fontsize=14, fontweight='bold')

plt.savefig('sst_analysis_code/plots/sst_overalltrend.png')
plt.show()

# Basin statistics
mean_trend = np.nanmean(significant_trends)
std_trend = np.nanstd(significant_trends)
print(f"Average warming rate: {mean_trend:.2f} ± {std_trend:.2f} °C/decade")
