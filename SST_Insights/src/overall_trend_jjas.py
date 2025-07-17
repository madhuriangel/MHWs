import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from joblib import Parallel, delayed
import pymannkendall as mk
import pandas as pd

file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'

# Load the dataset
ds = xr.open_dataset(file_path, decode_times=False)

# Convert 'time' to a real datetime index
start_date = "1982-01-01"
time = pd.date_range(start=start_date, periods=ds.sizes["time"], freq="D")
ds = ds.assign_coords(time=("time", time))

# Deseasonalize SST (monthly climatology)
monthly_clim = ds["sst"].groupby("time.month").mean(dim="time", skipna=True)
deseasonalized = ds["sst"].groupby("time.month") - monthly_clim

#  summer months (June, July, August, September) ----
summer_months = [6, 7, 8, 9]
sst_summer = deseasonalized.sel(time=deseasonalized.time.dt.month.isin(summer_months))
time_summer = sst_summer.time  # pandas.DatetimeIndex

# Trend‐calculation functions
def calculate_trend_vectorized(data, time):
    x = np.arange(len(time))
    slopes = np.full(data.shape[1:], np.nan)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            y = data[:, i, j]
            mask = ~np.isnan(y)
            if mask.sum() > 1 and np.ptp(y[mask]) > 0:
                slope, _, _, _, _ = linregress(x[mask], y[mask])
                slopes[i, j] = slope * 365 * 10
    return slopes

def mann_kendall_test_parallel(data):
    def test_pt(y):
        yv = y[~np.isnan(y)]
        if len(yv) > 1:
            return mk.original_test(yv).p
        else:
            return np.nan
    flat = data.reshape(data.shape[0], -1).T
    pvals = Parallel(n_jobs=-1)(delayed(test_pt)(y) for y in flat)
    return np.array(pvals).reshape(data.shape[1:])

# ---- APPLY to JJAS data ----
sst_data_summer = sst_summer.values
trends_summer = calculate_trend_vectorized(sst_data_summer, time_summer)
sig_summer = mann_kendall_test_parallel(sst_data_summer)
significant_trends_summer = np.ma.masked_where(sig_summer >= 0.05, trends_summer)

# ---- PLOT summer trends ----
lon, lat = np.meshgrid(ds["lon"], ds["lat"])
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
pcm = ax.pcolormesh(lon, lat, significant_trends_summer,
                    cmap="coolwarm", vmin=0.1, vmax=0.5,
                    transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels = False; gl.right_labels = False
ax.set_title("JJAS SST Trends [1982–2024] (°C/decade)", fontsize=16, fontweight='bold', pad=10)
cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
cbar.set_label("°C/decade", fontsize=14, fontweight='bold')
#plt.savefig('sst_analysis_code/plots/sst_trend_JJAS.png')
plt.show()

# ---- AVERAGE warming rate over summer grid ----
mean_trend_summer = np.nanmean(significant_trends_summer)
std_trend_summer  = np.nanstd(significant_trends_summer)
print(f"Average JJAS warming rate: {mean_trend_summer:.2f} ± {std_trend_summer:.2f} °C/decade")
