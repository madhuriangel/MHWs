
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymannkendall as mk
from scipy.stats import linregress
from matplotlib import gridspec

def load_mhw_dataset(file_path):
    return xr.open_dataset(file_path, decode_times=False)

def compute_mean_and_trend(mhw_data, time):
    mean_data = mhw_data.mean(dim='time')
    trend_data = np.full(mean_data.shape, np.nan)
    mean_p_values = np.full(mean_data.shape, np.nan)
    trend_p_values = np.full(mean_data.shape, np.nan)

    for i in range(mean_data.shape[0]):
        for j in range(mean_data.shape[1]):
            series = mhw_data[:, i, j].values
            if np.all(np.isnan(series)) or np.sum(~np.isnan(series)) < 2:
                continue

            valid_mask = ~np.isnan(series)
            mk_mean = mk.original_test(series[valid_mask])
            mk_trend = mk.original_test(series[valid_mask])
            mean_p_values[i, j] = mk_mean.p
            trend_p_values[i, j] = mk_trend.p

            slope, _, _, _, _ = linregress(time[valid_mask], series[valid_mask])
            trend_data[i, j] = slope * 10  # Convert to per decade

    return mean_data, trend_data, mean_p_values, trend_p_values

def plot_mhw_maps(mean_data, trend_data, mean_p, trend_p, lat, lon, var_label, units, row, gs, fig):
    mean_mask = np.where(mean_p > 0.05, 1, np.nan)
    trend_mask = np.where(trend_p > 0.05, 1, np.nan)

    vmin_mean, vmax_mean = np.nanmin(mean_data), np.nanmax(mean_data)
    vmin_trend, vmax_trend = np.nanmin(trend_data), np.nanmax(trend_data)

    # Font settings
    title_fontsize = 15
    label_fontsize = 14
    tick_fontsize = 12

    # Plot Mean
    ax_mean = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
    mean_plot = ax_mean.pcolormesh(lon, lat, mean_data, cmap="plasma", vmin=vmin_mean, vmax=vmax_mean)
    ax_mean.add_feature(cfeature.COASTLINE, linewidth=1)
    ax_mean.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax_mean.set_title(f"Mean {var_label} (1982–2024)(alter)", fontsize=title_fontsize)
    lat_idx, lon_idx = np.where(mean_mask == 1)
    ax_mean.scatter(lon[lon_idx], lat[lat_idx], color='black', s=6)
    cbar = plt.colorbar(mean_plot, ax=ax_mean, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"{units}", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    ax_mean.tick_params(labelsize=tick_fontsize)

    # Plot Trend
    ax_trend = fig.add_subplot(gs[row, 1], projection=ccrs.PlateCarree())
    trend_plot = ax_trend.pcolormesh(lon, lat, trend_data, cmap="coolwarm", vmin=vmin_trend, vmax=vmax_trend)
    ax_trend.add_feature(cfeature.COASTLINE, linewidth=1)
    ax_trend.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax_trend.set_title(f"{var_label} Trend (1982–2024)(alter)", fontsize=title_fontsize)
    lat_idx, lon_idx = np.where(trend_mask == 1)
    ax_trend.scatter(lon[lon_idx], lat[lat_idx], color='black', s=6)
    cbar = plt.colorbar(trend_plot, ax=ax_trend, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"{units}/decade", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    ax_trend.tick_params(labelsize=tick_fontsize)

def main():
    file_path = 'mhw_results/mhw_stats_alterhobday34.nc'
    data = load_mhw_dataset(file_path)

    mhw_variables = {
        "MHW Duration": "mhw_duration",
        "MHW Frequency": "mhw_count",
        "MHW Intensity": "mhw_intensity"
        #"MHW Intensity": "mhw_meanintensity"
    }
    units = {
        "MHW Frequency": "Count",
        "MHW Duration": "Days",
        "MHW Intensity": "°C"
    }

    time = data['time'].values
    lat = data['lat'].values
    lon = data['lon'].values

    fig = plt.figure(figsize=(14, 13))  # Slightly larger to fit bigger fonts
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1], wspace=0.15, hspace=0.25)

    for idx, (var_label, var_name) in enumerate(mhw_variables.items()):
        mhw_data = data[var_name]
        mean_data, trend_data, mean_p, trend_p = compute_mean_and_trend(mhw_data, time)
        plot_mhw_maps(mean_data, trend_data, mean_p, trend_p, lat, lon, var_label, units[var_label], idx, gs, fig)

    plt.savefig("plots/mhwmetricsalterhobday1982_2024sig.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

