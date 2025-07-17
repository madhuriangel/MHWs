import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymannkendall as mk
from scipy.stats import linregress
from matplotlib import gridspec

# Paths to your two datasets
DATASETS = {
    'Hobday':   'mhw_analysis_hobday/mhw_results/hobday1991_2024/mhw_stats_hobday34.nc',
    'Darmaraki':'mhw_analysis_alterhob/mhw_results/mhw_stats_alterhobday34.nc'
}

METRICS = {
    'Duration':  ('mhw_duration', 'Days'),
    'Frequency': ('mhw_count',    'Count'),
    'Intensity': ('mhw_intensity','°C')
}

# Minimum years to trust a trend (only applied to Duration & Intensity)
MIN_EVENTS = 5

def load_dataset(path):
    return xr.open_dataset(path, decode_times=False)

def compute_mean_and_trend(da, time):
    """
    Returns 2D arrays: mean, trend, p_mean, p_trend
    """
    mean_map   = da.mean(dim='time').values
    trend_map  = np.full_like(mean_map, np.nan, dtype=float)
    p_mean_map = np.full_like(mean_map, np.nan, dtype=float)
    p_tr_map   = np.full_like(mean_map, np.nan, dtype=float)

    # loop over grid
    for i in range(mean_map.shape[0]):
        for j in range(mean_map.shape[1]):
            series = da[:, i, j].values
            valid  = ~np.isnan(series)
            if valid.sum() < 2:
                continue

            # MK test for p-values
            p_mean_map[i,j] = mk.original_test(series[valid]).p
            p_tr_map[i,j]   = mk.original_test(series[valid]).p

            # OLS slope *10 -> per decade
            slope, *_ = linregress(time[valid], series[valid])
            trend_map[i,j] = slope * 10

    return mean_map, trend_map, p_mean_map, p_tr_map

# 1) Load & compute for both methods & all metrics
results = {}
for method, path in DATASETS.items():
    ds = load_dataset(path)
    t201 = ds['time'].values
    results[method] = {}
    for met, (var, unit) in METRICS.items():
        da = ds[var]
        m, tr, pm, ptr = compute_mean_and_trend(da, t201)

        # apply event‐count mask for Duration & Intensity only
        if met in ('Duration','Intensity'):
            counts = da.notnull().sum(dim='time').values
            tr = np.where(counts >= MIN_EVENTS, tr, np.nan)

        results[method][met] = {
            'mean': m, 'trend': tr,
            'p_mean': pm, 'p_tr': ptr,
            'unit': unit
        }
    # grab coords
    lat = ds['lat'].values
    lon = ds['lon'].values

# 2) Find global vmin/vmax across both methods for each metric
vlims = {}
for met in METRICS:
    all_means  = np.hstack([results[m][met]['mean'].ravel()  for m in DATASETS])
    all_trends = np.hstack([results[m][met]['trend'].ravel() for m in DATASETS])
    vlims[met] = {
        'mean':  (np.nanmin(all_means),  np.nanmax(all_means)),
        'trend': (np.nanmin(all_trends), np.nanmax(all_trends))
    }
    
# 4) Identify significant increases for each method & metric
####
# significant positive‐trend cells/total grid‐cells×100%.
# sig_increases = {}

# for method in results:
#     sig_increases[method] = {}
#     for met in METRICS:
#         tr = results[method][met]['trend']
#         p  = results[method][met]['p_tr']

#         # mask: only keep cells with a positive trend AND p < 0.05
#         sig_inc = np.where((tr > 0) & (p < 0.05), tr, np.nan)
#         sig_increases[method][met] = sig_inc

#         # quick stats
#         total   = tr.size               # <— use .size instead of np.product
#         n_sig   = np.count_nonzero(~np.isnan(sig_inc))
#         frac_sig = n_sig / total * 100
#         print(f"{method:>9} {met:>9}: {n_sig} cells ({frac_sig:.1f} %) with significant increase")
        
sig_increases = {}

for method in results:
    sig_increases[method] = {}
    for met in METRICS:
        tr   = results[method][met]['trend']
        p    = results[method][met]['p_tr']
        unit = results[method][met]['unit']

        # mask: only keep cells with a positive trend AND p < 0.05
        sig_inc = np.where((tr > 0) & (p < 0.05), tr, np.nan)
        sig_increases[method][met] = sig_inc

        # quick stats
        total   = tr.size               # total grid‑cells
        n_sig   = np.count_nonzero(~np.isnan(sig_inc))
        frac_sig = n_sig / total * 100
        print(f"{method:>9} {met:>9}: {n_sig} cells ({frac_sig:.1f} %) with significant increase")

        # mean significant trend ----
        if n_sig > 0:
            mean_sig = np.nanmean(sig_inc)
            print(f"           → Mean significant {met.lower()} trend: {mean_sig:.2f} {unit}/decade")
        else:
            print(f"           → No significant {met.lower()}‑trend cells to average.")
        

# 3) Plot one figure per method, without a global title
for method in DATASETS:
    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1],hspace=0.25, wspace=0.15)

    for i, met in enumerate(METRICS):
        dat = results[method][met]
        vmin_m, vmax_m = vlims[met]['mean']
        vmin_t, vmax_t = vlims[met]['trend']
        unit = dat['unit']

        # subplot: mean
        axm = fig.add_subplot(gs[i,0], projection=ccrs.PlateCarree())
        pcm = axm.pcolormesh(lon, lat, dat['mean'], cmap='plasma',
                             vmin=vmin_m, vmax=vmax_m)
        axm.add_feature(cfeature.LAND, edgecolor='k', zorder=100)
        axm.add_feature(cfeature.COASTLINE)
        axm.set_title(f"Mean MHW {met}:{method} ", fontsize=14)
        #axm.set_title(f"{method}: Mean MHW {met}", fontsize=14)

        # stipple non‐significant
        yi, xi = np.where(dat['p_mean'] > 0.05)
        axm.scatter(lon[xi], lat[yi], s=6, color='k')

        cbm = plt.colorbar(pcm, ax=axm, orientation='vertical',shrink=0.8, pad=0.02)
        cbm.set_label(unit, fontsize=12)

        # subplot: trend
        axtr = fig.add_subplot(gs[i,1], projection=ccrs.PlateCarree())
        pct = axtr.pcolormesh(lon, lat, dat['trend'], cmap='coolwarm',
                              vmin=vmin_t, vmax=vmax_t)
        axtr.add_feature(cfeature.LAND, edgecolor='k', zorder=100)
        axtr.add_feature(cfeature.COASTLINE)
        axtr.set_title(f"MHW {met} Trend:{method}", fontsize=14)

        # stipple non‐significant
        yi, xi = np.where(dat['p_tr'] > 0.05)
        axtr.scatter(lon[xi], lat[yi], s=6, color='k')

        cbtr = plt.colorbar(pct, ax=axtr, orientation='vertical',shrink=0.8, pad=0.02)
        cbtr.set_label(f"{unit}/decade", fontsize=12)

    plt.show()
