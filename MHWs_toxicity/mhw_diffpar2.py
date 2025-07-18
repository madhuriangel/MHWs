import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define analysis windows and MHW periods
windows = {
    '2021': dict(win_start='2021-07-04', win_end='2021-08-18',
                 mhw_start='2021-07-18', mhw_end='2021-08-02'),
    '2023': dict(win_start='2023-05-22', win_end='2023-07-08',
                 mhw_start='2023-06-05', mhw_end='2023-06-24')
}

# File paths for each variable group
files = {
    't2m_d2m': {'2021':'era5/d2m_t2m_2021.nc', 
                '2023':'era5/d2m_t2m_2023.nc'},
    'msl':     {'all':'era5/msl2021_23.nc'},
    'slhf_ssr':{'2021':'era5/slhf_ssr2021.nc',
                '2023':'era5/slhf_ssr2023.nc'},
    'str_sshf':{'2021':'era5/str_sshf2021.nc',
                '2023':'era5/str_sshf2023.nc'},
    'uv10':    {'2021':'era5/uv102021.nc',
                '2023':'era5/uv102023.nc'}
}

def open_and_fix(path):
    ds = xr.open_dataset(path)
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

def daily_means(path, win):
    ds = open_and_fix(path).sel(time=slice(win['win_start'], win['win_end']))
    return ds.resample(time='1D').mean()

ds_daily = {}
for yr in ['2021','2023']:
    w = windows[yr]
    parts = [
        daily_means(files['t2m_d2m'][yr], w),    # 't2m', 'd2m'
        daily_means(files['msl']['all'], w),     # 'msl'
        daily_means(files['slhf_ssr'][yr], w),   # 'slhf', 'ssr'
        daily_means(files['str_sshf'][yr], w),   # 'str', 'sshf'
        daily_means(files['uv10'][yr], w)        # 'u10', 'v10'
    ]
    ds = xr.merge(parts)

    # compute wind speed
    ds['wind_speed'] = np.sqrt(ds.u10**2 + ds.v10**2)

    # compute Qnet = ssr + str − slhf − sshf
    ds['Qnet'] = ds.ssr + ds.str - ds.slhf - ds.sshf

    ds_daily[yr] = ds

# Define font sizes
title_fontsize = 15
label_fontsize = 14
tick_fontsize = 12

for yr, ds in ds_daily.items():
    w = windows[yr]
    vars_to_plot = ['t2m', 'd2m', 'msl', 'Qnet', 'wind_speed']
    units = {
        't2m': '°C',
        'd2m': '°C',
        'msl': 'Pa',
        'Qnet': 'W/m²',
        'wind_speed': 'm/s'
    }

    fig, axes = plt.subplots(len(vars_to_plot), 1,
                             figsize=(12, 2.5 * len(vars_to_plot)),
                             sharex=True)

    for ax, var in zip(axes, vars_to_plot):
        da_mean = ds[var].mean(dim=['latitude', 'longitude'])
        da_std = ds[var].std(dim=['latitude', 'longitude'])

        # Convert Kelvin to Celsius
        if var in ['t2m', 'd2m']:
            da_mean -= 273.15

        #da_mean.plot(ax=ax, linewidth=1.5, label='Mean')
        da_mean.plot(ax=ax, linewidth=1.5, label='', add_legend=False)
        ax.set_title("") 


        # Plot shaded std dev band
        # ax.fill_between(da_mean['time'].values,
        #                 da_mean - da_std,
        #                 da_mean + da_std,
        #                 alpha=0.3, color='gray', label='±1 STD')

        peak_idx = da_mean.argmax().item()
        peak_time = da_mean['time'].values[peak_idx]
        peak_value = da_mean.values[peak_idx]
        ax.annotate(f'{peak_value:.2f} {units[var]}',
                    xy=(peak_time, peak_value),
                    xytext=(12, 12),
                    textcoords='offset points',
                    fontsize=12,
                    arrowprops=dict(arrowstyle='->', color='black'))

        ax.set_ylabel(f"{var} ({units[var]})", fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Shade MHW periods
        ax.axvspan(np.datetime64(w['win_start']), np.datetime64(w['mhw_start']),
                   alpha=0.2, color='lightgray', label='Before MHW')
        ax.axvspan(np.datetime64(w['mhw_start']), np.datetime64(w['mhw_end']),
                   alpha=0.4, color='red', label='During MHW')
        ax.axvspan(np.datetime64(w['mhw_end']), np.datetime64(w['win_end']),
                   alpha=0.2, color='lightgray', label='After MHW')

    axes[-1].set_xlabel('Date', fontsize=label_fontsize)
    axes[-1].tick_params(axis='x', labelsize=tick_fontsize)
    fig.autofmt_xdate(rotation=30)
    plt.suptitle(f"MHW window {yr}: before, during, after", fontsize=title_fontsize)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
