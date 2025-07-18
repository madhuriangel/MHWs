import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

periods = {
    'Before MHW': ('2023-05-22', '2023-06-04'),
    'During MHW': ('2023-06-05', '2023-06-24'),
    'After MHW':  ('2023-06-25', '2023-07-08'),
}

def open_and_fix(path):
    ds = xr.open_dataset(path)
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

ds_t2m  = open_and_fix('era5/d2m_t2m_2023.nc')[['t2m','d2m']]
ds_msl  = open_and_fix('era5/msl2021_23.nc')[['msl']]
ds_flx  = open_and_fix('era5/slhf_ssr2023.nc')[['slhf','ssr']]
ds_tfl  = open_and_fix('era5/str_sshf2023.nc')[['str','sshf']]
ds_uv   = open_and_fix('era5/uv102023.nc')[['u10','v10']]

# Define font sizes
title_fontsize = 15
label_fontsize = 14
tick_fontsize = 12

ds_Qnet = (ds_flx['ssr'] + ds_tfl['str'] - ds_flx['slhf'] - ds_tfl['sshf']) \
           .to_dataset(name='Qnet')
ds_uv['wind_speed'] = np.sqrt(ds_uv['u10']**2 + ds_uv['v10']**2)

ds = xr.merge([ds_t2m, ds_msl, ds_Qnet, ds_uv])

def mean_over(ds, var, t0, t1):
    da = ds[var].sel(time=slice(t0, t1)).mean(dim='time')
    if var in ('t2m','d2m'):
        da = da - 273.15
    return da

plot_specs = {
    't2m':        dict(title='2 m air T (°C)',    cmap='coolwarm'),
    'd2m':        dict(title='2 m dew T (°C)',    cmap='viridis'),
    'msl':        dict(title='MSLP (hPa)',        cmap='plasma'),
    'Qnet':       dict(title='Net heat flux (W/m²)', cmap='inferno'),
    'wind_speed': dict(title='Wind speed (m/s)',   cmap='Blues'),
}

lon0, lon1 = -12, -5
lat0, lat1 = 49, 56

for var, spec in plot_specs.items():
    fig, axes = plt.subplots(
        1, 3, figsize=(15,5),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    for ax, (label, (t0, t1)) in zip(axes, periods.items()):
        da = mean_over(ds, var, t0, t1)
        if var == 'msl':
            da = da/100.0
        vmin, vmax = float(da.min()), float(da.max())

        pcm = ax.pcolormesh(
            da.longitude, da.latitude, da,
            cmap=spec['cmap'], vmin=vmin, vmax=vmax, shading='auto',
            transform=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        ax.set_title(f"{label} ({t0} to {t1})", fontsize=12)

        if var == 'wind_speed':
            pm = ds[['u10','v10']].sel(time=slice(t0, t1)).mean(dim='time')
            skip = (slice(None,None,5), slice(None,None,5))
            ax.quiver(
                pm.longitude[skip[1]], pm.latitude[skip[0]],
                pm.u10[skip], pm.v10[skip],
                transform=ccrs.PlateCarree(),
                scale=200,    # smaller scale → longer arrows
                width=0.005,  # thicker arrows
                color='black',
                regrid_shape=20
            )

    # Horizontal colorbar below plots
    # cbar = fig.colorbar(
    #     pcm, ax=axes, orientation='horizontal',
    #     fraction=0.03,   # reduces bar height
    #     pad=0.15         # pushes bar lower
    # )
    # cbar.set_label(spec['title'], fontsize=12)
    cbar = fig.colorbar(pcm, ax=axes, orientation='horizontal', fraction=0.08, pad=0.35)
    cbar.set_label(spec['title'], fontsize=label_fontsize, fontweight='bold')
    cbar.ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout(rect=[0, 0.18, 1, 0.95])
    plt.show()
