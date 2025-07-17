import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import alternew_hobday1 as mhw   # your detect() implementation

SST_FILE   = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"
CLIM_YEARS = [1991, 2024]
LAT0, LON0 = 55.625, 355.375    # 0–360° longitude
PCTILE     = 99               
PLOT_START = "2021-07-01"
PLOT_END   = "2021-08-20"

ds = xr.open_dataset(SST_FILE)
if ds.lon.max() > 180:
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)

lon_wrapped = LON0 - 360 if LON0 > 180 else LON0
sst = ds.sst.sel(lat=LAT0, lon=lon_wrapped, method="nearest")
times = pd.to_datetime(sst.time.values)
temps = sst.values

ordinals = np.array([t.date().toordinal() for t in times])
mhw_dict, clim = mhw.detect(
    ordinals, temps,
    climatologyPeriod=CLIM_YEARS,
    pctile=PCTILE,   
    Ly=False
)

df_ev = pd.DataFrame({
    "start":   mhw_dict["date_start"],
    "end":     mhw_dict["date_end"],
    "dur":     mhw_dict["duration"],
    "Imax":    mhw_dict["intensity_max"],
    "Imean":   mhw_dict["intensity_mean"],
    "Icum":    mhw_dict["intensity_cumulative"]
})
df_ev["start"] = pd.to_datetime(df_ev["start"])
df_ev["end"]   = pd.to_datetime(df_ev["end"])
df_ev["year"]  = df_ev["start"].dt.year

# Filter for 2021, pick the *longest* event
df21 = df_ev[df_ev.year == 2021]
if df21.empty:
    raise RuntimeError("No 2021 events found!")
best = df21.loc[df21.dur.idxmax()]

evt_start = best.start
evt_end   = best.end
Dur        = int(best.dur)
Imax       = best.Imax
Imean      = best.Imean
Icum       = best.Icum

print(f"2021 longest event: {evt_start.date()} → {evt_end.date()}  dur={Dur} d  Imax={Imax:.2f}°C")

df = pd.DataFrame({
    "SST":    temps,
    "Clim":   clim["seas"],
    "Thresh": clim["thresh"]
}, index=times).loc[PLOT_START:PLOT_END]

mask_evt = (df.index >= evt_start) & (df.index <= evt_end)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(df.index, df.Clim,   color="blue",      lw=2, label="Climatology")
ax.plot(df.index, df.Thresh, color="orange",    lw=2, label=f"{PCTILE}th percentile")
ax.plot(df.index, df.SST,    color="green",     lw=1.5, label="SST")

ax.fill_between(df.index, df.Thresh, df.SST,
                where=mask_evt, color="red", alpha=0.4, label="MHW event")

# dashed verticals at start/end
ax.axvline(evt_start, color="k", ls="--")
ax.axvline(evt_end,   color="k", ls="--")

# annotate the peak SST during the event
peak_idx = df.SST[mask_evt].idxmax()
peak_val = df.SST.loc[peak_idx]
ax.annotate(f"Peak {peak_val:.2f}°C",
            xy=(peak_idx, peak_val),
            xytext=(14,2),
            textcoords="offset points",
            arrowprops=dict(color="black"))


# Title & labels
ax.set_title(
    f"MHW Event at {LAT0:.3f}°N, {(-360+LON0 if LON0>180 else LON0):.3f}°E  "
    f"[{evt_start.date()} → {evt_end.date()}]  dur={Dur} d, Iₘₐₓ={Imax:.2f}°C",
    fontsize=14
)
ax.set_xlim(pd.to_datetime(PLOT_START), pd.to_datetime(PLOT_END))
ax.set_xlabel("Date")
ax.set_ylabel("SST (°C)")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.tick_params(rotation=30)
ax.grid(True, ls="--", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
