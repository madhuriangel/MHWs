import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------
# 1. LOAD & PREPARE THE DATA
# -----------------------------------------------------------------------------
# MHW events (Jan, July–September 2023)
events = pd.read_csv(
    "mhw_analysis_alterhob/mhw_results/events_2023_jan_jun_sept.csv",
    parse_dates=["Start Date", "End Date"],
    dayfirst=True
)

# Toxicity sites (already in 0–360 lon format)
tox = pd.read_csv(
    "mhw_analysis_alterhob/hab2023_conv.csv",
    parse_dates=["periodstart_date", "periodend_date"],
    dayfirst=True
).rename(columns={
    "periodstart_date": "Start Date",
    "periodend_date":   "End Date"
})

# (No need to convert longitudes—already 0–360)

# -----------------------------------------------------------------------------
# 2. SELECT TOP 500 EVENTS AND SAVE THEM
# -----------------------------------------------------------------------------
top_strong = events.nlargest(500, "Max Intensity (°C)")
top_long   = events.nlargest(500, "Duration (days)")

top_strong.to_csv("mhw_analysis_alterhob/mhw_results/top500_strongest_2023_jnjul_sep.csv", index=False)
top_long.to_csv("mhw_analysis_alterhob/mhw_results/top500_longest_2023_janjul_sep.csv",   index=False)

# -----------------------------------------------------------------------------
# 3. SPATIAL PLOT WITH CARTOPY
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(10, 8),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# — Map extent around Ireland
ax.set_extent([-12, -4, 49, 56], crs=ccrs.PlateCarree())

# — Background layers
ax.add_feature(cfeature.OCEAN,      facecolor="lightblue", zorder=0)
ax.add_feature(cfeature.LAND, zorder=1)
ax.add_feature(cfeature.COASTLINE,  linewidth=1,           zorder=2)
ax.add_feature(cfeature.BORDERS,    linestyle=":",         zorder=2)

# — Plot top 500 strongest events
ax.scatter(
    top_strong["Longitude"], top_strong["Latitude"],
    s=80, marker="o", facecolor="white", edgecolor="darkred",
    linewidth=1, alpha=0.8, label="Strongest", zorder=3
)

# — Plot top 500 longest events
ax.scatter(
    top_long["Longitude"], top_long["Latitude"],
    s=80, marker="s", facecolor="white", edgecolor="darkgreen",
    linewidth=1, alpha=0.8, label="Longest", zorder=4
)

# — Plot toxicity sites
ax.scatter(
    tox["Longitude"], tox["Latitude"],
    s=100, marker="^", facecolor="orange", edgecolor="black",
    alpha=0.9, label="Toxicity Sites", zorder=5
)

# — Title, labels, legend
ax.set_title(
    "2023 Jan, July–Oct: Strongest & Longest MHW & Toxicity Sites",
    fontsize=15
)
ax.set_xlabel("Longitude (°E)", fontsize=14)
ax.set_ylabel("Latitude (°N)",  fontsize=14)
ax.legend(loc="upper left", fontsize=12, frameon=True)

# — Ticks in degrees, no gridliner (avoid unsupported xinline/yinline)
ax.set_xticks(range(-12, -3, 2))
ax.set_yticks(range(49, 57, 2))
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x)}°E")
)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda y, _: f"{int(y)}°N")
)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("mhw_analysis_alterhob/mhw_results/spatial_top500_2023_janjul_sep.png", dpi=300)
plt.show()
