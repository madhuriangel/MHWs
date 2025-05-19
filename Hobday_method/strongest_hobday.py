import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

# Load dataset
##on the mhw_analysis_hobday folder all the code running and saving done
#file_path = "mhw_results/hobday1991_2020/mhw_stats_hobday30.nc"
file_path = 'mhw_results/hobday1991_2024/mhw_stats_hobday34.nc'
#file_path = "mhw_results/hobday1982_2020/mhw_stats_hobday39.nc"
ds = xr.open_dataset(file_path,decode_times=False)

# Extract variables
lat = ds['lat'].values
lon = ds['lon'].values

# Extract strongest and longest MHW event data
ev_max_intensity = ds['ev_max_max'].values
ev_max_duration = ds['ev_max_dur'].values
ev_max_start = ds['ev_max_start'].values
ev_max_end = ds['ev_max_end'].values

ev_dur_intensity = ds['ev_dur_max'].values
ev_dur_duration = ds['ev_dur_dur'].values
ev_dur_start = ds['ev_dur_start'].values
ev_dur_end = ds['ev_dur_end'].values

# Convert ordinal start and end dates to YYYY-MM-DD
convert_ordinal_to_date = np.vectorize(lambda x: datetime.fromordinal(int(x)).strftime('%Y-%m-%d') if not np.isnan(x) else np.nan)

ev_max_start_dates = convert_ordinal_to_date(ev_max_start)
ev_max_end_dates = convert_ordinal_to_date(ev_max_end)
ev_dur_start_dates = convert_ordinal_to_date(ev_dur_start)
ev_dur_end_dates = convert_ordinal_to_date(ev_dur_end)

# Flatten arrays to extract top 5 events
ev_max_flat = list(zip(ev_max_intensity.flatten(), ev_max_duration.flatten(), ev_max_start_dates.flatten(), ev_max_end_dates.flatten(), lat.repeat(len(lon)), np.tile(lon, len(lat))))
ev_dur_flat = list(zip(ev_dur_intensity.flatten(), ev_dur_duration.flatten(), ev_dur_start_dates.flatten(), ev_dur_end_dates.flatten(), lat.repeat(len(lon)), np.tile(lon, len(lat))))

# Remove NaNs
ev_max_filtered = [x for x in ev_max_flat if not np.isnan(x[0])]
ev_dur_filtered = [x for x in ev_dur_flat if not np.isnan(x[0])]

# Sort based on intensity for strongest event
ev_max_sorted = sorted(ev_max_filtered, key=lambda x: x[0], reverse=True)[:20]

# Sort based on duration for longest event
ev_dur_sorted = sorted(ev_dur_filtered, key=lambda x: x[1], reverse=True)[:20]

# Create DataFrames
df_ev_max = pd.DataFrame(ev_max_sorted, columns=['Intensity (°C)', 'Duration (days)', 'Start Date', 'End Date', 'Latitude', 'Longitude'])
df_ev_dur = pd.DataFrame(ev_dur_sorted, columns=['Intensity (°C)', 'Duration (days)', 'Start Date', 'End Date', 'Latitude', 'Longitude'])

# # Save DataFrames as CSV files
df_ev_max.to_csv("mhw_results/strongest/strongest_mhw_eventstop20_hobday_2.csv", index=False)
df_ev_dur.to_csv("mhw_results/longest/longest_mhw_eventstop20_hobday_2.csv", index=False)

# print("CSV files saved successfully:")
# print(" - strongest_mhw_events.csv")
# print(" - longest_mhw_events.csv")

# Save DataFrames as formatted TXT tables
# df_ev_max.to_string("mhw_results/strongest_mhw_events_hobday.txt", index=False)
# df_ev_dur.to_string("mhw_results/longest_mhw_events_hobday.txt", index=False)



import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

# Define font sizes
title_fontsize = 15
label_fontsize = 14
tick_fontsize = 12

# Plot the strongest and longest events on a spatial map
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Set extent for full Ireland view
ax.set_extent([-12, -4, 49, 56], crs=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
ax.set_title("Top 20 Strongest and Longest MHW Events (Hobday)", fontsize=title_fontsize)

# Plot strongest events (red bubbles)
ax.scatter(df_ev_max['Longitude'], df_ev_max['Latitude'], s=100, color='red',
           edgecolor='black', label="Strongest Events", alpha=0.8)

# Plot longest events (green bubbles)
ax.scatter(df_ev_dur['Longitude'], df_ev_dur['Latitude'], s=100, color='green',
           edgecolor='black', label="Longest Events", alpha=0.8)

# Add legend in the top-left corner
ax.legend(loc="upper left", frameon=True, fontsize=label_fontsize)

# Set labels with specified font sizes
ax.set_xlabel("Longitude (°E)", fontsize=label_fontsize)
ax.set_ylabel("Latitude (°N)", fontsize=label_fontsize)

# Set ticks and tick font size
ax.set_xticks(range(-12, -3, 2))
ax.set_yticks(range(49, 57, 2))
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Format the ticks with degree symbols
lon_formatter = mticker.FuncFormatter(lambda x, _: f"{int(x)}°E")
lat_formatter = mticker.FuncFormatter(lambda y, _: f"{int(y)}°N")

ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

# Save and show the plot
plt.savefig('plots/mhw_metrics/top20_strong_long_mhw_hobday_2.png')
plt.show()
