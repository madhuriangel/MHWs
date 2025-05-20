
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

# Define font sizes
title_fontsize = 15
label_fontsize = 14
tick_fontsize = 12

# =============================================================================
# 1. LOAD NETCDF DATA AND EXTRACT MHW EVENT INFORMATION
# =============================================================================
file_path = 'mhw_analysis_alterhob/mhw_results/mhw_stats_alterhobday34.nc'
ds = xr.open_dataset(file_path, decode_times=False)

# Extract spatial coordinates from netCDF
lat = ds['lat'].values
lon = ds['lon'].values

# Extract variables for the strongest and longest events
ev_max_intensity39 = ds['ev_max_max'].values
ev_max_duration39 = ds['ev_max_dur'].values
ev_max_start39 = ds['ev_max_start'].values
ev_max_end39 = ds['ev_max_end'].values

ev_dur_intensity39 = ds['ev_dur_max'].values
ev_dur_duration39 = ds['ev_dur_dur'].values
ev_dur_start39 = ds['ev_dur_start'].values
ev_dur_end39 = ds['ev_dur_end'].values

# Function to convert ordinal dates to 'YYYY-MM-DD'
convert_ordinal_to_date = np.vectorize(lambda x: datetime.fromordinal(int(x)).strftime('%Y-%m-%d') if not np.isnan(x) else np.nan)

ev_max_start_dates39 = convert_ordinal_to_date(ev_max_start39)
ev_max_end_dates39   = convert_ordinal_to_date(ev_max_end39)
ev_dur_start_dates39 = convert_ordinal_to_date(ev_dur_start39)
ev_dur_end_dates39   = convert_ordinal_to_date(ev_dur_end39)

# Flatten arrays to combine all grid points
ev_max_flat39 = list(zip(ev_max_intensity39.flatten(),
                         ev_max_duration39.flatten(),
                         ev_max_start_dates39.flatten(),
                         ev_max_end_dates39.flatten(),
                         lat.repeat(len(lon)),
                         np.tile(lon, len(lat))))

ev_dur_flat39 = list(zip(ev_dur_intensity39.flatten(),
                         ev_dur_duration39.flatten(),
                         ev_dur_start_dates39.flatten(),
                         ev_dur_end_dates39.flatten(),
                         lat.repeat(len(lon)),
                         np.tile(lon, len(lat))))

# Filter out events with NaN intensity
ev_max_filtered39 = [x for x in ev_max_flat39 if not np.isnan(x[0])]
ev_dur_filtered39 = [x for x in ev_dur_flat39 if not np.isnan(x[0])]

# Sort the events (sorting is optional; here we keep all events)
ev_max_sorted39 = sorted(ev_max_filtered39, key=lambda x: x[0], reverse=True)
ev_dur_sorted39 = sorted(ev_dur_filtered39, key=lambda x: x[1], reverse=True)

# Create DataFrames for strongest and longest events (using all events)
df_ev_max39 = pd.DataFrame(ev_max_sorted39, 
                           columns=['Intensity (°C)', 'Duration (days)', 'Start Date', 'End Date', 'Latitude', 'Longitude'])
df_ev_dur39 = pd.DataFrame(ev_dur_sorted39, 
                           columns=['Intensity (°C)', 'Duration (days)', 'Start Date', 'End Date', 'Latitude', 'Longitude'])

# Convert "Start Date" columns to datetime for filtering by year
df_ev_max39['Start Date'] = pd.to_datetime(df_ev_max39['Start Date'])
df_ev_dur39['Start Date'] = pd.to_datetime(df_ev_dur39['Start Date'])

# =============================================================================
# 1a. FILTER TO ONLY 2023 EVENTS (based on Start Date)
# =============================================================================
df_ev_max39_2023 = df_ev_max39[df_ev_max39['Start Date'].dt.year == 2023].copy()
df_ev_dur39_2023 = df_ev_dur39[df_ev_dur39['Start Date'].dt.year == 2023].copy()

# =============================================================================
# 2. CONVERT MHW EVENT LONGITUDES FROM 0–360 TO NEGATIVE IF > 180
# =============================================================================
def convert_longitude(lon_val):
    return lon_val - 360 if lon_val > 180 else lon_val

df_ev_max39_2023['Longitude'] = df_ev_max39_2023['Longitude'].apply(convert_longitude)
df_ev_dur39_2023['Longitude'] = df_ev_dur39_2023['Longitude'].apply(convert_longitude)

# =============================================================================
# 3. LOAD THE EARLIER CSV DATA (HABS / Toxicity Locations)
# =============================================================================
csv_df = pd.read_csv("mhw_analysis_alterhob/hab2023_d.csv")
# (Assumes CSV contains columns 'longitude' and 'latitude' for toxicity sites)

# =============================================================================
# 4. FILTER MHW EVENTS BY PROXIMITY TO HABS DATA (within 100 km)
# =============================================================================
def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth's radius in km
    return km

def filter_by_proximity(df_events, csv_df, threshold_km=100):
    filtered_events = []
    for idx, event in df_events.iterrows():
        event_lon = event['Longitude']
        event_lat = event['Latitude']
        for _, hab in csv_df.iterrows():
            d = haversine(event_lon, event_lat, hab['longitude'], hab['latitude'])
            if d <= threshold_km:
                filtered_events.append(event)
                break  # Found a close habs event, no need to check further
    return pd.DataFrame(filtered_events)

df_ev_max39_close = filter_by_proximity(df_ev_max39_2023, csv_df, threshold_km=100)
df_ev_dur39_close = filter_by_proximity(df_ev_dur39_2023, csv_df, threshold_km=100)

# Save filtered DataFrames to CSV (optional)
df_ev_max39_close.to_csv("mhw_analysis_alterhob/plots/strongest_events_2023_close100km.csv", index=False)
df_ev_dur39_close.to_csv("mhw_analysis_alterhob/plots/longest_events_2023_close100km.csv", index=False)

# =============================================================================
# 5. CREATE THE SPATIAL PLOT USING CARTOPY
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Set extent for a full Ireland view
ax.set_extent([-12, -4, 49, 56], crs=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")

# Set plot title using title_fontsize
ax.set_title("2023 MHW Events Close to Toxicity Locations", fontsize=title_fontsize)

ax.scatter(df_ev_max39_close['Longitude'], df_ev_max39_close['Latitude'], s=200,
           color='#DCFFF6', edgecolor='black', label="Strongest MHW Events", alpha=0.8, zorder=101)

# Longest events (green) – uncomment to plot if needed
ax.scatter(df_ev_dur39_close['Longitude'], df_ev_dur39_close['Latitude'], s=100,
           color="#005E63", edgecolor='black', label="Longest MHW Events", alpha=0.8, zorder=101)

# Plot toxicity locations (blue)
ax.scatter(csv_df['longitude'], csv_df['latitude'], s=100,
           color="#FF6600", edgecolor='black', label="Toxicity Location", alpha=0.7, zorder=101)

# Add legend
ax.legend(loc="upper left", frameon=True, fontsize=tick_fontsize)

# Set axis labels using label_fontsize
ax.set_xlabel("Longitude (°E)", fontsize=label_fontsize)
ax.set_ylabel("Latitude (°N)", fontsize=label_fontsize)

# Configure tick marks with degree symbols and tick_fontsize
ax.set_xticks(range(-12, -3, 2))
ax.set_yticks(range(49, 57, 2))
lon_formatter = mticker.FuncFormatter(lambda x, _: f"{int(x)}°E")
lat_formatter = mticker.FuncFormatter(lambda y, _: f"{int(y)}°N")
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(axis='both', labelsize=tick_fontsize)

plt.savefig('mhw_analysis_alterhob/plots/stronglong_tox2023.png')
plt.show()

# (Optional) Process total events data, converting zeros to NaN
tot_event = ds['mhw_total_events'].values
tot_event[tot_event == 0] = np.nan
