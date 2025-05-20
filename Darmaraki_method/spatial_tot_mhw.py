import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib.ticker as mticker

# Load the full MHW events file
df_all = pd.read_csv("mhw_results/all_mhw_events_by_grid.csv")

# Count MHW events at each grid point
event_counts = df_all.groupby(['Latitude', 'Longitude']).size().reset_index(name='Event Count')

# Create map with natural earth features and ocean background
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Set extent over Ireland
ax.set_extent([-12, -4, 49, 56], crs=ccrs.PlateCarree())



# Plot the event count with colored markers
sc = ax.scatter(
    event_counts['Longitude'] - 360,  # adjust from 0–360 to -180–180
    event_counts['Latitude'],
    c=event_counts['Event Count'],
    cmap='hot_r',
    s=80,
    edgecolor='black',
    transform=ccrs.PlateCarree(),
    zorder=1  # Draw below the land
)

#Add geographic features
# THEN: Add land/ocean on top (higher zorder)
#ax.add_feature(cfeature.OCEAN, zorder=2)
ax.add_feature(cfeature.LAND,  zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=4)
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black", zorder=4)


# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Number of MHW Events')

# Format ticks and labels
ax.set_title("MHWs Density (Darmaraki Method)", fontsize=15)
ax.set_xticks(range(-12, -3, 2))
ax.set_yticks(range(49, 57, 2))
ax.tick_params(labelsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}°E"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}°N"))

plt.tight_layout()
plt.show()
