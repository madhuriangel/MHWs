import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# Load the dataset
#file_path = 'mhw_results/hobday/mhw_stats_hobday.nc'
file_path = 'mhw_results/hobday1991_2024/mhw_stats_hobday34.nc'
ds = xr.open_dataset(file_path, decode_times=False)

#output_dir = 'mhw_results/hobday/plots'
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

def plot_spatial_variable_per_year(variable, ds, var_name, units, cmap='viridis'):
    years = ds['time'].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Define global min and max for standardized colorbar across all years
    vmin = np.nanmin(variable)
    vmax = np.nanmax(variable)

    # Create a figure for each year
    for i, year in enumerate(years):
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        cs = ax.contourf(lon, lat, variable[i, :, :], levels=np.linspace(vmin, vmax, 11), cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label(f'{var_name} ({units})')
        
        # Add map features
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        
        # Set the title and axis labels
        ax.set_title(f'{var_name} in {int(year)}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Save the plot as an image file
        output_file = os.path.join(output_dir, f'{var_name}_Year_{int(year)}.png')
        plt.savefig(output_file)
        plt.close()

def plot_mean_spatial_variable(variable, ds, var_name, units, cmap='viridis'):
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Compute the mean across the time dimension
    variable_mean = np.nanmean(variable, axis=0)
    
    # Define global min and max for standardized colorbar
    vmin = np.nanmin(variable_mean)
    vmax = np.nanmax(variable_mean)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cs = ax.contourf(lon, lat, variable_mean, levels=np.linspace(vmin, vmax, 11), cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(f'Mean {var_name} ({units})')
    
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    
    ax.set_title(f'Mean {var_name} over all years')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    output_file = os.path.join(output_dir, f'Mean_{var_name}.png')
    plt.savefig(output_file)
    plt.close()

# Function to plot trends for variables like MHW count, intensity, and duration
def plot_trend_variable(variable, ds, var_name, units, cmap):
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Define global min and max for standardized colorbar
    vmin = np.nanmin(variable)
    vmax = np.nanmax(variable)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cs = ax.contourf(lon, lat, variable, levels=np.linspace(vmin, vmax, 11), cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(f'Trend {var_name} ({units})')
    
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    
    ax.set_title(f'Trend {var_name} over all years')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    output_file = os.path.join(output_dir, f'Trend_{var_name}.png')
    plt.savefig(output_file)
    plt.close()

# Function to plot spatial statistics for strongest/longest events
def plot_spatial_variable(variable, ds, var_name, units, cmap='viridis'):
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Define global min and max for standardized colorbar
    vmin = np.nanmin(variable)
    vmax = np.nanmax(variable)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cs = ax.contourf(lon, lat, variable, levels=np.linspace(vmin, vmax, 11), cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(f'{var_name} ({units})')
    
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    
    ax.set_title(f'{var_name} Spatial Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    output_file = os.path.join(output_dir, f'{var_name}.png')
    plt.savefig(output_file)
    plt.close()

plot_spatial_variable_per_year(ds['mhw_count'], ds, 'MHW Count', 'Events', cmap='coolwarm')
plot_spatial_variable_per_year(ds['mhw_intensity'], ds, 'MHW Intensity', '°C', cmap='viridis')
plot_spatial_variable_per_year(ds['mhw_duration'], ds, 'MHW Duration', 'Days', cmap='hot')

plot_mean_spatial_variable(ds['mhw_count'], ds, 'MHW Count', 'Events', cmap='coolwarm')
plot_mean_spatial_variable(ds['mhw_intensity'], ds, 'MHW Intensity', '°C', cmap='viridis')
plot_mean_spatial_variable(ds['mhw_duration'], ds, 'MHW Duration', 'Days', cmap='hot')

plot_trend_variable(ds['mhw_count_tr'], ds, 'MHW Count Trend', 'Events/year', cmap='Blues')
plot_trend_variable(ds['mhw_intensity_tr'], ds, 'MHW Intensity Trend', '°C/year', cmap='Purples')
plot_trend_variable(ds['mhw_duration_tr'], ds, 'MHW Duration Trend', 'Days/year', cmap='Greens')

plot_spatial_variable(ds['ev_max_max'], ds, 'Max Intensity of Strongest Event', '°C', cmap='plasma')
plot_spatial_variable(ds['ev_max_dur'], ds, 'Duration of Strongest Event', 'Days', cmap='inferno')
plot_spatial_variable(ds['ev_dur_max'], ds, 'Max Intensity of Longest Event', '°C', cmap='plasma')
plot_spatial_variable(ds['ev_dur_dur'], ds, 'Duration of Longest Event', 'Days', cmap='inferno')
