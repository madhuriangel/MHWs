

#####
##This is comparison of hobday vs alter from baseline 1991_2024
#mhw_analysis_alterhob\mhw_results\combined_barplots1991_2024hobvsalt (plotsare saved here)

import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file paths
# file1_path = 'mhw_results/hobday1991_2024_res1/mhw_stats_hobday30.nc'
# file2_path = 'mhw_results/alter_hobday1991_2024/mhw_stats_alterhobday30.nc'

file1_path = 'mhw_analysis_hobday/mhw_results/hobday1991_2024/mhw_stats_hobday34.nc'
file2_path = 'mhw_analysis_alterhob/mhw_results/mhw_stats_alterhobday34.nc'

# Open the datasets (decode_times=False since time is in years)
ds1 = xr.open_dataset(file1_path, decode_times=False)
ds2 = xr.open_dataset(file2_path, decode_times=False)

# Define the output directory for figures and create it if it doesn't exist
#output_dir = 'mhw_results/combined_barplots1991_2024hobvsalt'
output_dir = 'mhw_analysis_alterhob/mhw_results/combined_barplots1991_2024hobvsalt'
os.makedirs(output_dir, exist_ok=True)

def plot_combined_variable_bar(variable1, variable2, ds1, ds2, var_name, units, label1, label2, output_dir):
    """
    Plots the comparison of yearly mean values of a specified variable from two datasets
    as side-by-side grouped bar plots, and saves the plot to the specified output directory.
    
    Parameters:
    ----------
    variable1 : np.ndarray
        Yearly mean values of the specified variable from the first dataset.
    variable2 : np.ndarray
        Yearly mean values of the specified variable from the second dataset.
    ds1 : xarray.Dataset
        The first dataset containing the variable and time coordinates.
    ds2 : xarray.Dataset
        The second dataset containing the variable and time coordinates.
    var_name : str
        Name of the variable being plotted (used for titles and labels).
    units : str
        Units of the variable (used for labeling y-axis).
    label1 : str
        Label for the first dataset (e.g., 'hobday1991_2024').
    label2 : str
        Label for the second dataset (e.g., 'hobday1991_2020').
    output_dir : str
        Directory where the plot image will be saved.
    """
    # Extract the 'time' coordinate (assumed to be years) from both datasets
    years1 = ds1['time'].values
    years2 = ds2['time'].values

    # Ensure both datasets have the same number of years
    if len(years1) != len(years2):
        raise ValueError("The two datasets must have the same number of years")
    
    # Convert to a NumPy array of integers (if needed)
    x_years = years1.astype(int)
    
    # Create x positions for each group of bars
    x = np.arange(len(x_years))
    width = 0.4  # width of each bar

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dataset 1 as bars shifted to the left
    ax.bar(x - width/2, variable1, width, label=label1, color='#ffcd92')
    
    # Plot dataset 2 as bars shifted to the right
    ax.bar(x + width/2, variable2, width, label=label2, color='#042759')
    
    # Set titles and axis labels
    ax.set_title(f'Yearly {var_name}', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{var_name} ({units})', fontsize=12)
    
    # Set x-axis tick positions and labels
    # ax.set_xticks(x)
    # ax.set_xticklabels(x_years, rotation=45, ha='right')
    
    step = 5
    tick_positions = x[::step]
    tick_labels    = x_years[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, ha='right')
    # Add legend and tight layout for better spacing
    ax.legend()
    plt.tight_layout()
    
    # Save the figure to the output directory
    output_file = os.path.join(output_dir, f'{var_name}_combined_bar.png')
    plt.savefig(output_file, dpi=150)
    plt.close()

# Compute the spatial mean over lat and lon for each variable in both datasets.
# Note: Variable names can be adjusted if needed. Here we assume:
#   - For intensity, the variable is named 'mhw_meanintensity'
#   - For duration, 'mhw_duration'
#   - For count, 'mhw_count'
mhw_intensity1 = ds1['mhw_meanintensity'].mean(dim=('lat', 'lon')).values
mhw_intensity2 = ds2['mhw_meanintensity'].mean(dim=('lat', 'lon')).values

mhw_duration1 = ds1['mhw_duration'].mean(dim=('lat', 'lon')).values
mhw_duration2 = ds2['mhw_duration'].mean(dim=('lat', 'lon')).values

mhw_count1 = ds1['mhw_count'].mean(dim=('lat', 'lon')).values
mhw_count2 = ds2['mhw_count'].mean(dim=('lat', 'lon')).values

# Plot the combined variables as grouped bar charts
plot_combined_variable_bar(mhw_intensity1, mhw_intensity2, ds1, ds2, 
                           var_name='MHW Intensity', units='°C', 
                           label1='hobday1991_2024', label2='darmaraki1991_2024', 
                           output_dir=output_dir)

plot_combined_variable_bar(mhw_duration1, mhw_duration2, ds1, ds2, 
                           var_name='MHW Duration', units='Days', 
                           label1='hobday1991_2024', label2='darmaraki1991_2024', 
                           output_dir=output_dir)

plot_combined_variable_bar(mhw_count1, mhw_count2, ds1, ds2, 
                           var_name='MHW Count', units='Events', 
                           label1='hobday1991_2024', label2='darmaraki1991_2024', 
                           output_dir=output_dir)
