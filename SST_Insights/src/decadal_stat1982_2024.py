import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(file_path):
    """
    Load SST dataset from NetCDF and assign proper datetime coordinates.

    Parameters
    ----------
    file_path : str
        Path to the SST NetCDF file.

    Returns
    -------
    xr.Dataset
        Dataset with 'time' coordinate parsed as datetime.
    """
    ds = xr.open_dataset(file_path, decode_times=False)
    start = pd.Timestamp("1982-01-01")
    times = pd.date_range(start=start, periods=ds.dims["time"], freq="D")
    return ds.assign_coords(time=("time", times))


def define_periods():
    """
    Define decadal periods for analysis.

    Returns
    -------
    dict
        Mapping of period labels to time slices.
    """
    return {
        "D1 (1982-1992)": slice("1982-01-01", "1992-12-31"),
        "D2 (1993-2003)": slice("1993-01-01", "2003-12-31"),
        "D3 (2004-2013)": slice("2004-01-01", "2013-12-31"),
        "D4 (2014-2024)": slice("2014-01-01", "2024-12-31"),
    }


def calculate_monthly_stats(ds, periods):
    """
    Compute monthly mean SST and overall stats for each period.

    Parameters
    ----------
    ds : xr.Dataset
        SST dataset with time, lat, lon dimensions.
    periods : dict
        Period name to time slice mapping.

    Returns
    -------
    monthly_means : dict
        Period to list of 12 monthly mean SST values.
    period_stats : dict
        Summary stats (mean, min, max SST) per period.
    """
    monthly_means = {}
    period_stats = {}

    for label, tr in periods.items():
        subset = ds.sel(time=tr)
        monthly_avg = (
            subset['sst']
            .groupby('time.month')
            .mean(dim=['time', 'lat', 'lon'])
            .values.tolist()
        )
        monthly_means[label] = monthly_avg

        overall_mean = round(float(subset['sst'].mean(dim=['time', 'lat', 'lon']).item()), 2)
        overall_min = round(float(subset['sst'].min(dim=['time', 'lat', 'lon']).item()), 2)
        overall_max = round(float(subset['sst'].max(dim=['time', 'lat', 'lon']).item()), 2)

        period_stats[label] = {
            'Mean SST [°C]': overall_mean,
            'Min SST [°C]': overall_min,
            'Max SST [°C]': overall_max,
        }

    return monthly_means, period_stats


def plot_monthly_means(monthly_means, output_path):
    """
    Plot decadal monthly SST means and save figure.

    Parameters
    ----------
    monthly_means : dict
        Period to monthly mean values.
    output_path : str
        Path to save PNG.
    """
    months = list(range(1, 13))
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = {
        'D1 (1982-1992)': 'purple',
        'D2 (1993-2003)': 'orange',
        'D3 (2004-2013)': 'blue',
        'D4 (2014-2024)': 'green',
    }

    plt.figure(figsize=(10, 6))
    for label, values in monthly_means.items():
        plt.plot(months, values, label=label, color=colors[label])

    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('SST [°C]', fontsize=14, fontweight='bold')
    plt.title('Decadal Change of Area-Averaged SST', fontsize=16, fontweight='bold')
    plt.xticks(months, labels, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(title='Periods', fontsize=12, title_fontsize=12, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    """
    Main function to execute full-year SST analysis: calculates stats and plots trends.
    """
    #file_path = 'Data_noaa_copernicus/noaa_avhrr/noaa_combined_1982_2024_1res.nc'
    #output_figure = "sst_analysis_code/plots/decadal_evolution_sst_AllMonths.png"
    # file_path = 'noaaonly1982_2024_1.nc'
    # output_figure = "sst_analysis_code/plots/NOAAdecadal_evolution_sst_AllMonths.png"
    file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'
    ds = load_dataset(file_path)
    periods = define_periods()
    monthly_means, stats = calculate_monthly_stats(ds, periods)

    # Print stats
    stats_df = pd.DataFrame(stats).T
    print('Basic Statistics for Each Period:')
    print(stats_df)

    # Plot
    plot_monthly_means(
        monthly_means,
        'sst_analysis_code/plots/decadal_evolution_sst_AllMonths.png'
    )
