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
    start_date = pd.Timestamp("1982-01-01")
    times = pd.date_range(start=start_date, periods=ds.dims['time'], freq='D')
    return ds.assign_coords(time=('time', times))


def define_periods():
    """
    Define decadal periods for JJAS SST analysis.

    Returns
    -------
    dict
        Mapping of period labels to time slices.
    """
    return {
        'D1 (1982-1992)': slice('1982-01-01', '1992-12-31'),
        'D2 (1993-2003)': slice('1993-01-01', '2003-12-31'),
        'D3 (2004-2013)': slice('2004-01-01', '2013-12-31'),
        'D4 (2014-2024)': slice('2014-01-01', '2024-12-31'),
    }


def calculate_jjas_stats(ds, periods):
    """
    Compute monthly JJAS mean SST and summary statistics for each period.

    Parameters
    ----------
    ds : xr.Dataset
        SST dataset with time, lat, lon dimensions.
    periods : dict
        Period name to time slice mapping.

    Returns
    -------
    monthly_means : dict
        Period to array of JJAS monthly mean SST values.
    period_stats : dict
        Summary stats (mean, min, max SST) per period.
    """
    monthly_means = {}
    period_stats = {}

    for label, tr in periods.items():
        subset = ds.sel(time=tr)
        jjas = subset.sel(time=subset['time.month'].isin([6, 7, 8, 9]))

        # Monthly mean per JJAS month
        mean_vals = (
            jjas['sst']
            .groupby('time.month')
            .mean(dim=['time', 'lat', 'lon'])
            .values
        )
        monthly_means[label] = mean_vals

        # Overall JJAS statistics
        mean_val = jjas['sst'].mean(dim=['time', 'lat', 'lon']).item()
        min_val = jjas['sst'].min(dim=['time', 'lat', 'lon']).item()
        max_val = jjas['sst'].max(dim=['time', 'lat', 'lon']).item()

        period_stats[label] = {
            'Mean SST [°C]': round(mean_val, 2),
            'Min SST [°C]': round(min_val, 2),
            'Max SST [°C]': round(max_val, 2)
        }

    return monthly_means, period_stats


def plot_jjas_monthly_means(monthly_means, output_path):
    """
    Plot JJAS monthly mean SST for each period and save figure.

    Parameters
    ----------
    monthly_means : dict
        Period to JJAS monthly mean SST arrays.
    output_path : str
        Path to save the plot as a PNG.
    """
    months = [6, 7, 8, 9]
    labels = ['Jun', 'Jul', 'Aug', 'Sep']
    colors = {
        'D1 (1982-1992)': 'purple',
        'D2 (1993-2003)': 'orange',
        'D3 (2004-2013)': 'blue',
        'D4 (2014-2024)': 'green'
    }

    plt.figure(figsize=(10, 6))
    for label, vals in monthly_means.items():
        plt.plot(months, vals, label=label, color=colors.get(label))

    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('SST [°C]', fontsize=14, fontweight='bold')
    plt.title('Decadal JJAS SST Evolution', fontsize=16, fontweight='bold')
    plt.xticks(months, labels, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(title='Periods', fontsize=12, title_fontsize=12, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    file_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'
    ds = load_dataset(file_path)
    periods = define_periods()
    monthly_means, stats = calculate_jjas_stats(ds, periods)

    stats_df = pd.DataFrame(stats).T
    print('JJAS Period Statistics:')
    print(stats_df)

    plot_jjas_monthly_means(
        monthly_means,
        'sst_analysis_code/plots/decadal_evolution_sst_JJAS.png'
    )
