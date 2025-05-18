import xarray as xr
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from typing import Tuple


def load_sst(data_path: str) -> xr.DataArray:
    """
    Load sea surface temperature (SST) data from a NetCDF file.

    Parameters
    ----------
    data_path : str
        Path to the NetCDF dataset.

    Returns
    -------
    xr.DataArray
        SST data array.
    """
    ds = xr.open_dataset(data_path)
    return ds['sst']


def compute_mean_sst(sst_data: xr.DataArray) -> pd.Series:
    """
    Aggregate SST over latitude and longitude into a daily mean series.

    Parameters
    ----------
    sst_data : xr.DataArray
        SST data array with dimensions ('time', 'lat', 'lon').

    Returns
    -------
    pd.Series
        Daily mean SST indexed by datetime.
    """
    mean_da = sst_data.mean(dim=['lat', 'lon'])
    df = mean_da.to_dataframe(name='SST').dropna()
    df.index = pd.to_datetime(df.index)
    return df['SST']


def decompose_sst(sst_series: pd.Series, period: int = 365) -> seasonal_decompose:
    """
    Perform additive seasonal decomposition on the SST series.

    Parameters
    ----------
    sst_series : pd.Series
        Daily mean SST series.
    period : int
        Seasonal period in days (default is 365).

    Returns
    -------
    DecomposeResult
        Object containing observed, trend, seasonal, and residual components.
    """
    return seasonal_decompose(sst_series, model='additive', period=period)


def plot_decomposition(result: seasonal_decompose, output_path: str) -> None:
    """
    Plot and save the components of an SST seasonal decomposition.

    Parameters
    ----------
    result : DecomposeResult
        Output from `decompose_sst` containing decomposition components.
    output_path : str
        File path for saving the resulting plot.
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Seasonal Decomposition of SST", y=1.02)

    labels = ['Observed', 'Trend', 'Seasonal', 'Residual']
    comps = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, comp, label in zip(axs, comps, labels):
        ax.plot(comp, color='#042759')
        ax.set_ylabel(label)
    axs[-1].set_xlabel("Time")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    plt.show()


if __name__ == "__main__":
    # Example usage
    data_path = 'noaa_icesmi_combinefile_FINAL_1res1982_2024.nc'
    output_fig = 'sst_analysis_code/plots/seasonal_decompose.png'

    sst = load_sst(data_path)
    mean_series = compute_mean_sst(sst)
    result = decompose_sst(mean_series)
    plot_decomposition(result, output_fig)
