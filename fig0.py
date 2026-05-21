"""
Publication figures for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
from scipy.interpolate import CubicSpline

from fig1 import STYLE, TWO_COL

# %%
def figure0(ds, rolling_window=240, start=0, end=None,
            save=False, savepath=None, savename='figure0.pdf'):
    """
    Figure 0: SPICAV SO2 time series establishing decadal variability.

    Top row: full lat-weighted SPICAV time series with error bars and a
    cubic spline connecting the observations. Bottom row: rolling mean
    with standard error envelope.

    Args:
        ds (xr.Dataset): SPICAV observations dataset with dims
            (start_date, latitude, longitude).
        rolling_window (int): Rolling mean window in days.
        start (int): Start index into ds.start_date.
        end (int): End index into ds.start_date. Defaults to all observations.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    ds_sl     = ds.isel(start_date=slice(start, end))
    obs_dates = pd.to_datetime(ds_sl.start_date.values)

    # Lat-weighted mean SO2 and absolute error
    w        = np.cos(np.deg2rad(ds_sl.latitude))
    so2_obs  = ds_sl.so2_ppbv.weighted(w).mean(dim=['latitude', 'longitude'])
    err_mean = ds_sl.rel_error_pct.mean(dim=['latitude', 'longitude'])
    err_abs  = (err_mean * so2_obs) / 100.0

    so2_vals = so2_obs.values
    err_vals = err_abs.values

    # Rolling mean and standard error, time-based window in days
    smoothed      = ds_sl.so2_ppbv.weighted(w).mean(dim=['latitude', 'longitude']).rolling(
                        start_date=rolling_window, center=False,
                        min_periods=rolling_window // 4).mean()
    roll          = so2_obs.rolling(start_date=rolling_window, center=True,
                                    min_periods=rolling_window // 4)
    smoothed_err  = roll.std() / np.sqrt(roll.count())

    # --- Layout ---
    fig = plt.figure(figsize=(TWO_COL, 5.5))
    gs  = fig.add_gridspec(2, 1, hspace=0.20)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # --- Top: scatter + spline connecting the dots ---
    ax_top.errorbar(obs_dates, so2_vals, yerr=err_vals,
                    fmt='o', markerfacecolor='firebrick', markeredgecolor='k',
                    markersize=3, markeredgewidth=0.4,
                    ecolor='k', linewidth=0.6, capsize=2, capthick=0.8,
                    zorder=3, label='SPICAV')
    ax_top.plot(obs_dates, so2_vals, color='grey', linestyle='--', linewidth=1.0)

    ax_top.set_yscale('log')
    ax_top.set_ylabel('SO$_2$ vmr (ppbv)')
    ax_top.set_title('a)', loc='left', pad=2)
    ax_top.spines['top'].set_visible(True)
    ax_top.spines['right'].set_visible(True)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax_top.xaxis.set_major_locator(locator)
    ax_top.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_top.xaxis.set_minor_locator(mdates.MonthLocator())

    # --- Bottom: rolling mean with error bars ---
    ax_bot.errorbar(ds_sl.start_date.values, smoothed.values, yerr=smoothed_err.values,
                    fmt='o', markerfacecolor='firebrick', markeredgecolor='k',
                    markersize=3, markeredgewidth=0.4, linestyle='-',
                    ecolor='k', linewidth=0.8, capsize=2, capthick=0.8,
                    color='firebrick')

    ax_bot.set_ylabel('SO$_2$ vmr (ppbv)')
    ax_bot.set_xlabel('Date')
    ax_bot.set_title(f'b) {rolling_window}-day rolling mean', loc='left', pad=2)
    ax_bot.spines['top'].set_visible(True)
    ax_bot.spines['right'].set_visible(True)

    locator2 = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax_bot.xaxis.set_major_locator(locator2)
    ax_bot.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator2))
    ax_bot.xaxis.set_minor_locator(mdates.MonthLocator())

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
