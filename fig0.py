"""
Publication figures for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
import pandas as pd
from fig1 import STYLE, TWO_COL
from tools import spicav_campaigns, trapezoid, fit_trapezoid

# %%
def figure0(ds, gap_days=21, weighted=True,
            save=False, savepath=None, savename='figure0.pdf'):
    """
    Figure 0: SPICAV SO2 time series establishing decadal variability.

    Top row: full lat-weighted SPICAV time series with error bars.
    Bottom row: one point per observing campaign (median of log10 SO2,
    with the 16th-84th percentile spread as the error bar) and a
    weighted piecewise-linear rise-fall-plateau fit in log space.
    The rise and fall durations are the fitted breakpoints, annotated
    with arrows; the rise is an upper limit because the record starts
    mid-rise.

    Args:
        ds (xr.Dataset): SPICAV observations dataset with dims
            (start_date, latitude, longitude).
        gap_days (int): Campaign-splitting gap in days.
        weighted (bool): Weight the fit by the campaign medians' standard
            errors if True.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.

    Returns:
        dict: Fitted rise/fall durations, breakpoint dates, and levels.
    """
    plt.rcParams.update(STYLE)

    obs_dates = pd.to_datetime(ds.start_date.values)
    w        = np.cos(np.deg2rad(ds.latitude))
    so2_obs  = ds.so2_ppbv.weighted(w).mean(dim=['latitude', 'longitude'])
    err_mean = ds.rel_error_pct.mean(dim=['latitude', 'longitude'])
    err_abs  = (err_mean * so2_obs) / 100.0

    camp, t0 = spicav_campaigns(ds, gap_days)
    (cost_val, p), = fit_trapezoid(camp, weighted, n_solutions=1)
    t_peak, t_plateau, y_start, y_peak, y_plateau = p

    peak_date    = t0 + pd.Timedelta(days=t_peak)
    plateau_date = t0 + pd.Timedelta(days=t_plateau)
    rise_days    = int(round(t_peak))
    fall_days    = int(round(t_plateau - t_peak))

    t_line = np.linspace(0, camp['t'].values[-1], 500)
    dates_line = t0 + pd.to_timedelta(t_line, unit='D')

    # --- Layout ---
    fig = plt.figure(figsize=(TWO_COL, 5.5))
    gs  = fig.add_gridspec(2, 1, hspace=0.20)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # --- Top: every observation ---
    ax_top.errorbar(obs_dates, so2_obs.values, yerr=err_abs.values,
                    fmt='o', markerfacecolor='firebrick', markeredgecolor='k',
                    markersize=3, markeredgewidth=0.4,
                    ecolor='k', linewidth=0.6, capsize=2, capthick=0.8,
                    zorder=3, label='SPICAV')
   # ax_top.plot(obs_dates, so2_obs.values, color='grey', linestyle='--',
   #             linewidth=1.0, zorder=2)
    ax_top.set_yscale('log')
    ax_top.set_ylabel('SO$_2$ vmr (ppbv)')
    ax_top.set_title('a) SPICAV SO$_2$ observations, latitude-weighted mean',
                     loc='left', pad=2)

    # --- Bottom: campaign medians and the fit ---
    ax_bot.vlines(camp['date'], 10 ** camp['lo'], 10 ** camp['hi'],
                  color='k', linewidth=0.6, zorder=3)
    ax_bot.plot(camp['date'], 10 ** camp['log'], 'o',
                markerfacecolor='firebrick', markeredgecolor='k',
                markersize=3, markeredgewidth=0.4, linestyle='none', zorder=4)
    ax_bot.plot(dates_line, 10 ** trapezoid(t_line, *p),
                color='k', linewidth=1.4, linestyle='dashed', zorder=5)

    ax_bot.set_yscale('log')
    ax_bot.set_ylabel('SO$_2$ vmr (ppbv)')
    ax_bot.set_xlabel('Date')
    ax_bot.set_title('b) Campaign medians with rise-fall-plateau fit',
                     loc='left', pad=2)

    # --- Rise and fall arrows ---
    trans = transforms.blended_transform_factory(ax_bot.transData,
                                                 ax_bot.transAxes)
    mid_rise = t0 + (peak_date - t0) / 2
    ax_bot.annotate('', xy=(mdates.date2num(peak_date), 0.93),
                    xytext=(mdates.date2num(t0), 0.93),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='<->', color='k', lw=0.8))
    ax_bot.text(mdates.date2num(mid_rise), 0.88, f'$\\leq$ {rise_days}\ndays',
                transform=trans, ha='center', va='top', fontsize=8,
                fontweight='bold')

    mid_fall = peak_date + (plateau_date - peak_date) / 2
    ax_bot.annotate('', xy=(mdates.date2num(plateau_date), 0.08),
                    xytext=(mdates.date2num(peak_date), 0.08),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='<->', color='k', lw=0.8))
    ax_bot.text(mdates.date2num(mid_fall), 0.13, f'{fall_days} days',
                transform=trans, ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    for ax in (ax_top, ax_bot):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.set_xlim(t0 - pd.Timedelta(days=90),
                    obs_dates[-1] + pd.Timedelta(days=90))

    if save:
        fig.savefig(savepath + savename)

    plt.show()

    return {
        'rise_days':    rise_days,
        'fall_days':    fall_days,
        'peak_date':    peak_date,
        'plateau_date': plateau_date,
        'start_ppbv':   10 ** y_start,
        'peak_ppbv':    10 ** y_peak,
        'plateau_ppbv': 10 ** y_plateau,
    }

# %%
