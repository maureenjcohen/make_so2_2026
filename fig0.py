"""
Publication figures for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms
import pandas as pd
from fig1 import STYLE, TWO_COL

# %%
def figure0(ds, rolling_window=240, start=0, end=None,
            peak_search_before=None, trough_threshold=None, slope_offset=0,
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
        peak_search_before (str): If given (e.g. '2008-01-01'), draws a slope
            line from the first valid rolling-mean point to the peak before
            this date, annotated with the duration.
        trough_threshold (float): If given, draws a second slope line from the
            peak to the first point after the peak where the rolling mean falls
            below this value (ppbv), annotated with the duration.
        slope_offset (float): Subtract this value (ppbv) from both slope lines
            so they don't sit on top of the data points.
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

    # Rolling mean and standard error, point-based window (xarray)
    smoothed     = so2_obs.rolling(start_date=rolling_window, center=True,
                                   min_periods=rolling_window // 4).mean()
    roll         = so2_obs.rolling(start_date=rolling_window, center=True,
                                   min_periods=rolling_window // 4)
    smoothed_err = roll.std() / np.sqrt(roll.count())

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
    ax_top.set_title('a) SPICAV SO$_2$ observations', loc='left', pad=2)
    ax_top.spines['top'].set_visible(True)
    ax_top.spines['right'].set_visible(True)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax_top.xaxis.set_major_locator(locator)
    ax_top.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_top.xaxis.set_minor_locator(mdates.MonthLocator())

    # --- Bottom: rolling mean with error bars ---
    ax_bot.errorbar(obs_dates, smoothed.values, yerr=smoothed_err.values,
                    fmt='o', markerfacecolor='firebrick', markeredgecolor='k',
                    markersize=3, markeredgewidth=0.4, linestyle='--', color='grey',
                    ecolor='k', linewidth=1.0, capsize=2, capthick=0.8)

    ax_bot.set_ylabel('SO$_2$ vmr (ppbv)')
    ax_bot.set_xlabel('Date')
    ax_bot.set_title(f'b) {rolling_window}-point rolling mean', loc='left', pad=2)
    ax_bot.spines['top'].set_visible(True)
    ax_bot.spines['right'].set_visible(True)

    locator2 = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax_bot.xaxis.set_major_locator(locator2)
    ax_bot.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator2))
    ax_bot.xaxis.set_minor_locator(mdates.MonthLocator())

    # --- Rising slope annotation on subplot b) ---
    if peak_search_before is not None:
        smoothed_vals = smoothed.values

        valid_mask  = ~np.isnan(smoothed_vals)
        first_valid = np.where(valid_mask)[0][0]

        before_mask = obs_dates < pd.Timestamp(peak_search_before)
        search_mask = valid_mask & before_mask.values
        peak_idx    = np.where(search_mask)[0][np.argmax(smoothed_vals[search_mask])]

        start_ann = obs_dates[first_valid]
        peak_ann  = obs_dates[peak_idx]
        duration  = (peak_ann - start_ann).days
        mid_ann   = start_ann + (peak_ann - start_ann) / 2

        trans = transforms.blended_transform_factory(ax_bot.transData, ax_bot.transAxes)

        # Rising slope line, offset downward
        ax_bot.plot([start_ann, peak_ann],
                    [smoothed_vals[first_valid] - slope_offset,
                     smoothed_vals[peak_idx]    - slope_offset],
                    color='k', linewidth=1.0, linestyle='-', zorder=5)

        # Double-headed arrow + duration label for the rise
        ax_bot.annotate('', xy=(peak_ann, 0.08), xytext=(start_ann, 0.08),
                        xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle='<->', color='k', lw=0.8))
        ax_bot.text(mid_ann, 0.13, f'{duration} days ({duration / 365.25:.1f} yr)',
                    transform=trans, ha='center', va='bottom', fontsize=7)

        # --- Declining slope: peak to first crossing below trough_threshold ---
        if trough_threshold is not None:
            after_peak   = np.where(valid_mask & (np.arange(len(smoothed_vals)) > peak_idx))[0]
            below_thresh = after_peak[smoothed_vals[after_peak] < trough_threshold]

            if len(below_thresh) > 0:
                trough_idx  = below_thresh[0]
                trough_ann  = obs_dates[trough_idx]
                duration_dt = (trough_ann - peak_ann).days
                mid_dt      = peak_ann + (trough_ann - peak_ann) / 2

                # Declining slope line, offset downward
                ax_bot.plot([peak_ann, trough_ann],
                            [smoothed_vals[peak_idx]   - slope_offset,
                             smoothed_vals[trough_idx] - slope_offset],
                            color='k', linewidth=1.0, linestyle='-', zorder=5)

                # Double-headed arrow + duration label for the decline
                ax_bot.annotate('', xy=(trough_ann, 0.08), xytext=(peak_ann, 0.08),
                                xycoords=trans, textcoords=trans,
                                arrowprops=dict(arrowstyle='<->', color='k', lw=0.8))
                ax_bot.text(mid_dt, 0.13,
                            f'{duration_dt} days ({duration_dt / 365.25:.1f} yr)',
                            transform=trans, ha='center', va='bottom', fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
