"""
Publication figures for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import colors
import pandas as pd
import cartopy.crs as ccrs

from tools import plume_dict

# %%
# --- Publication style ---
# Matches Science / Science Advances figure specifications.
# Apply at the top of any script that uses these functions:
#   plt.rcParams.update(STYLE)

STYLE = {
    # Font: Arial preferred; falls back to Helvetica, then DejaVu Sans
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':            7,

    # Axes
    'axes.labelsize':       7,
    'axes.titlesize':       8,
    'axes.linewidth':       0.5,
    'axes.spines.top':      False,
    'axes.spines.right':    False,

    # Ticks: inward-facing, no top/right
    'xtick.labelsize':      7,
    'ytick.labelsize':      7,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.major.width':    0.5,
    'ytick.major.width':    0.5,
    'xtick.minor.width':    0.4,
    'ytick.minor.width':    0.4,
    'xtick.major.size':     3.0,
    'ytick.major.size':     3.0,
    'xtick.minor.size':     1.5,
    'ytick.minor.size':     1.5,

    # Lines and markers
    'lines.linewidth':      1.0,
    'lines.markersize':     3.0,

    # Legend
    'legend.fontsize':      7,
    'legend.frameon':       False,
    'legend.handlelength':  1.5,

    # Output: 300 dpi raster; Type 42 fonts embed cleanly in PDF/EPS
    'figure.dpi':           300,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.02,
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
}

# Figure widths in inches (Science two-column layout: ~8.7 cm / ~17.4 cm)
ONE_COL       = 3.43   # 8.7 cm  — single column
ONE_HALF_COL  = 5.20   # 13.2 cm — 1.5 column
TWO_COL       = 6.85   # 17.4 cm — full width


# %%
def figure1(sim, ds, snap_times, long_start, long_end, zoom_start, zoom_end,
            plume_anchor, lev=35, central_lon=0,
            vmin=100.0, vmax=1000.0, n_quiver=4,
            save=False, savepath=None, savename='figure1.pdf'):
    """
    Figure 1: four orthographic SO2 globe snapshots (left column) and two
    SPICAV SO2 time series with simulated plume overlaid (right column).

    Args:
        sim (PlumeSim): Simulation object containing model output.
        ds (xr.Dataset): SPICAV observations dataset with dims
            (start_date, latitude, longitude).
        snap_times (list of 4 int): Simulation time indices for globe panels.
        long_start, long_end (int): Obs index range for the long time series.
        zoom_start, zoom_end (int): Obs index range for the zoomed time series.
        plume_anchor (int): Index into ds.start_date that aligns with the
            moment of plume injection. Slide this to reposition the simulated
            plume on the time axis without affecting the observations.
        lev (int): Vertical level index (default 35 ≈ 70 km).
        central_lon (float): Central longitude for orthographic projection.
        vmin, vmax (float): SO2 colormap limits in ppbv for globe panels.
        n_quiver (int): Subsampling interval for wind quivers.
        save (bool): Save the figure if True.
        savepath (str): Directory to save the figure.
        savename (str): Filename for saved figure.
    """

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    height_km = np.round(sim.heights[lev], 1)

    # --- Pre-compute lat-weighted global-mean SO2 for all simulation timesteps ---
    # This matches the SPICAV weighting: cos(lat)-weighted mean over lat and lon.
    sim_so2_3d   = sim.data['so2'][:, lev, :, :] * 1e9          # ppbv (time, lat, lon)
    sim_lat_w    = np.cos(np.deg2rad(sim.data['lat']))
    sim_so2_mean = sim_so2_3d.weighted(sim_lat_w).mean(dim=['lat', 'lon'])  # (time,)
    sim_times    = sim.data.time_counter.values                   # seconds from model start

    plume_inject_time = sim_times[plume_dict['plume_1']['start_time']]
    anchor_date       = pd.Timestamp(ds.start_date.values[plume_anchor])

    def match_plume_to_obs(obs_dates):
        """
        For each SPICAV observation date, find the nearest simulation timestep
        and return the pre-computed weighted-mean SO2 value (ppbv).
        Returns NaN for observations outside the simulation time range.
        """
        matched = np.full(len(obs_dates), np.nan)
        for i, obs_date in enumerate(obs_dates):
            delta_s = (pd.Timestamp(obs_date) - anchor_date).total_seconds()
            target  = plume_inject_time + delta_s
            if sim_times[0] <= target <= sim_times[-1]:
                nearest      = int(np.argmin(np.abs(sim_times - target)))
                matched[i]   = float(sim_so2_mean.values[nearest])
        return matched

    # --- Layout ---
    # Left column: 4 globe panels + shared colorbar below.
    # Right column: long time series (top) + zoomed time series (bottom).
    fig      = plt.figure(figsize=(TWO_COL, 7.5))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.4)
    left_gs  = outer_gs[0].subgridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.08], hspace=0.08)
    right_gs = outer_gs[1].subgridspec(2, 1, hspace=0.4)

    globe_axes = [fig.add_subplot(left_gs[i], projection=proj) for i in range(4)]
    ax_cbar    = fig.add_subplot(left_gs[4])
    ax_long    = fig.add_subplot(right_gs[0])
    ax_zoom    = fig.add_subplot(right_gs[1])

    # --- Globe snapshots ---
    norm   = colors.LogNorm(vmin=vmin, vmax=vmax)
    cf     = None
    labels = ['a', 'b', 'c', 'd']

    for i, (ax, t) in enumerate(zip(globe_axes, snap_times)):
        so2_snap = sim.data['so2'][t, lev, :, :].values * 1e9
        u_snap   = sim.data['vitu'][t, lev, :, :].values
        v_snap   = sim.data['vitv'][t, lev, :, :].values

        cf = ax.pcolormesh(sim.lons, sim.lats, so2_snap,
                           norm=norm, cmap='Greens', transform=plate)

        q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                      -u_snap[::n_quiver, ::n_quiver], v_snap[::n_quiver, ::n_quiver],
                      transform=plate, color='black',
                      width=0.003, headwidth=3)

        # Reference arrow on the first panel only
        if i == 0:
            ax.quiverkey(q, X=0.88, Y=1.06, U=50, label='50 m s$^{-1}$',
                         labelpos='E', coordinates='axes',
                         fontproperties={'size': 6})

        ax.set_global()
        gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey',
                          alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

        # Time since plume injection
        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{labels[i]}) {hrs:+.0f} h', loc='left', pad=2)

    # Shared colorbar for all globe panels
    cb = fig.colorbar(cf, cax=ax_cbar, orientation='horizontal')
    cb.set_label(f'SO$_2$ vmr at {height_km} km (ppbv)')
    cb.ax.xaxis.set_tick_params(which='both', direction='in')

    # --- Time series panels ---
    def plot_ts(ax, start, end, label, zoom=False):
        """Plot SPICAV observations with error bars and matched simulated plume."""
        ds_sl      = ds.isel(start_date=slice(start, end))
        obs_dates  = ds_sl.start_date.values

        # Lat-weighted mean SO2 and absolute error (matches SPICAV weighting)
        w        = np.cos(np.deg2rad(ds_sl.latitude))
        so2_obs  = ds_sl.so2_ppbv.weighted(w).mean(dim=['latitude', 'longitude'])
        err_mean = ds_sl.rel_error_pct.mean(dim=['latitude', 'longitude'])
        err_abs  = (err_mean * so2_obs) / 100.0

        ax.errorbar(obs_dates, so2_obs.values, yerr=err_abs.values,
                    fmt='o', markerfacecolor='r', markeredgecolor='k',
                    markersize=3, markeredgewidth=0.4,
                    ecolor='k', linewidth=0.6, capsize=2, capthick=0.8,
                    zorder=3, label='SPICAV')

        # Simulated plume: each obs date matched to nearest simulation timestep
        plume = match_plume_to_obs(obs_dates)
        mask  = ~np.isnan(plume)
        if mask.any():
            ax.plot(obs_dates[mask], plume[mask],
                    color='steelblue', linestyle='--', linewidth=1.0,
                    marker='o', markersize=2,
                    label='Simulated plume', zorder=4)

        ax.set_yscale('log')
        ax.set_ylabel('SO$_2$ vmr (ppbv)')
        ax.set_xlabel('Date')
        ax.set_title(label, loc='left', pad=2)

        if zoom:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
            ax.tick_params(axis='x', labelrotation=30)
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plot_ts(ax_long, long_start, long_end, label='e)')
    plot_ts(ax_zoom, zoom_start, zoom_end, label='f)', zoom=True)

    ax_long.legend(loc='upper right')

    if save:
        fig.savefig(savepath + savename)

    plt.show()
