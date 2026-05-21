"""
Publication figures for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import cartopy.crs as ccrs

from fig1 import STYLE, TWO_COL
from tools import plume_dict

# %%
def figure2(sim, snap_times, ts_levs, ts_start, ts_end,
            lev=35, central_lon=130,
            lat_idx=None, lon_idx=None,
            n_quiver=4, vmin=None, vmax=None,
            save=False, savepath=None, savename='figure2.pdf'):
    """
    Figure 2: six orthographic vertical-wind globe snapshots (top two rows) and
    a multi-level time series of vertical wind at a single gridpoint (bottom row).

    Args:
        sim (PlumeSim): Simulation object.
        snap_times (list of 6 int): Time indices for globe panels (3 per row,
            top row first then middle row).
        ts_levs (list of 5 int): Vertical level indices for the time series.
        ts_start (int): Start time index for the time series.
        ts_end (int): End time index for the time series.
        lev (int): Vertical level for globe snapshots.
        central_lon (float): Central longitude for orthographic projection.
        lat_idx (int): Latitude index for time series gridpoint. Defaults to
            plume injection point.
        lon_idx (int): Longitude index for time series gridpoint. Defaults to
            plume injection point.
        n_quiver (int): Quiver subsampling interval.
        vmin, vmax (float): Colormap limits for vitwz (m/s). If None, set
            symmetrically from the max absolute value across all six snapshots.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    if lat_idx is None:
        lat_idx = plume_dict['hotspot']['lat_idx']
    if lon_idx is None:
        lon_idx = plume_dict['hotspot']['lon_idx']

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    height_km = np.round(sim.heights[lev], 1)
    sim_times = sim.data.time_counter.values
    plume_inject_time = sim_times[plume_dict['hotspot']['start_time']]

    # Colormap limits: symmetric about zero if not specified
    if vmin is None or vmax is None:
        all_snaps = np.array([sim.data['vitwz'][t, lev, :, :].values for t in snap_times])
        abs_max = np.nanmax(np.abs(all_snaps))
        vmin = -abs_max if vmin is None else vmin
        vmax =  abs_max if vmax is None else vmax

    # --- Layout ---
    fig = plt.figure(figsize=(TWO_COL, 7.0))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.08)

    globe_axes = []
    for row in range(2):
        for col in range(3):
            globe_axes.append(fig.add_subplot(gs[row, col], projection=proj))
    ax_ts = fig.add_subplot(gs[2, :])

    # --- Globe snapshots ---
    norm   = colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    cf     = None
    labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for i, (ax, t) in enumerate(zip(globe_axes, snap_times)):
        vitwz_snap = sim.data['vitwz'][t, lev, :, :].values
        u_snap     = sim.data['vitu'][t, lev, :, :].values
        v_snap     = sim.data['vitv'][t, lev, :, :].values

        cf = ax.contourf(sim.lons, sim.lats, vitwz_snap,
                         levels=20, norm=norm, cmap='seismic', transform=plate)

        ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                  -u_snap[::n_quiver, ::n_quiver], v_snap[::n_quiver, ::n_quiver],
                  transform=plate, color='black',
                  width=0.005, headwidth=4, scale=800)

        ax.set_global()
        gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{labels[i]}) {hrs:+.0f} h', loc='left', pad=3)

    # Shared colorbar for all globe panels, placed to the right
    sm = plt.cm.ScalarMappable(norm=norm, cmap='seismic')
    sm.set_array([])
    cb = fig.colorbar(sm, ax=globe_axes, orientation='vertical',
                      fraction=0.025, pad=0.03, shrink=0.85)
    cb.set_label(f'w at {height_km} km (m s$^{{-1}}$)')
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())

    # --- Time series ---
    ts_times = sim_times[ts_start:ts_end]
    ts_hours = (ts_times - plume_inject_time) / 3600.0

    line_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ts_levs)))

    for ts_lev, c in zip(ts_levs, line_colors):
        vitwz_ts  = sim.data['vitwz'][ts_start:ts_end, ts_lev, lat_idx, lon_idx].values
        alt_label = f'{sim.heights[ts_lev]:.0f} km'
        ax_ts.plot(ts_hours, vitwz_ts, color=c, linewidth=1.0, label=alt_label)

    ax_ts.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax_ts.set_xlabel('Time from injection (h)')
    ax_ts.set_ylabel('Vertical wind (m s$^{-1}$)')
    ax_ts.set_title('g)', loc='left', pad=2)
    ax_ts.legend(loc='upper right', fontsize=7,
                 title='Altitude', title_fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
