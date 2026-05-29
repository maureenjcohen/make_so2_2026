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
from tools import plume_dict_master

# %%
def figure3(sim, snap_times, ts_levs, ts_start, ts_end,
            lev=22, central_lon=320,
            n_quiver=4, vmin=None, vmax=None,
            save=False, savepath=None, savename='figure3.pdf'):
    """
    Figure 3: four orthographic H2O globe snapshots (top row) and a
    multi-level time series of the global H2O maximum (bottom row).

    Args:
        sim (PlumeSim): Simulation object.
        snap_times (list of 4 int): Time indices for the globe panels.
        ts_levs (list of int): Vertical level indices for the time series.
        ts_start (int): Start time index for the time series.
        ts_end (int): End time index for the time series.
        lev (int): Vertical level for globe snapshots.
        central_lon (float): Central longitude for orthographic projection.
        n_quiver (int): Quiver subsampling interval.
        vmin, vmax (float): Colormap limits for H2O (ppmv). If None,
            auto-scaled from the data range across all four snapshots.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    height_km = np.round(sim.heights[lev], 1)
    sim_times = sim.data.time_counter.values
    plume_inject_time = sim_times[sim.plumes['start_time']]

    # Colormap limits: auto-scale from all four snapshots if not specified
    if vmin is None or vmax is None:
        all_snaps = np.array([sim.data['h2o'][t, lev, :, :].values * 1e6
                              for t in snap_times])
        vmin = np.nanmin(all_snaps) if vmin is None else vmin
        vmax = np.nanmax(all_snaps) if vmax is None else vmax

    # --- Layout ---
    fig = plt.figure(figsize=(TWO_COL, 5.0))
    gs  = fig.add_gridspec(2, 4, height_ratios=[1, 0.75], hspace=0.1, wspace=0.08)

    globe_axes = [fig.add_subplot(gs[0, col], projection=proj) for col in range(4)]
    ax_ts = fig.add_subplot(gs[1, :])

    # --- Globe snapshots ---
    norm   = colors.Normalize(vmin=vmin, vmax=vmax)
    labels = ['a', 'b', 'c', 'd']

    for i, (ax, t) in enumerate(zip(globe_axes, snap_times)):
        h2o_snap = sim.data['h2o'][t, lev, :, :].values * 1e6
        u_snap   = sim.data['vitu'][t, lev, :, :].values
        v_snap   = sim.data['vitv'][t, lev, :, :].values

        ax.contourf(sim.lons, sim.lats, h2o_snap,
                    levels=np.linspace(vmin, vmax, 21),
                    norm=norm, cmap='Blues', transform=plate)

        q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                  -u_snap[::n_quiver, ::n_quiver], v_snap[::n_quiver, ::n_quiver],
                  transform=plate, color='black', alpha=0.75,
                  width=0.005, headwidth=4, scale=400)
        # Reference arrow on the first panel
        if i == 0:
            ax.quiverkey(q, X=0.65, Y=1.06, U=40, label='40 m/s',
                         labelpos='E', coordinates='axes',
                         fontproperties={'size': 7})

        ax.set_global()
        gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{labels[i]}) {hrs:+.0f} h', loc='left', pad=3)

    # Shared colorbar to the right of the globe row
    sm = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
    sm.set_array([])
    cb = fig.colorbar(sm, ax=globe_axes, orientation='horizontal',
                      fraction=0.025, pad=0.03, shrink=0.85)
    cb.set_label(f'H$_2$O at {height_km} km (ppmv)')
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())

    # --- H2O maximum time series ---
    ts_times    = sim_times[ts_start:ts_end]
    ts_hours    = (ts_times - plume_inject_time) / 3600.0
    line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ts_levs)))

    for ts_lev, c in zip(ts_levs, line_colors):
        h2o_max   = sim.data['h2o'][ts_start:ts_end, ts_lev, :, :].values.max(axis=(1, 2)) * 1e6
        alt_label = f'{sim.heights[ts_lev]:.0f} km'
        ax_ts.plot(ts_hours, h2o_max, color=c, linewidth=1.0, label=alt_label)

        background = sim.data['h2o'][sim.plumes['start_time'] - 1,
                                     ts_lev, :, :].values.mean() * 1e6
        ax_ts.axhline(background, color=c, linewidth=0.8, linestyle='--')

    x_inj_end = ts_hours[sim.plumes['end_time'] - ts_start]
    ax_ts.axvline(x_inj_end, color='darkgrey', linestyle='--')
    ax_ts.text(x_inj_end + 2.0, 0.40, 'Injection stops',
               transform=ax_ts.get_xaxis_transform(),
               va='top', ha='left', fontsize=7, color='darkgrey')

    ax_ts.set_xlabel('Time from injection (h)')
    ax_ts.set_ylabel('H$_2$O (ppmv)')
    ax_ts.set_title('e)', loc='left', pad=2)
    ax_ts.legend(loc='upper right', fontsize=7, title='Altitude', title_fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
