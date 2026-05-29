"""
Animations for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib import colors as mcolors
import cartopy.crs as ccrs

from fig1 import STYLE, ONE_COL, ONE_HALF_COL

# %%
def anim1(sim, t_start, t_end, lev=35, central_lon=130,
          n_quiver=4, qscale=800, vmin=1.0, vmax=1000.0,
          save=False, savepath=None, savename='anim1_so2.mp4'):
    """
    Animation of SO2 mixing ratio on an orthographic globe.

    Animated equivalent of the globe snapshots in Figure 1.
    Horizontal winds shown as quivers.

    Args:
        sim (PlumeSim): Simulation object.
        t_start (int): First frame time index.
        t_end (int): Last frame time index (exclusive).
        lev (int): Vertical level index. Defaults to 35 (~70 km).
        central_lon (float): Central longitude for orthographic projection.
        n_quiver (int): Quiver subsampling interval.
        qscale (float): Quiver scale (larger = shorter arrows).
        vmin, vmax (float): SO2 colormap limits in ppbv.
        save (bool): Save if True.
        savepath (str): Directory to save.
        savename (str): Output filename (.mp4).
    """
    plt.rcParams.update(STYLE)
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'axes.titlesize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10})

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    sim_times         = sim.data.time_counter.values
    plume_inject_time = sim_times[sim.plumes['start_time']]
    height_km         = np.round(sim.heights[lev], 1)
    norm              = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # --- Initial frame ---
    so2_0 = sim.data['so2'][t_start, lev, :, :].values * 1e9
    u_0   = sim.data['vitu'][t_start, lev, :, :].values
    v_0   = sim.data['vitv'][t_start, lev, :, :].values

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL + 0.6),
                           subplot_kw={'projection': proj})

    mesh = ax.pcolormesh(sim.lons, sim.lats, so2_0, norm=norm,
                         cmap='YlOrBr_r', transform=plate)
    q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                  -u_0[::n_quiver, ::n_quiver], v_0[::n_quiver, ::n_quiver],
                  transform=plate, color='black', alpha=0.75,
                  width=0.005, headwidth=4, scale=qscale)
    qk = ax.quiverkey(q, X=0.88, Y=1.06, U=100, label='100 m s$^{-1}$',
                      labelpos='E', coordinates='axes')

    ax.set_global()
    gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label(f'SO$_2$ at {height_km} km (ppbv)')
    cb.ax.xaxis.set_minor_locator(mticker.NullLocator())

    hrs_0 = (sim_times[t_start] - plume_inject_time) / 3600.0
    ax.set_title(f'{hrs_0:+.0f} h', loc='left', pad=3)

    def update(frame):
        nonlocal mesh, q, qk
        t = t_start + frame

        so2 = sim.data['so2'][t, lev, :, :].values * 1e9
        u   = sim.data['vitu'][t, lev, :, :].values
        v   = sim.data['vitv'][t, lev, :, :].values

        mesh.remove()
        q.remove()
        qk.remove()

        mesh = ax.pcolormesh(sim.lons, sim.lats, so2, norm=norm,
                             cmap='YlOrBr_r', transform=plate)
        q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                      -u[::n_quiver, ::n_quiver], v[::n_quiver, ::n_quiver],
                      transform=plate, color='black', alpha=0.75,
                      width=0.005, headwidth=4, scale=qscale)
        qk = ax.quiverkey(q, X=0.88, Y=1.06, U=100, label='100 m s$^{-1}$',
                          labelpos='E', coordinates='axes')

        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{hrs:+.0f} h', loc='left', pad=3)

    ani = animation.FuncAnimation(fig, update, frames=t_end - t_start,
                                  interval=200, blit=False)

    if save:
        ani.save(savepath + savename, writer='ffmpeg', dpi=150)

    plt.show()

# %%
def anim2(sim, t_start, t_end, lev=35, central_lon=130,
          n_quiver=4, qscale=1000, vmin=None, vmax=None,
          save=False, savepath=None, savename='anim2_vwind.mp4'):
    """
    Animation of vertical wind on an orthographic globe.

    Animated equivalent of the globe snapshots in Figure 2.
    Diverging seismic colormap centred on zero. Horizontal winds as quivers.

    Args:
        sim (PlumeSim): Simulation object.
        t_start (int): First frame time index.
        t_end (int): Last frame time index (exclusive).
        lev (int): Vertical level index. Defaults to 35 (~70 km).
        central_lon (float): Central longitude for orthographic projection.
        n_quiver (int): Quiver subsampling interval.
        qscale (float): Quiver scale.
        vmin, vmax (float): Colormap limits in m/s. If None, set symmetrically
            from the maximum absolute value over the full time range.
        save (bool): Save if True.
        savepath (str): Directory to save.
        savename (str): Output filename (.mp4).
    """
    plt.rcParams.update(STYLE)
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'axes.titlesize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10})

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    sim_times         = sim.data.time_counter.values
    plume_inject_time = sim_times[sim.plumes['start_time']]
    height_km         = np.round(sim.heights[lev], 1)

    if vmin is None or vmax is None:
        all_w   = sim.data['vitwz'][t_start:t_end, lev, :, :].values
        abs_max = np.nanmax(np.abs(all_w))
        vmin    = -abs_max if vmin is None else vmin
        vmax    =  abs_max if vmax is None else vmax

    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    # --- Initial frame ---
    w_0 = sim.data['vitwz'][t_start, lev, :, :].values
    u_0 = sim.data['vitu'][t_start, lev, :, :].values
    v_0 = sim.data['vitv'][t_start, lev, :, :].values

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL + 0.6),
                           subplot_kw={'projection': proj})

    mesh = ax.pcolormesh(sim.lons, sim.lats, w_0, norm=norm,
                         cmap='seismic', transform=plate)
    q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                  -u_0[::n_quiver, ::n_quiver], v_0[::n_quiver, ::n_quiver],
                  transform=plate, color='black', alpha=0.75,
                  width=0.005, headwidth=4, scale=qscale)
    qk = ax.quiverkey(q, X=0.88, Y=1.06, U=100, label='100 m s$^{-1}$',
                      labelpos='E', coordinates='axes')

    ax.set_global()
    gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label(f'Vertical wind at {height_km} km (m s$^{{-1}}$)')
    cb.ax.xaxis.set_minor_locator(mticker.NullLocator())

    hrs_0 = (sim_times[t_start] - plume_inject_time) / 3600.0
    ax.set_title(f'{hrs_0:+.0f} h', loc='left', pad=3)

    def update(frame):
        nonlocal mesh, q, qk
        t = t_start + frame

        w = sim.data['vitwz'][t, lev, :, :].values
        u = sim.data['vitu'][t, lev, :, :].values
        v = sim.data['vitv'][t, lev, :, :].values

        mesh.remove()
        q.remove()
        qk.remove()

        mesh = ax.pcolormesh(sim.lons, sim.lats, w, norm=norm,
                             cmap='seismic', transform=plate)
        q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                      -u[::n_quiver, ::n_quiver], v[::n_quiver, ::n_quiver],
                      transform=plate, color='black', alpha=0.75,
                      width=0.005, headwidth=4, scale=qscale)
        qk = ax.quiverkey(q, X=0.88, Y=1.06, U=100, label='100 m s$^{-1}$',
                          labelpos='E', coordinates='axes')

        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{hrs:+.0f} h', loc='left', pad=3)

    ani = animation.FuncAnimation(fig, update, frames=t_end - t_start,
                                  interval=200, blit=False)

    if save:
        ani.save(savepath + savename, writer='ffmpeg', dpi=150)

    plt.show()

# %%
def anim3(sim, t_start, t_end, lev=22, central_lon=320,
          n_quiver=4, qscale=400, vmin=None, vmax=None,
          save=False, savepath=None, savename='anim3_h2o.mp4'):
    """
    Animation of H2O mixing ratio on an orthographic globe.

    Animated equivalent of the globe snapshots in Figure 3.
    Horizontal winds shown as quivers.

    Args:
        sim (PlumeSim): Simulation object.
        t_start (int): First frame time index.
        t_end (int): Last frame time index (exclusive).
        lev (int): Vertical level index. Defaults to 22 (~47 km).
        central_lon (float): Central longitude for orthographic projection.
        n_quiver (int): Quiver subsampling interval.
        qscale (float): Quiver scale.
        vmin, vmax (float): H2O colormap limits in ppmv. If None, auto-scaled
            from the data range over the full time range.
        save (bool): Save if True.
        savepath (str): Directory to save.
        savename (str): Output filename (.mp4).
    """
    plt.rcParams.update(STYLE)
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'axes.titlesize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10})

    plate = ccrs.PlateCarree()
    proj  = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0)

    sim_times         = sim.data.time_counter.values
    plume_inject_time = sim_times[sim.plumes['start_time']]
    height_km         = np.round(sim.heights[lev], 1)

    if vmin is None or vmax is None:
        all_h2o = sim.data['h2o'][t_start:t_end, lev, :, :].values * 1e6
        vmin    = np.nanmin(all_h2o) if vmin is None else vmin
        vmax    = np.nanmax(all_h2o) if vmax is None else vmax

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Initial frame ---
    h2o_0 = sim.data['h2o'][t_start, lev, :, :].values * 1e6
    u_0   = sim.data['vitu'][t_start, lev, :, :].values
    v_0   = sim.data['vitv'][t_start, lev, :, :].values

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, ONE_HALF_COL + 0.6),
                           subplot_kw={'projection': proj})

    mesh = ax.pcolormesh(sim.lons, sim.lats, h2o_0, norm=norm,
                         cmap='Blues', transform=plate)
    q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                  -u_0[::n_quiver, ::n_quiver], v_0[::n_quiver, ::n_quiver],
                  transform=plate, color='black', alpha=0.75,
                  width=0.005, headwidth=4, scale=qscale)
    qk = ax.quiverkey(q, X=0.65, Y=1.06, U=40, label='40 m s$^{-1}$',
                      labelpos='E', coordinates='axes')

    ax.set_global()
    gl = ax.gridlines(crs=plate, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label(f'H$_2$O at {height_km} km (ppmv)')
    cb.ax.xaxis.set_minor_locator(mticker.NullLocator())

    hrs_0 = (sim_times[t_start] - plume_inject_time) / 3600.0
    ax.set_title(f'{hrs_0:+.0f} h', loc='left', pad=3)

    def update(frame):
        nonlocal mesh, q, qk
        t = t_start + frame

        h2o = sim.data['h2o'][t, lev, :, :].values * 1e6
        u   = sim.data['vitu'][t, lev, :, :].values
        v   = sim.data['vitv'][t, lev, :, :].values

        mesh.remove()
        q.remove()
        qk.remove()

        mesh = ax.pcolormesh(sim.lons, sim.lats, h2o, norm=norm,
                             cmap='Blues', transform=plate)
        q = ax.quiver(sim.lons[::n_quiver], sim.lats[::n_quiver],
                      -u[::n_quiver, ::n_quiver], v[::n_quiver, ::n_quiver],
                      transform=plate, color='black', alpha=0.75,
                      width=0.005, headwidth=4, scale=qscale)
        qk = ax.quiverkey(q, X=0.65, Y=1.06, U=40, label='40 m s$^{-1}$',
                          labelpos='E', coordinates='axes')

        hrs = (sim_times[t] - plume_inject_time) / 3600.0
        ax.set_title(f'{hrs:+.0f} h', loc='left', pad=3)

    ani = animation.FuncAnimation(fig, update, frames=t_end - t_start,
                                  interval=200, blit=False)

    if save:
        ani.save(savepath + savename, writer='ffmpeg', dpi=150)

    plt.show()

# %%
