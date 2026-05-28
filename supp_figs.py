"""
Supplementary and Materials & Methods figures for
'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from fig1 import STYLE, TWO_COL, ONE_COL, ONE_HALF_COL
from tools import logistic_decay

# %%
def fig_so2_prof(sim, times=None, t_start=None, t_end=None,
                 labels=None, xscale='log',
                 save=False, savepath=None, savename='fig_so2_prof.pdf'):
    """
    Supplementary figure: global area-weighted mean SO2 vertical profile.

    Always plots the long-term time mean over t_start:t_end (defaults to
    all timesteps). If `times` is provided, individual snapshot profiles
    are overlaid. A horizontal dashed line marks level 35, below which SO2
    is semi-prescribed and above which it evolves freely.

    Args:
        sim (PlumeSim): Simulation object.
        times (list of int): Time indices for snapshot profiles. If None,
            only the time mean is plotted.
        t_start (int): Start index for the time mean. Defaults to first timestep.
        t_end (int): End index for the time mean. Defaults to last timestep.
        labels (list of str): Legend labels for each snapshot. If None,
            auto-labels with hours elapsed since the first timestep.
        xscale (str): 'log' or 'linear' for the SO2 axis.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    aire      = sim.data['aire'].values       # (lat, lon)  gridbox areas
    heights   = sim.heights                   # (lev,)  altitudes in km
    sim_times = sim.data.time_counter.values  # (time,)  seconds from model start
    t0        = sim_times[0]

    fig, ax = plt.subplots(figsize=(ONE_COL, 4.5))

    # --- Snapshot profiles ---
    if times is not None:
        line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))
        for i, t in enumerate(times):
            so2_3d = sim.data['so2'][t, :, :, :].values * 1e9  # (lev, lat, lon)
            so2_snap = (so2_3d * aire[np.newaxis, :, :]).sum(axis=(1, 2)) / aire.sum()

            if labels is not None:
                label = labels[i]
            else:
                hrs = (sim_times[t] - t0) / 3600.0
                label = f'{hrs:.0f} h'

            ax.plot(so2_snap, heights, color=line_colors[i], linewidth=0.8,
                    alpha=0.7, label=label)

    # --- Long-term time mean ---
    so2_4d = sim.data['so2'][t_start:t_end, :, :, :].values * 1e9  # (time, lev, lat, lon)
    so2_area = (so2_4d * aire[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3)) / aire.sum()
    so2_tmean = so2_area.mean(axis=0)  # (lev,)
    ax.plot(so2_tmean, heights, color='k', linewidth=1.5, label='Time mean', zorder=5)

    # --- Prescribed/free boundary at level 35 ---
    ax.axhline(heights[35], color='k', linewidth=0.8, linestyle='--', zorder=4)
    ax.text(0.97, heights[35], 'Prescribed below',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=6,
            color='k')

    ax.set_xscale(xscale)
    ax.set_xlabel('SO$_2$ vmr (ppbv)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Area-weighted mean SO$_2$ profile', loc='left', pad=2)
    ax.legend(loc='upper right', fontsize=7, title='Time', title_fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
def fig_temp_profile(sim, t=0,
                     ylim_bottom=None, ylim_top=None,
                     save=False, savepath=None, savename='fig_temp_profile.pdf'):
    """
    Supplementary figure: latitude-altitude cross-section of temperature.

    Plots temperature at a single timestep and the longitude index specified
    in the simulation's plume_dict (sim.plumes['lon_idx']), as a filled
    contour plot with latitude on the x-axis and altitude on the y-axis.

    Args:
        sim (PlumeSim): Simulation object.
        t (int): Time index to plot. Defaults to 0.
        ylim_bottom (float): Lower limit of the altitude axis in km.
        ylim_top (float): Upper limit of the altitude axis in km.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    lon_idx = sim.plumes['lon_idx']
    lon_val = sim.lons[lon_idx]

    temp = sim.data['temp'][t, :, :, lon_idx].values  # (lev, lat)

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, 4.0))

    cf = ax.contourf(sim.lats, sim.heights, temp,
                     levels=20, cmap='inferno')

    cb = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.03)
    cb.set_label('Temperature (K)')
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title(f'Temperature, lon = {lon_val:.0f}°',
                 loc='left', pad=2)
    ax.set_xlim(sim.lats[0], sim.lats[-1])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    if ylim_bottom is not None or ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
def fig_logistic(sim, L=115, K=760, x0=20, k=0.2,
                 save=False, savepath=None, savename='fig_logistic.pdf'):
    """
    Supplementary figure: logistic decay curve vs. simulated temperature profile.

    Plots two lines on the same axes (level index on y, temperature on x):
      - Global area-weighted long-term mean temperature from the simulation.
      - Logistic decay curve with the given parameters.

    Args:
        sim (PlumeSim): Simulation object.
        L (float): Temperature at the top level (K) for the logistic curve.
        K (float): Temperature at the bottom level (K) for the logistic curve.
        x0 (float): Level index at the midpoint of the transition.
        k (float): Steepness of the transition.
        save (bool): Save the figure if True.
        savepath (str): Directory to save.
        savename (str): Filename for saved figure.
    """
    plt.rcParams.update(STYLE)

    levs   = len(sim.heights)
    levels = np.arange(levs)

    # --- Global area-weighted long-term mean temperature ---
    aire     = sim.data['aire'].values  # (lat, lon)
    temp_4d  = sim.data['temp'].values  # (time, lev, lat, lon)
    temp_area = (temp_4d * aire[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3)) / aire.sum()
    temp_mean = temp_area.mean(axis=0)  # (lev,)

    # --- Logistic decay curve ---
    temp_logistic = logistic_decay(levs=levs, L=L, K=K, x0=x0, k=k)

    fig, ax = plt.subplots(figsize=(ONE_COL, 4.0))

    ax.plot(temp_mean,     levels, color='steelblue',  linewidth=1.0, label='Simulated mean')
    ax.plot(temp_logistic, levels, color='firebrick',  linewidth=1.0,
            linestyle='--', label='Logistic decay')

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Level index')
    ax.set_title('Temperature profile', loc='left', pad=2)
    ax.legend(loc='upper right', fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
