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
