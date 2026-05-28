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
def fig_so2_prof(sim, times=None, tmean=False, t_start=None, t_end=None,
                 labels=None, xscale='log',
                 save=False, savepath=None, savename='fig_so2_prof.pdf'):
    """
    Supplementary figure: global area-weighted mean SO2 vertical profile.

    The area weighting uses the 'aire' field (gridbox areas in m²).

    Two modes:
      tmean=False  Plot one profile per entry in `times`.
      tmean=True   Plot a single time-mean profile averaged over
                   t_start:t_end (defaults to all timesteps).

    Args:
        sim (PlumeSim): Simulation object.
        times (list of int): Time indices to plot. Required when tmean=False;
            ignored when tmean=True.
        tmean (bool): If True, plot the long-term time mean instead of
            individual snapshots.
        t_start (int): Start time index for the mean (tmean=True only).
            Defaults to the first timestep.
        t_end (int): End time index for the mean (tmean=True only).
            Defaults to the last timestep.
        labels (list of str): Legend labels, one per time index
            (tmean=False only). If None, auto-labels with hours elapsed
            since the first timestep.
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

    if tmean:
        so2_4d = sim.data['so2'][t_start:t_end, :, :, :].values * 1e9  # (time, lev, lat, lon)
        so2_area = (so2_4d * aire[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3)) / aire.sum()
        so2_mean = so2_area.mean(axis=0)  # (lev,)
        ax.plot(so2_mean, heights, color='firebrick', linewidth=1.0, label='Time mean')
    else:
        line_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))
        for i, t in enumerate(times):
            so2_3d = sim.data['so2'][t, :, :, :].values * 1e9  # (lev, lat, lon)
            so2_mean = (so2_3d * aire[np.newaxis, :, :]).sum(axis=(1, 2)) / aire.sum()

            if labels is not None:
                label = labels[i]
            else:
                hrs = (sim_times[t] - t0) / 3600.0
                label = f'{hrs:.0f} h'

            ax.plot(so2_mean, heights, color=line_colors[i], linewidth=1.0, label=label)

    ax.set_xscale(xscale)
    ax.set_xlabel('SO$_2$ vmr (ppbv)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Area-weighted mean SO$_2$ profile', loc='left', pad=2)
    ax.legend(loc='upper right', fontsize=7,
              title=None if tmean else 'Time', title_fontsize=7)

    if save:
        fig.savefig(savepath + savename)

    plt.show()

# %%
