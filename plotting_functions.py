"""
Plotting functions for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).
"""
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import pandas as pd

from tools import plume_dict, max_dispersal

# %%
def marcq_reproduce(ds, save=False, window=240, min_valid_days=60,
               savepath='/exomars/projects/mc5526/VPCM_decadal_so2/scratch_plots/'):
    """
    Combined plot: top - latitude-weighted SO_2 at 70km (from marcq_fig1), 
    bottom - 240-day rolling mean of SO_2 at 70km (from marcq_fig3)
    """
    # Compute latitude-weighted SO_2
    weights = np.cos(np.deg2rad(ds.latitude))
    so2_weighted = ds.so2_ppbv.weighted(weights).mean(dim=['latitude', 'longitude'])
    error_mean = ds.rel_error_pct.mean(dim=['latitude', 'longitude'])
    error_absolute = error_mean * so2_weighted / 100.0

    # Compute smoothed rolling mean
    smoothed = so2_weighted.rolling(start_date=window, center=True, min_periods=min_valid_days).mean()
    roll = so2_weighted.rolling(start_date=window, center=True, min_periods=min_valid_days)
    smoothed_error = roll.std() / np.sqrt(roll.count())

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Top subplot: latitude-weighted SO_2
    ax1.errorbar(ds.start_date.values, so2_weighted, yerr=error_absolute, 
                 fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=3, markeredgewidth=0.5, 
                 ecolor='k', linewidth=1, capsize=5, capthick=2)
    ax1.set_title('a) Latitude-weighted SO$_2$ at 70 km', fontsize=16)
    ax1.set_ylabel('SO$_2$ / ppbv', fontsize=16)
    ax1.set_yscale('log')

    # Bottom subplot: rolling mean
    ax2.errorbar(ds.start_date.values, smoothed, yerr=smoothed_error, linestyle='-', c='r',
                 fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=4, markeredgewidth=0.5, 
                 ecolor='k', linewidth=1, capsize=3, capthick=1)
    ax2.set_title(f'b) {window}-day rolling mean of SO$_2$ at 70 km', fontsize=16)
    ax2.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('SO$_2$ / ppbv', fontsize=16)

    # Format x-axis like marcq_fig1
    ax2.set_xlim(pd.to_datetime('2006-01-01'), pd.to_datetime('2015-01-01'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.subplots_adjust(hspace=0.15)

    if save:
        fig.savefig(savepath+'marcq_reproduce.png', bbox_inches='tight')

    plt.show()

# %%
def time_series(ds, start, end, save=False, savepath='/exomars/projects/mc5526/VPCM_decadal_so2/scratch_plots/'):

    # 1. Slice the dataset temporally right away to save memory and keep code DRY
    ds_sliced = ds.isel(start_date=slice(start, end))
    time_axis_mean = ds_sliced.start_date.values

    # --- ax1: Latitude-weighted spatial mean ---
    weights = np.cos(np.deg2rad(ds_sliced.latitude))
    so2_weighted = ds_sliced.so2_ppbv.weighted(weights).mean(dim=['latitude', 'longitude'])
    
    # Calculate error mean natively in xarray to preserve dimensions
    error_mean = ds_sliced.rel_error_pct.mean(dim=['latitude', 'longitude'])
    error_absolute = (error_mean * so2_weighted) / 100.0

    # --- ax2: Individual unweighted observations ---
    # Convert only the variables we need to a single DataFrame to ensure dropna() aligns them
    df_unweighted = ds_sliced[['so2_ppbv', 'rel_error_pct']].to_dataframe().dropna().reset_index()
    
    # Calculate absolute error explicitly on the numeric columns, avoiding the datetime column
    df_unweighted['abs_error'] = (df_unweighted['rel_error_pct'] * df_unweighted['so2_ppbv']) / 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Aggregated Mean
    ax1.errorbar(time_axis_mean, so2_weighted, yerr=error_absolute, 
                 fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=4, markeredgewidth=0.5, 
                 ecolor='k', linewidth=1, capsize=3, capthick=1)
    ax1.set_title('a) Latitude-weighted SO$_2$ at 70 km', fontsize=16)
    ax1.set_ylabel('SO$_2$ / ppbv', fontsize=16)
    ax1.set_yscale('log')

    ylim = ax1.get_ylim()  

    # Plot 2: Scatter of individual points
    # We must use the time axis from the dataframe, not the 1D aggregated time_axis_mean
    ax2.errorbar(df_unweighted['start_date'], df_unweighted['so2_ppbv'], yerr=df_unweighted['abs_error'], 
                 fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=4, markeredgewidth=0.5, 
                 ecolor='k', linewidth=1, capsize=3, capthick=1)
    ax2.set_title('b) Unweighted SO$_2$ at 70 km', fontsize=16)
    ax2.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('SO$_2$ / ppbv', fontsize=16)
    ax2.set_yscale('log')
    ax2.set_ylim(ylim)
    
    plt.subplots_adjust(hspace=0.15)
    
    if save:
        fig.savefig(savepath+f'so2_timeseries_{start}_{end}.png', bbox_inches='tight')
        
    plt.show()

# %%
def zmage(plobject, hmin=0, hmax=None, time_slice=-1, convert2yr=True,
          plume_markers=None, levels=None, savepath=None,
          save=False, sformat='png', savename='zmage.png'):
    """
    Plot the zonal mean age of air.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        hmin (int): Minimum height index. Defaults to 0.
        hmax (int): Maximum height index. Defaults to None.
        time_slice (int): Time index to select. Defaults to -1.
        convert2yr (bool): Whether to convert units to years. Defaults to True.
        levels (list, optional): Contour levels. Defaults to None.
        savepath (str): Directory path to save the plot. Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        saveformat (str): Format to save the plot. Defaults to 'png'.
        savename (str): Filename for the saved plot. Defaults to 'zmage.png'.
    """
    ageo = plobject.data['age']
    # Select time slice
    ageo = ageo[time_slice,:,:,:]
    
    # Calculate zonal mean
    zmageo = np.mean(ageo, axis=-1)

    if convert2yr:
        zmageo = zmageo/(60*60*24*360)
        cunit = 'Earth years'
    else:
        cunit = 'seconds'
 
    zmslice = zmageo[hmin:hmax,:]
    
    fig = plt.figure(figsize=(6, 6))
    plt.contourf(plobject.lats, plobject.heights[hmin:hmax],
                 zmslice,
                 levels=levels,
                 cmap='plasma')
    if plume_markers is not None:
        for plume in plume_markers:
            plt.plot(plobject.lats[plume_markers[plume]['lat_idx']], plobject.heights[plume_markers[plume]['lev']], marker='*', color='black', markersize=10)
    plt.title('Zonal mean age of air', fontsize=16)
    plt.xlabel('Latitude / deg', fontsize=16)
    plt.ylabel('Height / km', fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}', fontsize=16)
    
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def plume_cross_section(plobject, key, lev, times=[54,104,174], 
                        save=False, savename='plume_cross_section.png',
                        savepath=None, sformat='png'):
    """Plot a cross section of the plume for a given variable.  
    Creates a 2x3 subplot figure: top row horizontal (lon/lat) cross sections,
    bottom row vertical (lat/altitude) cross sections at three different times.
    """
    if len(times) != 3:
        raise ValueError("times must be a list of exactly 3 time indices")
    
    # Get plume longitude for vertical cross section
    lon_idx = plobject.plumes['plume_6']['lon_idx']
    lat_idx = plobject.plumes['plume_6']['lat_idx']
    interval = np.diff(plobject.data[key].time_counter.values)[0]/(60*60)
    start_time = plobject.plumes['plume_6']['start_time']
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharey='row', sharex='col')
    
    # prepare letters for all 6 subplots
    letters = [chr(97 + k) for k in range(6)]  # ['a','b',...,'f']
    for i, time_idx in enumerate(times):
        # Horizontal cross section (top row)
        time_hrs = (time_idx - start_time) * interval
        ax_h = axes[0, i]
        data_h = plobject.data[key][time_idx, lev, 65:, 40:70]*1e6
        cs_h = ax_h.contourf(plobject.lons[40:70], plobject.lats[65:], data_h, cmap='Blues',
                             levels=np.arange(28,46,1))
        ax_h.axhline(plobject.lats[lat_idx+1], color='red', linestyle='dashed', label='Plume latitude')
        ax_h.grid(True, alpha=0.5)
        ax_h.set_title(f'{letters[i]}) {time_hrs:.0f} hrs', fontsize=16)
        
        if i == 0:
            ax_h.set_ylabel('Latitude / deg', fontsize=16)
        
        # Vertical cross section (bottom row)
        ax_v = axes[1, i]
        data_v = plobject.data[key][time_idx, 15:25, lat_idx+1, 40:70]*1e6
        cs_v = ax_v.contourf(plobject.lons[40:70], plobject.heights[15:25], data_v, cmap='Blues',
                             levels=np.arange(28,46,1))
        ax_v.axhline(plobject.heights[lev], color='red', linestyle='dashed', label='Plume altitude')
        # add letter label to bottom titles as well
        ax_v.set_title(f'{letters[i+3]})', fontsize=16, pad=4)
        if i == 0:
            ax_v.set_ylabel('Height / km', fontsize=16)
        ax_v.set_xlabel('Longitude / deg', fontsize=16)
        ax_v.grid(True, alpha=0.5)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.15)

    # Add single colorbar for all subplots
    cbar = fig.colorbar(cs_v, ax=axes.ravel(), orientation='horizontal', pad=0.1)
    cbar.set_label('ppm', fontsize=16)
    fig.suptitle(r'H$_2$O plume cross-sections', y=0.97, fontsize=18)
    
    if save:
        if savepath is None:
            savepath = ''
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def dispersal_time(plobject, lev, keys, lats, lons, name_dict, threshold,
                   axis_len=500, save=False, plot=True,
                   savename='plume_dispersal.png',
                   savepath=None,
                   sformat='png'):
    """
    Find the time for the tracer value in the gridbox to return to the background value.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        key (str): Dictionary key of the data variable.
        lev (int): Vertical level index.
        threshold (float or None): Threshold value for defining the plume.
        save (bool): Whether to save the plot. Defaults to False.
        savename (str): Filename for the saved plot. Defaults to 'plume_dispersal.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    # Extract data at desired model level
    if isinstance(keys, str):
        keys = [keys]
    num_subplots = len(keys)
    if num_subplots == 1:
        num_cols, num_rows = 1, 1
    elif num_subplots == 2:
        num_cols, num_rows = 2, 1
    elif num_subplots == 4:
        num_cols, num_rows = 2, 2

    position = range(1, num_subplots+1)
    interval = np.diff(plobject.data.time_counter.values)[0]
    time_axis = np.arange(plobject.plumes['plume_1']['start_time']-2, axis_len)*interval/(60*60)

    if plot==True:
        fig = plt.figure(figsize=(num_cols*4, num_rows*4), tight_layout=True)

    disp_times = []
    for i, key in enumerate(keys):
        series1 = plobject.data[key][:,lev,lats[0],lons[0]]
        series2 = plobject.data[key][:,lev,lats[1],lons[1]]
        # Get background value of tracer before plume starts
        background_val1 = series1[plobject.plumes['plume_1']['start_time']-1].values*threshold
        background_val2 = series2[plobject.plumes['plume_2']['start_time']-1].values*threshold
    
        counter1 = 0
        for t in range(plobject.plumes['plume_1']['end_time'], series1.shape[0]):
            if series1[t].values > background_val1:
                # Count how many time steps tracer value remains above background
                counter1 = counter1 + 1
            else:
                # Stop loop when tracer returns to background
                break

        counter2 = 0
        for t in range(plobject.plumes['plume_1']['end_time'], series2.shape[0]):
            if series2[t].values > background_val2:
                # Count how many time steps tracer value remains above background
                counter2 = counter2 + 1
            else:
                # Stop loop when tracer returns to background
                break
        
        # Total time in seconds until tracer valuer returns to background
        disp_time1 = counter1 * interval
        disp_time2 = counter2 * interval
        
        # Convert to hours for convenience
        disp_hours1 = disp_time1 / (60*60)
        disp_hours2 = disp_time2 / (60*60)
        disp_dict = {'species': key, 'disp_time_eq_hrs': disp_hours1, 'disp_time_hl_hrs': disp_hours2}
        disp_times.append(disp_dict)

        # Get data, including 5 time steps before and after plume
        data1 = series1[plobject.plumes['plume_1']['start_time']-2:axis_len]*1e6
        data2 = series2[plobject.plumes['plume_1']['start_time']-2:axis_len]*1e6
    
        if plot==True:
            ax = fig.add_subplot(num_rows, num_cols, position[i])
            ax.plot(time_axis, data1, color='red', label=rf'{np.round(plobject.lats[lats[0]],2)}°N: $\tau_d$={np.round(disp_hours1,2)} h')
            ax.plot(time_axis, data2, color='blue', label=rf'{np.round(plobject.lats[lats[1]],2)}°N: $\tau_d$={np.round(disp_hours2,2)} h')
            ax.plot(time_axis, np.ones_like(data1)*background_val1*1e6, color='red',
                    linestyle='dashed')
            ax.plot(time_axis, np.ones_like(data1)*background_val2*1e6, color='blue',
                    linestyle='dashed')
            title = name_dict.get(key, key)
            if num_subplots > 1:
                letter = chr(97 + i)
                title = f'{letter}) {title}'
            ax.set_title(title, fontsize=16)
            ax.set_ylabel(f'{name_dict[key]} vmr / ppm', fontsize=16)
            ax.set_ylim([background_val1*1e6*0.8, data1.max()*1.2])
            ax.set_xlabel('Time / hours', fontsize=16)
            plt.legend(loc='upper right')
    if plot==True:
        fig.suptitle(f'Plume dispersal times, h = {np.round(plobject.heights[lev], 2)} km', y=0.97, fontsize=18)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    if plot==True:
        plt.show()

    return disp_times

# %%
def dispersal_map(plobject, lev, keys, name_dict, threshold,
                   save=False,
                   savename='dispersal_map.png',
                   savepath=None,
                   sformat='png'):
    """Create maps showing how long plume tracer remains above background.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        lev (int): Vertical level index.
        keys (str or list): Variable key(s) to plot.
        labels (dict, optional): Legacy mapping used for plot titles. If
            ``name_dict`` is provided it takes precedence. Defaults to None.
        name_dict (dict, optional): Preferred mapping from variable key to
            pretty name, used exactly like in ``summ_stats``. Defaults to None.
        threshold (float): Threshold multiplier for background levels.
        save (bool): Whether to save the figure. Defaults to False.
        savename (str): Filename for saved figure. Defaults to
            'dispersal_map.png'.
        savepath (str): Directory to save in. Defaults to None.
        sformat (str): File format (png, pdf, etc). Defaults to 'png'.

    """
    if isinstance(keys, str):
        keys = [keys]

    # Define styles
    styles = {
        'h2o': {'cmap': 'Blues', 'vmax': 45.0, 'background': 30.0},
        'co':  {'cmap': 'Purples', 'vmax': 12.0, 'background': 7.35},
        'ocs': {'cmap': 'YlOrBr', 'vmax': 4.5, 'background': 1.0},
        'hcl': {'cmap': 'Reds', 'vmax': 0.6, 'background': 0.4},
        'so2': {'cmap': 'Greens', 'vmax': 1.3, 'background': 0.13}
    }

    # Calculate layout
    num_subplots = len(keys)
    if num_subplots == 1:
        num_cols, num_rows = 1, 1
        figsize = (6, 5)
    elif num_subplots == 2:
        num_cols, num_rows = 2, 1
        figsize = (12, 5)
    elif num_subplots == 4:
        num_cols, num_rows = 2, 2
        figsize = (12, 10)
    else:
        num_cols = int(np.ceil(np.sqrt(num_subplots)))
        num_rows = int(np.ceil(num_subplots / num_cols))
        figsize = (num_cols*8, num_rows*5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=False, sharey=True, tight_layout=True)
    if num_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get eruption coords
    lat_eq = plobject.plumes['plume_1']['lat_idx']
    lon_eq = plobject.plumes['plume_1']['lon_idx']
    lat_hl = plobject.plumes['plume_2']['lat_idx']
    lon_hl = plobject.plumes['plume_2']['lon_idx']

    for i, key in enumerate(keys):
        ax = axes[i]
        
        cube = plobject.data[key][:,lev,:,:]
        interval = np.diff(cube.time_counter.values)[0]
        background = styles[key]['background']*1e-6*threshold
        #background = cube[plobject.plumes['plume_1']['start_time']-1,:,:].values*threshold
        post_eruption = cube[plobject.plumes['plume_1']['end_time']:,:,:]
        # Only consider data after plume forcing has finished
        mask = post_eruption.values > background
        # Boolean array with same dimension as cube, True is where values are above threshold
        flattened = np.count_nonzero(mask, axis=0)
        # Count how many time outputs are True for each lon x lat point
        map_hours = flattened * interval / (60*60)
        # Convert time outputs to hours

        # Determine style
        if key in styles:
            cmap = styles[key]['cmap']
        else:
            cmap = 'viridis'

        cf = ax.contourf(plobject.lons, plobject.lats, map_hours, cmap=cmap)
        ax.plot(plobject.lons[lon_eq], plobject.lats[lat_eq], 'ro', label='Equatorial eruption')
        ax.plot(plobject.lons[lon_hl], plobject.lats[lat_hl], 'ro', label='High-latitude eruption')

        if name_dict is not None and key in name_dict:
            title_name = name_dict[key]
        else:
            title_name = cube.long_name or cube.name

        # add subplot letter if more than one panel
        if num_subplots > 1:
            letter = chr(97 + i)
            title_name = f'{letter}) {title_name}'

        ax.set_title(f'{title_name} vmr above {background*1e6:.2f} ppm', fontsize=16)
        ax.grid()
        
        # Labels
        if i % num_cols == 0:
            ax.set_ylabel('Latitude / deg', fontsize=16)
        if i >= num_subplots - num_cols:
            ax.set_xlabel('Longitude / deg', fontsize=16)

        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal')
        cbar.set_label('Hours', fontsize=16)
        # ax.legend()
    fig.suptitle(f'Plume dispersal maps, h = {np.round(plobject.heights[lev], 2)} km', y=0.97, fontsize=18)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # Remove unused axes
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %%
def animate_chem_plume(plobject, lev, keys, name_dict, t0, tf, n=4, qscale=1,
                  savename='test.png', savepath=None, snapshot=None):
    """
    Create an animation of a chemical plume.

    Args:
        plobject (PlumeSim): PlumeSim class object containing the data.
        lev (int): Vertical level index to be visualised.
        keys (str or list): Dictionary key(s) of the data variable(s).
        t0 (int): First frame (time index).
        tf (int): Final frame (time index).
        n (int): Quiver plot sampling interval. Defaults to 4.
        qscale (float): Scale for quiver plot. Defaults to 1.
        savepath (str): Directory path to save the output. Defaults to None.
        snapshot (int): Frame to save as a still image. Defaults to None.
    """
    # Get height in km rounded to 2 decimal points (for title)
    height = np.round(plobject.heights[lev],2)
    
    if isinstance(keys, str):
        keys = [keys]

    # Define styles
    styles = {
        'h2o': {'cmap': 'Blues', 'vmax': 45.0},
        'co':  {'cmap': 'Purples', 'vmax': 11.0},
        'ocs': {'cmap': 'Greens', 'vmax': 1.5},
        'hcl': {'cmap': 'Reds', 'vmax': 0.6},
        'so2': {'cmap': 'Greens', 'vmax': 0.2}
    }

    # Extract data
    cubes = {}
    for key in keys:
        if key == 'n2':
             cubes[key] = plobject.data[key][t0:tf,lev,:,:]*100
        else:
             cubes[key] = plobject.data[key][t0:tf,lev,:,:]*1e6

    # Extract zonal and meridional wind for desired altitude
    u = plobject.data['vitu'][t0:tf,lev,:,:]
    v = plobject.data['vitv'][t0:tf,lev,:,:]

    # Calculate layout
    num_subplots = len(keys)
    if num_subplots == 1:
        num_cols, num_rows = 1, 1
        figsize = (6, 5)
    elif num_subplots == 2:
        num_cols, num_rows = 2, 1
        figsize = (12, 5)
    elif num_subplots == 4:
        num_cols, num_rows = 2, 2
        figsize = (12, 10)
    else:
        num_cols = int(np.ceil(np.sqrt(num_subplots)))
        num_rows = int(np.ceil(num_subplots / num_cols))
        figsize = (num_cols*8, num_rows*5)

    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=False, sharey=True, tight_layout=True)
    if num_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    X, Y = np.meshgrid(plobject.lons, plobject.lats)

    interval = np.diff(cubes[keys[0]].time_counter.values)[0]/(60*60)
    time_axis = np.round(np.arange(0,tf-t0)*interval,0)
 
    quiv_args = {
    'angles': 'xy',
    'scale_units': 'xy',
    'scale': qscale,
    'color': 'black'
    }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        for i, key in enumerate(keys):
            ax = axes[i]
            ax.clear()
            
            cube = cubes[key]
            
            # Determine style
            if key in styles:
                cmap = styles[key]['cmap']
                vmax = styles[key]['vmax']
                if vmax is None: vmax = cube.max()
                vmin = cube.min()
            else:
                cmap = 'viridis'
                vmax = cube.max()
                vmin = cube.min()

            cf = ax.contourf(plobject.lons, plobject.lats, cube[frame,:,:],
                                  cmap=cmap,
                                  vmin=vmin, vmax=vmax)
            
            q = ax.quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                       v[frame,::n,::n], **quiv_args)
            
            ax.quiverkey(ax.quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                       v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label=f'{qscale*10} m/s',
                     labelpos='E', coordinates='axes', color='black')
            
            # determine title text including optional letter prefix
            title_text = name_dict[key]
            if num_subplots > 1:
                letter = chr(97 + i)
                title_text = f'{letter}) {title_text}'
            ax.set_title(title_text, color='black', y=1.05, fontsize=16)
            
            # Labels
            if i % num_cols == 0:
                ax.set_ylabel('Latitude / deg', fontsize=16)
            if i >= num_subplots - num_cols:
                 ax.set_xlabel('Longitude / deg', fontsize=16)

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.suptitle(f'Volcanic plume at {height} km, {time_axis[frame]} hrs', y=0.97, fontsize=18)

        if snapshot is not None and frame == snapshot:
            if savepath:
                fig.savefig(savepath + f'{savename}_snapshot_{frame}.pdf', bbox_inches='tight')
            else:
                fig.savefig(f'{savename}_snapshot_{frame}.pdf', bbox_inches='tight')

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,tf-t0), interval=200, repeat=False)
    
    # Add colorbars (based on first frame of plume)
    for i, key in enumerate(keys):
        ax = axes[i]
        cube = cubes[key]
        if key in styles:
             cmap = styles[key]['cmap']
             vmax = styles[key]['vmax']
             if vmax is None: vmax = cube.max()
             vmin = cube.min()
        else:
             cmap = 'viridis'
             vmax = cube.max()
             vmin = cube.min()
             
        cf = ax.contourf(plobject.lons, plobject.lats, cube[4,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal')
        if key == 'n2':
             cbar.set_label('%', color='black', fontsize=16)
        else:
             cbar.set_label('ppm', color='black', fontsize=16)

    # Remove unused axes
    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])

    # Save the animation as an mp4 file
    ani.save(savepath + f'{savename}.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative

# %%
def summ_stats(plobject, keys, lev, t0, tf, name_dict=None, savename='stats.png',
               savepath=None,
               save=False, sformat='png'):
    """
    Compute variability of chemical species (standard deviation on lon-lat grid).

    The plot titles are taken from ``name_dict`` when provided; otherwise the
    long name or variable name from the dataset is used.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        keys (str or list): Dictionary key(s) of the data variable(s).
        lev (int): Vertical level index.
        t0 (int): Start time index.
        tf (int): End time index.
        name_dict (dict, optional): Mapping from variable key to pretty name.
            Defaults to None.
        savename (str): Filename for the saved plot. Defaults to 'stats.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    if isinstance(keys, str):
        keys = [keys]

    num_rows = len(keys)
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows), sharex=False, sharey=True, tight_layout=True)
    
    # Ensure axes is 2D array [row, col]
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)

    for i, key in enumerate(keys):
        # Extract data for the specified time range and level
        cube = plobject.data[key][t0:tf,lev,:,:]
        
        # Determine species name (use dictionary lookup if provided)
        if name_dict is not None and key in name_dict:
            title_name = name_dict[key]
        else:
            title_name = cube.long_name or cube.name
        title_height = np.round(plobject.heights[lev],2)
        
        # Convert units based on species
        if key=='n2':
            cube = cube*100
            unit = '%'
        else:
            cube = cube*1e6
            unit = 'ppm'

        # Calculate standard deviation and mean over time
        std = cube.std(dim='time_counter',skipna=True, keep_attrs=True)
        avg = cube.mean(dim='time_counter', skipna=True, keep_attrs=True)

        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        
        # determine subplot labels (a), b), etc) for each axes
        # index ordering: row-major, left-to-right
        base_index = i * num_cols
        label1 = f"{chr(97 + base_index)})"  # a), b), ...
        label2 = f"{chr(97 + base_index + 1)})"
        
        # Plot mean abundance
        abd_plot = ax1.contourf(plobject.lons, plobject.lats, avg, cmap='afmhot')
        ax1.set_title(f'{label1} Mean {title_name} at {title_height} km', fontsize=16)
        ax1.set_ylabel('Latitude / deg', fontsize=16)
        cbar1 = plt.colorbar(abd_plot, orientation='horizontal', ax=ax1)
        cbar1.set_label(unit, fontsize=16)
        cbar1.ax.tick_params(rotation=45)

        # Plot coefficient of variation
        std_plot = ax2.contourf(plobject.lons, plobject.lats, 100*std/avg, cmap='copper')
        ax2.set_title(f'{label2} Coefficient of variation in {title_name} at {title_height} km', fontsize=16)
        cbar2 = plt.colorbar(std_plot, orientation='horizontal',ax=ax2)
        cbar2.set_label('%', fontsize=16)
        cbar2.ax.tick_params(rotation=45)
        
        # Only set xlabel on bottom plots
        if i == num_rows - 1:
            ax1.set_xlabel('Longitude / deg', fontsize=16)
            ax2.set_xlabel('Longitude / deg', fontsize=16)
    fig.suptitle(f'Background chemical variability at h = {np.round(plobject.heights[lev],2)} km', y=0.99, fontsize=18)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %%
def sensitivity_test(plume5, plume4, plume3, plume2, plume1, plume0,
                     name_dict, levs=[10,14,18], key='h2o',
                     save=False,
                     savename='sensitivity_test.png',
                     savepath=None,
                     sformat='png'):
    """
    Plot relationship between plume strength and dispersal time for H2O for five plumes.

    Args:
        plume5 (PlumeSim): PlumeSim object for strongest plume.
        plume4 (PlumeSim): PlumeSim object for 2nd strongest plume.
        plume3 (PlumeSim): PlumeSim object for 3rd strongest plume.
        plume2 (PlumeSim): PlumeSim object for 4th strongest plume.
        plume1 (PlumeSim): PlumeSim object for weakest plume.
        lev (int): Vertical level index.
        key (str): Dictionary key of the data variable.
        save (bool): Whether to save the plot. Defaults to False.
        savename (str): Filename for the saved plot. Defaults to 'sensitivity_test.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sens_tests = []
    for i, l in enumerate(levs):
        times0 = dispersal_time(plume0, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        times1 = dispersal_time(plume1, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        times2 = dispersal_time(plume2, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        times3 = dispersal_time(plume3, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        times4 = dispersal_time(plume4, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        times5 = dispersal_time(plume5, lev=l, keys=[key], name_dict=name_dict, threshold=1.005, lats=[49,82], lons=[92,47], axis_len=500,
                   save=False, plot=False)
        
        disp_times_eq_hrs = [times0[0]['disp_time_eq_hrs'], times1[0]['disp_time_eq_hrs'], times2[0]['disp_time_eq_hrs'], times3[0]['disp_time_eq_hrs'],
                      times4[0]['disp_time_eq_hrs'], times5[0]['disp_time_eq_hrs']]
        disp_times_hl_hrs = [times0[0]['disp_time_hl_hrs'], times1[0]['disp_time_hl_hrs'], times2[0]['disp_time_hl_hrs'], times3[0]['disp_time_hl_hrs'],
                      times4[0]['disp_time_hl_hrs'], times5[0]['disp_time_hl_hrs']]
        strengths = [1.03, 1.1, 1.2, 1.3, 1.4, 1.5]
        sens_dict = {'level': l, 'disp_times_eq_hrs': disp_times_eq_hrs,
                     'disp_times_hl_hrs': disp_times_hl_hrs}
        sens_tests.append(sens_dict)

        # Plot sensitivity test
        ax[i].plot(strengths, disp_times_eq_hrs, marker='o', color='blue', label='Equatorial eruption')
        ax[i].plot(strengths, disp_times_hl_hrs, marker='o', color='green', label='High-latitude eruption')
        ax[i].invert_xaxis()
        letter = chr(97 + i)
        ax[i].set_title(f'{letter}) {np.round(plume1.heights[l],2)} km', fontsize=16)
        ax[i].set_xlabel('Plume scaling factor', fontsize=16)
        ax[i].set_xticks(strengths)
        ax[i].set_ylabel('Dispersal time / hours', fontsize=16)
        ax[i].legend(loc='lower left')

    fig.suptitle(f'Sensitivity test for {name_dict[key]} plume dispersal time', y=1.01, fontsize=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
    else:
        plt.show()

    return sens_tests

# %%
def sensitivity_test_maximum_dispersal(plume5, plume4, plume3, plume2, plume1,
                     name_dict, levs=[10,14,18], key='h2o',
                     save=False,
                     savename='sensitivity_test_max_dispersal.png',
                     savepath=None,
                     sformat='png'):
    """
    Plot relationship between plume strength and maximum dispersal for H2O for six plumes.

    Args:
        plume5 (PlumeSim): PlumeSim object for strongest plume.
        plume4 (PlumeSim): PlumeSim object for 2nd strongest plume.
        plume3 (PlumeSim): PlumeSim object for 3rd strongest plume.
        plume2 (PlumeSim): PlumeSim object for 4th strongest plume.
        plume1 (PlumeSim): PlumeSim object for 5th strongest plume.
        plume0 (PlumeSim): PlumeSim object for weakest plume.
        name_dict (dict): Dictionary mapping variable names to display names.
        levs (list): Vertical level indices. Defaults to [10,14,18].
        key (str): Dictionary key of the data variable.
        save (bool): Whether to save the plot. Defaults to False.
        savename (str): Filename for the saved plot. Defaults to 'sensitivity_test_max_dispersal.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sens_tests = []
    for i, l in enumerate(levs):
        #max_disp0 = max_dispersal(plume0, lev=l, lat=70, threshold=1.05)
        max_disp1 = max_dispersal(plume1, lev=l, lat=70, threshold=1.05)
        max_disp2 = max_dispersal(plume2, lev=l, lat=70, threshold=1.05)
        max_disp3 = max_dispersal(plume3, lev=l, lat=70, threshold=1.05)
        max_disp4 = max_dispersal(plume4, lev=l, lat=70, threshold=1.05)
        max_disp5 = max_dispersal(plume5, lev=l, lat=70, threshold=1.05)
        
        max_disp_eq_hrs = [max_disp1['eq_plume_max'], 
                           max_disp2['eq_plume_max'], max_disp3['eq_plume_max'],
                           max_disp4['eq_plume_max'], max_disp5['eq_plume_max']]
        max_disp_hl_hrs = [max_disp1['hl_plume_max'], 
                           max_disp2['hl_plume_max'], max_disp3['hl_plume_max'],
                           max_disp4['hl_plume_max'], max_disp5['hl_plume_max']]
        strengths = [1.1, 1.2, 1.3, 1.4, 1.5]
        sens_dict = {'level': l, 'max_disp_eq_hrs': max_disp_eq_hrs,
                     'max_disp_hl_hrs': max_disp_hl_hrs}
        sens_tests.append(sens_dict)

        # Plot sensitivity test
        ax[i].plot(strengths, max_disp_eq_hrs, marker='o', color='blue', label='Equatorial eruption')
        ax[i].plot(strengths, max_disp_hl_hrs, marker='o', color='green', label='High-latitude eruption')
        ax[i].invert_xaxis()
        letter = chr(97 + i)
        ax[i].set_title(f'{letter}) {np.round(plume1.heights[l],2)} km', fontsize=16)
        ax[i].set_xlabel('Plume scaling factor', fontsize=16)
        ax[i].set_xticks(strengths)
        ax[i].set_ylabel('Maximum dispersal time / hours', fontsize=16)
        ax[i].legend(loc='lower left')

    fig.suptitle(f'Maximum persistence of {name_dict[key]} enhancements', y=1.01, fontsize=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
    else:
        plt.show()

    return sens_tests
# %%
