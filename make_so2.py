"""
Plotting functions for 'Decadal sulphur dioxide variability on Venus not caused by volcanic injection'
(Cohen et al. 2027).
"""
# %% Filepaths
eqday_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_day.nc'
eqnight_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_night.nc'
so2_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_so2_big.nc'
chem_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_chem.nc'
hotspot_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/hotspot_96x96x50/Xins_hotspot1.nc'
retrievals = '/exomars/projects/mc5526/VPCM_decadal_so2/netcdfs/SO2_lons.nc'
savedir = '/exomars/projects/mc5526/VPCM_decadal_so2/scratch_plots/'
filetype = 'png'

# %%
# Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
# Import local
from classes import PlumeSim
from plotting_functions import dispersal_time, dispersal_map, animate_chem_plume, summ_stats, time_series
from plotting_functions import data_compare, sensitivity_test, marcq_reproduce
from fig1 import figure1, STYLE
from fig2 import figure2 
from fig3 import figure3 
from fig0 import figure0
from tools import venusdict, plume_dict_master, calculate_so2_mass, name_dict
from supp_figs import fig_so2_prof, fig_temp_profile, fig_logistic, fig_orbit_summary, fig_orbit_path
from animations import anim1, anim2, anim3


# %% Main code block
if __name__ == "__main__":

    # %%
    # Open retrievals dataset
    ds_retrievals = xr.open_dataset(retrievals)

    # %%
    figure0(ds_retrievals, rolling_window=60, 
            peak_search_before='2008-01-01', trough_threshold=30, 
            save=True, savepath=savedir, savename='fig0_ref_30.png')

    # %%
    # Create PlumeSim object and load data for plume run
    eqday = PlumeSim(venusdict, plume_dict_master['plume_1'], 'equatorial dayside plume')
    eqday.load_file(eqday_plume)
    eqday.set_resolution()

    # %%
    figure1(eqday, ds_retrievals, snap_times=[8, 158, 208, 278], 
            long_start=0, long_end=100, zoom_start=38, zoom_end=72, 
            plume_anchor=39, lev=35, central_lon=130, save=True,
            savepath=savedir, savename='fig1_ref.png')
    
    # %%
    eqday.close()
    del eqday
    
    # %%
    # Create PlumeSim object and load data for plume run
    so2 = PlumeSim(venusdict, plume_dict_master['so2'], 'equatorial dayside plume')
    so2.load_file(so2_plume)
    so2.set_resolution()
    
    # %%
    figure1(so2, ds_retrievals, snap_times=[250, 316, 404, 608], 
            long_start=0, long_end=334, zoom_start=38, zoom_end=72, 
            plume_anchor=38, lev=35, central_lon=-40, save=True,
            savepath=savedir, savename='fig1_ref_big.png')

    # %%
    so2.close()
    del so2

    # %%
    hotspot = PlumeSim(venusdict, plume_dict_master['hotspot'], 'pillar of fire')
    hotspot.load_file(hotspot_plume)
    hotspot.set_resolution()
    
    # %% Late start
    figure2(hotspot, snap_times=[130, 196, 230, 256, 300, 368], 
            ts_levs=[22, 28, 34, 40], ts_start=100, ts_end=700, 
            lev=22, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-1.2, vmax=3.0)
    
    # %% Early start
    figure2(hotspot, snap_times=[10, 30, 90, 150, 312, 508], 
            ts_levs=[22, 28, 34, 40], ts_start=0, ts_end=700, 
            zu_start=0, zu_end=1000,
            lev=35, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-2.0, vmax=3.0, save=True, savepath=savedir, savename='fig2_ref.png')

    # %%
    hotspot.close()
    del hotspot

    # %%
    chem = PlumeSim(venusdict, plume_dict_master['h2o'], 'potential absorber plume')
    chem.load_file(chem_plume)
    chem.set_resolution()

    # %%
    figure3(chem, snap_times=[10, 140, 304, 364], ts_levs=[21, 22, 23], ts_start=0, ts_end=2000, central_lon=320)
    # %%
   # Animation + cover image of SO2 plume at 70 km
    animate_chem_plume(so2, lev=35, keys=['so2'], name_dict=name_dict, t0=0, tf=1200, qscale=5, log=False, savepath=savedir, savename='eqday_plume_no_log', snapshot=100) 

    # %%
    # Calculation of SO2 mass in plume at 70 km for reference
    so2_mass_kg = calculate_so2_mass(so2, vmr_ppm=0.195)
    print(f'SO2 mass in plume at 70 km: {so2_mass_kg:.2f} kg')

    # %%
    eqnight_sim = PlumeSim(venusdict, plume_dict, 'equatorial nightside plume')
    eqnight_sim.load_file(eqnight_plume)
    eqnight_sim.set_resolution()

    # %%
    animate_chem_plume(eqnight_sim, lev=35, keys=['so2'], name_dict=name_dict, t0=0, tf=1200, qscale=5, log=False, savepath=savedir, savename='eqnight_plume_no_log', snapshot=100)

    # %% Reload so2 plume for supplementary figures and animations
    so2 = PlumeSim(venusdict, plume_dict_master['so2'], 'equatorial dayside plume')
    so2.load_file(so2_plume)
    so2.set_resolution()

    # %%
    fig_so2_prof(so2, save=True, savepath=savedir, savename='figS_so2_prof.pdf')

    # %%
    fig_temp_profile(so2, save=True, savepath=savedir, savename='figS_temp_profile.pdf')

    # %%
    fig_logistic(so2, save=True, savepath=savedir, savename='figS_logistic.pdf')

    # %%
    fig_orbit_summary(ds_retrievals, orbit_idx=0,
                      save=True, savepath=savedir, savename='figS_orbit_summary.pdf')

    # %%
    fig_orbit_path(ds_retrievals, orbit_idx=0,
                   save=True, savepath=savedir, savename='figS_orbit_path.pdf')

    # %%
    anim1(so2, t_start=plume_dict_master['so2']['start_time'],
          t_end=plume_dict_master['so2']['end_time'],
          lev=35, central_lon=-40,
          save=True, savepath=savedir, savename='anim1_so2.mp4')

    # %%
    hotspot = PlumeSim(venusdict, plume_dict_master['hotspot'], 'pillar of fire')
    hotspot.load_file(hotspot_plume)
    hotspot.set_resolution()

    # %%
    anim2(hotspot, t_start=plume_dict_master['hotspot']['start_time'],
          t_end=plume_dict_master['hotspot']['end_time'],
          lev=22, central_lon=0,
          save=True, savepath=savedir, savename='anim2_vwind.mp4')

    # %%
    hotspot.close()
    del hotspot

    # %%
    anim3(chem, t_start=plume_dict_master['h2o']['start_time'],
          t_end=plume_dict_master['h2o']['end_time'],
          lev=22, central_lon=320,
          save=True, savepath=savedir, savename='anim3_h2o.mp4')

    # %%
    so2.close()
    del so2
# %%
