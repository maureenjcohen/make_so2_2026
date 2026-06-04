"""
Plotting functions for 'Decadal sulphur dioxide variability on Venus not caused by volcanic injection'
(Cohen et al. 2027).
"""
# %% Filepaths
eqday_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_day.nc'
eqnight_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_night.nc'
so2_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_so2.nc'
so2_plume_big = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_so2_big.nc'
chem_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_chem.nc'
hotspot2_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/hotspot_96x96x50/Xins_hotspot2.nc'
hotspot1_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/hotspot_96x96x50/Xins_hotspot1.nc'
pillar_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/hotspot_96x96x50/Xins_pillar.nc'
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
from tools import venusdict, plume_dict_master, calculate_so2_mass, name_dict, source_dispersal_time, mass_flux
from supp_figs import fig_so2_prof, fig_temp_profile, fig_logistic, fig_orbit_summary, fig_orbit_path
from animations import anim1, anim2, anim3


# %% Main code block
if __name__ == "__main__":

    # %%
    # Open retrievals dataset
    ds_retrievals = xr.open_dataset(retrievals)

    # %%
    figure0(ds_retrievals, rolling_window=60, 
            peak_search_before='2008-01-01', trough_threshold=20, 
            save=True, savepath=savedir, savename='fig0_ref_20.' + filetype)
    # %%
    fig_orbit_summary(ds_retrievals, orbit_idx=(41, 50, 60),
                      save=True, savepath=savedir, savename='figS_orbit_summary.' + filetype)

    # %%
    # Create PlumeSim object and load data for plume run
    eqday = PlumeSim(venusdict, plume_dict_master['plume_1'], 'equatorial dayside plume')
    eqday.load_file(eqday_plume)
    eqday.set_resolution()

    eqnight = PlumeSim(venusdict, plume_dict_master['plume_2'], 'equatorial nightside plume')
    eqnight.load_file(eqnight_plume)
    eqnight.set_resolution()

    # %%
    eqday_mass = mass_flux(eqday, vmr_ppm=1.0, key='so2', threshold=1.05)
    eqnight_mass = mass_flux(eqnight, vmr_ppm=1.0, key='so2', threshold=1.05)

    # %%
    eqday.close()
    eqnight.close()
    del eqday, eqnight
 
    # %%
    # Create PlumeSim object and load data for plume run
    so2 = PlumeSim(venusdict, plume_dict_master['so2'], 'equatorial dayside plume')
    so2.load_file(so2_plume)
    so2.set_resolution()
    
    # %%
    figure1(so2, ds_retrievals, snap_times=[250, 316, 404, 608], 
            long_start=0, long_end=334, zoom_start=38, zoom_end=72, 
            plume_anchor=38, lev=35, central_lon=-40, save=True,
            savepath=savedir, savename='fig1_ref.' + filetype)
    
    # %%
    # Create PlumeSim object and load data for plume run
    so2_big = PlumeSim(venusdict, plume_dict_master['so2'], 'BIG equatorial dayside plume')
    so2_big.load_file(so2_plume_big)
    so2_big.set_resolution()
    
    # %%
    figure1(so2_big, ds_retrievals, snap_times=[250, 316, 404, 608], 
            long_start=0, long_end=334, zoom_start=38, zoom_end=72, 
            plume_anchor=38, lev=35, central_lon=-40, save=True,
            savepath=savedir, savename='fig1_ref_big.' + filetype)
    
    # %%
    so2_big.close()
    del so2_big

    # %%
    hotspot2 = PlumeSim(venusdict, plume_dict_master['hotspot2'], 'simple hotspot')
    hotspot2.load_file(hotspot2_plume)
    hotspot2.set_resolution()
    
    # %% Late start
    figure2(hotspot2, snap_times=[130, 196, 230, 256, 300, 368], 
            ts_levs=[21, 28, 35, 40], ts_start=100, ts_end=700, 
            zu_start=0, zu_end=1500, zu_anomaly=True,
            lev=35, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-1.0, vmax=1.0, save=True, savepath=savedir, 
            savename='fig2_hotspot2.' + filetype)
    
    # %%
    hotspot1 = PlumeSim(venusdict, plume_dict_master['hotspot1'], 'increasing hotspot')
    hotspot1.load_file(hotspot1_plume)
    hotspot1.set_resolution()
    
    # %% Early start
    figure2(hotspot1, snap_times=[10, 30, 90, 150, 312, 508], 
            ts_levs=[21, 28, 35, 40], ts_start=0, ts_end=700, 
            zu_start=0, zu_end=3000, zu_anomaly=True,
            lev=35, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-3.0, vmax=3.0, save=True, savepath=savedir, 
            savename='fig2_hotspot1.' + filetype)
    
    # %%
    pillar = PlumeSim(venusdict, plume_dict_master['pillar'], 'pillar of fire')
    pillar.load_file(pillar_plume)
    pillar.set_resolution()
    
    # %% Late start
    figure2(pillar, snap_times=[130, 196, 256, 300, 512, 1842], 
            ts_levs=[21, 28, 35, 40], ts_start=100, ts_end=700, 
            zu_start=0, zu_end=3600, zu_anomaly=True,
            lev=35, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-3.0, vmax=3.0, save=True, savepath=savedir, 
            savename='fig2_pillar.' + filetype)

    # %%
    chem = PlumeSim(venusdict, plume_dict_master['h2o'], 'potential absorber plume')
    chem.load_file(chem_plume)
    chem.set_resolution()

    # %%
    figure3(chem, snap_times=[10, 140, 304, 364], ts_levs=[21, 22, 23], 
            ts_start=0, ts_end=2000, central_lon=320,
            save=True, savepath=savedir, savename='fig3_h2o.' + filetype)

    # %%
    fig_so2_prof(so2, times=[0, 1000, 2000, 3000, 4000, 5000],
                 save=True, savepath=savedir, savename='figS_so2_prof.' + filetype)

    # %%
    fig_temp_profile(hotspot2, ylim_bottom=20, ylim_top=35,
                     t=hotspot2.plumes['start_time']+6, save=True, savepath=savedir, 
                     savename='figS_temp_profile_hotspot2.' + filetype)
    
    # %%
    fig_temp_profile(hotspot1, ylim_bottom=20, ylim_top=35,
                    t=hotspot1.plumes['start_time']+6, save=True, savepath=savedir, 
                    savename='figS_temp_profile_hotspot1.' + filetype)
    # %%
    fig_temp_profile(pillar, ylim_bottom=20, ylim_top=35,
                    t=pillar.plumes['start_time']+6, save=True, savepath=savedir, 
                    savename='figS_temp_profile_pillar.' + filetype)

    # %%
    fig_logistic(pillar, save=True, savepath=savedir, savename='figS_logistic.' + filetype)

    # %%
    anim1(so2, t_start=plume_dict_master['so2']['start_time'],
          t_end=650,
          lev=35, central_lon=-40,
          save=True, savepath=savedir, savename='anim1_so2.mp4')

    # %%
    anim2(hotspot2, t_start=plume_dict_master['hotspot2']['start_time'],
          t_end=700, vmin=-1.0, vmax=1.0,
          lev=35, central_lon=0,
          save=True, savepath=savedir, savename='anim2_vwind_hotspot2.mp4')
    
    # %%
    anim2(hotspot1, t_start=plume_dict_master['hotspot1']['start_time'],
          t_end=700, vmin=-3.0, vmax=3.0,
          lev=35, central_lon=0,
          save=True, savepath=savedir, savename='anim2_vwind_hotspot1.mp4')
    
    # %%
    anim2(pillar, t_start=plume_dict_master['pillar']['start_time'],
          t_end=700, vmin=-3.0, vmax=3.0,
          lev=35, central_lon=0,
          save=True, savepath=savedir, savename='anim2_vwind_pillar.mp4')

    # %%
    anim3(chem, t_start=plume_dict_master['h2o']['start_time'],
          t_end=650,
          lev=22, central_lon=320,
          save=True, savepath=savedir, savename='anim3_h2o.mp4')

    # %%
    so2.close()
    del so2

    hotspot2.close()
    hotspot1.close()
    pillar.close()
    del hotspot2, hotspot1, pillar

    chem.close()
    del chem
# %%
