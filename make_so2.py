"""
Plotting functions for 'Decadal sulphur dioxide variability on Venus not caused by volcanic injection'
(Cohen et al. 2027).
"""
# %% Filepaths
eqday_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_day.nc'
eqnight_plume = '/exomars/data/internal/working/mc5526/VPCM_decadal/Xins_eq_night.nc'
so2_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_so2.nc'
chem_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/so2_plume_96x96x78/Xins_chem.nc'
hotspot_plume = '/exomars/projects/mc5526/VPCM_decadal_so2/hotspot_96x96x50/Xins_pillar.nc'
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
from fig1 import figure1, figure2, STYLE
from tools import venusdict, plume_dict, calculate_so2_mass, name_dict


# %% Main code block
if __name__ == "__main__":

    # %%
    # Open retrievals dataset
    ds_retrievals = xr.open_dataset(retrievals)

    # %%
    # Create PlumeSim object and load data for plume run
    eqday_sim = PlumeSim(venusdict, plume_dict, 'equatorial dayside plume')
    eqday_sim.load_file(so2_plume)
    eqday_sim.set_resolution()

    # %%
    figure1(eqday_sim, ds_retrievals, snap_times=[8, 158, 208, 278], 
            long_start=0, long_end=100, zoom_start=38, zoom_end=72, 
            plume_anchor=41, lev=35, central_lon=130)
    
    # %%
    eqday_sim.close()
    del eqday_sim

    # %%
    pillar = PlumeSim(venusdict, plume_dict, 'pillar of fire')
    pillar.load_file(hotspot_plume)
    pillar.set_resolution()
    
    # %%
    figure2(pillar, snap_times=[130, 196, 230, 256, 300, 368], 
            ts_levs=[22, 28, 34, 40], ts_start=100, ts_end=700, 
            lev=22, central_lon=0, lat_idx=49, lon_idx=47, 
            vmin=-1.2, vmax=3.0)

    # %%
   # Animation + cover image of SO2 plume at 70 km
    animate_chem_plume(eqday_sim, lev=35, keys=['so2'], name_dict=name_dict, t0=0, tf=1200, qscale=5, log=False, savepath=savedir, savename='eqday_plume_no_log', snapshot=100)
    
    # %%
    # Dispersal time of SO2 plumes at 70 km
    dispersal_time(eqday_sim, lev=35, keys=['so2'], lats=[49,82], lons=[92,47], name_dict=name_dict,
                   threshold=1.05, axis_len=500,
                   save=True, savename='eq_day_dispersal' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    
    # %%
    data_compare(ds_retrievals, 40, 71, eqday_sim, plume_dict, log=True, weighted=False, save=True)

    # %%
    # Calculation of SO2 mass in plume at 70 km for reference
    so2_mass_kg = calculate_so2_mass(eqday_sim, vmr_ppm=0.195)
    print(f'SO2 mass in plume at 70 km: {so2_mass_kg:.2f} kg')

    # %%
    eqnight_sim = PlumeSim(venusdict, plume_dict, 'equatorial nightside plume')
    eqnight_sim.load_file(eqnight_plume)
    eqnight_sim.set_resolution()

    # %%
    animate_chem_plume(eqnight_sim, lev=35, keys=['so2'], name_dict=name_dict, t0=0, tf=1200, qscale=5, log=False, savepath=savedir, savename='eqnight_plume_no_log', snapshot=100)
# %%
