"""
Plotting functions for 'Decadal sulphur dioxide variability on Venus not caused by volcanic injection'
(Cohen et al. 2027).
"""
# %% Filepaths
plumes = '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.5nocl.nc'
chem_dir = '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/'
chem = [chem_dir + 'Xins4.nc',  chem_dir + 'Xins5.nc', chem_dir + 'Xins6.nc', chem_dir + 'Xins7.nc']
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
from plotting_functions import zmage, dispersal_time, dispersal_map, animate_chem_plume, summ_stats, sensitivity_test, plume_cross_section
from tools import venusdict, plume_dict, calculate_so2_mass, name_dict


# %% Main code block
if __name__ == "__main__":

    # Create PlumeSim object and load data for chemistry run
    chem_sim = PlumeSim(venusdict, None, 'chem_4days')
    chem_sim.load_file(chem)
    chem_sim.set_resolution()

    summ_stats(chem_sim, keys=['so2'], lev=35, t0=0, tf=None,
                name_dict=name_dict,
                savename='fig16' + '.' + filetype,
                savepath=savedir,
                save=True, sformat=filetype)
    
    chem_sim.close()
    del chem_sim

    # Create PlumeSim object and load data for plume run
    plume_sim = PlumeSim(venusdict, plume_dict, 'scale_1.5')
    plume_sim.load_file(plumes)
    plume_sim.set_resolution()

   # Animation + cover image of SO2 plume at 70 km
    animate_chem_plume(plume_sim, lev=35, keys=['so2'], t0=0, tf=250, qscale=5, savepath=savedir, savename='fig13', snapshot=100)
    
    # Dispersal time of SO2 plumes at 70 km
    dispersal_time(plume_sim, lev=35, keys=['so2'], lats=[49,82], lons=[92,47], axis_len=500,
                   save=True, savename='fig12' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    
    # Calculation of SO2 mass in plume at 70 km for reference
    so2_mass_kg = calculate_so2_mass(plume_sim, vmr_ppm=0.195)
    print(f'SO2 mass in plume at 70 km: {so2_mass_kg:.2f} kg')