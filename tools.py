"""
Utility tools for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).
"""

import numpy as np
import xarray as xr

# %% Definition of basic Venus model constants
# radius: km, g: m/s^2, periods: days, psurf: bar
# molmass: kg/mol, R: J/(mol K), scaleh: km
venusdict = {'radius': 6051.3, 'g': 8.87, 'rotperiod' : 243.0,
             'revperiod': 224.7, 'rotrate': 2.99e-07, 'psurf': 92.,
             'molmass': 0.04401, 'R' : 8.3143, 'RCO2' : 188.92, 'rhoconst': 65.,
             'scaleh': 16.,
             'name': 'Venus'}
# Altitude levels in km for VPCM 50 level outputs
heights50 = [9.4504710e-03, 4.8338074e-02, 1.5300426e-01, 3.7324557e-01,
       7.5411582e-01, 1.3392169e+00, 2.1695750e+00, 3.2827427e+00,
       4.7116103e+00, 6.4832373e+00, 8.6177559e+00, 1.1128335e+01,
       1.4020479e+01, 1.7288918e+01, 2.0912828e+01, 2.4761955e+01,
       2.8558361e+01, 3.2130924e+01, 3.5487930e+01, 3.8643879e+01,
       4.1625511e+01, 4.4466003e+01, 4.7189983e+01, 4.9781677e+01,
       5.2210793e+01, 5.4474300e+01, 5.6580246e+01, 5.8551716e+01,
       6.0436192e+01, 6.2270138e+01, 6.4059181e+01, 6.5802261e+01,
       6.7509895e+01, 6.9198883e+01, 7.0857117e+01, 7.2468269e+01,
       7.4031113e+01, 7.5544807e+01, 7.7022598e+01, 7.8483002e+01,
       7.9931213e+01, 8.1360779e+01, 8.2763893e+01, 8.4131378e+01,
       8.5458458e+01, 8.6925591e+01, 8.8879539e+01, 9.1365417e+01,
       9.4282623e+01, 9.7128662e+01]
# Altitude levels in km for VPCM 78 level outputs
heights78 = [9.45306290e-03, 4.83510271e-02, 1.53046221e-01, 3.73352647e-01,
       7.54337728e-01, 1.33960199e+00, 2.17016840e+00, 3.28359985e+00,
       4.71279430e+00, 6.48478937e+00, 8.61961460e+00, 1.11300869e+01,
       1.40210838e+01, 1.72868519e+01, 2.09060841e+01, 2.47483273e+01,
       2.85367489e+01, 3.21012650e+01, 3.54501762e+01, 3.85974350e+01,
       4.15684090e+01, 4.43950195e+01, 4.71000137e+01, 4.96712570e+01,
       5.20832405e+01, 5.43336220e+01, 5.64306679e+01, 5.83991661e+01,
       6.02850800e+01, 6.21196480e+01, 6.39083824e+01, 6.56521988e+01,
       6.73622131e+01, 6.90550003e+01, 7.07164383e+01, 7.23295288e+01,
       7.38933105e+01, 7.54046097e+01, 7.68734818e+01, 7.83174973e+01,
       7.97435608e+01, 8.11459122e+01, 8.25166473e+01, 8.38473129e+01,
       8.51307755e+01, 8.65320282e+01, 8.83726730e+01, 9.07071533e+01,
       9.34497375e+01, 9.65266037e+01, 9.90068893e+01, 1.00908829e+02,
       1.02824753e+02, 1.04765404e+02, 1.06736778e+02, 1.08729904e+02,
       1.10737305e+02, 1.12745575e+02, 1.14689415e+02, 1.16525398e+02,
       1.18274094e+02, 1.19964310e+02, 1.21610733e+02, 1.23218185e+02,
       1.24797585e+02, 1.26358299e+02, 1.27900208e+02, 1.29420120e+02,
       1.30924683e+02, 1.32434402e+02, 1.33977646e+02, 1.35579971e+02,
       1.37256210e+02, 1.39007202e+02, 1.40822159e+02, 1.42683762e+02,
       1.44574203e+02, 1.46479889e+02]
# Dictionary of plume coordinates
plume_dict = {  'plume_1': {'name': 'h2o_lev10_eq', 'lev': 10, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_2': {'name': 'h2o_lev10_hl', 'lev': 10, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
                'plume_3': {'name': 'h2o_hcl_lev14_eq', 'lev': 14, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_4': {'name': 'h2o_hcl_lev14_hl', 'lev': 14, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
                'plume_5': {'name': 'four_gases_lev18_eq', 'lev': 18, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_6': {'name': 'four_gases_lev18_hl', 'lev': 18, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
#               'plume_7': {'name': 'so2_lev35_eq', 'lev': 35, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
#                'plume_8': {'name': 'so2_lev35_hl', 'lev': 35, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127}
            }

name_dict = {'h2o': 'H$_2$O', 'hcl': 'HCl', 'co': 'CO', 'ocs': 'OCS', 'so2': 'SO$_2$'}

def find_plume(plobject, key, lev, threshold):
    """
    Find array indices of maximum tracer value.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        key (str): Dictionary key of the data variable.
        lev (int): Vertical level index.
        threshold (float or None): Threshold value. If None, max value is used.

    Returns:
        tuple: (start_time, end_time, lat_idx, lon_idx) indices.
    """
    # Extract age of air tracer cube at desired model level
    cube = plobject.data[key][:,lev,:,:]
    
    # What is the peak mmr of the plume injection?
    if threshold is None:
        threshold = cube.max()
    
    mask = np.where(cube >= threshold)
    # print(mask) # Debug
    
    # Get 0th element of 0th dimension (first time occurrence)
    start_time = mask[0][0]
    
    # Get last element of 0th dimension (last time occurrence)
    end_time = mask[0][-1]
    
    # Get 0th element of 1st dimension (latitude) - all should be identical
    lat_idx = mask[1][0]
    
    # Get 0th element of 2nd dimension (longitude) - all should be identical
    lon_idx = mask[2][0]

    return start_time, end_time, lat_idx, lon_idx

def calculate_so2_mass(sim_object, vmr_ppm=0.195):
    """
    Calculates the mass of SO2 in a rectangular volume.
    
    Parameters:
    sim_object (PlumeSim): PlumeSim object containing the simulation data.
    vmr_ppm (float): Volume mixing ratio of SO2 in parts per million.
    
    Returns:
    float: Total mass of SO2 in kilograms.
    """
    # 1. Constants
    R = 8.31446      # Ideal gas constant (J / mol*K)
    M_SO2 = 0.06406   # Molar mass of SO2 in kg/mol
    
    # 2. Conversions to SI units
    area_m2 = sim_object.data['aire'][plume_dict['plume_7']['lat_idx'], plume_dict['plume_7']['lon_idx']] # m2
    volume_m3 = area_m2 * (sim_object.heights[plume_dict['plume_7']['lev']] * 1000 - sim_object.heights[plume_dict['plume_7']['lev']-1] * 1000)  # m3
    pressure_pa = sim_object.data['pres'][plume_dict['plume_7']['start_time'], plume_dict['plume_7']['lev'], plume_dict['plume_7']['lat_idx'], plume_dict['plume_7']['lon_idx']] # Pascals
    
    # 3. Calculate total moles of gas (n = PV / RT)
    n_total = (pressure_pa * volume_m3) / (R * sim_object.data['temp'][plume_dict['plume_7']['start_time'], plume_dict['plume_7']['lev'], plume_dict['plume_7']['lat_idx'], plume_dict['plume_7']['lon_idx']])
    # 4. Calculate moles of SO2
    n_so2 = n_total * (vmr_ppm/1e6)  # Convert ppm to fraction
    
    # 5. Convert moles to mass (kg)
    mass_kg = (n_so2 * M_SO2)
    
    return mass_kg

# %%
def max_dispersal(plobject, lev, lat, threshold=1.05):

    cube = plobject.data['h2o'][:,lev,:,:]
    interval = np.diff(cube.time_counter.values)[0]
    background = np.mean(cube.values)*threshold
    post_eruption = cube[plobject.plumes['plume_1']['end_time']:,:,:]
    # Only consider data after plume forcing has finished
    mask = post_eruption.values > background
    # Boolean array with same dimension as cube, True is where values are above threshold
    flattened = np.count_nonzero(mask, axis=0)
    # Count how many time outputs are True for each lon x lat point
    map_hours = flattened * interval / (60*60)

    top_half = np.max(map_hours[lat:, :])
    top_coords = np.argmax(map_hours[lat:, :])
    top_row, top_col = np.unravel_index(top_coords, map_hours.shape)
    bottom_half = np.max(map_hours[:lat, :])
    bottom_coords = np.argmax(map_hours[:lat, :])
    bottom_row, bottom_col = np.unravel_index(bottom_coords, map_hours.shape)

    results_dict = {'hl_plume_max': top_half, 'hl_max_coords': (lat + top_row, top_col), 'eq_plume_max': bottom_half, 'eq_max_coords': (bottom_row, bottom_col)}
    return results_dict
# %%
