"""
Utility tools for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).
"""
# %%
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit, minimize

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
plume_dict_master = {
              'plume_1': {'name': 'so2_lev35_eqday', 'lev': 35, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
              'plume_2': {'name': 'so2_lev35_eqnight', 'lev': 35, 'lat_idx': 49, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
              'hotspot1': {'name': 'hotspot1', 'lev': 22, 'lat_idx': 49, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
              'hotspot2': {'name': 'hotspot2', 'lev': 22, 'lat_idx': 49, 'lon_idx': 47, 'start_time': 128, 'end_time': 254},
              'pillar': {'name': 'pillar', 'lev': 22, 'lat_idx': 49, 'lon_idx': 47, 'start_time': 128, 'end_time': 254},
              'h2o': {'name': 'h2o', 'lev': 22, 'lat_idx': 49, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
              'so2' : {'name': 'so2', 'lev': 35, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 128, 'end_time': 254}
            }

name_dict = {'h2o': 'H$_2$O', 'hcl': 'HCl', 'co': 'CO', 'ocs': 'OCS', 'so2': 'SO$_2$'}

# %%
def logistic_decay(levs=22, L=115, K=760, x0=20, k=0.2):
    """
    levs = number of levels
    K = bottom level temp
    L = top level temp
    x0 = midpoint of transition (lev)
    k = decay rate
    """
    x = np.arange(levs)

    return L + (K - L) / (1 + np.exp(k * (x - x0)))

# %%
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
    
    lev       = sim_object.plumes['lev']
    lat_idx   = sim_object.plumes['lat_idx']
    lon_idx   = sim_object.plumes['lon_idx']
    start_time = sim_object.plumes['start_time']

    # 2. Conversions to SI units
    area_m2    = sim_object.data['aire'][lat_idx, lon_idx]  # m2
    volume_m3  = area_m2 * (sim_object.heights[lev] * 1000 - sim_object.heights[lev - 1] * 1000)  # m3
    pressure_pa = sim_object.data['pres'][start_time, lev, lat_idx, lon_idx]  # Pascals

    # 3. Calculate total moles of gas (n = PV / RT)
    n_total = (pressure_pa * volume_m3) / (R * sim_object.data['temp'][start_time, lev, lat_idx, lon_idx])
    # 4. Calculate moles of SO2
    n_so2 = n_total * (vmr_ppm/1e6)  # Convert ppm to fraction
    
    # 5. Convert moles to mass (kg)
    mass_kg = (n_so2 * M_SO2)
    
    return mass_kg

# %%
def source_dispersal_time(plobject, key='so2', threshold=1.05):
    """
    Time in seconds for the source gridbox tracer to return to background
    after the eruption ends.

    The background level is the pre-eruption value at start_time - 1,
    multiplied by threshold. Counting stops at the first timestep after
    end_time where the value falls at or below that level.

    Args:
        plobject (PlumeSim): Simulation object.
        key (str): Tracer variable key. Defaults to 'so2'.
        threshold (float): Multiplier on the pre-eruption value that defines
            "returned to background". 1.05 means within 5% of baseline.

    Returns:
        float: Dispersal time in seconds.
    """
    lev        = plobject.plumes['lev']
    lat_idx    = plobject.plumes['lat_idx']
    lon_idx    = plobject.plumes['lon_idx']
    start_time = plobject.plumes['start_time']
    end_time   = plobject.plumes['end_time']

    series   = plobject.data[key][:, lev, lat_idx, lon_idx]
    interval = float(np.diff(plobject.data.time_counter.values)[0])
    background = float(series[start_time - 1].values) * threshold

    counter = 0
    for t in range(end_time, series.shape[0]):
        if float(series[t].values) > background:
            counter += 1
        else:
            break

    return counter * interval


# %%
def mass_flux(plobject, vmr_ppm=0.195, key='so2', threshold=1.05):
    """
    Mass flux required to maintain the enhanced SO2 concentration in the
    source gridbox, and total SO2 mass injected during the active eruption.

    The outflow rate is the SO2 mass in the source gridbox divided by the
    dispersal time (the time for the source gridbox value to return to
    background once forcing stops). Total injected mass is that rate
    integrated over the active eruption period (start_time to end_time).

    Args:
        plobject (PlumeSim): Simulation object.
        vmr_ppm (float): SO2 vmr in ppm used for the mass calculation.
            Defaults to 0.195.
        key (str): Tracer variable key. Defaults to 'so2'.
        threshold (float): Threshold multiplier for background definition
            passed to source_dispersal_time. Defaults to 1.05.

    Returns:
        dict with keys:
            'mass_in_box_kg'    (float): SO2 mass in source gridbox (kg).
            'dispersal_time_s'  (float): Time to return to background (s).
            'mass_outflow_kgs'  (float): Required outflow rate to maintain
                                         enhanced concentration (kg/s).
            'duration_s'        (float): Active eruption duration (s).
            'total_injected_kg' (float): Total SO2 injected over eruption
                                         period (kg).
    """
    mass_kg = calculate_so2_mass(plobject, vmr_ppm=vmr_ppm)
    tau_s   = source_dispersal_time(plobject, key=key, threshold=threshold)

    interval   = float(np.diff(plobject.data.time_counter.values)[0])
    start_time = plobject.plumes['start_time']
    end_time   = plobject.plumes['end_time']
    duration_s = (end_time - start_time) * interval

    mass_outflow_kgs  = mass_kg / tau_s
    total_injected_kg = mass_outflow_kgs * duration_s

    return {
        'mass_in_box_kg':    mass_kg,
        'dispersal_time_s':  tau_s,
        'mass_outflow_kgs':  mass_outflow_kgs,
        'duration_s':        duration_s,
        'total_injected_kg': total_injected_kg,
    }


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
def local_time(plobject, time_slice=-1, silent='no'):
    
    """ A function that calculates the local time for a
    snapshot from a given timestep."""
    equator = np.argmin(np.abs(plobject.lats))
    # Find row number of latitude closest to 0,
    # aka the equator
    rad_toa = plobject.data['tops']
    # Solar radiation at top of atmosphere
    subsol = int(rad_toa[time_slice,equator,:].argmax(dim='lon').values)
    # Find column number of longitude where solar
    # radiation is currently at a maximum
    if silent=='no':
        print('Local noon is at col ' + str((subsol)))
        print('Local noon is at lon ' + str(plobject.lons[subsol]))
    else:
        pass
    dt = 24/len(plobject.lons)
    hours = np.arange(24,0,-dt)
    # Array of hour coordinates with same
    # dimension as longitude coordinates
    roll_step = int(subsol - (len(plobject.lons)/2))
    new_hours = list(np.roll(hours, roll_step))

    return new_hours

# %%
def spicav_max_so2(ds):
    """
    Find the maximum SO2 abundance in a SPICAV dataset.

    Returns the peak SO2 value and its associated orbit number, date,
    latitude, and spatial coordinate (longitude or local solar time).

    Args:
        ds (xr.Dataset): SPICAV SO2 dataset containing a so2_ppbv variable.

    Returns:
        dict with keys:
            'so2_ppbv'  (float): Peak SO2 value in ppbv.
            'orbit'     (int):   VEX orbit number.
            'date':              Observation date (numpy datetime64).
            'latitude'  (float): Latitude in degrees.
            'longitude' (float) or 'local_solar_time' (float):
                                 Spatial coordinate in degrees or hours.
    """
    df = ds['so2_ppbv'].to_dataframe().dropna().reset_index()

    if df.empty:
        raise ValueError("Dataset contains no valid SO2 observations.")

    row = df.loc[df['so2_ppbv'].idxmax()]

    orb_num = int(ds.sel(start_date=row['start_date']).vex_orbit.values)

    result = {
        'so2_ppbv': float(row['so2_ppbv']),
        'orbit':    orb_num,
        'date':     row['start_date'],
        'latitude': float(row['latitude']),
    }

    if 'longitude' in df.columns:
        result['longitude'] = float(row['longitude'])
    elif 'local_solar_time' in df.columns:
        result['local_solar_time'] = float(row['local_solar_time'])

    return result

# %%

# %%
def spicav_campaigns(ds, gap_days=21):
    """
    Group the SPICAV observations into observing campaigns.

    Computes the latitude-weighted mean SO2 per observation, then groups
    consecutive observations separated by less than gap_days into campaigns.
    Each campaign is summarised by the median of log10(SO2), the 16th-84th
    percentile spread, and the standard error of the median.

    Args:
        ds (xr.Dataset): SPICAV observations dataset with dims
            (start_date, latitude, longitude).
        gap_days (int): A gap longer than this starts a new campaign.

    Returns:
        (pd.DataFrame, pd.Timestamp): One row per campaign (date, t in days
            since first observation, log, lo, hi, sem, n), and the date of
            the first observation.
    """
    dates = pd.to_datetime(ds.start_date.values)
    w     = np.cos(np.deg2rad(ds.latitude))
    so2   = ds.so2_ppbv.weighted(w).mean(dim=['latitude', 'longitude']).values

    df = pd.DataFrame({'date': dates, 'so2': so2}).dropna()
    df = df[df['so2'] > 0].sort_values('date').reset_index(drop=True)

    t0 = df['date'].iloc[0]
    t  = (df['date'] - t0).dt.days.values.astype(float)
    y  = np.log10(df['so2'].values)

    label = np.concatenate([[0], np.cumsum(np.diff(t) > gap_days)])

    rows = []
    for k in range(label.max() + 1):
        m  = label == k
        yk = y[m]
        sig = (np.percentile(yk, 84) - np.percentile(yk, 16)) / 2
        rows.append({
            'date': t0 + pd.Timedelta(days=float(t[m].mean())),
            't':    t[m].mean(),
            'log':  np.median(yk),
            'lo':   np.percentile(yk, 16),
            'hi':   np.percentile(yk, 84),
            'sem':  max(1.25 * sig / np.sqrt(m.sum()), 0.05),
            'n':    int(m.sum()),
        })
    return pd.DataFrame(rows), t0

# %%
def trapezoid(t, t_peak, t_plateau, y_start, y_peak, y_plateau):
    """
    Continuous piecewise-linear rise-fall-plateau model in log space:
    linear rise from (0, y_start) to (t_peak, y_peak), linear fall to
    (t_plateau, y_plateau), constant afterwards.
    """
    y = np.empty_like(t, dtype=float)
    rise = t <= t_peak
    fall = (t > t_peak) & (t <= t_plateau)
    flat = t > t_plateau
    y[rise] = y_start + (y_peak - y_start) * t[rise] / t_peak
    y[fall] = y_peak + (y_plateau - y_peak) * (t[fall] - t_peak) / (t_plateau - t_peak)
    y[flat] = y_plateau
    return y

# %%
def fit_trapezoid(camp, weighted=True, n_solutions=1):
    """
    Weighted least-squares fit of the rise-fall-plateau model to the
    campaign medians, from multiple starting points.

    The fit has two nearly-degenerate solutions, differing in whether the
    deep 2010-11 minimum is treated as part of the decline or the plateau.
    With n_solutions=2, both are returned (best first).

    Args:
        camp (pd.DataFrame): Campaign summary from spicav_campaigns().
        weighted (bool): Weight campaigns by 1/sem^2 if True, equally if False.
        n_solutions (int): 1 for the best fit, 2 to also return the
            alternative local minimum.

    Returns:
        list of (float, np.ndarray): (cost, params) per solution, where
            params = (t_peak, t_plateau, y_start, y_peak, y_plateau).
    """
    t = camp['t'].values
    y = camp['log'].values
    w = 1.0 / camp['sem'].values**2 if weighted else np.ones_like(y)
    t_end = t[-1]

    def cost(p):
        t_peak, t_plateau, y_start, y_peak, y_plateau = p
        if not (30 < t_peak < t_plateau < t_end) or y_peak <= y_start:
            return 1e12
        return np.sum(w * (y - trapezoid(t, *p))**2)

    sols = []
    for tp0 in [150, 263, 400]:
        for tpl0 in [700, 900, 1200, 1600, 2200]:
            p0 = [tp0, tpl0, np.min(y[:5]), np.max(y), np.median(y[t > tpl0])]
            res = minimize(cost, p0, method='Nelder-Mead',
                           options={'maxiter': 50000, 'xatol': 0.5,
                                    'fatol': 1e-10})
            sols.append((res.fun, res.x))
    sols.sort(key=lambda s: s[0])

    best = [sols[0]]
    if n_solutions > 1:
        for fun, x in sols[1:]:
            if abs(x[1] - best[0][1][1]) > 200:
                best.append((fun, x))
                break
    return best
# %%
