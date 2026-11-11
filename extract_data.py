"""
Extract the plume injection level and the 4 model levels above and below it
from the main-paper VPCM simulations, for deposit in an open data repository.

Only the perturbed fields are kept, together with the grid and metadata:
    SO2 plume      -> so2
    Hotspot (simple) -> temp, vitu, vitv, vitwz
    H2O plume      -> h2o

Values are copied unchanged. Files are opened with decode_cf=False so raw
values and attributes are written back exactly as stored in the model output.
Data are copied in time blocks so that memory use stays bounded without dask.
"""
# %%
import os

import netCDF4
import xarray as xr

from config import so2_plume, hotspot2_plume, chem_plume, data_out
from tools import plume_dict_master

# %%
LEV_DIM = 'presnivs'
TIME_DIM = 'time_counter'
HALF_WIDTH = 4          # levels kept either side of the injection level
TIME_BLOCK = 100        # time steps copied per write

RUNS = {
    'so2':      {'path': so2_plume,      'vars': ['so2'],
                 'outname': 'vpcm_so2_plume'},
    'hotspot2': {'path': hotspot2_plume, 'vars': ['temp', 'vitu', 'vitv', 'vitwz'],
                 'outname': 'vpcm_hotspot', 'levels': (22,35), 'ntime': 1600},
    'h2o':      {'path': chem_plume,     'vars': ['h2o'],
                 'outname': 'vpcm_h2o_plume'},
}


# %%
def ancillary_vars(ds, names):
    """
    Find data variables that describe the grid of the selected variables:
    those listed in their 'coordinates' attribute and the bounds of any
    coordinate. These are data variables rather than coordinates because the
    file is opened with decode_cf=False.
    """
    found = set()
    for name in names:
        for ref in ds[name].attrs.get('coordinates', '').split():
            if ref in ds.data_vars:
                found.add(ref)
    for name in list(ds.coords) + sorted(found):
        bnds = ds[name].attrs.get('bounds')
        if bnds in ds.data_vars:
            found.add(bnds)
    return sorted(found)


def extract_run(key, run, outdir, half_width=HALF_WIDTH, time_block=TIME_BLOCK):
    """
    Write the injection level +/- half_width levels of the selected variables
    of one simulation to a new netCDF file.

    Args:
        key (str): Key of the run in plume_dict_master.
        run (dict): Entry of RUNS with input path, variables and output name.
        outdir (str): Output directory.
        half_width (int): Number of levels kept above and below the injection level.
        time_block (int): Number of time steps copied per write.

    Returns:
        str: Path of the written file.
    """
    lev = plume_dict_master[key]['lev']
    lev_lo, lev_hi = run.get('levels'), (lev - half_width, lev + half_width)

    ds = xr.open_dataset(run['path'], decode_cf=False)
    nlev = ds.sizes[LEV_DIM]
    if lev_lo < 0 or lev_hi >= nlev:
        raise ValueError(f'Levels {lev_lo}-{lev_hi} outside 0-{nlev - 1} for {key}')

    keep = run['vars'] + ancillary_vars(ds, run['vars'])
    out = ds[keep].isel({LEV_DIM: slice(lev_lo, lev_hi + 1),
                         TIME_DIM: slice(0, run.get('ntime'))})
    out.attrs['extract_description'] = (
        f'Model levels {lev_lo}-{lev_hi} (0-based indices into the original '
        f'{nlev}-level {LEV_DIM} axis); the injection level is {lev}. '
        f'First {out.sizes[TIME_DIM]} of {ds.sizes[TIME_DIM]} output times. '
        f'Values are unmodified model output.')
    out.attrs['injection_level_index'] = lev
    out.attrs['injection_lat_index'] = plume_dict_master[key]['lat_idx']
    out.attrs['injection_lon_index'] = plume_dict_master[key]['lon_idx']

    # Fresh encoding: the original chunk sizes do not fit the sliced arrays
    encoding = {}
    for name in list(out.data_vars) + list(out.coords):
        out[name].encoding = {}
        if out[name].ndim > 0 and out[name].dtype.kind in 'fiu':
            encoding[name] = {'zlib': True, 'complevel': 4}

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{run['outname']}_lev{lev_lo}-{lev_hi}.nc")
    nt = out.sizes[TIME_DIM]
    print(f'{key}: writing {keep} levels {lev_lo}-{lev_hi}, {nt} time steps -> {outpath}')

    # First block creates the file with all grid variables and metadata
    out.isel({TIME_DIM: slice(0, time_block)}).to_netcdf(
        outpath, unlimited_dims=[TIME_DIM], encoding=encoding)

    # Remaining blocks are appended along the unlimited time dimension
    timed = [n for n in list(out.variables) if TIME_DIM in out[n].dims]
    with netCDF4.Dataset(outpath, 'a') as nc:
        nc.set_auto_maskandscale(False)
        for t0 in range(time_block, nt, time_block):
            t1 = min(t0 + time_block, nt)
            block = out.isel({TIME_DIM: slice(t0, t1)})
            for name in timed:
                axis = out[name].dims.index(TIME_DIM)
                index = [slice(None)] * out[name].ndim
                index[axis] = slice(t0, t1)
                nc[name][tuple(index)] = block[name].values
            print(f'  {t1}/{nt}')

    ds.close()
    return outpath


# %%
if __name__ == '__main__':
    for key, run in RUNS.items():
        extract_run(key, run, data_out)
