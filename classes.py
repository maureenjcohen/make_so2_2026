"""
Class definitions for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).
"""

import xarray as xr
import numpy as np

from tools import venusdict, heights50, heights78

class PlumeSim:
    """
    A PlumeSim object which contains the output data for a simulation.
    """
    
    def __init__(self, planetdict, plume_dict, run):
        """
        Initiate a PlumeSim object.

        Args:
            planetdict (dict): Dictionary of planet constants.
            model (str): Name of the model ('vpcm').
            run (str): Name of the run, should be the scaling factor.
        """
        self.name = planetdict['name']
        self.plumes = plume_dict
        self.run = run
        # Easter egg
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset')
        for key, value in planetdict.items():
            setattr(self, key, value)

    def load_file(self, fn):
        """
        Load a netCDF file using the xarray package and store it in the object.

        Lists dictionary key, name, dimensions, and shape of each data cube
        and stores text in a reference list.

        Args:
            fn (str or list): Filename or list of filenames to load.
        """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time_counter', decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = f"{key}: {ds[key].long_name}, {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
            else:
                keystring = f"{key}: {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """
        Close the netCDF file packaged in the PlumeSim data object.
        """
        self.data.close()
        print('PlumeSim object associated dataset has been closed')

    def set_resolution(self):
        """
        Automatically detect file resolution and assign aesthetically
        pleasing coordinate arrays to the object for use in labelling plots.
        """
        self.lons = np.round(self.data.variables['lon'].values)
        self.lats = np.round(self.data.variables['lat'].values)
        self.tinterval = np.diff(self.data['time_counter'][0:2])[0]
        self.areas = self.data.variables['aire'].values
        if len(self.data.variables['presnivs'][:]) == 50:
            self.heights = np.array(heights50)
        elif len(self.data.variables['presnivs'][:]) == 78:
            self.heights = np.array(heights78)
        else:
            print('Altitude in km not available')       
        self.set_vertical()
        print(f"Resolution is {len(self.lats)} lats, {len(self.lons)} lons, {self.vert} levs")
        print(f'Vertical axis is {self.vert_axis}')

    def set_vertical(self):
        """
        Identify and set vertical axis and units.
        """
        self.levs = self.data['presnivs'].values
        self.vert = len(self.levs)
        self.vert_unit = self.data['presnivs'].units
        try:
            self.vert_axis = self.data['presnivs'].long_name
        except:
            self.vert_axis = self.data['presnivs'].standard_name