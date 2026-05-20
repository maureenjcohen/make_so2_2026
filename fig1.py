"""
Plotting functions for 'Decadal sulphur dioxide variability on Venus is not caused by volcanic activity'
(Cohen et al. 2026).
"""
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib import colors
import pandas as pd

from tools import plume_dict, max_dispersal

# %%
def sulfur_plume():