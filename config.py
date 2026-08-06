import os
from dotenv import load_dotenv

load_dotenv()

# %% VPCM plume simulations
eqday_plume = os.getenv('EQDAY_PLUME')
eqnight_plume = os.getenv('EQNIGHT_PLUME')
so2_plume = os.getenv('SO2_PLUME')
so2_plume_big = os.getenv('SO2_PLUME_BIG')
chem_plume = os.getenv('CHEM_PLUME')
hotspot1_plume = os.getenv('HOTSPOT1_PLUME')
hotspot2_plume = os.getenv('HOTSPOT2_PLUME')
pillar_plume = os.getenv('PILLAR_PLUME')

# %% SPICAV SO2 retrievals
retrievals = os.getenv('RETRIEVALS')

# %% Output
savedir = os.getenv('SAVEDIR')
filetype = os.getenv('FILETYPE', 'png')
