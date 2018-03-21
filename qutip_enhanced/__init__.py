# import matplotlib
# matplotlib.use('Qt5Agg')

import pandas as pd #THIS IS NECESSARY FOR PYTHON 3, LEAVE IT HERE
from .qutip_enhanced import *

from . import analyze
from . import coordinates
from . import nv_hamilton
from . import sequence_creator
from . import sequence_creator as sc
from . import dynamo_helpers
from . import qtgui
from . import data_handling
from . import data_handling as dh
from . import data_generation
from . import data_generation as dg
from . import qutrit_sensing

import sys
if sys.version_info[0] == 3:
    from . import lmfit_models
