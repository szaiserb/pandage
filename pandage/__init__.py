# import matplotlib
# matplotlib.use('Qt5Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from . import qtgui
from . import data_handling
from . import data_handling as dh
from . import data_generation as dg
from . import lmfit_models
from . import pddata
from . import pdplotddata