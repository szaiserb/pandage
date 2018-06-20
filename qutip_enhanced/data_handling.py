# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type


import os, sys

import numpy as np
import pandas as pd
import itertools

import datetime
import traceback
import collections
import subprocess
import threading
from PyQt5.QtWidgets import QListWidgetItem
import PyQt5.QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from .util import ret_property_array_like_types
import qutip_enhanced.qtgui.gui_helpers
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import functools
import time

if sys.version_info.major == 2:
    import __builtin__
else:
    import builtins as __builtin__
import lmfit

def printexception(func):
    @functools.wraps(func)
    def wrapper(*a, **kw):
        try:
            return func(*a, **kw)
        except:
            print('THROWN_FROM_PRINTEXCEPTIONWRAPPER')
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
    return wrapper

def ptrepack(file, folder, tempfile=None, lock=None):
    # C:\Users\yy3\AppData\Local\conda\conda\envs\py27\Scripts\ptrepack.exe -o --chunkshape=auto --propindexes --complevel=0 --complib=blosc data.hdf data_tmp.hdf

    if lock is not None and not lock.acquire(False):
        print('Trying to run ptrepack on {}'.format(os.path.join(folder, file)))
        print('Waiting for hdf_lock..')
        lock.acquire()
        print('Ok.. hdf_lock acquired.')
    tempfile = 'temp.hdf' if tempfile is None else tempfile
    ptrepack = r"{}\Scripts\ptrepack.exe".format(os.path.dirname(sys.executable))
    command = [ptrepack, "-o", "--chunkshape=auto", "--propindexes", "--complevel=9", "--complib=blosc", file, tempfile]
    _ = subprocess.call(command, cwd=folder)
    i = 0
    t0 = time.time()
    while i < 1000:
        try:
            os.remove(os.path.join(folder, file))
            break
        except:
            print("Trying to remove {} again. (Try {}, {}s)".format(os.path.join(folder, file), i, time.time() - t0))
            i += 1
    i = 0
    t0 = time.time()
    while i < 1000:
        try:
            os.rename(os.path.join(folder, tempfile), os.path.join(folder, file))
            if i > 0:
                print("Renaming {} to {} was successful after {} tries and a total of {}s)".format(os.path.join(folder, tempfile), os.path.join(folder, file), i, time.time() - t0))
            break
        except:
            print("Trying to rename {} to {} again. (Try {}, {}s)".format(os.path.join(folder, tempfile), os.path.join(folder, file), i, time.time() - t0))
            i += 1
    if lock is not None: lock.release()


def ptrepack_thread(**kwargs):
    t = threading.Thread(name='run_ptrepack', target=ptrepack, kwargs=kwargs)
    t.start()


def ptrepack_all(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".hdf"):
                ptrepack(file, root)


def df_pop(df, n):
    out = df.head(n)
    return df.iloc[n:, :], out

def l_to_df(l):
    """
    creates DataFrame from list of OrderedDict

    :param l: collections.OrderedDict, dict or list thereof
    :return:
    """
    if type(l) == pd.DataFrame:
        return l
    else:
        if type(l) in [collections.OrderedDict, dict]:
            l = [l]
        return pd.concat([pd.Series(i) for i in l], axis=1).transpose()

def df_access(df, other):
    if type(other) == pd.Series:
        other = pd.DataFrame(other).transpose()
    df_all = df.merge(other.drop_duplicates(), on=list(other.columns), how='left', indicator=True)
    return df_all[df_all._merge == 'both'].iloc[:, :-1].reset_index(drop=True)

def df_delete(df, other):
    if type(other) == pd.Series:
        other = pd.DataFrame(other).transpose()
    df_all = df.merge(other.drop_duplicates(), on=list(other.columns), how='left', indicator=True)
    return df_all[df_all._merge == 'left_only'].iloc[:, :-1].reset_index(drop=True)

def dict_access(df, l):
    return df_access(df=df, other=l_to_df(l))

def dict_delete(df, l):
    return df_delete(df=df, other=l_to_df(l))

class Data:
    def __init__(self, parameter_names=None, observation_names=None, dtypes=None, **kwargs):
        if parameter_names is not None:
            self.parameter_names = parameter_names
        if observation_names is not None:
            self.observation_names = observation_names
        if dtypes is not None:
            self.dtypes = dtypes
        if len(kwargs) > 0:
            self.init(**kwargs)
        self.hdf_lock = threading.Lock()

    parameter_names = ret_property_array_like_types('parameter_names', [str, unicode])
    observation_names = ret_property_array_like_types('observation_names', [str, unicode])

    @property
    def dtypes(self):
        if not hasattr(self, '_dtypes'):
            self._dtypes = dict([(key, 'float') for key in self.observation_names])
        return self._dtypes

    @dtypes.setter
    def dtypes(self, val):
        if val is not None:
            for k, v in val.items():
                if not k in self.observation_names or type(v) != str:  # k may not be in parameter_names, as these are inferred correctly from the passed values
                    raise Exception("Error: {}, {}, {}, {}, {}".format(val, k, v, type(v), self.observation_names))
            self._dtypes = val

    def change_column_dtype(self, column_name, new_dtype=None):
        if new_dtype is not None:
            self._df[[column_name]] = self.df[[column_name]].astype(new_dtype)
        if column_name in self.observation_names:
            self.dtypes[column_name] = new_dtype
        self.reinstate_integrity()

    def rename_column(self, column_name, column_name_new):
        if column_name not in self.variables:
            raise Exception('Error: column does not exist.')
        self.parameter_names = [i if i != column_name else column_name_new for i in self.parameter_names]
        self.observation_names = [i if i != column_name else column_name_new for i in self.observation_names]
        self._df = self.df.rename(columns={column_name: column_name_new})
        self.reinstate_integrity()

    def delete_columns(self, column_names):
        not_in = [cn for cn in column_names if not cn in self.variables]
        if len(not_in) > 0:
            raise Exception("Error: column_names {} can not be deleted because they do not exist.".format(not_in))
        nupn = self.non_unary_parameter_names
        non_unary_parameter_names_to_be_deleted = [cn for cn in column_names if cn in self.parameter_names and cn in nupn]
        if len(non_unary_parameter_names_to_be_deleted) != 0:
            raise Exception('Error: column_names {} are in parameter_names but are non_unary. Columns can not be deleted.'.format(non_unary_parameter_names_to_be_deleted))
        self._df = self.df.drop(column_names, axis=1)
        self.reinstate_integrity()

    @property
    def variables(self):
        return self.parameter_names + self.observation_names

    @property
    def number_of_variables(self):
        return len(self.variables)

    def backwards_compatibility_last_parameter(self, cn):
        for n in ['tau', 'point', 'phi', 'x2', 'x1']:  # order of x2, x1, x0.. is important
            if n in cn:
                next_n = cn[cn.index(n) + 1]
                if next_n == 'trace' or '_idx' in next_n:
                    return n
                else:
                    raise Exception('Incompatibility Error: {}'.format(cn))
        else:
            return None

    def get_last_parameter(self, df, last_parameter=None):
        if hasattr(self, '_parameter_names'):
            if last_parameter is not None:
                if last_parameter != self.parameter_names[-1]:
                    raise Exception('Error: {}, {}'.format(self.parameter_names[-1], last_parameter))
            else:
                last_parameter = self.parameter_names[-1]
        elif last_parameter is None:
            cn = list(df.columns)
            bclp = self.backwards_compatibility_last_parameter(cn)
            if bclp is not None:
                last_parameter = bclp
            else:
                l_idx = [cni for cni in cn if cni.endswith('_idx')]
                if len(l_idx) == 0:
                    raise Exception('Error: Could not figure out last_parameter, thus parameter_names and observation_names could not be determined: {}'.format(cn))
                l = [i[:-4] for i in l_idx]
                if all(np.array(cn[:2 * len(l_idx)]) == l + l_idx):
                    last_parameter = l_idx[-1]
                else:
                    raise Exception('Error: Could not figure out last_parameter, thus parameter_names and observation_names could not be determined: {}'.format(cn))
        elif last_parameter not in df.columns:
                raise Exception("last_parameter {} ist not in dataframe columns {}".format(last_parameter, df.columns))
        return last_parameter

    def read_hdf(self, init_from_file):
        store = pd.HDFStore(init_from_file)
        if hasattr(store, 'df'):
            df = store['df']
            for key in ["parameter_names", 'observation_names', 'dtypes']:
                attr_name = "_{}".format(key)
                if hasattr(store.get_storer('df').attrs, key):
                    setattr(self, attr_name, getattr(store.get_storer('df').attrs, key))
                else:
                    if hasattr(self, attr_name):
                        delattr(self, attr_name)
        else:
            df = store.get('/a')
        store.close()
        return df

    def read_csv(self, init_from_file):
        return pd.read_csv(init_from_file, compression='gzip')

    def init_metadata(self, df, last_parameter):
        if False in [hasattr(self, i) for i in ['_parameter_names', '_observation_names', '_dtypes']]:
            if last_parameter is None:
                last_parameter = self.get_last_parameter(df)
            lpi = list(self.df.columns.values).index(last_parameter) + 1
            if not hasattr(self, '_parameter_names'):
                self.parameter_names = list(self.df.columns[:lpi])
            if not hasattr(self, '_observation_names'):
                self.observation_names = list(self.df.columns[lpi:])
            if not hasattr(self, '_dtypes'):
                self._dtypes = list(self.df.dtypes[lpi:])

    def init(self, last_parameter=None, df=None, init_from_file=None, iff=None, path=None):
        init_from_file = iff if iff is not None else init_from_file
        init_from_file = path if path is not None else init_from_file
        if init_from_file is not None:
            self.filepath = init_from_file
            if init_from_file.endswith('.hdf'):
                df = self.read_hdf(init_from_file)
            elif init_from_file.endswith('.csv'):
                df = self.read_csv(init_from_file)
        if df is None:
            self._df = pd.DataFrame(columns=self.variables)
        else:
            self._df = df[pd.notnull(df)]
        if init_from_file is not None:
            self.init_metadata(df=df, last_parameter=last_parameter)

    def dropnan(self, max_expected_rows=None):
        ldf = len(self.df)
        self.df.dropna(axis=0, how='any', inplace=True)
        if max_expected_rows is not None and ldf - len(self.df) > max_expected_rows:
            raise Exception('Error: A maximum of {} rows with nan values can be expected from unfinished measurements ({}, {}). Whats wrong?'.format(max_expected_rows, ldf, len(self.df)))

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self.reinstate_integrity()

    def save(self, filepath, notify=False):
        if filepath.endswith('.csv'):
            t0 = time.time()
            self._df.to_csv(filepath, index=False, compression='gzip')
            print("csv data saved ({:.2f}s).\n If speed is an issue, use hdf. csv format is useful ony for py2-py3-compatibility.".format(time.time() - t0))
        elif filepath.endswith('.hdf'):
            t0 = time.time()
            if not self.hdf_lock.acquire(False):
                print('Trying to save data to hdf.')
                print('Waiting for hdf_lock..')
                self.hdf_lock.acquire()
                print('Ok.. hdf_lock acquired.')
            store = pd.HDFStore(filepath)
            store.put('df', self._df, table=True)
            for key in ["parameter_names", 'observation_names', 'dtypes']:
                setattr(store.get_storer('df').attrs, key, getattr(self, key))
            store.close()
            self.hdf_lock.release()
            ptrepack_thread(file=os.path.split(filepath)[1], folder=os.path.split(filepath)[0], lock=self.hdf_lock)
            self.filepath = filepath
            if notify:
                print("hdf data saved in ({:.2f})".format(time.time() - t0))

    def get_parameters_df(self, l):
        df_append = l_to_df(l)
        if not (df_append.columns == self.parameter_names).all():
            raise Exception('Error: Dataframe columns dont match: '.format(df_append.columns, self.parameter_names))
        return df_append.reset_index(drop=True)

    def initialize_observations_df(self, n):
        l_obs = []
        for idx in range(n):
            l_obs.append(collections.OrderedDict())
            for k, v in self.dtypes.items():
                if v == 'datetime':
                    l_obs[idx][k] = datetime.datetime(1900, 1, 1)  # datetime.datetime.min is also an option, but leads to OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 1-01-01 00:00:00, when calling data = self.df.iloc[index.row(), index.column()]
                else:
                    l_obs[idx][k] = getattr(__builtin__, v)()
        return pd.DataFrame(columns=self.observation_names, data=l_obs)

    def append(self, l):
        parameters_df = self.get_parameters_df(l)
        df_append = pd.concat([parameters_df, self.initialize_observations_df(len(parameters_df))], axis=1)  # NECESSARY! Reason: sorting issues when appending df with missing columns
        if len(self.df) == 0:
            self._df = df_append
        else:
            self._df = self._df.append(df_append, ignore_index=True)
        self.check_integrity()

    def set_observations(self, l, start_idx=None):
        start_idx = len(self._df) - 1 if start_idx is None else start_idx
        l = l_to_df(l)
        for idx, row in l.iterrows():
            for obs, val in zip(l.columns, row):
                self._df.at[start_idx - len(l) + idx + 1, obs] = val
        self.check_integrity()

    def df_access(self, other):
        return df_access(self.df, other)

    def df_delete(self, other):
        self._df = df_delete(self.df, other)
        self.check_integrity()

    def dict_access(self, l):
        return dict_access(df=self.df, l=l)

    def dict_delete(self, l):
        self._df = dict_delete(df=self.df, l=l)
        self.check_integrity()

    def column_product(self, column_names):
        return itertools.product(*[getattr(self.df, cn).unique() for cn in column_names])

    def sub(self, l):
        ldf = l_to_df(l)
        new_parameter_names = [i for i in self.parameter_names if not (i in ldf.columns and len(ldf[i].unique()) == 1)]
        sub = Data(parameter_names=new_parameter_names, observation_names=self.observation_names, dtypes=self.dtypes)
        sub._df = self.dict_access(ldf)
        return sub

    def iterator(self, column_names, output_data_instance=False):
        for idx, p in enumerate(self.column_product(column_names=column_names)):
            d = collections.OrderedDict(zip(column_names, p))
            d_idx = collections.OrderedDict([(cn, np.argwhere(getattr(self.df, cn).unique() == p)[0, 0]) for cn, p in d.items()])
            sub = self.sub(l=d)
            if not output_data_instance:
                sub = sub.df
            yield d, d_idx, idx, sub

    @property
    def non_unary_parameter_names(self):
        """
        :return: all parameter_names that are indeed varied (with at least two different values in self.df)
        """
        out = []
        for cn in self.parameter_names:
            try:
                if len(self.df[cn].unique()) > 1:
                    out.append(cn)
            except:  # drops for example columns with numpy arrays
                pass
        return out

    def check_integrity(self):
        if len(self.parameter_names) + len(self.observation_names) != len(self.df.columns):
            cnl = [(i, j) for i, j in zip(self.parameter_names + self.observation_names, self.df.columns)]
            raise Exception('Error: Integrity corrupted: {}, {}, {}\n{}'.format(len(self.parameter_names), len(self.observation_names), len(self.df.columns), cnl))
        if not all([i == j for i, j in zip(self.df.columns[:len(self.parameter_names)], self.parameter_names)]):
            raise Exception('Error: Integrity corrupted: {}, {}'.format(self.df.columns, self.parameter_names))
        if not all([i == j for i, j in zip(self.df.columns[-len(self.observation_names):], self.observation_names)]):
            raise Exception('Error: Integrity corrupted: {}, {}'.format(self.df.columns, self.observation_names))
        for cn in self.df.columns:
            if cn in self.parameter_names and cn in self.observation_names:
                raise Exception('Error: Integrity corrupted! {}, {}'.format(self.parameter_names, self.observation_names))

    def reinstate_integrity(self):
        """
        After columns of self.df have been removed, remove these from other variables as well and vice versa
        """
        cnl = [cn for cn in self.df.columns if cn not in self.parameter_names + self.observation_names]
        self._df = self.df.drop(cnl, axis=1)
        self.parameter_names = [i for i in self.parameter_names if i in self.df.columns]
        self.observation_names = [i for i in self.observation_names if i in self.df.columns]
        for key in self.dtypes.keys():
            if key in self.dtypes and key not in self.df.columns:
                del self.dtypes[key]
        self.check_integrity()

    def observations_to_parameters(self, observation_names, new_names, new_parameter_name, new_observation_name='value', new_parameter_dtype=None):
        df = self.df.rename(dict([(old, new) for old, new in zip(observation_names, new_names)]), axis='columns')
        df = df.melt(id_vars=[cn for cn in df.columns if cn not in new_names])
        self._df = df.rename({'value': new_observation_name, 'variable': new_parameter_name}, axis=1)
        self.parameter_names.append(new_parameter_name)
        previous_observation_location = min([self.observation_names.index(i) for i in observation_names])
        self.observation_names = [i for i in self.observation_names if i not in observation_names]
        self.observation_names.insert(previous_observation_location, new_observation_name)
        new_column_names = self.parameter_names + self.observation_names
        self._df = self._df[new_column_names]
        self.change_column_dtype(new_parameter_name, new_parameter_dtype)
        self.check_integrity()

    def check_pn_on_helper(self, parameter_name, observation_names):
        if not parameter_name in self.parameter_names:
            raise Exception('Error: chosen parameter_name {} is not in self.parameter_names {}.'.format(parameter_name, self.parameter_names))
        if len(observation_names) == 0:
            raise Exception('Error: Chose at least on observation.')
        if any([i not in self.observation_names for i in observation_names]):
            raise Exception('Error: at least one of the chosen observation_names {} is not in self.observation_names {}'.format(observation_names, self.observation_names))

    def subtract_parameter(self, parameter_name, observation_names):
        self.check_pn_on_helper(parameter_name, observation_names)
        lnp = len(getattr(self.df, parameter_name).unique())
        if lnp != 2:
            raise Exception('Error: Only parameters with 2 unique items can be subtracted. Parameter {} has {}'.format(parameter_name, lnp))
        self.delete_columns([i for i in self.observation_names if i not in observation_names])
        self._df.dropna(inplace=True)
        self.parameter_names = [key for key in self.parameter_names if key != parameter_name]
        self._df = self.df.groupby(self.parameter_names).filter(lambda x: len(x) == 2).groupby(self.parameter_names).agg(dict([(obs, lambda x: -1 * np.diff(x)) for obs in observation_names])).reset_index()
        self.check_integrity()

    def average_parameter(self, parameter_name, observation_names):
        self.check_pn_on_helper(parameter_name, observation_names)
        self.delete_columns([i for i in self.observation_names if i not in observation_names])
        self._df.dropna(inplace=True)
        self.parameter_names = [key for key in self.parameter_names if key != parameter_name]
        self._df = self.df.groupby(self.parameter_names).agg(dict([(obs, np.mean) for obs in observation_names])).reset_index()
        self.check_integrity()

def extend_columns(df, other, columns=None):
    """
    Extends df horizontally to include all columns of other
    Necessary workaround due to pandas deficiencies regarding missing data (only float has a realy missing data type)
    Order of columns is important.

    Example:
        df.columns = ['A', 'B', 'C']
        other.columns = ['B', 'C', 'D']
        result: df.columns = ['A', 'B', 'C', 'D']

    :param df: pandas.DataFrame
        DataFrame whose columns should be extended
    :param other: pandas.DataFrame
        DataFrame to take extra columns from
    :return: pandas.DataFrame
    """
    columns = None
    columns = other.columns if columns is None else columns
    if type(columns) != pd.core.indexes.base.Index:
        raise Exception('Error: unexpected behaviour.')
    if not all([i in other.columns for i in df.columns]):
        raise Exception('Error: all columns of other ({}) must be in df ({})'.format(other.columns, df.columns))
    if len(df.columns) == len(columns) and (df.columns == columns).all():
        return df
    else:
        print('This should work but is untested and to my knowledge not used right now. (data_handling.extend_columns)')
        df_missing_columns = pd.Index([i for i in other.columns if i not in df.columns])  # df.columns.append(other.columns).drop_duplicates(keep=False)
        return pd.concat([df, pd.concat([other.loc[:, df_missing_columns].iloc[0:1, :]] * len(df)).reset_index(drop=True)], axis=1).reset_index(drop=True)

def df_take_duplicate_rows(df, other):
    df_columns = df.columns  # here for security reasons, leave it!
    df_dtypes = df.dtypes  # here for security reasons, leave it!
    df_all = df.merge(other.drop_duplicates(), on=list(other.columns), how='left', indicator=True)
    out = df_all[df_all._merge == 'both'].iloc[:, :-1].reset_index(drop=True)
    if not (out.columns == df_columns).all():  # here for security reasons, leave it!
        print(out.columns, df_columns)
    if not (out.dtypes == df_dtypes).all():  # here for security reasons, leave it!
        print(out.columns, df.columns)
    return out

def df_drop_duplicate_rows(df, other):
    """

    :param df: dataframe
    :param other: dataframe
    :param columns: pd.core.indexes.base.Index
        columns to compare
        default: right.columns
    :return:
    """
    df_columns = df.columns  # here for security reasons, leave it!
    df_dtypes = df.dtypes  # here for security reasons, leave it!
    df_all = df.merge(other.drop_duplicates(), on=list(other.columns), how='left', indicator=True)
    out = df_all[df_all._merge == 'left_only'].iloc[:, :-1].reset_index(drop=True)
    if not (out.columns == df_columns).all():  # here for security reasons, leave it!
        print(out.columns, df_columns)
    if not (out.dtypes == df_dtypes).all():  # here for security reasons, leave it!
        print(out.columns, df.columns)
    return out

class Base:

    def __init__(self, delete_names=None):
        self.delete_names = [] if delete_names is None else delete_names

    @printexception
    def delete_attributes(self):
        for attr_name in self.delete_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    @property
    @printexception
    def data(self):
        return getattr(self, '_data')

    @printexception
    def update_data(self):
        self._data = self.get_data()
        if hasattr(self.parent, '_gui'):
            getattr(getattr(self.parent.gui, self.name), "update_data")(self.data)
        self.update_selected_indices()

    @property
    def selected_indices(self):
        return self._selected_indices

class SelectableList(Base):
    def __init__(self, name='noname', parent=None, **kwargs): # kwargs =dict(delete_names=['_data','_selected_indices',])
        super(SelectableList, self).__init__(**kwargs)
        self.parent = parent
        self.name = name

    @property
    @printexception
    def selected_index(self):
        if len(self._selected_indices) != 1:
            raise Exception("Error: No single index selected.")
        return self._selected_indices[0]

    @printexception
    def update_selected_indices(self, val=None):
        if val is not None:
            self._selected_indices = val
        elif not hasattr(self, '_selected_indices') or (any([i not in self.data for i in self.selected_data]) and len(self.data) > 0):
            self._selected_indices = self.selected_indices_default()
        else:
            return
        if hasattr(self.parent, '_gui'):
            getattr(getattr(self.parent.gui, self.name), "update_selected_indices")(self.selected_indices)

    @property
    @printexception
    def selected_data(self):
        return [self.data[i] for i in self.selected_indices]

    @property
    @printexception
    def selected_value(self):
        return self.data[self.selected_index]

    @selected_value.setter
    @printexception
    def selected_value(self, val):
        self.update_selected_indices([self.data.index(val)])

class XAxisParameterComboBox(SelectableList):

    def __init__(self, **kwargs):
        super(XAxisParameterComboBox, self).__init__(name='x_axis_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self):
        return [np.argmax([len(getattr(self.parent.data.df, p).unique()) for p in self.data])]

class ColAxParameterComboBox(SelectableList):

    def __init__(self, **kwargs):
        super(ColAxParameterComboBox, self).__init__(name='col_ax_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self):
        return [0]

class RowAxParameterComboBox(SelectableList):

    def __init__(self, **kwargs):
        super(RowAxParameterComboBox, self).__init__(name='row_ax_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self):
        return [0]

class SubtractParameterComboBox(SelectableList):

    def __init__(self, **kwargs):
        super(SubtractParameterComboBox, self).__init__(name='subtract_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self):
        return [0]

class ObservationList(SelectableList):

    def __init__(self, **kwargs):
        super(ObservationList, self).__init__(name='observation_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return [i for i in self.parent.data.observation_names if not i in ['trace', 'start_time', 'end_time', 'thresholds']]

    @printexception
    def selected_indices_default(self):
        return [self.data.index('sweeps')] if 'sweeps' in self.data else []

class AverageParameterList(SelectableList):

    def __init__(self, **kwargs):
        super(AverageParameterList, self).__init__(name='average_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return self.parent.data.parameter_names

    @printexception
    def selected_indices_default(self):
        return [self.data.index('sweeps')] if 'sweeps' in self.data else []

class DataTable(Base):

    def __init__(self, name='noname', parent=None, **kwargs):
        super(DataTable, self).__init__(**kwargs)
        self.parent = parent
        self.name = name

    @printexception
    def update_selected_indices(self, selected_indices=None):
        if isinstance(selected_indices, collections.OrderedDict) or selected_indices is None:
            self._selected_indices = self.selected_indices_default(selected_indices)
        else:
            raise Exception(type(selected_indices), selected_indices)
        if hasattr(self.parent, '_gui'):
            getattr(getattr(self.parent.gui, self.name), "update_selected_indices")(self.selected_indices)

    @property
    @printexception
    def selected_data(self):
        out = collections.OrderedDict()
        for cn, val in self.selected_indices.items():
            data_full = getattr(self.parent.data.df, cn).unique()
            out[cn] = [data_full[i] for i in val]
        return out

    @printexception
    def update_selected_data(self, selected_data):
        out = collections.OrderedDict()
        for cn, val in selected_data.items():
            indices = []
            for i in val:
                indices.append(np.where(self.data[cn] == i)[0][0])
            out[cn] = indices
        self.update_selected_indices(out)

class ParameterTable(DataTable):
    def __init__(self, **kwargs):
        super(ParameterTable, self).__init__(name='parameter_table', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        out = collections.OrderedDict()
        for cn in self.parent.parameter_names_reduced():
            out[cn] = getattr(self.parent.data.df, cn).unique()
        return out

    @printexception
    def selected_indices_default(self, selected_indices):
        out = collections.OrderedDict([(key, range(len(val))) for key, val in self._data.items()])
        if selected_indices is not None:
            for cn, val in selected_indices.items():
                out[cn] = val
        return out

class PlotData(qutip_enhanced.qtgui.gui_helpers.WithQt):
    def __init__(self, title=None, parent=None, gui=True, **kwargs):
        """

        :param title:
        :param parent:
        :param gui:
        :param kwargs:
            path: path to hdf file
            data: Data instance or
        """

        super(PlotData, self).__init__(parent=parent, gui=gui, QtGuiClass=PlotDataQt)
        self.observation_list = ObservationList(parent=self)
        self.average_parameter_list = AverageParameterList(parent=self)
        self.x_axis_parameter_list = XAxisParameterComboBox(parent=self)
        self.col_ax_parameter_list = ColAxParameterComboBox(parent=self)
        self.row_ax_parameter_list = RowAxParameterComboBox(parent=self)
        self.subtract_parameter_list = SubtractParameterComboBox(parent=self)
        self.parameter_table = ParameterTable(parent=self)
        self.set_data(**kwargs)
        if title is not None:
            self.update_window_title(title)

    fit_function = 'cosine'
    show_legend = False


    @printexception
    def set_data(self, **kwargs):
        if 'path' in kwargs:
            self.data = Data(iff=kwargs['path'])
        elif 'data' in kwargs:
            self.data = kwargs['data']
        if hasattr(self.data, 'filepath'):
            self.update_window_title(self.data.filepath)
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(self.data.filepath)

    def set_data_from_path(self, path):
        self.set_data(path=path)

    @property
    @printexception
    def data(self):
        if hasattr(self, '_data'):
            return self._data

    @data.setter
    @printexception
    def data(self, val):
        self.delete_attributes()
        if hasattr(self, '_gui'):
            self.gui.clear()
        self._data = val
        self.new_data_arrived()

    @printexception
    def new_data_arrived(self):
        if hasattr(self.data, 'filepath') and matplotlib.rcParams["savefig.directory"] != os.path.dirname(self.data.filepath):
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(self.data.filepath)
        self.x_axis_parameter_list.update_data()
        self.col_ax_parameter_list.update_data()
        self.row_ax_parameter_list.update_data()
        self.subtract_parameter_list.update_data()
        self.parameter_table.update_data()
        self.observation_list.update_data()
        self.average_parameter_list.update_data()
        self.update_plot()

    @printexception
    def delete_attributes(self):
        for attr_name in [
            '_data',
            '_selected_plot_items',
            '_fit_select_table_data',
            '_fit_select_table_selected_rows',
            '_fit_results',
            '_fit_result_table_data'
        ]:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        for attr_name in [
            'observation_list',
            'average_parameter_list',
            'x_axis_parameter_list',
            'col_ax_parameter_list',
            'row_ax_parameter_list',
            'subtract_parameter_list',
            'parameter_table',
        ]:
            getattr(self, attr_name).delete_attributes()

    @printexception
    def parameter_names_reduced(self, data=None):
        data = self.data if data is None else data
        return [i for i in data.parameter_names if not '_idx' in i]

    @property
    def data_selected(self):
        other = pd.DataFrame(list(itertools.product(*self.parameter_table.selected_data.values())))
        other.rename(columns=collections.OrderedDict([(key, val) for key, val in enumerate(self.parameter_table.selected_data.keys())]), inplace=True)
        data = Data(parameter_names=self.data.parameter_names, observation_names=self.data.observation_names)
        data._df = df_access(self.data.df, other)
        if self.subtract_parameter_list.selected_value != '__none__':
            data.subtract_parameter(self.subtract_parameter_list.selected_value, observation_names=self.observation_list.selected_data)
        for pn in self.average_parameter_list.selected_data:
            data.average_parameter(pn, observation_names=self.observation_list.selected_data)
        return data

    @property
    @printexception
    def selected_plot_items(self):
        return self._selected_plot_items

    @printexception
    def update_selected_plot_items(self):
        out = self.parameter_table.selected_data
        if all([len(i) == 0 for i in out.values()]):
            self._selected_plot_items = []
            return
        spl = [self.x_axis_parameter_list.selected_value, self.col_ax_parameter_list.selected_value, self.row_ax_parameter_list.selected_value, self.subtract_parameter_list.selected_value] + self.average_parameter_list.selected_data
        spl = [i for i in spl if i != '__none__']
        if len(spl) != len(set(spl)):
            self._selected_plot_items = []
            print('Selected plot options are invalid, parameters have been selected multiple times. {}'.format(spl))
            return
        for key, val in out.items():
            if key == self.x_axis_parameter_list.selected_value or key == self.subtract_parameter_list.selected_value:
                out[key] = ['__all__']
            if key in self.average_parameter_list.selected_data:
                out[key] = ['__average__']
        self._selected_plot_items = list(itertools.product(*out.values()))

    @property
    @printexception
    def fit_select_table_data(self):
        return self._fit_select_table_data

    @printexception
    def update_fit_select_table_data(self):
        cpd = collections.OrderedDict()
        for idx, spi in enumerate(self.line_plot_data()):
            for column_idx, column_name in enumerate(self.parameter_names_reduced()):
                if column_name in spi['condition_dict_reduced']:
                    if not column_name in cpd:
                        cpd[column_name] = []
                    cpd[column_name].append(spi['condition_dict_reduced'][column_name])
                elif column_name == self.subtract_parameter_list.selected_value:
                    cpd[column_name] = ['diff'.format(self.subtract_parameter_list.selected_value)]
        self._fit_select_table_data = cpd
        if hasattr(self, '_gui'):
            self.gui.update_fit_select_table_data(self.fit_select_table_data)
        self.update_fit_select_table_selected_rows(range(len(self.fit_select_table_data.values()[0])))


    @property
    @printexception
    def fit_select_table_selected_rows(self):
        return self._fit_select_table_selected_rows

    @printexception
    def update_fit_select_table_selected_rows(self, fit_select_table_selected_rows=None):
        self._fit_select_table_selected_rows = [] if fit_select_table_selected_rows is None else fit_select_table_selected_rows  # self.fit_select_table_widget.selected_items_unique_column_indices()
        if hasattr(self, '_gui'):
            self.gui.update_fit_select_table_selected_rows(self.fit_select_table_selected_rows)

    @property
    @printexception
    def fit_results(self):
        return self._fit_results

    @printexception
    def update_fit_results(self):
        try:
            spi = self.line_plot_data()
            if hasattr(self, 'custom_model'):
                mod = self.custom_model
            elif self.fit_function == 'cosine':
                from . import lmfit_models
                mod = lmfit_models.CosineModel()
            elif self.fit_function == 'exp':
                pass
            elif self.fit_function == 'lorentz':
                from . import lmfit_models
                mod = lmfit.models.LorentzianModel()
            self._fit_results = []
            for idx in self.fit_select_table_selected_rows:
                i = spi[idx]
                params = mod.guess(data=np.array(i['y']), x=np.array(i['x']))
                fp = getattr(self, 'fix_params', {})
                if all(key in params for key in fp.keys()):
                    for key, val in fp.items():
                        params[key].vary = False
                        params[key].value = val
                self._fit_results.append([i, mod.fit(np.array(i['y']), params, x=np.array(i['x']))])
            self.update_fit_result_table_data()
        except ValueError:
            print("Can not fit, input contains nan values")
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    @printexception
    def fit_result_table_data(self):
        return self._fit_result_table_data

    @printexception
    def update_fit_result_table_data(self):
        if len(self.fit_results) == 0:
            out = collections.OrderedDict()
        else:
            header = self.fit_results[0][0]['condition_dict_reduced'].keys() + ['observation_name'] + self.fit_results[0][1].params.keys()
            out = collections.OrderedDict([(key, []) for key in header])
            for fri in self.fit_results:
                for key, val in fri[0]['condition_dict_reduced'].items() + [('observation_name', fri[0]['observation_name'])] + [(key, val.value) for key, val in fri[1].params.items()]:
                    out[key].append(val)
        self._fit_result_table_data = out
        if hasattr(self, '_gui'):
            self.gui.update_fit_result_table_data(self.fit_result_table_data)

    @printexception
    def ret_line_plot_data_single(self, condition_dict, observation_name):
        cl = [self.data.df[key] == val for key, val in condition_dict.items() if val not in ['__all__', '__average__']]
        out = self.data.df[functools.reduce(np.logical_and, cl)] if len(cl) != 0 else self.data.df
        # TODO: wont work for dates as can not be averaged
        return out.groupby([key for key, val in condition_dict.items() if val != '__average__']).agg({observation_name: np.mean}).reset_index()

    @printexception
    def line_plot_data(self):
        plot_data = []
        if len(self.data.df) > 0:
            for p in self.selected_plot_items:
                condition_dict = collections.OrderedDict([(ni, pi) for ni, pi in zip(self.parameter_names_reduced(), p)])
                for observation_name in self.observation_list.selected_data:
                    if self.subtract_parameter_list.selected_value == '__none__':
                        dfxy = self.ret_line_plot_data_single(condition_dict, observation_name).dropna(subset=[observation_name])
                        condition_dict_reduced = collections.OrderedDict([(key, val) for key, val in condition_dict.items() if val not in ['__average__', '__all__']])
                        plot_data.append(
                            dict(
                                condition_dict_reduced=condition_dict_reduced,
                                observation_name=observation_name,
                                x=getattr(dfxy, self.x_axis_parameter_list.selected_value),
                                y=getattr(dfxy, observation_name)
                            )
                        )
                    else:
                        keys = [key for key, val in condition_dict.items() if val not in ['__average__', '__all__']] + [self.x_axis_parameter_list.selected_value]
                        dfxys = self.ret_line_plot_data_single(condition_dict, observation_name).dropna(subset=[observation_name])
                        dfxy = dfxys.groupby(keys).filter(lambda x: len(x) ==2).groupby(keys).agg({observation_name: lambda x: -1 * np.diff(x)}).reset_index()
                        if len(dfxy) > 1:
                            condition_dict_reduced = collections.OrderedDict([(key, val) for key, val in condition_dict.items() if val not in ['__average__', '__all__']])
                            plot_data.append(
                                dict(
                                    condition_dict_reduced=condition_dict_reduced,
                                    observation_name=observation_name,
                                    x=getattr(dfxy, self.x_axis_parameter_list.selected_value),
                                    y=getattr(dfxy, observation_name)
                                )
                            )
        return plot_data

    def plot_label(self, condition_dict_reduced):
        label = ""
        for key in self.data.non_unary_parameter_names:
            if key in condition_dict_reduced:
                val = condition_dict_reduced[key]
                if type(val) is str:
                    label += "{}:{}, ".format(key, val)
                else:
                    label += "{}:{:g}, ".format(key, val)
        return label[:-2]

    @staticmethod
    def get_subplot_grid_location_parameters(nx_tot, ny_tot, n_tot, **kwargs):
        """
        NOTE: index numbers (e.g. n, nx, ny start from 0 as is common in python, not from 1 as is common in matlab and partially in matplotlib
        :param nx_tot: int
            number of axes from left to right
        :param ny_tot: int
            number of axes from bottom to top
        :param n_tot: int
            total number of plots
        :param kwargs: either give (nx, ny) or n
            'nx': int
                x (horizontal) position in grid
            'ny': int
                y position in grid
            'n' : int
                position in flattened grid
        :return: dict
        """
        if nx_tot*ny_tot < n_tot:
            raise Exception('Error: {}, {}, {}'.format(nx_tot, ny_tot, n_tot))
        if 'n' in kwargs:
            n = kwargs['n']
            nx = n%nx_tot
            ny = int(np.floor(n/float(nx_tot)))
        elif 'nx' in kwargs and 'ny' in kwargs:
            nx = kwargs['nx']
            ny = kwargs['ny']
            n = nx_tot * ny + nx
        else:
            raise Exception('Error: {}'.format(kwargs))
        if n > n_tot - 1:
            raise Exception('Error: given n (or (nx, ny)) does not index an axis within the grid')
        most_bottom_axis = True if n + nx_tot > n_tot - 1 else False
        most_left_axis = True if nx == 0 else False
        return n, nx, ny, most_bottom_axis, most_left_axis

    def update_plot_new(self, fig):
        if len(self.observation_list.selected_data) == 0:
            print('Please select the items from observation_list that you want to plot.')
            return
        self.update_selected_plot_items()
        if len(self.selected_plot_items) == 0:
            return
        self.update_fit_select_table_data()
        fig.clear()
        if self.col_ax_parameter_list.selected_value != '__none__' and self.row_ax_parameter_list.selected_value == '__none__':
            ax_p_list = self.parameter_table_selected_data[self.col_ax_parameter_list.selected_value]
            n_tot = len(ax_p_list)
            nx_tot = int(np.ceil(np.sqrt(n_tot)))
            ny_tot = n_tot/float(nx_tot)
            ny_tot = int(ny_tot + 1) if int(ny_tot) == ny_tot else int(np.ceil(ny_tot))
        elif self.col_ax_parameter_list.selected_value != '__none__' and self.row_ax_parameter_list.selected_value != '__none__':
            ax_p_list = list(itertools.product(self.parameter_table_selected_data[self.row_ax_parameter_list.selected_value], self.parameter_table_selected_data[self.col_ax_parameter_list.selected_value]))
            n_tot = len(ax_p_list)
            nx_tot = len(self.parameter_table_selected_data[self.col_ax_parameter_list.selected_value])
            ny_tot = len(self.parameter_table_selected_data[self.row_ax_parameter_list.selected_value])
            if n_tot < len(self.line_plot_data()):# +1 is for legend
                ny_tot += 1
        else:
            ax_p_list = []
            nx_tot = ny_tot = n_tot = 1
        axes = []
        for ni in range(1, n_tot+1):
            pd = {}if len(axes) == 0 else {'sharex': axes[0], 'sharey': axes[0]}
            axes.append(fig.add_subplot(ny_tot, nx_tot, ni, **pd))
        for idx, pdi in enumerate(self.line_plot_data()):
            if len(ax_p_list) == 0 or self.row_ax_parameter_list.selected_value == '__none__': #len(ax_p_list[0]) == 1:
                n = np.argwhere(np.array(ax_p_list) == (pdi['condition_dict_reduced'][self.col_ax_parameter_list.selected_value]))[0,0] if self.col_ax_parameter_list.selected_value != '__none__' else 0
            else:
                for n, cr in enumerate(ax_p_list):
                    if cr[1] == pdi['condition_dict_reduced'][self.col_ax_parameter_list.selected_value] and cr[0] == pdi['condition_dict_reduced'][self.row_ax_parameter_list.selected_value]:
                        break
                else:
                    continue
            abcdefg = collections.OrderedDict([(key, val) for key, val in pdi['condition_dict_reduced'].items() if key not in [self.subtract_parameter_list.selected_value, self.col_ax_parameter_list.selected_value, self.row_ax_parameter_list.selected_value]])
            axes[n].plot(pdi['x'], pdi['y'], '-o', markersize=3, label=self.plot_label(abcdefg))
            if len(ax_p_list) != 0:
                title_list = []
                for column_name in [self.col_ax_parameter_list.selected_value, self.row_ax_parameter_list.selected_value]:
                    if column_name != '__none__' and len(self.data.df[self.col_ax_parameter_list.selected_value].unique()) > 1:
                        title_list.append("{}: {}".format(column_name, pdi['condition_dict_reduced'][column_name]))
                if len(title_list) > 0:
                    axes[n].set_title(", ".join(title_list))

        for n in range(n_tot):
            n, nx, ny, most_bottom_axis, most_left_axis = self.get_subplot_grid_location_parameters(nx_tot, ny_tot, n_tot, n=n)
            axes[n].set_xlabel(self.x_axis_parameter_list.selected_value)
            if not most_left_axis:
                plt.setp(axes[n].get_yticklabels(), visible=False)
            else:
                if len(self.observation_list.selected_data) == 1:
                    axes[n].set_ylabel(self.observation_list.selected_data[0])
        if 1 < len(axes[0].lines): # <= 12:  # NOT PYTHON 3 SAFE
            if len(ax_p_list) != 0:
                axes.append(fig.add_subplot(ny_tot, nx_tot, ny_tot*nx_tot))
                plt.setp(axes[-1].get_xticklabels(), visible=False)
                plt.setp(axes[-1].get_yticklabels(), visible=False)
                axes[-1].xaxis.set_ticks_position('none')
                axes[-1].yaxis.set_ticks_position('none')
                axes[-1].axis('off')
                axes[-1].legend(
                    axes[0].lines,
                    [l.get_label() for l in axes[0].lines],
                    borderaxespad=0.1,  # Small spacing around legend box
                    loc='upper left',
                    fontsize=20 / len(axes[0].lines) ** .25,
                )
            else:
                axes[0].legend(fontsize=20/len(axes[0].lines)**.25,)
        fig.tight_layout()
        return axes

    @printexception
    def update_plot(self, *args, **kwargs):
        if hasattr(self, '_data') and hasattr(self, '_gui'):
            self.gui.axes = self.update_plot_new(fig=self.gui.fig)
            self.gui.canvas.draw()

    @printexception
    def update_plot_fit(self, fit_results=None):
        if fit_results is None:
            self.update_fit_results()
        else:
            self._fit_results = fit_results
        if hasattr(self, '_gui'):
            self.gui.fig_fit.clear()
            self.gui.ax_fit = self.gui.fig_fit.add_subplot(111)
            for idx, fi in enumerate(self.fit_results):
                color = self.gui.ax_fit._get_lines.get_next_color()
                r = fi[1]
                x = r.userkws['x']
                y = r.eval(params=r.params, x=x)
                self.gui.ax_fit.plot(x, y, '-', color=color)
                self.gui.ax_fit.plot(x, r.data, 'o', color=color, markersize=3.5)
                self.gui.fig_fit.tight_layout()
                self.gui.canvas_fit.draw()

    @printexception
    def save_plot(self, filepath):
        plt.ioff()
        fig, ax = plt.subplots(1, 1)
        cn = self.parameter_names_reduced()
        if cn[0] == 'sweeps' and self.x_axis_parameter_list.selected_value != 'sweeps':
            del cn[0]
        cn.remove(self.x_axis_parameter_list.selected_value)
        for d, d_idx, idx, df_sub in self.data.iterator(cn):
            for observation in self.observation_list.selected_data:
                dfagg = df_sub.groupby([self.x_axis_parameter_list.selected_value]).agg({observation: np.mean}).reset_index()
                ax.plot(getattr(dfagg, self.x_axis_parameter_list.selected_value), getattr(dfagg, observation))
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
        plt.ion()

    @property
    @printexception
    def info_text(self):
        return self._info_text

    def update_info_text(self, info_text):
        self._info_text = info_text
        if hasattr(self, '_gui'):
            self.gui.update_info_text(info_text)

class BaseQt(PyQt5.QtWidgets.QWidget):

    def __init__(self, name, widget_name, parent, *args, **kwargs):
        super(BaseQt, self).__init__(*args, **kwargs)
        self.name = name
        self.widget_name = widget_name
        self.parent = parent

    def update_data(self, data):
        self.update_data_signal.emit(data)

    def update_selected_indices(self, selected_indices):
        getattr(self, "update_selected_indices_signal").emit(selected_indices)

    def connect_signals(self):
        for name in [
            'data'.format(self.name),
            'selected_indices'.format(self.name),
        ]:
            getattr(getattr(self, "update_{}_signal".format(name)), 'connect')(getattr(self, "update_{}_signal_emitted".format(name)))

class SelectableListQt(BaseQt):

    update_data_signal = pyqtSignal(list)
    update_selected_indices_signal = pyqtSignal(list)

    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        if widget.count() != 0:
            self.parent.clear_widget(self.widget_name, 'clear')
        for val in data:
            widget.addItem(QListWidgetItem(val))

    def update_selected_indices_signal_emitted(self, selected_indices):
        getattr(self.parent, self.widget_name).blockSignals(True)
        for i in selected_indices:
            getattr(self.parent, self.widget_name).item(i).setSelected(True)
        getattr(self.parent, self.widget_name).blockSignals(False)

    @printexception
    def update_selected_indices_from_gui(self, *args, **kwargs):
        getattr(self.parent.no_qt, self.name).update_selected_indices([i.row() for i in getattr(self.parent, self.widget_name).selectedIndexes()])

class ParameterCombobox(BaseQt):

    update_data_signal = pyqtSignal(list)
    update_selected_indices_signal = pyqtSignal(list)

    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        if widget.count() != 0:
            self.parent.clear_widget(self.widget_name, 'clear')
        widget.addItems(data)
        widget.blockSignals(False)
        if hasattr(getattr(self.parent.no_qt, self.name), '_selected_indices'):
            getattr(self.parent.parameter_table, 'update_item_flags')()

    def update_selected_indices_signal_emitted(self, selected_indices):
        getattr(self.parent, self.widget_name).blockSignals(True)
        getattr(self.parent, self.widget_name).setCurrentText(getattr(self.parent.no_qt, self.name).data[selected_indices[0]])
        getattr(self.parent, self.widget_name).blockSignals(False)

    @printexception
    def update_selected_indices_from_gui(self, *args, **kwargs):
        idx = getattr(self.parent.no_qt, self.name).data.index(str(getattr(self.parent, self.widget_name).currentText()))
        getattr(self.parent.no_qt, self.name).update_selected_indices([idx])


class ParameterTableQt(BaseQt):

    update_data_signal = pyqtSignal(collections.OrderedDict)
    update_selected_indices_signal = pyqtSignal(collections.OrderedDict)

    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        if widget.columnCount() == 0:
            widget.set_column_names(data.keys())
        for column_name, val in data.items():
            widget.set_column(column_name, val)
        self.update_item_flags()
        widget.blockSignals(False)

    def update_item_flags(self):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        if len(self.parent.no_qt.x_axis_parameter_list.selected_indices) == 1:
            for cidx in range(widget.columnCount()):
                cn = widget.column_name(cidx)
                if cn == self.parent.no_qt.subtract_parameter_list.selected_value:
                    flag = Qt.NoItemFlags
                else:
                    flag = Qt.ItemIsSelectable | Qt.ItemIsEnabled
                for ridx in widget.column_data_indices(cn):
                    widget.set_cell(ridx, cidx, flag=flag)
        widget.blockSignals(False)

    def update_selected_indices_signal_emitted(self, selected_indices):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        for cn, val in selected_indices.items():
            cidx = widget.column_index(cn)
            for ridx in widget.column_data_indices(cn):
                if val == '__all__' or (not isinstance(val, basestring) and ridx in val):
                    widget.item(ridx, cidx).setSelected(True)
                else:
                    widget.item(ridx, cidx).setSelected(False)
        widget.blockSignals(False)

    def update_selected_indices_from_gui(self):
        widget = getattr(self.parent, self.widget_name)
        out = collections.OrderedDict()
        column_names = widget.column_names
        for item in widget.selectedItems():
            cn = column_names[item.column()]
            if not cn in out:
                out[cn] = []
            out[cn].append(item.row())
        getattr(self.parent.no_qt, self.name).update_selected_indices(out)

class PlotDataQt(qutip_enhanced.qtgui.gui_helpers.QtGuiClass):
    def __init__(self, parent=None, no_qt=None):
        self.observation_list = SelectableListQt(name='observation_list', widget_name='observation_widget', parent=self)
        self.average_parameter_list = SelectableListQt(name='average_parameter_list', widget_name='average_parameter_widget', parent=self)
        self.x_axis_parameter_list = ParameterCombobox(name='x_axis_parameter_list', widget_name='x_axis_parameter_comboBox', parent=self)
        self.col_ax_parameter_list = ParameterCombobox(name='col_ax_parameter_list', widget_name='col_ax_parameter_comboBox', parent=self)
        self.row_ax_parameter_list = ParameterCombobox(name='row_ax_parameter_list', widget_name='row_ax_parameter_comboBox', parent=self)
        self.subtract_parameter_list = ParameterCombobox(name='subtract_parameter_list', widget_name='subtract_parameter_comboBox', parent=self)
        self.parameter_table = ParameterTableQt(name='parameter_table', widget_name='parameter_table_widget', parent=self)
        super(PlotDataQt, self).__init__(parent=parent, no_qt=no_qt, ui_filepath=os.path.join(os.path.dirname(__file__), 'qtgui/plot_data.ui'))


    update_fit_select_table_data_signal = pyqtSignal(collections.OrderedDict)
    update_fit_select_table_selected_rows_signal = pyqtSignal(list)
    update_fit_result_table_data_signal = pyqtSignal(collections.OrderedDict)
    update_info_text_signal = pyqtSignal(str)

    def clear_widget(self, name, clear_method_name):
        getattr(self, name).blockSignals(True)
        getattr(getattr(self, name), clear_method_name)()
        getattr(self, name).blockSignals(False)


    def clear_signal_emitted(self):
        for item in [
            ['x_axis_parameter_comboBox', 'clear'],
            ['col_ax_parameter_comboBox', 'clear'],
            ['row_ax_parameter_comboBox', 'clear'],
            ['subtract_parameter_comboBox', 'clear'],
            ['parameter_table_widget', 'clear_table_contents'],
            ['observation_widget', 'clear'],
            ['average_parameter_widget', 'clear'],
            ['fit_select_table_widget', 'clear_table_contents'],
            ['fit_result_table_widget', 'clear_table_contents'],
        ]:
            self.clear_widget(item[0], item[1])
        try:
            self.fig.clf()
            self.canvas.draw()
        except:
            pass
        try:
            self.fig_fit.clf()
            self.canvas_fit.draw()
        except:
            pass

    # def update_parameter_table_data(self, parameter_table_data):
    #     self.update_parameter_table_data_signal.emit(parameter_table_data)
    #
    # def update_parameter_table_data_signal_emitted(self, parameter_table_data):
    #     self.parameter_table_widget.blockSignals(True)
    #     if self.parameter_table_widget.columnCount() == 0:
    #         self.parameter_table_widget.set_column_names(parameter_table_data.keys())
    #     for column_name, val in parameter_table_data.items():
    #         self.parameter_table_widget.set_column(column_name, val)
    #     self.update_parameter_table_item_flags()
    #     self.parameter_table_widget.blockSignals(False)
    #
    # def update_parameter_table_item_flags(self):
    #     self.parameter_table_widget.blockSignals(True)
    #     if len(self.no_qt.x_axis_parameter_list.selected_indices) == 1: #hasattr(self.no_qt, '_x_axis_parameter'):
    #         for cidx in range(self.parameter_table_widget.columnCount()):
    #             cn = self.parameter_table_widget.column_name(cidx)
    #             if cn == self.no_qt.x_axis_parameter_list.selected_value:
    #                 flag = Qt.NoItemFlags
    #             else:
    #                 flag = Qt.ItemIsSelectable | Qt.ItemIsEnabled
    #             for ridx in self.parameter_table_widget.column_data_indices(cn):
    #                 self.parameter_table_widget.set_cell(ridx, cidx, flag=flag)
    #     self.parameter_table_widget.blockSignals(False)
    #
    # def update_parameter_table_selected_indices(self, selected_indices):
    #     self.update_parameter_table_selected_indices_signal.emit(selected_indices)
    #
    # def update_parameter_table_selected_indices_signal_emitted(self, parameter_table_selected_indices):
    #     self.parameter_table_widget.blockSignals(True)
    #     for cn, val in parameter_table_selected_indices.items():
    #         cidx = self.parameter_table_widget.column_index(cn)
    #         for ridx in self.parameter_table_widget.column_data_indices(cn):
    #             if val == '__all__' or (not isinstance(val, basestring) and ridx in val):
    #                 self.parameter_table_widget.item(ridx, cidx).setSelected(True)
    #             else:
    #                 self.parameter_table_widget.item(ridx, cidx).setSelected(False)
    #     self.parameter_table_widget.blockSignals(False)
    #
    # def update_parameter_table_selected_indices_from_gui(self):
    #     out = collections.OrderedDict()
    #     column_names = self.parameter_table_widget.column_names
    #     for item in self.parameter_table_widget.selectedItems():
    #         cn = column_names[item.column()]
    #         if not cn in out:
    #             out[cn] = []
    #         out[cn].append(item.row())
    #     self.no_qt.update_parameter_table_selected_indices(out)

    def update_fit_select_table_data(self, fit_select_table_data):
        self.update_fit_select_table_data_signal.emit(fit_select_table_data)

    def update_fit_select_table_data_signal_emitted(self, fit_select_table_data):
        self.fit_select_table_widget.blockSignals(True)
        self.fit_select_table_widget.clear_table_contents()
        self.fit_select_table_widget.set_column_names(fit_select_table_data.keys())
        for column_name, val in fit_select_table_data.items():
            self.fit_select_table_widget.set_column(column_name, val, [Qt.ItemIsSelectable | Qt.ItemIsEnabled for _ in range(len(val))])
        self.fit_select_table_widget.blockSignals(False)
    @printexception
    def update_fit_select_table_selected_rows_from_gui(self):
        self.no_qt.update_fit_select_table_selected_rows(self.fit_select_table_widget.selected_items_unique_row_indices())

    def update_fit_select_table_selected_rows(self, selected_rows):
        self.update_fit_select_table_selected_rows_signal.emit(selected_rows)

    def update_fit_select_table_selected_rows_signal_emitted(self, fit_select_table_selected_rows):
        self.fit_select_table_widget.blockSignals(True)
        for ridx in range(self.fit_select_table_widget.rowCount()):
            if self.fit_select_table_widget.item(ridx, 0).isSelected() ^ (ridx in fit_select_table_selected_rows):
                self.fit_select_table_widget.selectRow(ridx)
        self.fit_select_table_widget.blockSignals(False)

    def update_fit_result_table_data(self, fit_result_table_data):
        self.update_fit_result_table_data_signal.emit(fit_result_table_data)

    def update_fit_result_table_data_signal_emitted(self, fit_result_table_data):
        self.fit_result_table_widget.blockSignals(True)
        self.fit_result_table_widget.clear_table_contents()
        self.fit_result_table_widget.set_column_names(fit_result_table_data.keys())
        for column_name, val in fit_result_table_data.items():
            self.fit_result_table_widget.set_column(column_name, val)
        self.fit_result_table_widget.blockSignals(False)

    def update_info_text(self, info_text):
        self.update_info_text_signal.emit(info_text)

    def update_info_text_signal_emitted(self, info_text):
        self.info.setPlainText(info_text)

    def redraw_plot(self):
        self.fig.tight_layout()
        self.canvas.draw()

    def redraw_plot_fit(self):
        self.fig_fit.tight_layout()
        self.canvas_fit.draw()

    def init_gui(self):
        super(PlotDataQt, self).init_gui()
        for name in [
            'update_fit_select_table_data',
            'update_fit_select_table_selected_rows',
            'update_fit_result_table_data',
            'update_info_text'
        ]:
            getattr(getattr(self, "{}_signal".format(name)), 'connect')(getattr(self, "{}_signal_emitted".format(name)))
        for name in [
            'observation_list',
            'average_parameter_list',
            'x_axis_parameter_list',
            'row_ax_parameter_list',
            'col_ax_parameter_list',
            'subtract_parameter_list',
            'parameter_table'

        ]:
            getattr(self, name).connect_signals()

        # # Figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()
        self.plot_layout.addWidget(self.canvas, 1, 1, 20, 20)
        self.toolbar_layout.addWidget(self.toolbar, 21, 1, 1, 20)

        # Figure fit
        self.fig_fit = Figure()
        self.canvas_fit = FigureCanvas(self.fig_fit)
        self.toolbar_fit = NavigationToolbar(self.canvas_fit, self)

        self.ax_fit = self.fig_fit.add_subplot(111)
        self.canvas_fit.draw()
        self.plot_fit_layout.addWidget(self.canvas_fit, 1, 1, 20, 20)
        self.toolbar_fit_layout.addWidget(self.toolbar_fit, 21, 1, 1, 20)

        self.update_plot_button.clicked.connect(self.no_qt.update_plot)
        self.update_fit_result_button.clicked.connect(lambda: self.no_qt.update_plot_fit(None))
        self.update_plot_button.setAcceptDrops(True)

        self.parameter_table_widget.hdf_file_dropped.connect(self.no_qt.set_data_from_path)
        self.parameter_table_widget.itemSelectionChanged.connect(self.parameter_table.update_selected_indices_from_gui)
        self.fit_select_table_widget.itemSelectionChanged.connect(self.update_fit_select_table_selected_rows_from_gui)
        self.observation_widget.itemSelectionChanged.connect(self.observation_list.update_selected_indices_from_gui)
        self.average_parameter_widget.itemSelectionChanged.connect(self.average_parameter_list.update_selected_indices_from_gui)
        self.open_code_button.clicked.connect(self.open_measurement_code)
        self.open_explorer_button.clicked.connect(self.open_explorer)

        self.x_axis_parameter_comboBox.currentIndexChanged.connect(self.x_axis_parameter_list.update_selected_indices_from_gui)
        self.col_ax_parameter_comboBox.currentIndexChanged.connect(self.col_ax_parameter_list.update_selected_indices_from_gui)
        self.row_ax_parameter_comboBox.currentIndexChanged.connect(self.row_ax_parameter_list.update_selected_indices_from_gui)
        self.subtract_parameter_comboBox.currentIndexChanged.connect(self.subtract_parameter_list.update_selected_indices_from_gui)

    def open_measurement_code(self):
        if hasattr(self.no_qt.data, 'filepath'):
            subprocess.Popen(r"start {}/meas_code.py".format(os.path.dirname(self.no_qt.data.filepath)), shell=True)
        else:
            print('No filepath.')

    def open_explorer(self):
        if hasattr(self.no_qt.data, 'filepath'):
            subprocess.Popen("explorer {}".format(os.path.abspath(os.path.dirname(self.no_qt.data.filepath))), shell=True)
        else:
            print('No filepath.')


def cpd():
    hdfl = [fn for fn in os.listdir(os.getcwd()) if fn.endswith('.hdf')]
    if not 'data.hdf' in hdfl and len(hdfl) != 1:
        raise Exception('Error: {}'.format(hdfl))
    data = Data(iff=os.path.join(os.getcwd(), hdfl[0]))
    if 'prepare_data.py' in os.listdir(os.getcwd()):
        from prepare_data import prepare_data
        data = prepare_data(data)
    # out = PlotData(path=os.path.join(os.getcwd(), hdfl[0]))
    out = PlotData(data=data)
    out.gui.show_gui()
    return out

def cpd_thread():
    import threading
    out = PlotData()

    def run():

        for fn in os.listdir(os.getcwd()):
            if fn.endswith('.hdf'):
                out.set_data_from_path(fn)
        out.gui.show_gui()

    t = threading.Thread(target=run)
    t.start()


def subfolders_with_hdf(folder):
    l = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".hdf"):
                l.append(root)
    return l


def number_of_points_of_hdf_files_in_subfolders(folder):
    l = subfolders_with_hdf(folder)
    out_openable = []
    out_failed = []
    for subdir in l:
        for root, dirs, files in os.walk(subdir):
            for file in files:
                if file.endswith(".hdf"):
                    try:
                        d = Data(iff=os.path.join(root, file))
                        out_openable.append({'root': root, 'file': file, 'points': len(d.df)})
                    except:
                        out_failed.append({'root': root, 'file': file})
    return out_openable, out_failed


def hdf_files_in_subfolders_with_less_than_n_points(folder, n):
    out_openable, out_failed = number_of_points_of_hdf_files_in_subfolders(folder)
    out_smaller = [i for i in out_openable if i['points'] < n]
    out_larger_equal = [i for i in out_openable if i['points'] >= n]
    return out_smaller, out_larger_equal, out_failed


def move_folder(folder_list_dict=None, destination_folder=None):
    import shutil
    failed = 0
    for i in folder_list_dict:
        try:
            src = i['root']
            dst = os.path.join(destination_folder, os.path.basename(i['root']))
            shutil.move(src, dst)
        except:
            failed += 1
            print("Folder {} could not be moved. Lets hope it has tbc in its name".format(i['root']))
    print("Successfully moved: {}. Failed: {}".format(len(folder_list_dict) - failed, failed))

def replot_all_hdf(folder):
    for root, dirs, files in os.walk(os.path.join(folder), topdown=False):
        for name in files:
            if name == 'data.hdf' and 'sim_code' in root:
                pld = PlotData(gui=False, data=Data(iff=os.path.join(root, name)))
                pld.save_plot(os.path.join(root, 'plot1.png'))
                print(root)



# if __name__ == '__main__':
#     df = pd.DataFrame({'a': [1, 2], 'b': [.5, 1.5], 'c': ['alpha', 'beta']})
#     other = pd.DataFrame({'a': [1], 'b': [.5]})
#     reload(dh)
#     data = dh.Data(parameter_names = ['a', 'b'], observation_names=['c'], dtypes=collections.OrderedDict([('c', "object")]))
#     data.init()
#     data.append(collections.OrderedDict([('a', 1), ('b', 13.5)]))
#     data.set_observations(l=collections.OrderedDict([('c', 'ffff')]))
#     data.append(collections.OrderedDict([('a', 1), ('b', 20.5)]))
#     data.set_observations(l=collections.OrderedDict([('c', 'ggggg')]))
#     data.dict_delete(l=collections.OrderedDict([('a', 1), ('b', 20.5)]))

