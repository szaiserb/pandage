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

from .util import ret_property_array_like_types, printexception
import qutip_enhanced.qtgui.gui_helpers
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import functools
import time
import logging

if sys.version_info.major == 2:
    import __builtin__
else:
    import builtins as __builtin__
import lmfit

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
def weight_mean(w):
    def _weight_mean(d):
        try:
            #             return d.mean()
            return (d * w).sum() / w[d.index].sum()
        except ZeroDivisionError:
            return np.NaN
    return _weight_mean

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
            self.df[[column_name]] = self.df[[column_name]].astype(new_dtype)
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

    def append_columns(self, values_dict, no_new_columns_warning=True, **kwargs):
        if len(kwargs) != 1:
            raise Exception('Error: give one dictionary of column_names or observation_names. {}'.format(kwargs))
        od = {'parameter_names': 'observation_names', 'observation_names': 'parameter_names'}
        if type(values_dict) != collections.OrderedDict:
            raise Exception("Error: {}".format(values_dict))
        is_in_other = [cn for cn in kwargs.values()[0] if cn in getattr(self, od[kwargs.keys()[0]])]
        if len(is_in_other) > 0:
            raise Exception("Error: {} is in od[kwargs.keys()[0]]".format(is_in_other))
        not_in = [cn for cn in kwargs.values()[0] if cn not in getattr(self, kwargs.keys()[0])]
        if len(not_in) == 0:
            if no_new_columns_warning:
                raise Exception("Error: No new columns given. {}".format(kwargs))
            else:
                return
        if len(values_dict) != len(not_in):
            raise Exception('Error: {}, {}'.format(values_dict, not_in))
        not_in_values_dict = [not_in_i for not_in_i in not_in if not_in_i not in values_dict.keys()]
        if len(not_in_values_dict) > 0:
            raise Exception('Error: {}'.format(not_in_values_dict))
        setattr(self, kwargs.keys()[0], kwargs.values()[0])
        for column, value in values_dict.items():
            loc = getattr(self, kwargs.keys()[0]).index(column)
            if kwargs.keys()[0] == 'observation_names':
                loc += len(self.parameter_names)
            self.df.insert(loc, column, value, allow_duplicates=False)
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

    def seconds_since_last_save(self, filepath):
        return os.stat(filepath).st_mtime - time.time()

    def save(self, filepath, notify=False):
        if filepath.endswith('.csv'):
            t0 = time.time()
            self.df.to_csv(filepath, index=False, compression='gzip')
            print("csv data saved ({:.2f}s).\n If speed is an issue, use hdf. csv format is useful ony for py2-py3-compatibility.".format(time.time() - t0))
        elif filepath.endswith('.hdf'):
            t0 = time.time()
            t = []
            if not self.hdf_lock.acquire(False):
                print('Trying to save data to hdf.')
                print('Waiting for hdf_lock..')
                self.hdf_lock.acquire()
                print('Ok.. hdf_lock acquired.')
            t.append(time.time() -t0)
            store = pd.HDFStore(filepath)
            store.put('df', self.df, table=True)
            for key in ["parameter_names", 'observation_names', 'dtypes']:
                setattr(store.get_storer('df').attrs, key, getattr(self, key))
            store.close()
            t.append(time.time() - t0)
            self.hdf_lock.release()
            t.append(time.time() - t0)
            if getattr(self, 'save_dur_list', [0])[-1] < 5. or self.seconds_since_last_save(filepath) > 3600.:
                ptrepack_thread(file=os.path.split(filepath)[1], folder=os.path.split(filepath)[0], lock=self.hdf_lock)
            else:
                logging.getLogger().info('hdf-file is not repacked. {} {}', self.save_dur_list[-1], self.seconds_since_last_save(filepath))
            t.append(time.time() - t0)
            self.filepath = filepath
            if notify:
                logging.getLogger().info("hdf data saved in ({})".format(" ".join("{:.2f}".format(x) for x in t)))
                if hasattr(self, 'save_dur_list'):
                    self.save_dur_list.append(sum(t))
                else:
                    self.save_dur_list = [sum(t)]

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
                elif v.startswith('numpy'):
                    l_obs[idx][k] = getattr(np, v.split('.')[1])()
                else:
                    l_obs[idx][k] = getattr(__builtin__, v)()
        return pd.DataFrame(columns=self.observation_names, data=l_obs)

    def append(self, l):
        parameters_df = self.get_parameters_df(l)
        df_append = pd.concat([parameters_df, self.initialize_observations_df(len(parameters_df))], axis=1)  # NECESSARY! Reason: sorting issues when appending df with missing columns
        if len(self.df) == 0:
            self._df = df_append
        else:
            self._df = self.df.append(df_append, ignore_index=True)
        self.check_integrity()

    def set_observations(self, l, start_idx=None):
        start_idx = len(self.df) - 1 if start_idx is None else start_idx
        l = l_to_df(l)
        for idx, row in l.iterrows():
            for obs, val in zip(l.columns, row):
                self.df.at[start_idx - len(l) + idx + 1, obs] = val
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

    def update_observation_names_combine(self, observation_names, new_observation_name):
        observation_names = [i for i in self.observation_names if i not in observation_names]
        observation_names.insert(min([self.observation_names.index(i) for i in self.observation_names]), new_observation_name)
        self.observation_names = observation_names

    def observations_to_parameters(self, observation_names, new_parameter_name, new_observation_name='value'):
        df = pd.melt(
            self.df,
            id_vars=[cn for cn in self.df.columns if cn not in observation_names],
            value_vars=observation_names,
            var_name=new_parameter_name,
            value_name=new_observation_name)
        self.parameter_names.append(new_parameter_name)
        self.update_observation_names_combine(observation_names, new_observation_name)
        self.df = df[self.parameter_names + self.observation_names]

    def observations_to_parameters(self, observation_names, new_parameter_name, new_observation_name='value'):
        df = pd.melt(
            self.df,
            id_vars=[cn for cn in self.df.columns if cn not in observation_names],
            value_vars=observation_names,
            var_name=new_parameter_name,
            value_name=new_observation_name)
        self.parameter_names.append(new_parameter_name)
        self.update_observation_names_combine(observation_names, new_observation_name)
        self.df = df[self.parameter_names + self.observation_names]

    def check_pn_on_helper(self, parameter_name, observation_names):
        if not parameter_name in self.parameter_names:
            raise Exception('Error: chosen parameter_name {} is not in self.parameter_names {}.'.format(parameter_name, self.parameter_names))
        if len(observation_names) == 0:
            raise Exception('Error: Chose at least on observation.')
        if any([i not in self.observation_names for i in observation_names]):
            raise Exception('Error: at least one of the chosen observation_names {} is not in self.observation_names {}'.format(observation_names, self.observation_names))

    def eliminate_parameter(self, parameter_name, observation_names, operations, dropna=False):
        self.check_pn_on_helper(parameter_name, observation_names)
        self.delete_columns([i for i in self.observation_names if i not in observation_names])
        self.parameter_names = [key for key in self.parameter_names if key != parameter_name]
        if len(operations) == 1:
            operations = itertools.repeat(operations[0])
        elif len(operations) != len(observation_names):
            raise Exception('Error: ', operations, observation_names)
        self._df = self.df.groupby(self.parameter_names).agg(collections.OrderedDict([(observation_name, operation) for observation_name, operation in zip(observation_names, operations)])).reset_index()
        self.check_integrity()

    def average_parameter(self, parameter_name, observation_names, observation_name_weight=None):
        if observation_name_weight is None:
            operations = [np.mean]
        else:
            operations = [weight_mean(self.df[observation_name_weight])]
        self.eliminate_parameter(operations=operations, parameter_name=parameter_name, observation_names=observation_names, dropna=False)

    def sum_parameter(self, parameter_name, observation_names):
        self.eliminate_parameter(operations=[np.sum], parameter_name=parameter_name, observation_names=observation_names, dropna=True)

    def subtract_parameter(self, parameter_name, observation_names):
        lnp = len(getattr(self.df, parameter_name).unique())
        if lnp != 2:
            raise Exception('Error: Only parameters with 2 unique items can be subtracted. Parameter {} has {}'.format(parameter_name, lnp))
        self.eliminate_parameter(operations=[np.diff], parameter_name=parameter_name, observation_names=observation_names, dropna=True)

    def subtract_observations(self, observation_names, new_observation_name, dropna=False):
        print('functionality untested')
        if len(observation_names) != 2:
            raise Exception('Error :', observation_names)
        self._df[[observation_names[0]]] = self.df[observation_names[1]] - self.df[observation_names[0]]
        if dropna:
            self.df.dropna(subset=observation_names)
        self.rename_column(observation_names[0], new_observation_name)
        self.delete_columns([self.observation_names[1]])

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

    @printexception
    def __init__(self, delete_names=None):
        self.delete_names = [] if delete_names is None else delete_names

    @printexception
    def delete_attributes(self):
        for attr_name in self.delete_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        if hasattr(self, '_gui'):
            getattr(self.parent.gui, self.name).clear()

    @property
    @printexception
    def data(self):
        return getattr(self, '_data')

    @printexception
    def update_data(self):
        self._data = self.get_data()
        if hasattr(self.parent, '_gui'):
            getattr(self.parent.gui, self.name).update_data(self.data)
        self.update_selected_indices()

    @property
    @printexception
    def selected_indices(self):
        return self._selected_indices


class SelectableList(Base):

    @printexception
    def __init__(self, name='noname', parent=None, **kwargs):  # kwargs =dict(delete_names=['_data','_selected_indices',])
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
        if hasattr(self.parent, '_gui'):
            getattr(getattr(self.parent.gui, self.name), "update_selected_indices")(self.selected_indices)

    @printexception
    def update_selected_data(self, val=None):
        selected_indices = val if val is None else [self.data.index(i) for i in val]
        self.update_selected_indices(selected_indices)

    @property
    @printexception
    def selected_data(self):
        return [self.data[i] for i in self.selected_indices]

    @property
    @printexception
    def selected_value(self):
        return self.data[self.selected_index]

    @printexception
    def update_selected_value(self, val=None):
        selected_data = val if val is None else [val]
        self.update_selected_data(selected_data)


class XAxisParameterComboBox(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(XAxisParameterComboBox, self).__init__(name='x_axis_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self, selected_indices=None, exclude_names=None):
        exclude_names = [] if exclude_names is None else exclude_names
        arr = np.array([[idx, len(getattr(self.parent.data.df, p).unique())] for idx, p in enumerate(self.data) if p not in exclude_names])
        return [arr[:, 0][np.argmax(arr[:, 1])]]


class ColAxParameterComboBox(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(ColAxParameterComboBox, self).__init__(name='col_ax_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return [0]


class RowAxParameterComboBox(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(RowAxParameterComboBox, self).__init__(name='row_ax_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return [0]


class SubtractParameterComboBox(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(SubtractParameterComboBox, self).__init__(name='subtract_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return ['__none__'] + [cn for cn in self.parent.parameter_names_reduced()]

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return [0]


class ObservationList(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(ObservationList, self).__init__(name='observation_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return [i for i in self.parent.data.observation_names if not i in ['trace', 'start_time', 'end_time', 'thresholds']]

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return []


class AverageParameterList(SelectableList):

    @printexception
    def __init__(self, **kwargs):
        super(AverageParameterList, self).__init__(name='average_parameter_list', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        return self.parent.data.parameter_names

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return []


class DataTable(Base):

    @printexception
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

    @printexception
    def __init__(self, **kwargs):
        super(ParameterTable, self).__init__(name='parameter_table', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        out = collections.OrderedDict()
        for cn in self.parent.parameter_names_reduced():
            out[cn] = getattr(self.parent.data.df, cn).unique()
        return out

    @printexception
    def selected_indices_default(self, selected_indices=None):
        out = collections.OrderedDict()
        if hasattr(self, '_data'):
            for key, val in self._data.items():
                if selected_indices is not None and key in selected_indices:
                    out[key] = selected_indices[key]
                else:
                    out[key] = range(len(val))
        return out


class FitResultTable(DataTable):
    def __init__(self, **kwargs):
        super(FitResultTable, self).__init__(name='fit_result_table', delete_names=['_data', '_selected_indices'], **kwargs)

    @printexception
    def get_data(self):
        if not hasattr(self.parent, '_data_fit_results') or len(self.parent.data_fit_results.df) == 0:
            out = collections.OrderedDict()
        else:
            header = self.parent.data_fit_results.parameter_names + self.parent.data_fit_results.df.fit_result[0].params.keys()
            out = self.parent.data_fit_results.df[header].to_dict('list', into=collections.OrderedDict)
        return out

    @printexception
    def selected_indices_default(self, selected_indices=None):
        return collections.OrderedDict()


class PlotData(qutip_enhanced.qtgui.gui_helpers.WithQt):

    @printexception
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
        self.fit_result_table = FitResultTable(parent=self)
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

    @printexception
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
        self.set_custom_default_settings()
        self.update_plot()

    @printexception
    def set_custom_default_settings(self):
        if hasattr(self, '_custom_default_settings'):
            self._custom_default_settings()

    @printexception
    def delete_attributes(self):
        for attr_name in [
            '_data',
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
            'fit_result_table',
        ]:
            getattr(self, attr_name).delete_attributes()

    @printexception
    def parameter_names_reduced(self, data=None):
        data = self.data if data is None else data
        return [i for i in data.parameter_names if not '_idx' in i]

    @property
    @printexception
    def data_selected(self):
        if len(self.data.df) == 0:
            return
        other = pd.DataFrame(list(itertools.product(*self.parameter_table.selected_data.values())))
        other.rename(columns=collections.OrderedDict([(key, val) for key, val in enumerate(self.parameter_table.selected_data.keys())]), inplace=True)
        data = Data(parameter_names=self.data.parameter_names, observation_names=self.observation_list.selected_data)
        data._df = df_access(self.data.df[self.data.parameter_names + self.observation_list.selected_data], other)
        if self.subtract_parameter_list.selected_value != '__none__':
            data.subtract_parameter(self.subtract_parameter_list.selected_value, observation_names=self.observation_list.selected_data)
        for pn in self.average_parameter_list.selected_data:
            data.average_parameter(pn, observation_names=self.observation_list.selected_data)
        return data

    @property
    @printexception
    def data_fit_results(self):
        return self._data_fit_results

    def update_data_fit_results(self, data_selected=None):
        data_selected = self.data_selected if data_selected is None else data_selected
        x_axis_parameter = self.x_axis_parameter_list.selected_value
        if not x_axis_parameter in data_selected.parameter_names:
            print('Can not fit!\nx_axis_parameter {} is not in data_selected'.format(x_axis_parameter))
            return
        try:
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
            pnl = [pn for pn in data_selected.parameter_names if pn != x_axis_parameter]
            for d, d_idx, idx, sub in data_selected.iterator(column_names=pnl, output_data_instance=True):
                for idx_obs, obs in enumerate(sub.observation_names):
                    x = np.array(sub.df[x_axis_parameter])
                    y = np.array(sub.df[obs])
                    params = mod.guess(data=y, x=x)
                    for key, val in getattr(self, 'custom_params', {}).items():
                        if key not in params:
                            raise Exception('Error: invalid custom_params {} for fit. Key {} not found'.format(self.custom_params, key))
                        params[key] = val
                    if idx == 0 and idx_obs == 0:
                        data_fit_results = Data(parameter_names=pnl + ['observation_name'],
                                                observation_names=['fit_result'],
                                                dtypes={'fit_result': 'object'})
                        data_fit_results.init()
                    data_fit_results.append(collections.OrderedDict([(key, val) for key, val in zip(d.keys() + ['observation_name'], d.values() + [obs])]))
                    data_fit_results.set_observations(collections.OrderedDict([('fit_result', mod.fit(y, params, x=x))]))
            self._data_fit_results = data_fit_results
            self.extend_data_fit_results_by_parameters()
            self.fit_result_table.update_data()
        except ValueError:
            print("Can not fit, input contains nan values")
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def extend_data_fit_results_by_parameters(self):
        new_observation_names = self._data_fit_results.df.loc[0, 'fit_result'].params.keys()
        values_dict = collections.OrderedDict([(key, np.nan) for key in new_observation_names])
        observation_names = self._data_fit_results.observation_names[:-1] + new_observation_names + [self._data_fit_results.observation_names[-1]]
        self._data_fit_results.append_columns(values_dict=values_dict, observation_names=observation_names)
        for idx, _I_ in self._data_fit_results.df.iterrows():
            for pn, pv in _I_['fit_result'].params.items():
                self._data_fit_results.df.loc[idx, pn] = pv


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
    @printexception
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
        if nx_tot * ny_tot < n_tot:
            raise Exception('Error: {}, {}, {}'.format(nx_tot, ny_tot, n_tot))
        if 'n' in kwargs:
            n = kwargs['n']
            nx = n % nx_tot
            ny = int(np.floor(n / float(nx_tot)))
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

    @printexception
    def number_of_lines(self, data_selected=None):
        data_selected = self.data_selected if data_selected is None else data_selected
        return len(data_selected.df[[i for i in data_selected.parameter_names if i != self.x_axis_parameter_list.selected_value]].drop_duplicates())

    @printexception
    def update_plot_new(self, fig, data_selected=None):
        data_selected = self.data_selected if data_selected is None else data_selected
        if data_selected is None or len(data_selected.df) == 0:
            print('Nothing to plot. You may select something.')
            return
        x_axis_parameter = self.x_axis_parameter_list.selected_value
        col_ax_parameter = self.col_ax_parameter_list.selected_value
        row_ax_parameter = self.row_ax_parameter_list.selected_value
        for name in ['x_axis_parameter', 'col_ax_parameter', 'row_ax_parameter']:
            val = locals()[name]
            if val != '__none__' and not val in data_selected.parameter_names:
                print('Can not plot!\n{} {} is not in data_selected'.format(name, val))
                return
        fig.clear()
        if col_ax_parameter != '__none__' and row_ax_parameter == '__none__':
            ax_p_list = self.parameter_table.selected_data[col_ax_parameter]
            n_tot = len(ax_p_list)
            nx_tot = int(np.ceil(np.sqrt(n_tot)))
            ny_tot = n_tot / float(nx_tot)
            ny_tot = int(ny_tot + 1) if int(ny_tot) == ny_tot else int(np.ceil(ny_tot))
        elif col_ax_parameter != '__none__' and row_ax_parameter != '__none__':
            ax_p_list = list(itertools.product(self.parameter_table.selected_data[row_ax_parameter], self.parameter_table.selected_data[col_ax_parameter]))
            n_tot = len(ax_p_list)
            nx_tot = len(self.parameter_table.selected_data[col_ax_parameter])
            ny_tot = len(self.parameter_table.selected_data[row_ax_parameter])
            if n_tot == nx_tot * ny_tot:  # self.number_of_lines:# +1 is for legend, I DONT THINK THIS DID WHAT IT SHOULD, SO I CHANGED IT
                ny_tot += 1
        else:
            ax_p_list = []
            nx_tot = ny_tot = n_tot = 1
        axes = []
        for ni in range(1, n_tot + 1):
            pd = {} if len(axes) == 0 else {'sharex': axes[0], 'sharey': axes[0]}
            axes.append(fig.add_subplot(ny_tot, nx_tot, ni, **pd))
        for d, d_idx, idx, sub in data_selected.iterator(column_names=[pn for pn in data_selected.parameter_names if pn != self.x_axis_parameter_list.selected_value]):
            if len(ax_p_list) == 0 or row_ax_parameter == '__none__':  # len(ax_p_list[0]) == 1:
                n = np.argwhere(np.array(ax_p_list) == (d[col_ax_parameter]))[0, 0] if col_ax_parameter != '__none__' else 0
            else:
                for n, cr in enumerate(ax_p_list):
                    if cr[1] == d[col_ax_parameter] and cr[0] == d[row_ax_parameter]:
                        break
                else:
                    continue
            abcdefg = collections.OrderedDict([(key, val) for key, val in d.items() if key not in [self.subtract_parameter_list.selected_value, col_ax_parameter, row_ax_parameter]])
            for obs in self.observation_list.selected_data:
                axes[n].plot(sub[self.x_axis_parameter_list.selected_data], sub[obs], '-o', markersize=3, label=self.plot_label(abcdefg))
            if len(ax_p_list) != 0:
                title_list = []
                for column_name in [col_ax_parameter, row_ax_parameter]:
                    if column_name != '__none__' and len(self.data.df[col_ax_parameter].unique()) > 1:
                        title_list.append("{}: {}".format(column_name, d[column_name]))
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
        if 1 < len(axes[0].lines):  # <= 12:  # NOT PYTHON 3 SAFE
            if len(ax_p_list) != 0:
                axes.append(fig.add_subplot(ny_tot, nx_tot, ny_tot * nx_tot))
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
                axes[0].legend(fontsize=20 / len(axes[0].lines) ** .25, )
        fig.tight_layout()
        return axes

    @printexception
    def update_plot(self, *args, **kwargs):
        if hasattr(self, '_data') and hasattr(self, '_gui'):
            self.gui.axes = self.update_plot_new(fig=self.gui.fig)
            self.gui.canvas.draw()

    @printexception
    def update_plot_fit(self, data_fit_results=None):
        if data_fit_results is None:
            self.update_data_fit_results()
        else:
            self._data_fit_results = data_fit_results
        if hasattr(self, '_gui'):
            self.gui.fig_fit.clear()
            self.gui.ax_fit = self.gui.fig_fit.add_subplot(111)
            for r in self.data_fit_results.df.fit_result:
                color = self.gui.ax_fit._get_lines.get_next_color()
                x = r.userkws['x']
                y = r.eval(params=r.params, x=x)
                self.gui.ax_fit.plot(x, y, '-', color=color)
                self.gui.ax_fit.plot(x, r.data, 'o', color=color, markersize=3.5)
                self.gui.fig_fit.tight_layout()
                self.gui.canvas_fit.draw()

    @printexception
    def save_plot(self, filepath, notify=False):
        t0 = time.time()
        plt.ioff()
        fig = plt.figure()
        axes = self.update_plot_new(fig=fig)
        fig.savefig(filepath)
        plt.close(fig)
        plt.ion()
        if notify:
            logging.getLogger().info('Plot saved to {} in ({:.3f}s)'.format(filepath, time.time() -t0))

    @property
    @printexception
    def info_text(self):
        return self._info_text

    @printexception
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

    @printexception
    def update_data(self, data):
        self.update_data_signal.emit(data)


    @printexception
    def update_selected_indices(self, selected_indices):
        getattr(self, "update_selected_indices_signal").emit(selected_indices)

    @printexception
    def connect_signals(self):
        for name in [
            'data'.format(self.name),
            'selected_indices'.format(self.name),
        ]:
            getattr(getattr(self, "update_{}_signal".format(name)), 'connect')(getattr(self, "update_{}_signal_emitted".format(name)))

    @printexception
    def clear(self):
        widget = getattr(self.parent, self.widget_name)
        try:
            widget.blockSignals(True)
            widget.clear()
        except:
            raise
        finally:
            widget.blockSignals(False)

class SelectableListQt(BaseQt):
    update_data_signal = pyqtSignal(list)
    update_selected_indices_signal = pyqtSignal(list)

    @printexception
    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            if widget.count() != 0:
                getattr(self.parent, self.widget_name).clear()
            for val in data:
                widget.addItem(QListWidgetItem(val))
        except:
            raise
        finally:
            widget.blockSignals(False)

    @printexception
    def update_selected_indices_signal_emitted(self, selected_indices):
        widget = getattr(self.parent, self.widget_name)
        try:
            widget.blockSignals(True)
            for i in selected_indices:
                widget.item(i).setSelected(True)
        except:
            raise
        finally:
            widget.blockSignals(False)

    @printexception
    def update_selected_indices_from_gui(self, *args, **kwargs):
        print('Item selection has been changed.')
        getattr(self.parent.no_qt, self.name).update_selected_indices([i.row() for i in getattr(self.parent, self.widget_name).selectedIndexes()])


class ParameterCombobox(BaseQt):
    update_data_signal = pyqtSignal(list)
    update_selected_indices_signal = pyqtSignal(list)

    @printexception
    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            if widget.count() != 0:
                getattr(self.parent, self.widget_name).clear()
            widget.addItems(data)
        except:
            raise
        finally:
            widget.blockSignals(False)
        if hasattr(getattr(self.parent.no_qt, self.name), '_selected_indices'):
            getattr(self.parent.parameter_table, 'update_item_flags')()

    @printexception
    def update_selected_indices_signal_emitted(self, selected_indices):
        getattr(self.parent, self.widget_name).blockSignals(True)
        getattr(self.parent, self.widget_name).setCurrentText(getattr(self.parent.no_qt, self.name).data[selected_indices[0]])
        getattr(self.parent, self.widget_name).blockSignals(False)

    @printexception
    def update_selected_indices_from_gui(self, *args, **kwargs):
        idx = getattr(self.parent.no_qt, self.name).data.index(str(getattr(self.parent, self.widget_name).currentText()))
        getattr(self.parent.no_qt, self.name).update_selected_indices([idx])


class TableQt(BaseQt):
    update_data_signal = pyqtSignal(collections.OrderedDict)
    update_selected_indices_signal = pyqtSignal(collections.OrderedDict)

    @printexception
    def update_data_signal_emitted(self, data):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            if widget.columnCount() != 0:
                getattr(self.parent, self.widget_name).clear_table_contents()
            if widget.columnCount() == 0:
                widget.set_column_names(data.keys())
            for column_name, val in data.items():
                widget.set_column(column_name, val)
            self.update_item_flags()
        except:
            raise
        finally:
            widget.blockSignals(False)

    @printexception
    def update_selected_indices_signal_emitted(self, selected_indices):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            for cn, val in selected_indices.items():
                cidx = widget.column_index(cn)
                for ridx in widget.column_data_indices(cn):
                    if val == '__all__' or (not isinstance(val, basestring) and ridx in val):
                        widget.item(ridx, cidx).setSelected(True)
                    else:
                        widget.item(ridx, cidx).setSelected(False)
        except:
            raise
        finally:
            widget.blockSignals(False)

    @printexception
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


class ParameterTableQt(TableQt):

    @printexception
    def update_item_flags(self):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            for cidx in range(widget.columnCount()):
                cn = widget.column_name(cidx)
                if cn == self.parent.no_qt.subtract_parameter_list.selected_value:
                    flag = Qt.NoItemFlags
                else:
                    flag = Qt.ItemIsSelectable | Qt.ItemIsEnabled
                for ridx in widget.column_data_indices(cn):
                    widget.set_cell(ridx, cidx, flag=flag)
        except:
            raise
        finally:
            widget.blockSignals(False)

class FitResultTableQt(TableQt):

    @printexception
    def update_item_flags(self):
        widget = getattr(self.parent, self.widget_name)
        widget.blockSignals(True)
        try:
            for cidx in range(widget.columnCount()):
                for ridx in widget.column_data_indices(widget.column_name(cidx)):
                    widget.set_cell(ridx, cidx, flag=Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        except:
            raise
        finally:
            widget.blockSignals(False)


class PlotDataQt(qutip_enhanced.qtgui.gui_helpers.QtGuiClass):
    def __init__(self, parent=None, no_qt=None):
        self.observation_list = SelectableListQt(name='observation_list', widget_name='observation_widget', parent=self)
        self.average_parameter_list = SelectableListQt(name='average_parameter_list', widget_name='average_parameter_widget', parent=self)
        self.x_axis_parameter_list = ParameterCombobox(name='x_axis_parameter_list', widget_name='x_axis_parameter_comboBox', parent=self)
        self.col_ax_parameter_list = ParameterCombobox(name='col_ax_parameter_list', widget_name='col_ax_parameter_comboBox', parent=self)
        self.row_ax_parameter_list = ParameterCombobox(name='row_ax_parameter_list', widget_name='row_ax_parameter_comboBox', parent=self)
        self.subtract_parameter_list = ParameterCombobox(name='subtract_parameter_list', widget_name='subtract_parameter_comboBox', parent=self)
        self.parameter_table = ParameterTableQt(name='parameter_table', widget_name='parameter_table_widget', parent=self)
        self.fit_result_table = FitResultTableQt(name='fit_result_table', widget_name='fit_result_table_widget', parent=self)
        super(PlotDataQt, self).__init__(parent=parent, no_qt=no_qt, ui_filepath=os.path.join(os.path.dirname(__file__), 'qtgui/plot_data.ui'))

    update_info_text_signal = pyqtSignal(str)

    @printexception
    def clear_signal_emitted(self):
        for name in [
            'x_axis_parameter_comboBox',
            'col_ax_parameter_comboBox',
            'row_ax_parameter_comboBox',
            'subtract_parameter_comboBox',
            'parameter_table_widget',
            'observation_widget',
            'average_parameter_widget',
            'fit_result_table_widget',
        ]:
            getattr(self, name).clear() #self.clear_widget(name)
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

    @printexception
    def update_info_text(self, info_text):
        self.update_info_text_signal.emit(info_text)

    @printexception
    def update_info_text_signal_emitted(self, info_text):
        self.info.setPlainText(info_text)

    @printexception
    def redraw_plot(self):
        self.fig.tight_layout()
        self.canvas.draw()

    @printexception
    def redraw_plot_fit(self):
        self.fig_fit.tight_layout()
        self.canvas_fit.draw()

    @printexception
    def init_gui(self):
        super(PlotDataQt, self).init_gui()
        for name in [
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
            'parameter_table',
            'fit_result_table'

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
        self.observation_widget.itemSelectionChanged.connect(self.observation_list.update_selected_indices_from_gui)
        self.average_parameter_widget.itemSelectionChanged.connect(self.average_parameter_list.update_selected_indices_from_gui)
        self.open_code_button.clicked.connect(self.open_measurement_code)
        self.open_explorer_button.clicked.connect(self.open_explorer)

        self.x_axis_parameter_comboBox.currentIndexChanged.connect(self.x_axis_parameter_list.update_selected_indices_from_gui)
        self.col_ax_parameter_comboBox.currentIndexChanged.connect(self.col_ax_parameter_list.update_selected_indices_from_gui)
        self.row_ax_parameter_comboBox.currentIndexChanged.connect(self.row_ax_parameter_list.update_selected_indices_from_gui)
        self.subtract_parameter_comboBox.currentIndexChanged.connect(self.subtract_parameter_list.update_selected_indices_from_gui)

    @printexception
    def open_measurement_code(self, *args, **kwargs):
        if hasattr(self.no_qt.data, 'filepath'):
            subprocess.Popen(r"start {}/meas_code.py".format(os.path.dirname(self.no_qt.data.filepath)), shell=True)
        else:
            print('No filepath.')

    @printexception
    def open_explorer(self, *args, **kwargs):
        if hasattr(self.no_qt.data, 'filepath'):
            subprocess.Popen("explorer {}".format(os.path.abspath(os.path.dirname(self.no_qt.data.filepath))), shell=True)
        else:
            print('No filepath.')


def cpd():
    hdfl = [fn for fn in os.listdir(os.getcwd()) if fn.endswith('.hdf')]
    if not 'data.hdf' in hdfl and len(hdfl) != 1:
        raise Exception('Error: No hdf file found in files {}'.format(os.listdir(os.getcwd())))
    data = Data(iff=os.path.join(os.getcwd(), hdfl[0]))
    if 'prepare_data.py' in os.listdir(os.getcwd()):
        from prepare_data import prepare_data
        data = prepare_data(data)
    out = PlotData(data=data)
    if 'sweeps' in data.parameter_names:
        out.x_axis_parameter_list.update_selected_indices(out.x_axis_parameter_list.selected_indices_default(exclude_names=['sweeps']))
        out.average_parameter_list.update_selected_data(['sweeps'])
    if any(['result_' in out.observation_list.data]):
        out.observation_list.update_selected_data([i for i in out.observation_list.data if i.startswith('result_')])
    else:
        out.observation_list.update_selected_indices([0])
    out.update_plot()
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


def number_of_points_of_hdf_files_in_subfolders(folder, endswith='data.hdf'):
    l = subfolders_with_hdf(folder)
    out_openable = []
    out_failed = []
    for subdir in l:
        for root, dirs, files in os.walk(subdir):
            for file in files:
                if file.endswith(endswith):
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