# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import os, sys, subprocess

import numpy as np
import pandas as pd
import itertools
from numbers import Number
import traceback
import collections
import subprocess
from PyQt5.QtWidgets import QListWidgetItem, QTableWidgetItem, QMainWindow

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.uic import compileUi

from .qtgui import plot_data_gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import functools

if sys.version_info.major == 2:
    import __builtin__
else:
    import builtins as __builtin__
import lmfit
from . import lmfit_models


class TC:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def getter_setter_gen(name, type_):
    def getter(self):
        return getattr(self, "_" + name)

    def setter(self, value):
        if not isinstance(value, type_):
            raise TypeError("{} attribute must be set to an instance of {} but was set to {}".format(name, type_, value))
        setattr(self, "_" + name, value)

    return property(getter, setter)


def getter_setter_gen_tc(name, tc):
    k = tc.kwargs

    def getter(self):
        return getattr(self, "_" + name)

    getter = k.get('getter', getter)
    if 'setter' in k:
        return property(getter, k['setter'])
    elif 'start' in k and 'stop' in k:
        def setter(self, val):
            setattr(self, '_' + name, check_range(check_type(val, name, k['typ']), name, k['start'], k['stop']))
    elif 'list_type' in k:
        def setter(self, val):
            setattr(self, '_' + name, check_array_like_typ(val, name, k['list_type']))
    elif 'list_element' in k:
        def setter(self, val):
            setattr(self, '_' + name, check_list_element(val, name, k['list_element']))
    else:
        raise ValueError('Error. {}'.format(k))
    return property(getter, setter)


def auto_attr_check(cls):
    new_dct = {}
    for key, val in cls.__dict__.items():
        if isinstance(val, type):
            val = getter_setter_gen(key, val)
        elif type(val) == TC:
            val = getter_setter_gen_tc(key, val)
        new_dct[key] = val
    # Creates a new class, using the modified dictionary as the class dict:
    return type(cls)(cls.__name__, cls.__bases__, new_dct)


def check_type(val, name, typ):
    if issubclass(type(val), typ):
        return val
    else:
        raise Exception("Property {} must be {} but is {}".format(name, typ, type(val)))


def check_types(val, name, types):
    if any([issubclass(type(val), typ) for typ in types]):
        return val
    else:
        raise Exception("Property {} must be in {} but is {}".format(name, types, type(val)))


def check_range(val, name, start, stop):
    if start <= val <= stop:
        return val
    else:
        raise Exception("Property {} must be in range ({}, {}) but has a value of {}".format(name, start, stop, val))


def check_range_type(val, name, typ, start, stop):
    return check_range(check_type(val, name, typ), name, start, stop)


def check_array_like(val, name):
    at = [list, np.ndarray]
    if type(val) in at:
        return val
    else:
        raise Exception("Type of property {} must be in list {}. Tried to assign val {} of type {}.".format(name, at, val, type(val)))


def check_array_like_typ(val, name, typ):
    val = [check_type(i, name + '_i', typ) for i in check_array_like(val, name)]
    if typ in [float, int, Number]:
        val = np.array(val)
    return val


def check_array_like_types(val, name, types):
    val = [check_types(i, name + '_i', types) for i in check_array_like(val, name)]
    if types[0] in [float, int, Number]:  # This assumes, that if type[0] is numeric, all items are numeric
        val = np.array(val)
    return val


def check_list_element(val, name, l):
    if val in l:
        return val
    else:
        raise Exception("Property {} must be in list {} but has a value of {}".format(name, l, val))


def ret_getter(name):
    def getter(self):
        return getattr(self, '_' + name)

    return getter


def ret_property_typecheck(name, typ):
    def setter(self, val):
        setattr(self, '_' + name, check_type(val, name, typ))

    return property(ret_getter(name), setter)


def ret_property_range(name, typ, start, stop):
    def setter(self, val):
        setattr(self, '_' + name, check_range(check_type(val, name, typ), name, start, stop))

    return property(ret_getter(name), setter)


def ret_property_list_element(name, l):
    def setter(self, val):
        setattr(self, '_' + name, check_list_element(val, name, l))

    return property(ret_getter(name), setter)


def ret_property_array_like(name):
    def setter(self, val):
        setattr(self, '_' + name, check_array_like(val, name))

    return property(ret_getter(name), setter)


def ret_property_array_like_typ(name, typ):
    def setter(self, val):
        setattr(self, '_' + name, check_array_like_typ(val, name, typ))

    return property(ret_getter(name), setter)


def ret_property_array_like_types(name, types):
    def setter(self, val):
        setattr(self, '_' + name, check_array_like_types(val, name, types))

    return property(ret_getter(name), setter)


def ptrepack(file, folder, tempfile=None):
    # C:\Users\yy3\AppData\Local\conda\conda\envs\py27\Scripts\ptrepack.exe -o --chunkshape=auto --propindexes --complevel=0 --complib=blosc data.hdf data_tmp.hdf
    tempfile = 'temp.hdf' if tempfile is None else tempfile
    ptrepack = r"{}\Scripts\ptrepack.exe".format(os.path.dirname(sys.executable))
    command = [ptrepack, "-o", "--chunkshape=auto", "--propindexes", "--complevel=9", "--complib=blosc", file, tempfile]
    _ = subprocess.call(command, cwd=folder)
    os.remove(os.path.join(folder, file))
    os.rename(os.path.join(folder, tempfile), os.path.join(folder, file))


def ptrepack_all(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".hdf"):
                ptrepack(file, root)


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

    def get_last_parameter(self, df):
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
        return last_parameter

    def init(self, init_from_file=None, iff=None, last_parameter=None, df=None):
        init_from_file = iff if iff is not None else init_from_file
        if init_from_file is not None or df is not None:
            if init_from_file is not None:
                if init_from_file.endswith('.hdf'):
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
                    self.hdf_filepath = init_from_file
                elif init_from_file.endswith('.csv'):
                    df = pd.read_csv(init_from_file, compression='gzip')

            self._df = df[pd.notnull(df)]
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
        else:
            self._df = pd.DataFrame(columns=self.variables)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if value is not None:
            raise Exception("You must not change the data!")

    def save(self, filepath):
        if filepath.endswith('.csv'):
            self._df.to_csv(filepath, index=False, compression='gzip')
        elif filepath.endswith('.hdf'):
            store = pd.HDFStore(filepath)
            store.put('df', self._df, table=True)
            for key in ["parameter_names", 'observation_names', 'dtypes']:
                setattr(store.get_storer('df').attrs, key, getattr(self, key))
            store.close()
            ptrepack(os.path.split(filepath)[1], os.path.split(filepath)[0])
            self.hdf_filepath = filepath

    def append(self, df_or_l):
        if type(df_or_l) == pd.DataFrame:
            df = df_or_l
            if not (df.columns == self.parameter_names).all():
                raise Exception('Error: Dataframe columns dont match: '.format(df.columns, self.parameter_names))
            df_append = df_or_l
        else:
            l = df_or_l
            if type(l) in [collections.OrderedDict, dict]:
                l = [l]
            for kwargs in l:
                if len(kwargs) != len(self.parameter_names):
                    raise Exception('Missing parameter for dataframe.')
                if not all([i in kwargs for i in self.parameter_names]):
                    raise Exception('Wrong parameter for dataframe.')
            df_append = pd.DataFrame(columns=self.parameter_names, data=l)
        if len(self.df) == 0:
            l_obs = []
            for idx in range(len(df_append)):
                l_obs.append(collections.OrderedDict())
                for k, v in self.dtypes.items():
                    l_obs[idx][k] = getattr(__builtin__, v)()
            df_append_obs = pd.DataFrame(columns=self.observation_names, data=l_obs)
            self._df = pd.concat([df_append, df_append_obs], axis=1)
        else:
            self._df = self._df.append(df_append, ignore_index=True)


    def parse_l(self, l):
        if type(l) in [collections.OrderedDict, dict]:
            l = [l]
        return l

    def dict_access(self, l, df=None):
        df = self.df if df is None else df
        if len(l) == 0:
            raise Exception('Nope!')
        else:
            l = self.parse_l(l)
            return df[functools.reduce(np.logical_or, [functools.reduce(np.logical_and, [df[key] == val for key, val in d.items()]) for d in l])]

    def dict_delete(self, l, df=None):
        df = self.df if df is None else df
        if len(l) > 0:
            df.drop(self.dict_access(l=l, df=df).index, inplace=True)
            df.reset_index(inplace=True)

    def set_observations(self, l, start_idx=None):
        start_idx = len(self._df) - 1 if start_idx is None else start_idx
        if type(l) in [collections.OrderedDict, dict]:
            l = [l]
        for idx, kwargs in enumerate(l):
            for obs, val in kwargs.items():
                if obs not in self.observation_names:
                    raise Exception('Error: {}'.format(l))
                self._df.set_value(start_idx - len(l) + idx + 1, obs, val)

    def column_product(self, column_names):
        return itertools.product(*[getattr(self.df, cn).unique() for cn in column_names])

    def iterator(self, column_names, output_data_instance=False):
        for idx, p in enumerate(self.column_product(column_names=column_names)):
            d = collections.OrderedDict([i for i in zip(column_names, p)])
            d_idx = collections.OrderedDict([(cn, np.argwhere(getattr(self.df, cn).unique() == p)[0, 0]) for cn, p in zip(column_names, p)])
            if output_data_instance:
                new_parameter_names = [i for i in self.parameter_names if not i in column_names]
                sub = Data(parameter_names=new_parameter_names, observation_names=self.observation_names, dtypes=self.dtypes)
                sub._df = self.dict_access(d)
            else:
                sub = self.dict_access(d)
            yield d, d_idx, idx, sub


def recompile_plotdata_ui_file():
    fold = "{}/qtgui".format(os.path.dirname(__file__))
    name = "plot_data"
    uipath = r"{}/{}.ui".format(fold, name)
    pypath = r"{}/{}_gui.py".format(fold, name)
    with open(pypath, 'w') as f:
        compileUi(uipath, f)
    reload(plot_data_gui)


class PlotData:

    def __init__(self, title=None, parent=None, gui=True, **kwargs):
        super(PlotData, self).__init__()
        if gui:
            self._gui = PlotDataQt(plot_data_no_qt=self, parent=parent)
            self.gui.show()
        if 'path' in kwargs:
            self.set_data_from_path(path=kwargs['path'])
        elif 'data' in kwargs:
            self.data = kwargs['data']
        if title is not None:
            self.update_window_title(title)

    fit_function = 'cosine'
    show_legend = False

    def set_data_from_path(self, path):
        try:
            self.data = Data(iff=path)
            self.data_path = path
            self.update_window_title(self.data_path)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def data_path(self):
        try:
            return self._data_path
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @data_path.setter
    def data_path(self, val):
        try:
            self._data_path = val
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(self.data_path)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)


    @property
    def window_title(self):
        try:
            return self._window_title
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_window_title(self, val):
        try:
            self._window_title = str(val)
            if hasattr(self, '_gui'):
                self.gui.update_window_title(self.window_title)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)


    @property
    def gui(self):
        try:
            return self._gui
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def data(self):
        try:
            return self._data
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @data.setter
    def data(self, val):
        try:
            self.delete_attributes()
            if hasattr(self, '_gui'):
                self.gui.clear()
            self._data = val
            self.new_data_arrived()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)


    def new_data_arrived(self):
        try:
            self.update_x_axis_parameter_list()
            self.update_parameter_table_data()
            self.update_parameter_table_selected_indices()
            self.update_observation_list_data()
            self.update_observation_list_selected_indices()
            if len(self.data.df) < 5000:
                self.update_plot()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def delete_attributes(self):
        for attr_name in [
            '_data_path',
            '_data',
            '_x_axis_parameter_list',
            '_parameter_table_data',
            '_parameter_table_selected_data',
            '_observation_list_data',
            '_observation_list_selected_indices',
            '_selected_plot_items',
            '_fit_select_table_data',
            '_fit_select_table_selected_rows',
            '_fit_results',
            '_fit_result_table_data'
        ]:
            if hasattr(self, attr_name):
                try:
                    delattr(self, attr_name)
                except:
                    print("Attribute ", attr_name, " could not be deleted.")
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def x_axis_parameter(self):
        try:
            return self._x_axis_parameter
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @x_axis_parameter.setter
    def x_axis_parameter(self, val):
        try:
            if getattr(self, '_x_axis_parameter', None) != val:
                self._x_axis_parameter = val
                if hasattr(self, '_gui'):
                    self.gui.update_x_axis_parameter_comboBox()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def x_axis_parameter_list(self):
        return self._x_axis_parameter_list

    def update_x_axis_parameter_list(self):
        try:
            self._x_axis_parameter_list = [cn for cn in self.parameter_names_reduced() if cn in self.data.df._get_numeric_data().columns]
            if not hasattr(self, '_x_axis_parameter') or (self.x_axis_parameter not in self.x_axis_parameter_list and len(self.x_axis_parameter_list) > 0):
                self._x_axis_parameter = self.x_axis_parameter_with_largest_dim()
            if hasattr(self, '_gui'):
                self.gui.update_x_axis_parameter_comboBox()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def parameter_names_reduced(self, data=None):
        try:
            data = self.data if data is None else data
            return [i for i in data.parameter_names if not '_idx' in i]
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def x_axis_parameter_with_largest_dim(self):
        return self.x_axis_parameter_list[np.argmax([len(getattr(self.data.df, p).unique()) for p in self.x_axis_parameter_list])]

    @property
    def parameter_table_data(self):
        try:
            val_none = collections.OrderedDict([(cn, []) for cn in self.parameter_names_reduced()])
            return getattr(self, '_parameter_table_data', val_none)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_parameter_table_data(self):
        try:
            ptd = collections.OrderedDict()
            for cn in self.parameter_names_reduced():
                ptd[cn] = getattr(self.data.df, cn).unique()
            self._parameter_table_data = ptd
            if hasattr(self, '_gui'):
                self.gui.update_parameter_table_data(parameter_table_data=self.parameter_table_data)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def observation_list_data(self):
        return getattr(self, '_observation_list_data')

    def update_observation_list_data(self):
        try:
            if not hasattr(self, '_observation_list_data'):
                self._observation_list_data = self.observation_names_reduced()
                if hasattr(self, '_gui'):
                    self.gui.update_observation_list_data(self.observation_list_data)
            elif self.observation_list_data != self.observation_names_reduced():
                raise Exception('Error: Data of observation list must not be changed after data was given to PlotData.', self.observation_list_data, self.observation_names_reduced())
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def observation_list_selected_indices(self):
        return self._observation_list_selected_indices

    def update_observation_list_selected_indices(self, val=None):
        try:
            self._observation_list_selected_indices = [0] if val is None else val
            if hasattr(self, '_gui'):
                self.gui.update_observation_list_selected_indices(self.observation_list_selected_indices)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def observation_list_selected_data(self):
        try:
            return [self.observation_list_data[i] for i in self.observation_list_selected_indices]
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def observation_names_reduced(self):
        try:
            return [i for i in self.data.observation_names if not i in ['trace', 'start_time', 'end_time', 'thresholds']]
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def parameter_table_selected_indices(self):
        try:
            return self._parameter_table_selected_indices
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_parameter_table_selected_indices(self, parameter_table_selected_indices=None):
        try:
            if parameter_table_selected_indices is None:
                # if len(self.data.df) > 1000:
                #     self._parameter_table_selected_data = collections.OrderedDict([(key, []) for key, val in self.parameter_table_data.items()])
                #     return
                self._parameter_table_selected_indices = collections.OrderedDict([(key, []) if key in ['sweeps', self.x_axis_parameter] else (key, '__all__') for key, val in self.parameter_table_data.items()])
            elif not isinstance(parameter_table_selected_indices, collections.OrderedDict):
                raise Exception(type(parameter_table_selected_indices), parameter_table_selected_indices)
            else:
                out = collections.OrderedDict([(key, []) if key in ['sweeps', self.x_axis_parameter] else (key, []) for key, val in self.parameter_table_data.items()])
                for cn, val in parameter_table_selected_indices.items():
                    out[cn] = val
                self._parameter_table_selected_indices = out
            if hasattr(self, '_gui'):
                self.gui.update_parameter_table_selected_indices(self.parameter_table_selected_indices)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def parameter_table_selected_data(self):
        try:
            out = collections.OrderedDict()
            for cn, val in self.parameter_table_selected_indices.items():
                data_full = getattr(self.data.df, cn).unique()
                if val == '__all__':
                    out[cn] = data_full
                else:
                    out[cn] = [data_full[i] for i in val]
            return out
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_parameter_table_selected_data(self, parameter_table_selected_data):
        try:
            out = collections.OrderedDict()
            for cn, val in parameter_table_selected_data.items():
                if val in ['__all__', '__average__']:
                    out[cn] = val
                else:
                    indices = []
                    for i in val:
                        indices.append(np.where(self.parameter_table_data[cn] == i)[0][0])
                    out[cn] = indices
            self.update_parameter_table_selected_indices(out)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def selected_plot_items(self):
        try:
            return self._selected_plot_items
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_selected_plot_items(self):
        try:
            if len(getattr(self.data.df, self.x_axis_parameter).unique()) == 1:
                self.x_axis_parameter = self.x_axis_parameter_with_largest_dim()
            out = self.parameter_table_selected_data
            if all([len(i) == 0 for i in out.values()]):
                self._selected_plot_items = []
                return
            for key, val in out.items():
                if key == self.x_axis_parameter:
                    out[key] = ['__all__']
                elif len(val) == 0:
                    out[key] = ['__average__']
            self._selected_plot_items = list(itertools.product(*out.values()))
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def fit_select_table_data(self):
        try:
            return self._fit_select_table_data
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_fit_select_table_data(self):
        try:
            cpd = collections.OrderedDict()
            for column_idx, column_name in enumerate(self.parameter_names_reduced()):
                cpd[column_name] = []
                for pi in self.selected_plot_items:
                    if not pi[column_idx] in ['__average__', '__all__']:
                        cpd[column_name].append(pi[column_idx])
                if len(cpd[column_name]) == 0:
                    del cpd[column_name]
            self._fit_select_table_data = cpd
            if hasattr(self, '_gui'):
                self.gui.update_fit_select_table_data(self.fit_select_table_data)
            self.update_fit_select_table_selected_rows(range(len(self.fit_select_table_data.values()[0])))
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def fit_select_table_selected_rows(self):
        try:
            return self._fit_select_table_selected_rows
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_fit_select_table_selected_rows(self, fit_select_table_selected_rows=None):
        try:
            self._fit_select_table_selected_rows = [] if fit_select_table_selected_rows is None else fit_select_table_selected_rows  # self.fit_select_table.selected_items_unique_column_indices()
            if hasattr(self, '_gui'):
                self.gui.update_fit_select_table_selected_rows(self.fit_select_table_selected_rows)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def fit_results(self):
        try:
            return self._fit_results
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_fit_results(self):
        try:
            spi = self.line_plot_data()
            if hasattr(self, 'custom_model'):
                mod = self.custom_model
            elif self.fit_function == 'cosine':
                mod = lmfit_models.CosineModel()
            elif self.fit_function == 'exp':
                pass
            self._fit_results = []
            for idx in self.fit_select_table_selected_rows:
                i = spi[idx]
                # try:
                params = mod.guess(data=np.array(i['y']), x=np.array(i['x']))
                fp = getattr(self, 'fix_params', {})
                if all(key in params for key in fp.keys()):
                    for key, val in fp.items():
                        params[key].vary = False
                        params[key].value = val
                self._fit_results.append([i, mod.fit(np.array(i['y']), params, x=np.array(i['x']))])
                # except:
                #     print('fitting failed: {}'.format(i))
            self.update_fit_result_table_data()
        except ValueError:
            print("Can not fit, input contains nan values")
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def fit_result_table_data(self):
        try:
            return self._fit_result_table_data
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_fit_result_table_data(self):
        try:
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
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def ret_line_plot_data_single(self, condition_dict, observation_name):
        try:
            out = self.data.df[functools.reduce(np.logical_and, [self.data.df[key] == val for key, val in condition_dict.items() if val not in ['__all__', '__average__']])]
            # TODO: wont work for dates as can not be averaged
            return out.groupby([key for key, val in condition_dict.items() if val != '__average__']).agg({observation_name: np.mean}).reset_index()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def line_plot_data(self):
        try:
            plot_data = []
            if len(self.data.df) > 0:
                for p in self.selected_plot_items:
                    condition_dict = collections.OrderedDict([(ni, pi) for ni, pi in zip(self.parameter_names_reduced(), p)])
                    for observation_name in self.observation_list_selected_data:
                        dfxy = self.ret_line_plot_data_single(condition_dict, observation_name)
                        condition_dict_reduced = collections.OrderedDict([(key, val) for key, val in condition_dict.items() if val not in ['__average__', '__all__']])
                        plot_data.append(
                            dict(
                                condition_dict_reduced=condition_dict_reduced,
                                observation_name=observation_name,
                                x=getattr(dfxy, self.x_axis_parameter),
                                y=getattr(dfxy, observation_name)
                            )
                        )
            return plot_data
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_plot(self):
        try:
            if hasattr(self, '_data'):
                self.update_selected_plot_items()
                self.update_fit_select_table_data()
                if hasattr(self, '_gui'):
                    try:
                        self.gui.fig.clear()
                    except:
                        pass
                    self.gui.ax = self.gui.fig.add_subplot(111)
                    for idx, pdi in enumerate(self.line_plot_data()):
                        self.gui.ax.plot(pdi['x'], pdi['y'], '-')
                    self.gui.fig.tight_layout()
                    self.gui.canvas.draw()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_plot_fit(self):
        try:
            self.update_fit_results()
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
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def save_plot(self, filepath):
        try:
            plt.ioff()
            fig, ax = plt.subplots(1, 1)
            if len(getattr(self.data.df, self.x_axis_parameter).unique()) == 1:
                self.x_axis_parameter = self.x_axis_parameter_with_largest_dim()
            cn = self.parameter_names_reduced()
            cn.remove(self.x_axis_parameter)
            for d, d_idx, idx, df_sub in self.data.iterator(cn):
                for observation in self.observation_list_selected_data:
                    dfagg = df_sub.groupby([self.x_axis_parameter]).agg({observation: np.mean}).reset_index()
                    ax.plot(getattr(dfagg, self.x_axis_parameter), getattr(dfagg, observation))
            fig.tight_layout()
            fig.savefig(filepath)
            plt.ion()
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def info_text(self):
        try:
            return self._info_text
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def update_info_text(self, info_text):
        try:
            self._info_text = info_text
            if hasattr(self, '_gui'):
                self.gui.update_info_text(info_text)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)


class PlotDataQt(QMainWindow, plot_data_gui.Ui_window):

    def __init__(self, parent=None, plot_data_no_qt=None):
        super(PlotDataQt, self).__init__(parent=parent)
        self.plot_data_no_qt = plot_data_no_qt
        self.setupUi(self)
        self.init_gui()

    update_x_axis_parameter_comboBox_signal = pyqtSignal()
    clear_signal = pyqtSignal()
    show_signal = pyqtSignal()
    close_signal = pyqtSignal()
    update_window_title_signal = pyqtSignal(str)
    update_parameter_table_data_signal = pyqtSignal(collections.OrderedDict)
    update_parameter_table_selected_indices_signal = pyqtSignal(collections.OrderedDict)
    update_observation_list_data_signal = pyqtSignal(list)
    update_observation_list_selected_indices_signal = pyqtSignal(list)
    update_fit_select_table_data_signal = pyqtSignal(collections.OrderedDict)
    update_fit_select_table_selected_rows_signal = pyqtSignal(list)
    update_fit_result_table_data_signal = pyqtSignal(collections.OrderedDict)
    update_info_text_signal = pyqtSignal(str)

    def clear(self):
        self.clear_signal.emit()

    def clear_signal_emitted(self):
        for item in [
            ['x_axis_parameter_comboBox', 'clear'],
            ['parameter_table', 'clear_table_contents'],
            ['observation_widget', 'clear'],
            ['fit_select_table', 'clear_table_contents'],
            ['fit_result_table', 'clear_table_contents'],
        ]:
            getattr(self, item[0]).blockSignals(True)
            getattr(getattr(self, item[0]), item[1])()
            getattr(self, item[0]).blockSignals(False)
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

    def update_window_title(self, val):
        self.update_window_title_signal.emit(val)

    def update_window_title_signal_emitted(self, val):
        self.setWindowTitle(val)

    def update_x_axis_parameter_comboBox(self):
        self.update_x_axis_parameter_comboBox_signal.emit()

    def update_x_axis_parameter_comboBox_signal_emitted(self):
        self.x_axis_parameter_comboBox.blockSignals(True)
        if hasattr(self.plot_data_no_qt, '_x_axis_parameter_list') and hasattr(self.plot_data_no_qt, '_x_axis_parameter'):
            if self.x_axis_parameter_comboBox.count() == 0:
                self.x_axis_parameter_comboBox.addItems(self.plot_data_no_qt.x_axis_parameter_list)  # currentIndexChanged is triggered, value is first item (e.g. sweeps)
            self.x_axis_parameter_comboBox.setCurrentText(self.plot_data_no_qt.x_axis_parameter)
        self.x_axis_parameter_comboBox.blockSignals(False)
        self.update_parameter_table_item_flags()

    def update_x_axis_parameteter_from_comboBox(self):
        self.plot_data_no_qt.x_axis_parameter = str(self.x_axis_parameter_comboBox.currentText())

    def update_parameter_table_data(self, parameter_table_data):
        self.update_parameter_table_data_signal.emit(parameter_table_data)

    def update_parameter_table_data_signal_emitted(self, parameter_table_data):
        self.parameter_table.blockSignals(True)
        if self.parameter_table.columnCount() == 0:
            self.parameter_table.set_column_names(parameter_table_data.keys())
        for column_name, val in parameter_table_data.items():
            self.parameter_table.set_column(column_name, val)
        self.update_parameter_table_item_flags()
        self.parameter_table.blockSignals(False)

    def update_parameter_table_item_flags(self):
        self.parameter_table.blockSignals(True)
        if hasattr(self.plot_data_no_qt, '_x_axis_parameter'):
            for cidx in range(self.parameter_table.columnCount()):
                cn = self.parameter_table.column_name(cidx)
                if cn == self.plot_data_no_qt.x_axis_parameter:
                    flag = Qt.NoItemFlags
                else:
                    flag = Qt.ItemIsSelectable | Qt.ItemIsEnabled
                for ridx in self.parameter_table.column_data_indices(cn):
                    self.parameter_table.set_cell(ridx, cidx, flag=flag)
        self.parameter_table.blockSignals(False)

    def update_observation_list_data(self, observation_list_data):
        self.update_observation_list_data_signal.emit(observation_list_data)

    def update_observation_list_data_signal_emitted(self, observation_list_data):
        if self.observation_widget.count() == 0:
            for obs in observation_list_data:
                self.observation_widget.addItem(QListWidgetItem(obs))
        elif len(self.observation_list_data) != self.observation_widget.count():
            raise Exception('Error: ', self.observation_widget.count(), observation_list_data)

    def update_parameter_table_selected_indices(self, selected_indices):
        self.update_parameter_table_selected_indices_signal.emit(selected_indices)

    def update_parameter_table_selected_indices_signal_emitted(self, parameter_table_selected_indices):
        self.parameter_table.blockSignals(True)
        for cn, val in parameter_table_selected_indices.items():
            cidx = self.parameter_table.column_index(cn)
            for ridx in self.parameter_table.column_data_indices(cn):
                if val == '__all__' or ridx in val:
                    self.parameter_table.item(ridx, cidx).setSelected(True)
                else:
                    self.parameter_table.item(ridx, cidx).setSelected(False)
        self.parameter_table.blockSignals(False)

    def update_parameter_table_selected_indices_from_gui(self):
        out = collections.OrderedDict()
        column_names = self.parameter_table.column_names
        for item in self.parameter_table.selectedItems():
            cn = column_names[item.column()]
            if not cn in out:
                out[cn] = []
            out[cn].append(item.row())
        self.plot_data_no_qt.update_parameter_table_selected_indices(out)

    def update_observation_list_selected_indices(self, selected_indices):
        self.update_observation_list_selected_indices_signal.emit(selected_indices)

    def update_observation_list_selected_indices_signal_emitted(self, update_observation_list_selected_indices):
        self.observation_widget.blockSignals(True)
        for i in update_observation_list_selected_indices:
            self.observation_widget.item(i).setSelected(True)
        self.observation_widget.blockSignals(False)

    def update_observation_list_selected_indices_from_gui(self):
        self.plot_data_no_qt.update_observation_list_selected_indices([i.row() for i in self.observation_widget.selectedIndexes()])

    def update_fit_select_table_data(self, fit_select_table_data):
        self.update_fit_select_table_data_signal.emit(fit_select_table_data)

    def update_fit_select_table_data_signal_emitted(self, fit_select_table_data):
        self.fit_select_table.blockSignals(True)
        self.fit_select_table.clear_table_contents()
        self.fit_select_table.set_column_names(fit_select_table_data.keys())
        for column_name, val in fit_select_table_data.items():
            self.fit_select_table.set_column(column_name, val, [Qt.ItemIsSelectable | Qt.ItemIsEnabled for _ in range(len(val))])
        self.fit_select_table.blockSignals(False)

    def update_fit_select_table_selected_rows_from_gui(self):
        self.plot_data_no_qt.update_fit_select_table_selected_rows(self.fit_select_table.selected_items_unique_row_indices())

    def update_fit_select_table_selected_rows(self, selected_rows):
        self.update_fit_select_table_selected_rows_signal.emit(selected_rows)

    def update_fit_select_table_selected_rows_signal_emitted(self, fit_select_table_selected_rows):
        self.fit_select_table.blockSignals(True)
        for ridx in range(self.fit_select_table.rowCount()):
            if self.fit_select_table.item(ridx, 0).isSelected() ^ (ridx in fit_select_table_selected_rows):
                self.fit_select_table.selectRow(ridx)
        self.fit_select_table.blockSignals(False)

    def update_fit_result_table_data(self, fit_result_table_data):
        self.update_fit_result_table_data_signal.emit(fit_result_table_data)

    def update_fit_result_table_data_signal_emitted(self, fit_result_table_data):
        self.fit_result_table.blockSignals(True)
        self.fit_result_table.clear_table_contents()
        self.fit_result_table.set_column_names(fit_result_table_data.keys())
        for column_name, val in fit_result_table_data.items():
            self.fit_result_table.set_column(column_name, val)
        self.fit_result_table.blockSignals(False)

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
        for name in [
            'clear',
            'update_window_title',
            'update_x_axis_parameter_comboBox',
            'update_parameter_table_data',
            'update_observation_list_data',
            'update_parameter_table_selected_indices',
            'update_observation_list_selected_indices',
            'update_fit_select_table_data',
            'update_fit_select_table_selected_rows',
            'update_fit_result_table_data',
            'update_info_text'
        ]:
            getattr(getattr(self, "{}_signal".format(name)), 'connect')(getattr(self, "{}_signal_emitted".format(name)))

        self.show_signal.connect(self.show)
        self.close_signal.connect(self.close)
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

        self.ax_fit = self.fig_fit.add_subplot(111)
        self.ax_fit.plot(np.linspace(-2., 2., 100), np.linspace(-2, 2., 100) ** 2)
        self.canvas_fit.draw()

        self.update_plot_button.clicked.connect(self.plot_data_no_qt.update_plot)
        self.update_fit_result_button.clicked.connect(self.plot_data_no_qt.update_plot_fit)
        self.update_plot_button.setAcceptDrops(True)
        # self.parameter_tab.setCurrentIndex(0)

        self.parameter_table.hdf_file_dropped.connect(self.plot_data_no_qt.set_data_from_path)
        self.parameter_table.itemSelectionChanged.connect(self.update_parameter_table_selected_indices_from_gui)
        self.fit_select_table.itemSelectionChanged.connect(self.update_fit_select_table_selected_rows_from_gui)
        self.observation_widget.itemSelectionChanged.connect(self.update_observation_list_selected_indices_from_gui)
        self.open_code_button.clicked.connect(self.open_measurement_code)
        self.open_explorer_button.clicked.connect(self.open_explorer)

        self.x_axis_parameter_comboBox.currentIndexChanged.connect(self.update_x_axis_parameteter_from_comboBox)

    def show_gui(self):
        self.show_signal.emit()

    def close_gui(self):
        self.close_signal.emit()

    def open_measurement_code(self):
        if hasattr(self.plot_data_no_qt.data, 'hdf_filepath'):
            subprocess.Popen(r"start {}/meas_code.py".format(os.path.dirname(self.plot_data_no_qt.data.hdf_filepath)), shell=True)
        else:
            print('No filepath.')

    def open_explorer(self):
        if hasattr(self.plot_data_no_qt.data, 'hdf_filepath'):
            subprocess.Popen("explorer {}".format(os.path.abspath(os.path.dirname(self.plot_data_no_qt.data.hdf_filepath))), shell=True)
        else:
            print('No filepath.')


def cpd():
    for fn in os.listdir(os.getcwd()):
        if fn.endswith('.hdf'):
            out = PlotData(path=os.path.join(os.getcwd(), fn))
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


if __name__ == '__main__':
    from qutip_enhanced import *

    self = dh.cpd()
