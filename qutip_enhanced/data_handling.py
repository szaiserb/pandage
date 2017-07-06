# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import os, sys

import numpy as np
import pandas as pd
import itertools
from numbers import Number
import collections
import subprocess
from PyQt5.QtWidgets import QWidget, QGridLayout, QListWidgetItem, QAbstractItemView, QTableWidget, QListWidget, QTabWidget, QPlainTextEdit, QFrame, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.uic import compileUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .qtgui import plot_data_gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as img
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

class Data:
    def __init__(self, parameter_names=None, observation_names=None, dtypes=None):
        if parameter_names is not None:
            self.parameter_names = parameter_names
        if observation_names is not None:
            self.observation_names = observation_names
        if dtypes is not None:
            self.dtypes = dtypes

    parameter_names = ret_property_array_like_typ('parameter_names', str)
    observation_names = ret_property_array_like_typ('observation_names', str)

    @property
    def dtypes(self):
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

    def init(self, init_from_file=None, last_parameter='x_idx'):
        if init_from_file is not None:
            if init_from_file.endswith('.hdf'):
                df = pd.read_hdf(init_from_file)
            elif init_from_file.endswith('.csv'):
                df = pd.read_csv(init_from_file, compression='gzip')
            self._df = df[pd.notnull(df)]
            if False in [hasattr(self, i) for i in ['_parameter_names', '_observation_names', '_dtypes']]:
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

    def load(self, filepath):
        pass

    def save(self, filepath):
        if filepath.endswith('.csv'):
            self._df.to_csv(filepath, index=False, compression='gzip')
        elif filepath.endswith('.hdf'):
            self._df.to_hdf(filepath, 'a', index=False, format='fixed')
            try:
                tempfile = 'temp.hdf'
                command = ["ptrepack", "-o", "--chunkshape=auto", "--propindexes", "--complevel=9", "--complib=blosc", os.path.split(filepath)[1], tempfile]
                _ = subprocess.call(command, cwd=os.path.split(filepath)[0])
                os.remove(filepath)
                os.rename("{}/{}".format(os.path.split(filepath)[0], tempfile), filepath)
            except WindowsError:
                pass

    def append(self, l):
        if type(l) in [collections.OrderedDict, dict]:
            l = [l]
        for kwargs in l:
            if len(kwargs) != len(self.parameter_names):
                raise Exception('Missing parameter for dataframe.')
            if not all([i in kwargs for i in self.parameter_names]):
                raise Exception('Wrong parameter for dataframe.')
        for idx, li in enumerate(l):
            for k, v in self.dtypes.items():
                if v != 'float':
                    l[idx][k] = getattr(__builtin__, v)()
        df_append = pd.DataFrame(columns=self.variables, data=l)
        if len(self._df) == 0:
            self._df = df_append
        else:
            self._df = self._df.append(df_append, ignore_index=True)

    def set_observations(self, l):
        if type(l) in [collections.OrderedDict, dict]:
            l = [l]
        for idx, kwargs in enumerate(l):
            for obs, val in kwargs.items():
                self._df.set_value(len(self._df) - 1 - len(l) + idx + 1, obs, val)

    def dict_access(self, d, df=None):
        df = self.df if df is None else df
        return df[functools.reduce(np.logical_and, [df[key] == val for key, val in d.items()])]

def recompile_plotdata_ui_file():
    fold = "{}/qtgui".format(os.path.dirname(__file__))
    uipath = r"{}/plot_data.ui".format(fold)
    pypath = r"{}/plot_data_gui.py".format(fold)
    with open(pypath, 'w') as f:
        compileUi(uipath, f)
    reload(plot_data_gui)

class PlotData(plot_data_gui.Ui_window):
    def __init__(self, title):
        self.window = QWidget()
        self.setupUi(self.window)
        self.init_gui()
        self.window.setWindowTitle(title)

    x_axis_name = 'x'

    def init_gui(self):

        # # Figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.window)

        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()
        self.plot_layout.addWidget(self.canvas, 1, 1, 20, 20)
        self.toolbar_layout.addWidget(self.toolbar, 21, 1, 1, 20)

        # Figure fit
        self.fig_fit = Figure()
        self.canvas_fit = FigureCanvas(self.fig_fit)
        self.toolbar_fit = NavigationToolbar(self.canvas_fit, self.window)

        self.ax_fit = self.fig_fit.add_subplot(111)
        self.canvas_fit.draw()
        self.plot_fit_layout.addWidget(self.canvas_fit, 1, 1, 20, 20)
        self.toolbar_fit_layout.addWidget(self.toolbar_fit, 21, 1, 1, 20)

        self.ax_fit = self.fig_fit.add_subplot(111)
        self.ax_fit.plot(np.linspace(-2., 2., 100), np.linspace(-2,2.,100)**2)
        self.canvas_fit.draw()

        self.update_plot_button.clicked.connect(self.update_fit_select_table_and_plot)
        self.update_fit_result_button.clicked.connect(self.update_fit_result_table)
        self.parameter_tab.setCurrentIndex(0)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if not hasattr(self, '_data'):
            self._data = val
            self.parameter_table.setColumnCount(len(self.parameter_names_reduced()))
            self.parameter_table.setHorizontalHeaderLabels(self.parameter_names_reduced())
            self.set_observations()
        else:
            self._data = val
        self.set_parameters()
        self.plot_all_if_none()
        self.update_fit_select_table_and_plot()

    def plot_all_if_none(self):
        if len(self.parameter_table.selectedItems()) == 0:
            column_names = self.parameter_names_reduced()[1:]
            column_names.remove(self.x_axis_name)
            for cn in column_names:
                self.parameter_table.selectColumn(self.parameter_table.column_index(cn))
        if len(self.observation_widget.selectedItems()) == 0:
            self.observation_widget.item(0).setSelected(True)

    def set_parameters(self):
        for column_name in  self.parameter_names_reduced():
            self.parameter_table.append_to_column_parameters(column_name, getattr(self.data.df, column_name).unique())
            if column_name == self.x_axis_name:
                self.parameter_table.set_column_flags(column_name, Qt.NoItemFlags)

    def set_observations(self):
        for obs in self.observation_names_reduced():
            self.observation_widget.addItem(QListWidgetItem(obs))

    def parameter_names_reduced(self):
        return [i for i in self.data.parameter_names if not '_idx' in i]

    def observation_names_reduced(self):
        return [i for i in self.data.observation_names if not i in ['trace', 'start_time', 'end_time', 'thresholds']]

    def ret_line_plot_data_single(self, condition_dict, observation_name):
        out = self.data.df[functools.reduce(np.logical_and, [self.data.df[key] == val for key, val in condition_dict.items() if val not in ['__all__', '__average__']])]
        # TODO: wont work for dates as can not be averaged
        return out.groupby([key for key, val in condition_dict.items() if val != '__average__']).agg({observation_name: np.mean}).reset_index()

    def selected_plot_items(self):
        out = [[i for i in self.parameter_table.selected_table_items(column_name)] for column_name in self.parameter_names_reduced()]
        if all([len(i) == 0 for i in out]):
            return []
        for idx, item, name in zip(range(len(out)), out, self.parameter_names_reduced()):
            if name == self.x_axis_name:
                out[idx] = ['__all__']
            elif len(item) == 0:
                out[idx] = ['__average__']
        return itertools.product(*out)

    def ret_line_plot_data(self):
        plot_data = []
        for p in self.selected_plot_items():
            condition_dict = collections.OrderedDict([(ni, pi) for ni, pi in zip(self.parameter_names_reduced(), p)])
            for observation_name in self.observation_widget.selectedItems():
                dfxy = self.ret_line_plot_data_single(condition_dict, observation_name.text())
                condition_dict_reduced = collections.OrderedDict([(key, val) for key, val in condition_dict.items() if val not in ['__average__', '__all__']])
                plot_data.append(dict(condition_dict_reduced=condition_dict_reduced, observation_name=observation_name.text(), x=getattr(dfxy, self.x_axis_name), y=getattr(dfxy, observation_name.text())))
        return plot_data

    def update_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        for idx, pdi in enumerate(self.ret_line_plot_data()):
            self.ax.plot(pdi['x'], pdi['y'], '-',
                         label='NONE',  # '{}\n{}\n{}'.format(self.setpoint1_v.itemData(self.setpoint1_v.currentIndex()), self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()), self.ddy.currentText() )
                         )
        self.canvas.draw()

    def update_plot_fit(self):
        self.fig_fit.clear()
        self.ax_fit = self.fig_fit.add_subplot(111)
        for idx, fi in enumerate(getattr(self, 'fit_results', [])):
            color = self.ax_fit._get_lines.get_next_color()
            r = fi[1]
            x = r.userkws['x']
            y = r.eval(params=r.params, x=x)
            self.ax_fit.plot(x,y, '-', color=color)
            self.ax_fit.plot(x, r.data, 'o', color=color, markersize=3.5)
            self.canvas_fit.draw()

    def update_fit_select_table_and_plot(self):
        self.fit_select_table.clear_table_contents()
        cpd = collections.OrderedDict()
        spt = list(self.selected_plot_items())
        for column_idx, column_name in enumerate(self.parameter_names_reduced()):
            cpd[column_name] = []
            for pi in spt:
                if not pi[column_idx] in ['__average__', '__all__']:
                    cpd[column_name].append(pi[column_idx])
            if len(cpd[column_name]) == 0:
                del cpd[column_name]
        self.fit_select_table.set_columns(len(cpd.keys()), cpd.keys())
        for column_name, parameters in cpd.items():
            self.fit_select_table.append_to_column_parameters(column_name, parameters)
        self.update_plot()

    def update_fit_result_table(self):
        self.fit_result_table.clear_table_contents()
        spi = list(self.ret_line_plot_data())
        mod = lmfit_models.CosineModel()
        self.fit_results = []
        for i in [spi[idx] for idx in self.fit_select_table.selected_items_unique_column_indices()]:
            try:
                params = mod.guess(data=i['y'], x=i['x'])
                self.fit_results.append([i, mod.fit(i['y'], params, x=i['x'])])
            except:
                self.fit_results.append([i, None])
                print('fitting failed: {}'.format(i))
        if len(self.fit_results) == 0:
            print("No curves could be fitted successfully.. exiting.")
            return
        header = self.fit_results[0][0]['condition_dict_reduced'].keys() + ['observation_name'] + self.fit_results[0][1].params.keys()
        self.fit_result_table.setColumnCount(len(header))
        self.fit_result_table.setHorizontalHeaderLabels(header)
        for ridx, fri in enumerate(self.fit_results):
            if fri[1] is not None:
                self.fit_result_table.add_rows(1)

                cidx = 0
                for key, val in fri[0]['condition_dict_reduced'].items() + [('observation_name', fri[0]['observation_name'])] + [(key, val.value) for key, val in fri[1].params.items()]:
                    new_item = QTableWidgetItem(str(val))
                    new_item.setData(0x0100, val)
                    new_item.setFlags(Qt.ItemIsSelectable)
                    self.fit_result_table.setItem(ridx, cidx, new_item)
                    cidx += 1
        self.update_plot_fit()

    def clear(self):
        print('Clearing..')
        self.fit_result_table.clearSelection()
        self.fit_result_table.clear()
        self.fit_result_table.setColumnCount(0)
        self.fit_result_table.setRowCount(0)
        self.fit_select_table.clearSelection()
        self.fit_select_table.clear()
        self.fit_select_table.setColumnCount(0)
        self.fit_select_table.setRowCount(0)
        self.parameter_table.clearSelection()
        self.parameter_table.clear()
        self.parameter_table.setColumnCount(0)
        self.parameter_table.setRowCount(0)
        if hasattr(self, '_data'):
            delattr(self, '_data')

