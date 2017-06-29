# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import os

import numpy as np
import pandas as pd
import itertools
from numbers import Number
import collections
import subprocess
from PyQt5.QtWidgets import QWidget, QGridLayout, QListWidgetItem, QAbstractItemView, QTableWidget, QListWidget, QLabel, QPlainTextEdit, QFrame, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as img
import functools
import __builtin__
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


class Highlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(Highlighter, self).__init__(parent)

        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(Qt.darkBlue)
        keywordFormat.setFontWeight(QFont.Bold)

        keywordPatterns = ["\\bchar\\b", "\\bclass\\b", "\\bconst\\b",
                           "\\bdouble\\b", "\\benum\\b", "\\bexplicit\\b", "\\bfriend\\b",
                           "\\binline\\b", "\\bint\\b", "\\blong\\b", "\\bnamespace\\b",
                           "\\boperator\\b", "\\bprivate\\b", "\\bprotected\\b",
                           "\\bpublic\\b", "\\bshort\\b", "\\bsignals\\b", "\\bsigned\\b",
                           "\\bslots\\b", "\\bstatic\\b", "\\bstruct\\b",
                           "\\btemplate\\b", "\\btypedef\\b", "\\btypename\\b",
                           "\\bunion\\b", "\\bunsigned\\b", "\\bvirtual\\b", "\\bvoid\\b",
                           "\\bvolatile\\b"]

        self.highlightingRules = [(QRegExp(pattern), keywordFormat)
                                  for pattern in keywordPatterns]

        classFormat = QTextCharFormat()
        classFormat.setFontWeight(QFont.Bold)
        classFormat.setForeground(Qt.darkMagenta)
        self.highlightingRules.append((QRegExp("\\bQ[A-Za-z]+\\b"),
                                       classFormat))

        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(Qt.red)
        self.highlightingRules.append((QRegExp("//[^\n]*"),
                                       singleLineCommentFormat))

        self.multiLineCommentFormat = QTextCharFormat()
        self.multiLineCommentFormat.setForeground(Qt.red)

        quotationFormat = QTextCharFormat()
        quotationFormat.setForeground(Qt.darkGreen)
        self.highlightingRules.append((QRegExp("\".*\""),
                                       quotationFormat))

        functionFormat = QTextCharFormat()
        functionFormat.setFontItalic(True)
        functionFormat.setForeground(Qt.blue)
        self.highlightingRules.append((QRegExp("\\b[A-Za-z0-9_]+(?=\\()"),
                                       functionFormat))

        self.commentStartExpression = QRegExp("/\\*")
        self.commentEndExpression = QRegExp("\\*/")

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.indexIn(text)

        while startIndex >= 0:
            endIndex = self.commentEndExpression.indexIn(text, startIndex)

            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = len(text) - startIndex
            else:
                commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()

            self.setFormat(startIndex, commentLength,
                           self.multiLineCommentFormat)
            startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength);


def format(color, style=''):
    """Return a QTextCharFormat with the given attributes.
    """
    _color = QColor()
    _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)
    if 'bold' in style:
        _format.setFontWeight(QFont.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)

    return _format


# Syntax styles that can be shared by all languages
STYLES = {
    'import': format('blue', 'italic'),
    'plaintext': format('black'),
    'keyword': format('blue'),
    'operator': format('red', 'bold'),
    'brace': format('darkGray'),
    'defclass': format('black', 'bold'),
    'string': format('magenta'),
    'string2': format('darkMagenta'),
    'comment': format('darkGreen', 'italic'),
    'self': format('green', 'italic'),
    'numbers': format('brown'),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
    """
    # Python keywords
    keywords = [
        'and', 'assert', 'break', 'class', 'continue', 'def',
        'del', 'elif', 'else', 'except', 'exec', 'finally',
        'for', 'from', 'global', 'if', 'import', 'in',
        'is', 'lambda', 'not', 'or', 'pass', 'print',
        'raise', 'return', 'try', 'while', 'yield',
        'None', 'True', 'False',
    ]

    # Python operators
    operators = [
        '=',
        # Comparison
        '==', '!=', '<', '<=', '>', '>=',
        # Arithmetic
        '\+', '-', '\*', '/', '//', '\%', '\*\*',
        # In-place
        '\+=', '-=', '\*=', '/=', '\%=',
        # Bitwise
        '\^', '\|', '\&', '\~', '>>', '<<',
    ]

    # Python braces
    braces = [
        '\{', '\}', '\(', '\)', '\[', '\]',
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegExp("'''"), 1, STYLES['string2'])
        self.tri_double = (QRegExp('"""'), 2, STYLES['string2'])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, STYLES['keyword'])
                  for w in PythonHighlighter.keywords]
        rules += [(r'%s' % o, 0, STYLES['operator'])
                  for o in PythonHighlighter.operators]
        rules += [(r'%s' % b, 0, STYLES['brace'])
                  for b in PythonHighlighter.braces]

        # All other rules
        rules += [
            (r'import', 0, STYLES['import']),
            # (r'.', 0, STYLES['plaintext']),
            # 'self'
            (r'\bself\b', 0, STYLES['self']),

            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES['string']),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES['string']),

            # 'def' followed by an identifier
            (r'\bdef\b\s*(\w+)', 1, STYLES['defclass']),
            # 'class' followed by an identifier
            (r'\bclass\b\s*(\w+)', 1, STYLES['defclass']),

            # From '#' until a newline
            (r'#[^\n]*', 0, STYLES['comment']),

            # Numeric literals
            (r'\b[+-]?[0-9]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b', 0, STYLES['numbers']),
        ]

        # Build a QRegExp for each pattern
        self.rules = [(QRegExp(pat), index, fmt)
                      for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = text.length() - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False


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

class QTableWidgetEnhanced(QTableWidget):

    def column_data(self, column_name):

        out = []
        for row in range(self.rowCount()):
            o = self.item(row, self.column_index(column_name))
            if o.text() != '':
                out.append(o.data(0x0100))
        return out

    @property
    def column_names(self):
        return [self.horizontalHeaderItem(i).text() for i in range(self.columnCount())]

    def column_index(self, column_name):
        return self.column_names.index(column_name)

    def add_rows(self, n_new):
        if n_new > 0:
            n_old = self.rowCount()
            self.setRowCount(n_old+n_new)
            for rc in itertools.product(range(n_old, n_old+n_new), range(self.columnCount())):
                new_item = QTableWidgetItem('')
                new_item.setData(0x0100, None)
                new_item.setFlags(Qt.NoItemFlags)
                self.setItem(rc[0], rc[1], new_item)

    def set_columns(self, desired_total_count, header):
        if desired_total_count != len(header):
            raise Exception("Error: {}{}".format(desired_total_count, header))
        self.setColumnCount(desired_total_count)
        self.setHorizontalHeaderLabels(header)
        for rc in itertools.product(range(self.rowCount()), range(desired_total_count)):
            new_item = QTableWidgetItem('')
            new_item.setData(0x0100, None)
            new_item.setFlags(Qt.NoItemFlags)
            self.setItem(rc[0], rc[1], new_item)

    def append_to_column_parameters(self, column_name, parameters):
        cd = self.column_data(column_name)
        new_params = [i for i in parameters if i not in cd]
        delta_n_row = len(parameters) - self.rowCount()
        if delta_n_row > 0:
            self.add_rows(delta_n_row)
        for row_idx, new_param in zip(len(parameters) - len(new_params) + np.arange(0, len(parameters)), new_params):
            self.item(row_idx, self.column_index(column_name)).setText(str(new_param))
            self.item(row_idx, self.column_index(column_name)).setData(0x0100, new_param)
            self.item(row_idx, self.column_index(column_name)).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def set_column_flags(self, column_name, flag):
        for row in range(self.rowCount()):
            self.item(row, self.column_index(column_name)).setFlags(flag)

    def selected_table_items(self, column_name):
        out = []
        for item in self.selectedItems():
            if item.column() == self.column_index(column_name=column_name):
                out.append(item.data(0x0100))
        return out

    def selected_items_unique_column_indices(self):
        return list(set([i.row() for i in self.selectedItems()]))

    def clear_table_contents(self):
        for idx in itertools.product(range(self.rowCount()), range(self.columnCount())):
            self.clearSelection()
            self.item(idx[0], idx[1]).setText('')
            self.item(idx[0], idx[1]).setData(0x0100, None)
            self.item(idx[0], idx[1]).setFlags(Qt.NoItemFlags)

class PlotData:
    def __init__(self, title):
        self.window = QWidget()
        self.layout = QGridLayout(self.window)
        self.init_gui()
        self.window.setWindowTitle(title)

    x_axis_name = 'x'

    def init_gui(self):
        # Figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.window)

        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()

        # infos
        self.info = QPlainTextEdit()
        self.info.setFrameStyle(QFrame.NoFrame | QFrame.Sunken)
        self.info.setBackgroundVisible(True)

        self.parameter_table = QTableWidgetEnhanced()
        # self.parameter_table.itemSelectionChanged.connect(self.update_fit_select_table_and_plot)

        self.fit_select_table = QTableWidgetEnhanced()
        # self.fit_select_table.itemSelectionChanged.connect(self.update_fit_result_table)

        self.fit_result_table = QTableWidgetEnhanced()

        self.observation_widget = QListWidget()
        self.observation_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.observation_widget.itemSelectionChanged.connect(self.update_fit_select_table_and_plot)

        self.layout.addWidget(self.canvas, 2, 1, 1, 15)
        self.layout.addWidget(self.parameter_table, 1, 1, 1, 15)
        self.layout.addWidget(self.fit_select_table, 3, 1, 1, 15)
        self.layout.addWidget(self.fit_result_table, 4, 1, 1, 15)
        self.layout.addWidget(self.toolbar, 3, 4, 1, 15)
        self.layout.addWidget(self.info, 2, 16, 1, 1)

        self.layout.addWidget(self.observation_widget, 1, 16, 1, 1)


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
        self.update_plot()

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
        psl = []
        for idx, pdi in enumerate(self.ret_line_plot_data()):
            if idx in self.fit_select_table.selected_items_unique_column_indices():
                psl.append('o')
            else:
                psl.append('o-')
            self.ax.plot(pdi['x'], pdi['y'], psl[-1],
                         label='NONE',  # '{}\n{}\n{}'.format(self.setpoint1_v.itemData(self.setpoint1_v.currentIndex()), self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()), self.ddy.currentText() )
                         )
        for idx, fi in enumerate(getattr(self, 'fit_results', [])):
            fi[1].plot_fit(ax=self.ax)
        self.canvas.draw()

    def update_fit_select_table_and_plot(self):
        self.fit_result_table.clear_table_contents()
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


        # self.fit_result_str = "\n".join(["{}: {}".format(key, val.value) for key, val in self.fit_result.params.items()])
        # if self.fit_function in ['Cosinus decay', 'Cosinus dec offset']:  # now always true because of outer if, but later useful
        #     self.contrast = 100 * abs(self.fit_result.params['amplitude'].value * 2)
        # self.fit = self.fit_result.best_fit

    def update_fit_result_table(self):
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
        self.update_plot()

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

