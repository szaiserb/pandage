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
from PyQt5.QtWidgets import QWidget, QGridLayout, QListWidgetItem, QAbstractItemView, QListWidget, QLabel, QPlainTextEdit, QFrame
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as img
import functools
import __builtin__


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

class PlotData:
    def __init__(self, title):
        self.window = QWidget()
        self.layout = QGridLayout(self.window)
        self.init_gui()
        self.window.setWindowTitle(title)
        self.clear()

    x_axis_name = 'x'

    def init_gui(self):
        # Figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.window)
        # infos
        self.info = QPlainTextEdit()
        self.info.setFrameStyle(QFrame.NoFrame | QFrame.Sunken)
        self.info.setBackgroundVisible(True)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if not hasattr(self, '_data'):
            print('No data available. Initializing..')
            self.assign_layout(val)
            self.set_parameters(val)
        else:
            self.set_parameters(val)
        self._data = val
        self.update_plot()

    def set_parameters(self, data):
        for name in self.parameter_names_reduced(data):
            for item in self.new_parameters_l(name, data):
                new_item = QListWidgetItem(str(item))
                new_item.setData(0x0100, item)
                getattr(self, name).addItem(new_item)

    def parameter_names_reduced(self, data=None):
        data = self.data if data is None else data
        return [i for i in data.parameter_names if not '_idx' in i]

    def observation_names_reduced(self, data=None):
        data = self.data if data is None else data
        return [i for i in data.observation_names if not i in ['trace', 'start_time', 'end_time', 'thresholds']]

    def widget_data_l(self, name):
        return [getattr(self, name).item(i).data(0x0100) for i in range(getattr(self, name).count())]

    def new_parameters_l(self, name, data):
        wd = self.widget_data_l(name)
        return [i for i in getattr(data.df, name).unique() if i not in wd]

    def ret_line_plot_data_single(self, condition_dict, observation_name):
        out = self.data.df[functools.reduce(np.logical_and, [self.data.df[key] == val for key, val in condition_dict.items() if val not in ['__all__', '__average__']])]
        # TODO: wont work for dates as can not be averaged
        return out.groupby([key for key, val in condition_dict.items() if val != '__average__']).agg({observation_name: np.mean}).reset_index()

    def ret_line_plot_data(self):
        selected_items_list = [[i.data(0x0100) for i in getattr(self, name).selectedItems()] for name in self.parameter_names_reduced()]
        if all([len(i) == 0 for i in selected_items_list]):
            return []
        for idx, item, name in zip(range(len(selected_items_list)), selected_items_list, self.parameter_names_reduced()):
            if name == self.x_axis_name:
                selected_items_list[idx] = ['__all__']
            elif len(item) == 0:
                selected_items_list[idx] = ['__average__']
        plot_data = []
        for p in itertools.product(*selected_items_list):
            condition_dict = collections.OrderedDict([(ni, pi) for ni, pi in zip(self.parameter_names_reduced(), p)])
            for observation_name in self.observations.selectedItems():
                dfxy = self.ret_line_plot_data_single(condition_dict, observation_name.text())
                plot_data.append(dict(condition_dict=condition_dict, observation_name=observation_name.text(), x=getattr(dfxy, self.x_axis_name), y=getattr(dfxy, observation_name.text())))
        return plot_data

    def assign_layout(self, data):
        # Lists
        for name in self.parameter_names_reduced(data) + ['observations']:
            setattr(self, name, QListWidget())
            getattr(self, name).setSelectionMode(QAbstractItemView.ExtendedSelection)
            getattr(self, name).itemSelectionChanged.connect(self.update_plot)

        for obs in data.observation_names:
            self.observations.addItem(QListWidgetItem(obs))

        self.layout.addWidget(self.canvas, 4, 1, 10, 10)
        self.layout.addWidget(self.toolbar, 14, 1, 1, 1)
        for ix, name in enumerate(self.parameter_names_reduced(data)):
            self.layout.addWidget(QLabel(name), 1, 0 + ix * 1, 1, 1)
            self.layout.addWidget(getattr(self, name), 2, 0 + ix * 1, 2, 1)
        self.layout.addWidget(self.observations, 2, 0 + len(self.parameter_names_reduced(data)) * 1, 2, 1)
        getattr(self, self.x_axis_name).setEnabled(False)
        self.layout.addWidget(self.info, 5, 11, 4, 4)

    def update_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        for pdi in self.ret_line_plot_data():
            self.ax.plot(pdi['x'], pdi['y'], 'o-',
                         label='NONE',  # '{}\n{}\n{}'.format(self.setpoint1_v.itemData(self.setpoint1_v.currentIndex()), self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()), self.ddy.currentText() )
                         )
        self.canvas.draw()

    def clear(self):
        self.fig.clear()

        self.fig.patch.set_visible(False)
        self.ax.axis('off')
        pic = img.imread("{}/images/panda.png".format(os.path.dirname(__file__)))
        self.ax.imshow(pic)
        self.canvas.draw()
        self.clean_fig = True

# class PlotDataOld(object):
#     def __init__(self, title):
#         self.window = QWidget()
#         self.layout = QGridLayout(self.window)
#         self.init_gui()
#         self.window.setWindowTitle(title)
#         self.clear()
#
#     def init_gui(self):
#
#         # Figure
#         self.fig = Figure()
#         self.canvas = FigureCanvas(self.fig)
#         self.toolbar = NavigationToolbar(self.canvas, self.window)
#         # self.toolbar.hide()
#
#         # Code
#         self.code = QTextEdit()
#
#         # Buttons
#         self.button = QPushButton('Plot')
#         self.fitbutton = QCheckBox('Fit')
#         self.clearbutton = QPushButton('Clear')
#         self.multiplot = QCheckBox('Add Plot')
#         self.agg_data = QCheckBox('Plot aggregated data')
#
#         self.loop_over_sp1 = QCheckBox('All setpoint 1')
#         self.loop_over_sp2 = QCheckBox('All setpoint 2')
#         self.loop_over_results = QCheckBox('results')
#
#         # set points of values for dataframe
#         self.setpoint1_l = QLabel('Setpoint 1')
#         self.setpoint2_l = QLabel('Setpoint 2')
#         self.setpoint1 = QComboBox()
#         self.setpoint2 = QComboBox()
#         self.setpoint1_v = QComboBox()
#         self.setpoint2_v = QComboBox()
#         self.test = QListWidget()
#
#         # data to be plotted
#         self.x_l = QLabel('X axis')
#         self.y_l = QLabel('Y axis')
#         self.ddx = QComboBox()
#         self.ddy = QComboBox()
#
#         # infos
#         self.info = QPlainTextEdit()
#         self.info.setFrameStyle(QFrame.NoFrame | QFrame.Sunken )
#         self.info.setBackgroundVisible(True)
#
#         # Button actions
#         self.button.clicked.connect(lambda: self.plot())  # update plot when choosing something from dropdown menu
#         self.clearbutton.clicked.connect(self.clear)  # clear figure
#         self.ddy.activated.connect(lambda: self.plot())  # if dropdown menu chances, update plot
#         self.setpoint1.activated.connect(lambda: self.setpoint_changed())  # if dropdown menu chances, update plot
#         self.setpoint2.activated.connect(lambda: self.setpoint_changed())  # if dropdown menu chances, update plot
#         self.setpoint1_v.activated.connect(lambda: self.plot())  # if dropdown menu chances, update plot
#         self.setpoint2_v.activated.connect(lambda: self.plot())  # if dropdown menu chances, update plot
#
#         self.assign_layout()
#
#     def assign_layout(self):
#
#         self.layout.addWidget(self.canvas, 1, 1, 10, 10)
#
#         self.layout.addWidget(self.toolbar, 12, 1, 1, 1)
#
#         self.layout.addWidget(self.button, 9, 14, 1, 1)
#         self.layout.addWidget(self.fitbutton, 9, 13, 1, 1)
#
#         self.layout.addWidget(self.loop_over_sp1, 8, 11, 1, 1)
#         self.layout.addWidget(self.loop_over_sp2, 8, 12, 1, 1)
#         self.layout.addWidget(self.loop_over_results, 8, 13, 1, 1)
#
#         self.layout.addWidget(self.setpoint1_l, 1, 11, 1, 1)
#         self.layout.addWidget(self.setpoint1, 1, 12, 1, 1)
#         self.layout.addWidget(self.setpoint1_v, 1, 13, 1, 1)
#
#         self.layout.addWidget(self.setpoint2_l, 2, 11, 1, 1)
#         self.layout.addWidget(self.setpoint2, 2, 12, 1, 1)
#         self.layout.addWidget(self.setpoint2_v, 2, 13, 1, 1)
#
#         self.layout.addWidget(self.x_l, 3, 11, 1, 1)
#         self.layout.addWidget(self.ddx, 3, 12, 1, 1)
#         self.layout.addWidget(self.y_l, 3, 13, 1, 1)
#         self.layout.addWidget(self.ddy, 3, 14, 1, 1)
#
#         self.layout.addWidget(self.multiplot, 9, 11, 1, 1)
#         self.layout.addWidget(self.agg_data, 9, 12, 1, 1)
#         self.layout.addWidget(self.clearbutton, 10, 11, 1, 4)
#
#         self.layout.addWidget(self.info, 4, 11, 4, 4)
#
#         # self.layout.addWidget(self.code, 5,11, 1, 1)
#
#     @property
#     def data(self):
#         return self._data
#
#     @data.setter
#     def data(self, val):
#         self._data = val
#         self.ddxy()
#
#     @property
#     def parameters(self):
#         if not hasattr(self, '_parameters'):
#             return ['seq_num', 'x', 'seq_num_idx', 'x_idx']
#         return self._parameters
#
#     @parameters.setter
#     def parameters(self, val):
#         self._parameters = val
#
#     @property
#     def fit_data(self):
#         if not hasattr(self, '_fit_data'):
#             return (np.array([0.]), np.array([0.]))
#         else:
#             return self._fit_data
#
#     @fit_data.setter
#     def fit_data(self, val):
#         self._fit_data = val
#
#     @property
#     def columns(self):
#         return self._data.columns
#
#     @property
#     def sp1(self):
#         return self.setpoint1.currentText()
#
#     @sp1.setter
#     def sp1(self, val):
#         self.setpoint1.setCurrentText(val)
#         # self.setpoint_changed()
#
#     @property
#     def sp2(self):
#         return self.setpoint2.currentText()
#
#     @sp2.setter
#     def sp2(self, val):
#         self.setpoint2.setCurrentText(val)
#         # self.setpoint_changed()
#
#     @property
#     def ax_x(self):
#         return self.ddx.currentText()
#
#     @ax_x.setter
#     def ax_x(self, val):
#         self.ddx.setCurrentText(val)
#
#     @property
#     def ax_y(self):
#         return self.ddy.currentText()
#
#     @ax_y.setter
#     def ax_y(self, val):
#         self.ddy.setCurrentText(val)
#     @property
#     def agg_column(self):
#         if not hasattr(self, '_agg_column'):
#             return ['seq_num', 'x']
#         else:
#             return self._agg_column
#
#     @agg_column.setter
#     def agg_column(self, val):
#         self._agg_column = val
#
#     @property
#     def agg_dict(self):
#         if not hasattr(self, '_agg_dict'):
#             return dict(result_0=np.nanmean,
#                         # result_1=np.nanmean,
#                         # result_2=np.nanmean,
#                         events=np.nanmean
#                         )
#         else:
#             return self._agg_dict
#
#     @agg_dict.setter
#     def agg_dict(self, val):
#         self._agg_dict = val
#
#     @property
#     def num_res(self):
#         if not hasattr(self, '_num_res'):
#             return ['result_0']
#         else:
#             return self._num_res
#
#     @num_res.setter
#     def num_res(self, val):
#         self._num_res = val
#
#     def ddxy(self):
#         self.setpoint1.addItems(self.columns)
#         self.setpoint1.setDuplicatesEnabled(False)
#         self.setpoint2.addItems(self.columns)
#         self.setpoint2.setDuplicatesEnabled(False)
#         self.ddx.addItems(self.columns)
#         self.ddx.setDuplicatesEnabled(False)
#         self.ddy.addItems(self.columns)
#         self.ddy.setDuplicatesEnabled(False)
#
#     def setpoint_changed(self):
#         self.setpoint1_v.clear()
#         self.setpoint2_v.clear()
#         for item in self.data[self.setpoint1.currentText()].drop_duplicates().values:
#             self.setpoint1_v.addItem('{}'.format(item), item)
#             self.setpoint1_v.setDuplicatesEnabled(False)
#         for item in self.data[self.setpoint2.currentText()].drop_duplicates().values:
#             self.setpoint2_v.addItem('{}'.format(item), item)
#             self.setpoint2_v.setDuplicatesEnabled(False)
#
#     def ret_line_plot_data(self):
#         try:
#             if self.agg_data.checkState() == 0:
#                 temp = self._data.loc[(self._data[self.setpoint1.currentText()] == self.setpoint1_v.itemData(self.setpoint1_v.currentIndex())) & (self._data[self.setpoint2.currentText()] == self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()))]
#                 x = temp[self.ddx.currentText()].values
#                 y = temp[self.ddy.currentText()].values
#             else:
#                 if self.loop_over_sp1.checkState() == 2:
#                     temp = self.data.groupby([self.setpoint1.currentText(), self.ddx.currentText()], as_index=False).agg(self.agg_dict)
#                     values_sp1 = [self.setpoint1_v.itemText(i) for i in range(self.setpoint1_v.count())]
#                     r = []
#                     for val in values_sp1:
#                         r.append(temp[(temp[self.setpoint1.currentText()] == float(val))][self.ddy.currentText()].values)
#                 if self.loop_over_sp2.checkState() == 2:
#                     temp = self.data.groupby([self.setpoint2.currentText(), self.ddx.currentText()], as_index=False).agg(self.agg_dict)
#                     values_sp2 = [self.setpoint2_v.itemText(i) for i in range(self.setpoint2_v.count())]
#                     r = []
#                     for val in values_sp2:
#                         r.append(temp[(temp[self.setpoint2.currentText()] == float(val))][self.ddy.currentText()].values)
#                 if self.loop_over_results.checkState() == 2:
#                     # temp = self._data.loc[(self._data[self.setpoint1.currentText()] == self.setpoint1_v.itemData(self.setpoint1_v.currentIndex())) & (self._data[self.setpoint2.currentText()] == self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()))]
#                     temp = self.data.groupby(self.agg_column, as_index=False).agg(self.agg_dict)[(self._data[self.setpoint2.currentText()] == self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()))]
#                     x = temp[self.ddx.currentText()].values
#                     y = temp[['result_{}'.format(i) for i in range(self.num_res)]].values
#                 else:
#                     temp = self.data.groupby(self.agg_column, as_index=False).agg(self.agg_dict)[(self._data[self.setpoint2.currentText()] == self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()))]
#                     x = temp[self.ddx.currentText()].values
#                     y = temp[self.ddy.currentText()].values
#             return x, y
#         except:
#             raise Exception
#
#     def fit(self):
#         try:
#             self.ax.plot(self.fit_data[0], self.fit_data[1], '-')
#         except:
#             raise Exception
#
#
#
#     def plot(self):
#         if self.setpoint1_v.currentText() == u'':
#             self.setpoint_changed()
#             self.setpoint1_v.setCurrentIndex(0)
#         if self.setpoint2_v.currentText() == u'':
#             self.setpoint_changed()
#             self.setpoint2_v.setCurrentIndex(0)
#         try:
#             x, y = self.ret_line_plot_data()
#             if self.clean_fig:
#                 self.fig.clear()
#                 self.clean_fig = False
#             if self.multiplot.checkState() == 0:
#                 self.fig.clear()
#             if not hasattr(self, 'ax') or self.multiplot.checkState() == 0:
#                 self.ax = self.fig.add_subplot(111)
#             self.ax.plot(x, y, 'o-', label='{}\n{}\n{}'.format(self.setpoint1_v.itemData(self.setpoint1_v.currentIndex()),
#                                                                self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()),
#                                                                self.ddy.currentText()
#                                                                )
#                          )
#             if self.fitbutton.checkState() == 2:
#                 self.fit()
#             self.ax.set_xlabel('{}'.format(self.ddx.currentText()))
#             self.ax.set_ylabel('{}'.format(self.ddy.currentText()))
#             # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             # box = ax.get_position()
#             # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#             self.canvas.draw()
#         except:
#             raise Exception
#
#     def toggle_setpoints(self):
#         if self.agg_data.checkState() == 2:
#             if self.loop_over_sp1.checkState() == 0:
#                 self.setpoint1.setEditable(False)
#                 self.setpoint1_v.setEditable(False)
#             if self.loop_over_sp2.checkState() == 0:
#                 self.setpoint2.setEditable(False)
#                 self.setpoint2_v.setEditable(False)
#             if self.loop_over_results.checkState() == 0:
#                 self.ddx.setEditable(False)
#                 self.ddy.setEditable(False)
#         else:
#             self.setpoint1.setEditable(True)
#             self.setpoint1_v.setEditable(True)
#             self.setpoint2.setEditable(True)
#             self.setpoint2_v.setEditable(True)
#             self.ddx.setEditable(True)
#             self.ddy.setEditable(True)
#
#     def clear(self):
#         self.fig.clear()
#         ax = self.fig.add_subplot(111)
#         self.fig.patch.set_visible(False)
#         ax.axis('off')
#         pic = img.imread("{}/images/panda.png".format(os.path.dirname(__file__)))
#         ax.imshow(pic)
#         self.canvas.draw()
#         self.clean_fig = True