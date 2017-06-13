# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

from qutip_enhanced import *
import itertools

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as img

import pandas as pd
import os


def purity(dm):
    """should work for qudits also"""
    s = range(len(dm.dims[0]))
    comb = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))[1:]
    return dict((i, ((dm.ptrace(i)) ** 2).tr()) for i in comb)

def test_states(gate, pure=False, **kwargs):
    dims = gate.dims[0]
    out = {}
    test_states_single = dict(
        x=basis(2, 0) + basis(2, 1),
        xm=basis(2, 0) - basis(2, 1),
        y=basis(2, 0) + 1j * basis(2, 1),
        ym=basis(2, 0) - 1j * basis(2, 1),
        z=basis(2, 0),
        zm=basis(2, 1),
    )

    if 'names_multi_list' in kwargs:
        names_multi_list = kwargs.get('names_multi_list')
    else:
        names_single = kwargs.get('names_single', ['x', 'xm', 'y', 'ym', 'z', 'zm'])
        names_multi_list = itertools.product(*[names_single] * len(dims))
    for names_multi in names_multi_list:
        name_multi = "".join(names_multi)
        out[name_multi] = tensor(*[test_states_single[name] for name in names_multi]).unit()
        if not pure:
            out[name_multi] = ket2dm(out[name_multi]).unit()
    return out

def propagate_test_states(gates, test_states):
    out = {}
    if type(test_states) is dict:
        for key, val in test_states.items():
            out[key] = gates * val * gates.dag()
    elif type(gates) is list:
        for key, val in enumerate(gates):
            out[key] = val * test_states * val.dag()
    return out

def single_quantum_transition(states):
    """
    list of single quantum transitions. higher quantum number is always first item
    """
    out = []
    for s in itertools.product(itertools.product(*states), itertools.product(*states)):
        delta = np.array(s[0]) - np.array(s[1])
        if sum(np.abs(delta) == 1) == 1 and sum(delta) == 1:
            out.append([s[0], s[1]])
    return out

def state_num2name(state, name_list):
    return tuple([name_list[i][val] for i, val in enumerate(state)])

def state_name2num(state, name_list):
    return tuple([name_list[i].index(val) for i, val in enumerate(state)])

def state_num_name(state, name_list):
    if type(state) is list:
        st = type(state[0][0])
    else:
        st = type(state[0])
    if st == str:
        f = state_name2num
    else:
        f = state_num2name
    if type(state) is list:
        return [f(s, name_list) for s in state]
    else:
        return f(state, name_list)

def get_transition_frequency(h, **kwargs):
    dims = h.dims[0]
    m_list = list(itertools.product(*[range(i) for i in dims]))
    h_diag = h.diag()

    def t(s0, s1):
        return h_diag[m_list.index(s1)] - h_diag[m_list.index(s0)]

    if 's0' in kwargs and 's1' in kwargs:
        return t(kwargs['s0'], kwargs['s1'])
    else:
        if 'name_list' in kwargs:
            states_list_string = [(state_num_name(s[0], kwargs['name_list']), state_num_name(s[1], kwargs['name_list'])) for s in
                                  kwargs['states_list']]
        else:
            states_list_string = kwargs['states_list']
        return dict([(str(sstr), t(s[0], s[1])) for sstr, s in zip(states_list_string, kwargs.get('states_list'))])

def plot_state(dml, var=None, qubit_names=None, vertical_lines_at=None):
    dims = dml[0][0].dims[0]
    n = len(dims)
    qubit_names = ['Qubit {}'.format(i) for i in range(n)] if qubit_names is None else qubit_names

    x_val = range(len(dml[0])) if (var is None or type(var[0]) is str) else var
    x_ticks = var if (var is not None and type(var[0]) is str) else x_val

    opl = [sigmax, sigmay, sigmaz]
    y_labels = ['x', 'y', 'z', 'purity']

    lim = [[-1.05, 1.05]] * (len(y_labels) - 1) + [[-0.05, 1.05]]
    fig, arr = plt.subplots(4, n, sharex='col', sharey='row')

    for i, ylabel in enumerate(y_labels):
        for j, x_label in enumerate(qubit_names):
            for k in range(len(dml)):
                if i < len(y_labels) - 1:
                    y_val = [expect(opl[i](), dm.ptrace(j)) for dm in dml[k]]
                else:
                    y_val = [(dm.ptrace(j) ** 2).tr() for dm in dml[k]]
                arr[i][j].plot(x_val, y_val, linewidth=2)
                if type(var) is list and type(var[0]) is str:
                    arr[i][j].set_xticks(x_val)
                    arr[i][j].set_xticklabels(x_ticks, fontdict=None, minor=False, size='small', rotation='vertical')
                if vertical_lines_at is not None:
                    for vla in vertical_lines_at:
                        arr[i][j].axvline(x=vla, linewidth=1, color='r')
                arr[i][j].grid(True)
                arr[i][j].set_ylim(lim[i])
                if i != len(y_labels) - 1:
                    arr[i][j].tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom='on',  # ticks along the bottom edge are off
                        top='off',  # ticks along the top edge are off
                        labelbottom='off')  # labels along the bottom edge are off
                if i == len(y_labels) - 1:
                    arr[i][j].axhline(y=0.5, linewidth=.5, color='r', linestyle='dashed')
                if j == 0:
                    arr[i][j].set_ylabel('{}'.format(ylabel))
                if i == len(y_labels) - 1:
                    arr[i][j].set_xlabel('{}'.format(qubit_names[j]))
    fig.set_size_inches(24, 12)
    return fig, arr

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


class PythonHighlighter (QSyntaxHighlighter):
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

class PlotData(object):
    def __init__(self):
        self.window = QWidget()
        self.layout = QGridLayout(self.window)
        self.init_gui()
        self.clear()

        # self.data = data

    def init_gui(self):

        # Figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.window)
        # self.toolbar.hide()

        # Code
        self.code = QTextEdit()

        # Buttons
        self.button = QPushButton('Plot')
        self.clearbutton = QPushButton('Clear')
        self.multiplot = QCheckBox('Add Plot')
        self.agg_data = QCheckBox('Plot aggregated data')

        # set points of values for dataframe
        self.setpoint1_l = QLabel('Setpoint 1')
        self.setpoint2_l = QLabel('Setpoint 2')
        self.setpoint1 = QComboBox()
        self.setpoint2 = QComboBox()
        self.setpoint1_v = QComboBox()
        self.setpoint2_v = QComboBox()
        self.test = QListWidget()

        # data to be plotted
        self.x_l = QLabel('X axis')
        self.y_l = QLabel('Y axis')
        self.ddx = QComboBox()
        self.ddy = QComboBox()

        # Button actions
        self.button.clicked.connect(lambda: self.plot())  # update plot when choosing something from dropdown menu
        self.clearbutton.clicked.connect(self.clear)  # clear figure
        self.ddy.activated.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot
        self.setpoint1.activated.connect(lambda: self.setpoint_changed())  # if dropdown menu chances, update plot
        self.setpoint2.activated.connect(lambda: self.setpoint_changed())  # if dropdown menu chances, update plot
        self.setpoint1_v.activated.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot
        self.setpoint2_v.activated.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot

        self.assign_layout()

    def assign_layout(self):

        self.layout.addWidget(self.canvas, 1, 1, 10, 10)

        self.layout.addWidget(self.toolbar, 12, 1, 1, 1)

        self.layout.addWidget(self.button, 9, 14, 1, 1)

        self.layout.addWidget(self.setpoint1_l, 1, 11, 1, 1)
        self.layout.addWidget(self.setpoint1, 1, 12, 1, 1)
        self.layout.addWidget(self.setpoint1_v, 1, 13, 1, 1)

        self.layout.addWidget(self.setpoint2_l, 2, 11, 1, 1)
        self.layout.addWidget(self.setpoint2, 2, 12, 1, 1)
        self.layout.addWidget(self.setpoint2_v, 2, 13, 1, 1)

        self.layout.addWidget(self.x_l, 3, 11, 1, 1)
        self.layout.addWidget(self.ddx, 3, 12, 1, 1)
        self.layout.addWidget(self.y_l, 3, 13, 1, 1)
        self.layout.addWidget(self.ddy, 3, 14, 1, 1)

        self.layout.addWidget(self.multiplot, 9, 11, 1, 1)
        self.layout.addWidget(self.multiplot, 9, 12, 1, 1)
        self.layout.addWidget(self.clearbutton, 10, 11, 1, 4)

        # self.layout.addWidget(self.code, 5,11, 1, 1)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self.ddxy()

    @property
    def columns(self):
        return self._data.columns

    @property
    def sp1(self):
        return self.setpoint1.currentText()

    @sp1.setter
    def sp1(self, val):
        self.setpoint1.setCurrentText(val)
        # self.setpoint_changed()

    @property
    def sp2(self):
        return self.setpoint2.currentText()

    @sp2.setter
    def sp2(self, val):
        self.setpoint2.setCurrentText(val)
        # self.setpoint_changed()

    @property
    def ax_x(self):
        return self.ddx.currentText()

    @ax_x.setter
    def ax_x(self, val):
        self.ddx.setCurrentText(val)

    @property
    def ax_y(self):
        return self.ddy.currentText()

    @ax_y.setter
    def ax_y(self, val):
        self.ddy.setCurrentText(val)

    def ddxy(self):
        self.setpoint1.addItems(self.columns)
        self.setpoint1.setDuplicatesEnabled(False)
        self.setpoint2.addItems(self.columns)
        self.setpoint2.setDuplicatesEnabled(False)
        self.ddx.addItems(self.columns)
        self.ddx.setDuplicatesEnabled(False)
        self.ddy.addItems(self.columns)
        self.ddy.setDuplicatesEnabled(False)

    def setpoint_changed(self):
        self.setpoint1_v.clear()
        self.setpoint2_v.clear()
        for item in self.data[self.setpoint1.currentText()].drop_duplicates().values:
            self.setpoint1_v.addItem('{}'.format(item), item)
            self.setpoint1_v.setDuplicatesEnabled(False)
        for item in self.data[self.setpoint2.currentText()].drop_duplicates().values:
            self.setpoint2_v.addItem('{}'.format(item), item)
            self.setpoint2_v.setDuplicatesEnabled(False)

    def ret_line_plot_data(self):
        if self.agg_data.checkState == 0:
            pass
        if self.agg_data.checkState == 1:
            pass
        temp = self._data.loc[(self._data[self.setpoint1.currentText()] == self.setpoint1_v.itemData(self.setpoint1_v.currentIndex())) & (self._data[self.setpoint2.currentText()] == self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()))]
        x = temp[self.ddx.currentText()].values
        y = temp[self.ddy.currentText()].values
        return x, y

    def plot(self, add=False):
        try:
            x, y = self.ret_line_plot_data()
            if not add or self.clean_fig:
                self.fig.clear()
                self.clean_fig = False
            if self.multiplot.checkState() == 0:
                self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.plot(x, y, 'o-', label='{}\n{}\n{}'.format(self.setpoint1_v.itemData(self.setpoint1_v.currentIndex()),
                                                          self.setpoint2_v.itemData(self.setpoint2_v.currentIndex()),
                                                          self.ddy.currentText()))
            ax.set_xlabel('{}'.format(self.ddx.currentText()))
            ax.set_ylabel('{}'.format(self.ddy.currentText()))
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            self.canvas.draw()
        except:
            raise Exception


    def clear(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.fig.patch.set_visible(False)
        ax.axis('off')
        pic = img.imread("{}/images/panda.png".format(os.path.dirname(__file__)))
        ax.imshow(pic)
        self.canvas.draw()
        self.clean_fig = True

if __name__ == '__main__':

    # app = QApplication(sys.argv)

    main = PlotData()
    main.data = pd.read_hdf(r"Y:\NuclearOPs\CNOT_KDD\cnot_kdd\20170505-h18m05s44_cnot_kdd\data.hdf")
    # main.data = pd.read_hdf(r"Y:\NuclearOPs\QFT\qutrit_qft\20170428-h08m29s31_qutrit_qft\data.hdf")
    main.data = main.data.dropna()
    # import syntax
    # syntax.PythonHighlighter(main.code.document())
    # main.code.setPlainText(open(r"Y:\NuclearOPs\CNOT_KDD\cnot_kdd\20170505-h18m05s44_cnot_kdd\meas_code.py").read())
    main.code.setText(open(r"Y:\NuclearOPs\CNOT_KDD\cnot_kdd\20170505-h18m05s44_cnot_kdd\meas_code.py").read())
    main.code.setFont('Courier')
    # main.code.setStyleSheet("background-color: black")
    main.highlighter = PythonHighlighter(main.code.document())
    main.code.show()
    # main.window.show()





