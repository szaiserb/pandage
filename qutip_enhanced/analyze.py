# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as img

from qutip_enhanced import *
import matplotlib.pyplot as plt
import itertools

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


class PlotData(object):
    """
    from qutip_enhanced import *
    from PySide.QtGui import *
    import sys
    app = QApplication(sys.argv)
    import pandas as pd
    p = analyze.PlotData()
    p.window.show()
    p.data=pd.read_hdf(r"\\PI3-PC161\data\NuclearOPs\CNOT_KDD\cnot_kdd\20170505-h18m05s44_cnot_kdd\data.hdf")
    """

    def __init__(self):
        self.window = QWidget()
        self.layout = QGridLayout(self.window)

        self.init_gui()

        self.clear()

    def init_gui(self):

        # Figure
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        # Buttons
        self.button = QPushButton('Plot')
        self.clearbutton = QPushButton('Clear')
        self.multiplot = QCheckBox('Add Plot')

        # set points of values for dataframe
        self.sp1_l = QLabel('Setpoint 1')
        self.sp2_l = QLabel('Setpoint 2')
        self.sp1 = QComboBox()
        self.sp2 = QComboBox()
        self.sp1_v = QSpinBox()
        self.sp2_v = QSpinBox()

        # data to be plotted
        self.x_l = QLabel('X axis')
        self.y_l = QLabel('Y axis')
        self.ddx = QComboBox()
        self.ddy = QComboBox()

        # Button actions
        self.button.clicked.connect(lambda: self.plot())  # update plot when choosing something from dropdown menu
        self.clearbutton.clicked.connect(self.clear)  # clear figure
        self.ddy.activated.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot
        self.sp1_v.valueChanged.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot
        self.sp2_v.valueChanged.connect(lambda: self.plot(add=True))  # if dropdown menu chances, update plot

        self.assign_layout()

    def assign_layout(self):

        self.layout.addWidget(self.canvas, 1, 1, 10, 10)
        self.layout.addWidget(self.button, 9, 14, 1, 1)

        self.layout.addWidget(self.sp1_l, 1, 11, 1, 1)
        self.layout.addWidget(self.sp1, 1, 12, 1, 1)
        self.layout.addWidget(self.sp1_v, 1, 13, 1, 1)

        self.layout.addWidget(self.sp2_l, 2, 11, 1, 1)
        self.layout.addWidget(self.sp2, 2, 12, 1, 1)
        self.layout.addWidget(self.sp2_v, 2, 13, 1, 1)

        self.layout.addWidget(self.x_l, 3, 11, 1, 1)
        self.layout.addWidget(self.ddx, 3, 12, 1, 1)
        self.layout.addWidget(self.y_l, 3, 13, 1, 1)
        self.layout.addWidget(self.ddy, 3, 14, 1, 1)

        self.layout.addWidget(self.multiplot, 9, 11, 1, 1)
        self.layout.addWidget(self.clearbutton, 10, 11, 1, 4)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self.ddxy()

    @property
    def columns(self):
        return self._data.columns.values

    def ddxy(self):
        for name in self.columns:
            sp1_items = [self.sp1.itemText(i) for i in range(self.sp1.count())]
            sp2_items = [self.sp2.itemText(i) for i in range(self.sp2.count())]
            ddx_items = [self.ddx.itemText(i) for i in range(self.ddx.count())]
            ddy_items = [self.ddy.itemText(i) for i in range(self.ddy.count())]
            if name not in sp1_items:
                self.sp1.addItem('{}'.format(name), name)
            if name not in sp2_items:
                self.sp2.addItem('{}'.format(name), name)
            if name not in ddx_items:
                self.ddx.addItem('{}'.format(name), name)
            if name not in ddy_items:
                self.ddy.addItem('{}'.format(name), name)
            else:
                pass

    def plot(self, add=False):
        try:
            temp = self._data.loc[(self._data[self.sp1.currentText()] == self.sp1_v.value()) & (self._data[self.sp2.currentText()] == self.sp2_v.value())]
            x = temp[self.ddx.currentText()].values
            y = temp[self.ddy.currentText()].values
            if not add or self.clean_fig:
                self.fig.clear()
                self.clean_fig = False
            if self.multiplot.checkState() == 0:
                self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.plot(x, y)
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