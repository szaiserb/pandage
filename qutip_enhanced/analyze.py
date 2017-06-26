# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

from qutip_enhanced import *
import itertools

import numpy as np

import matplotlib.pyplot as plt

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

from qutip_enhanced.data_handling import PlotData