# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type
from qutip import tensor, ket2dm, sigmax, sigmay, sigmaz, expect, jmat
from .data_generation import DataGeneration
from .data_handling import PlotData
from .sequence_creator import unitary_propagator_list_mult
from .qutip_enhanced import dim2spin
from collections import OrderedDict
from itertools import chain, combinations, product, izip, count

import traceback, sys

def purity(dm):
    """should work for qudits also"""
    s = range(len(dm.dims[0]))
    comb = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]
    return dict((i, ((dm.ptrace(i)) ** 2).tr()) for i in comb)

def test_states_single(dim, pure=False):
    m = (dim - 1.)/2.
    out = dict()
    for axis in ['x', 'y', 'z']:
        for idx, state in enumerate(jmat(m, axis).eigenstates()[1][::-1]):
            if not pure:
                state = ket2dm(state)
            out["{}{}".format(axis, idx)] = state
    return out

__TEST_STATES_SINGLE_PURE__ = dict([(dim, test_states_single(dim=dim, pure=True)) for dim in [2,3]])
__TEST_STATES_SINGLE__ = dict([(dim, test_states_single(dim=dim, pure=False)) for dim in [2,3]])

def test_states(dims=None, pure=False, **kwargs):
    """

    :param dims: e.g. [2,3]
    :param pure: boolean
    :param kwargs: either variable names_multi_list or names_multi_list_str must be given
        names_multi_list: e.g. [['z0', 'z1], ['x0', 'y2']], this will produce two teststates, each with total dim 2x3 (as given in dim=[2,3])
    :return:
    """
    if 'names_multi_list' in kwargs:
        names_multi_list = kwargs.get('names_multi_list')
        l_ok_list = [len(i) == len(dims) for i in names_multi_list]
        if not all(l_ok_list):#:for i in names_multi_list:
            raise Exception('Error: {}, {}, {}, {}, {}'.format(len(dims), dims, names_multi_list, l_ok_list, [[len(i), i, len(dims)] for i in names_multi_list]))
    elif 'names_multi_list_str' in kwargs:
        nmls = kwargs.get('names_multi_list_str')
        names_multi_list = []
        for i in nmls:
            if type(i) != str:
                raise Exception('Error: each item of names_multi_list_str must be {} (is {})'. format(str, type(i)))
            if len(i) != 2 * len(dims):
                raise Exception("Error: {}, {}, {}, {}".format(dims, 2 * len(dims), i, len(i)))
            names_multi_list.append([i+j for i, j in zip(i[::2], i[1::2])])
    else:
        raise Exception('Error: {}'.format(kwargs))
    out = {}
    for idx, names_multi in enumerate(names_multi_list):
        name_multi = "".join(names_multi)
        out[name_multi] = tensor(*[__TEST_STATES_SINGLE_PURE__[dim][name] for name, dim in zip(names_multi, dims)]).unit()
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

# def single_quantum_transition(states):
#     """
#     list of single quantum transitions. higher quantum number is always first item
#     """
#     out = []
#     for s in itertools.product(itertools.product(*states), itertools.product(*states)):
#         delta = np.array(s[0]) - np.array(s[1])
#         if sum(np.abs(delta) == 1) == 1 and sum(delta) == 1:
#             out.append([s[0], s[1]])
#     return out

def single_quantum_transition_hf_spins(states_list, hf_spin_list='all'):
    """
    :param states_list:
        example; states_list = [[0,1], [0,1], [0,1]] for e.g. two electron states [+1,0], two 14n states [+1,0], 13c
    :param hf_spin_list:
        example: hf_spin_list = [0]
        Gives all spins, that have a hyperfine coupling to multiple other spins and thus change their hyperfine frequencies
    :return:
        example: NOTE: only the electron transitions are listed, as only spin[0] is a hf_spin
        [[(0, 0, 0), (1, 0, 0)],
         [(0, 0, 1), (1, 0, 1)],
         [(0, 1, 0), (1, 1, 0)],
         [(0, 1, 1), (1, 1, 1)]]
    """
    if hf_spin_list is 'all':
        hf_spin_list = range(len(states_list))
    non_hf_spins = [i for i in range(len(states_list)) if i not in hf_spin_list]
    states_list_non_hf_spins = [state for idx, state in enumerate(states_list) if idx not in hf_spin_list]

    out = []
    for nhfl in product(*states_list_non_hf_spins):
        state_list_sub = states_list[:]
        for idx, state in zip(non_hf_spins, nhfl):
            state_list_sub[idx] = [state]
        for s in product(*state_list_sub):
            for sn in hf_spin_list:
                if s[sn] + 1 in state_list_sub[sn]:
                    ts = list(s)
                    ts[sn] += 1
                    out.append([s, tuple(ts)])
    return out

def single_quantum_transitions_non_hf_spins(states_list, hf_spin_list='all'):
    """
    :param states_list:
        example; states_list = [[0,1], [0,1], [0,1]] for e.g. two electron states [+1,0], two 14n states [+1,0], 13c
    :param hf_spin_list:
        example: hf_spin_list = [0]
        Gives all spins, that have a hyperfine coupling to multiple other spins and thus change their hyperfine frequencies
    :return:
        example: NOTE: only the electron transitions are listed, as only spin[0] is a hf_spin
    [[(0, 0, 0), (0, 1, 0)],
     [(0, 0, 0), (0, 0, 1)],
     [(1, 0, 0), (1, 1, 0)],
     [(1, 0, 0), (1, 0, 1)]]
    """
    if hf_spin_list is 'all':
        hf_spin_list = range(len(states_list))
    non_hf_spins = [i for i in range(len(states_list)) if i not in hf_spin_list]
    states_list_hf_spins = [states_list[i] for i in hf_spin_list]
    out = []
    for nhfl in product(*states_list_hf_spins):
        state_list_sub = states_list[:]
        for hf_spin, state in zip(hf_spin_list, nhfl):
            state_list_sub[hf_spin] = [state]
        for sn in non_hf_spins:
            for i in [k for k in non_hf_spins if k != sn]:
                state_list_sub[i] = [state_list_sub[i][0]]
            for s in product(*state_list_sub):
                if s[sn] + 1 in states_list[sn]:
                    ts = list(s)
                    ts[sn] += 1
                    out.append([s, tuple(ts)])
    return out

def flipped_spin_numbers(transition):
    """
    :param transition: example [(0, 0, 0), (0, 1, 0)]
    :return: int
        for above example, 1 is returned
    """
    l = []
    for i in range(len(transition[0])):
        if (transition[1][i] - transition[0][i]) == 1:
            l.append(i)
    return l

def single_quantum_transition(states_list, hf_spin_list='all'):
    return single_quantum_transition_hf_spins(states_list=states_list, hf_spin_list=hf_spin_list) + \
           single_quantum_transitions_non_hf_spins(states_list=states_list, hf_spin_list=hf_spin_list)

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

def get_transition_frequency(**kwargs):
    if 'h' in kwargs:
        dims = kwargs['h'].dims[0]
        h_diag = kwargs['h'].diag()
    else:
        dims = kwargs['dims']
        h_diag = kwargs['h_diag']
    m_list = list(product(*[range(i) for i in dims]))

    def t(s0, s1):
        return h_diag[m_list.index(s1)] - h_diag[m_list.index(s0)]
    if 'transition' in kwargs:
        return t(*kwargs['transition'])
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

class Simulate(DataGeneration):

    def __init__(self, gui=False, progress_bar=None, **kwargs):
        DataGeneration.__init__(self)
        self.parameters = kwargs.pop('parameters')
        self.times = kwargs.pop('times')  # dp.times_full
        self.fields = kwargs.pop('fields')  # dp.fields_full
        self.ret_h_mhz = kwargs.pop('ret_h_mhz')
        self.L_Bc = kwargs.pop('L_Bc')
        self.dims = kwargs.pop('dims')
        self.gui = gui
        self.pld = PlotData('analyze_pulse', gui=self.gui)
        self.pld.x_axis_parameter = 'to_bin'
        self.progress_bar = progress_bar
        self.number_of_simultaneous_measurements = len(self.parameters['to_bin']) * len(self.parameters['initial_state']) * len(self.parameters['spin_num']) * len(self.parameters['axis'])

    @property
    def observation_names(self):
        return ["expect"]

    @property
    def dtypes(self):
        return None

    def run(self, abort=None):
        self.init_run(init_from_file=None, iff=None)
        try:
            raise Exception('THE NEXT LINE PROBABLY IS BAD AND IT SHOULD BE for idx, self.current_iterator_df in enumerate(self.iterator()):')
            for idx, self.current_iterator_df in izip(count(0), self.iterator()):
                if abort is not None and abort.is_set(): break
                obs = self.f(current_iterator_df=self.current_iterator_df, abort=abort)
                self.data.set_observations(obs)
                if self.gui:
                    self.pld.new_data_arrived()
            self.pld.new_data_arrived()
            # self.save()
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            self.state = 'idle'
            self.update_current_str()

    def update_progress(self):
        super(Simulate, self).update_progress()
        if self.progress_bar is None:
            print(self.progress)
        else:
            self.progress_bar.value = self.progress

    def detunings(self, iterator_df_row):
        return dict([(key, val) for key, val in iterator_df_row.iteritems() if 'detuning' in key])

    def f(self, current_iterator_df, abort):
        observation_dict_list = []
        u_list_generated = False
        for idx, _I_ in current_iterator_df.iterrows():
            if abort.is_set(): break
            if not u_list_generated:
                u_list_mult = unitary_propagator_list_mult(
                    h_mhz=self.ret_h_mhz(**self.detunings(_I_)),
                    times_full=self.times,
                    fields_full=self.fields,
                    L_Bc=[_I_['ps'] * i for i in self.L_Bc],
                )
                u_list_generated = True
            ts = test_states(dims=self.dims, pure=False, names_multi_list_str=[_I_['initial_state']]).values()[0]
            pts = propagate_test_states([u_list_mult[_I_['to_bin']]], ts)
            obse = OrderedDict(
                [
                    ('expect',
                     expect(
                         jmat(
                             dim2spin(self.dims[_I_['spin_num']]),
                             _I_['axis']),
                         pts[0].ptrace(_I_['spin_num']))
                     )
                ]
            )
            observation_dict_list.append(obse)
        return observation_dict_list
