from qutip_enhanced import *
import matplotlib.pyplot as plt
import itertools

def purity(dm):
    """should work for qudits also"""
    s = range(len(dm.dims[0]))
    comb = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))[1:]
    return dict((i, ((dm.ptrace(i))**2).tr()) for i in comb)

def test_states(gate, pure=False, **kwargs):
    dims = gate.dims[0]
    out = {}
    x = basis(2, 0) + basis(2, 1)
    xm = basis(2, 0) - basis(2, 1)
    y = basis(2, 0) + 1j * basis(2, 1)
    ym = basis(2, 0) - 1j * basis(2, 1)
    z = basis(2, 0)
    zm = basis(2, 1)
    test_states_single = dict([(name, locals()[name]) for name in ['x', 'xm', 'y', 'ym', 'z', 'zm']])
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
    dims = dml[0].dims[0]
    n = len(dims)
    qubit_names = ['Qubit {}'.format(i) for i in range(n)] if qubit_names is None else qubit_names

    x_val = range(len(dml)) if (var is None or type(var[0]) is str) else var
    x_ticks = var if type(var[0]) is str else x_val

    opl = [sigmax, sigmay, sigmaz]
    y_labels = ['x', 'y', 'z', 'purity']

    lim = [[-1.05, 1.05]] * (len(y_labels) - 1) + [[-0.05, 1.05]]

    fig, arr = plt.subplots(4, n, sharex='col', sharey='row')

    for i, ylabel in enumerate(y_labels):
        for j, x_label in enumerate(qubit_names):
            if i < len(y_labels) - 1:
                y_val = [expect(opl[i](), dm.ptrace(j)) for dm in dml]
            else:
                y_val = [(dm.ptrace(j) ** 2).tr() for dm in dml]

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
                arr[i][j].axhline(y=0.5, linewidth=1, color='r', linestyle='dashed')
            if j == 0:
                arr[i][j].set_ylabel('{}'.format(ylabel))
            if i == len(y_labels) - 1:
                arr[i][j].set_xlabel('{}'.format(qubit_names[j]))

def plot_state_old(dml, var=None, qubit_names=None, vertical_lines_at=None):
    dims = dml[0].dims[0] if type(dml) is list else dml.dims[0]
    n = len(dims)
    qubit_names = ['Qubit {}'.format(i) for i in range(n)] if qubit_names is None else qubit_names



    if (type(var) is list) and (type(var[0]) is str):
        x_val = range(len(var))
    else:
        x_val = var

    if type(dml) is list:
        if type(var) is not None and len(var) != len(dml):
            raise Exception('list of density matrices must be as long as list of x-variables.')
    if type(dml) is not list and var is not None:
        raise Exception('only one density matrix given, list of x-variables not allowed.')

    opl = [sigmax, sigmay, sigmaz]
    if type(dml) is list:
        y_labels = ['x', 'y', 'z', 'purity']

    else:
        y_labels = ['traced out state', 'purity']
        b = []
    lim = [[-1.05, 1.05]] * (len(y_labels) - 1) + [[-0.05, 1.05]]

    fig = plt.figure()
    arr = [[[]] * n] * len(y_labels)
    for i, ylabel in enumerate(y_labels):
        for j, x_label in enumerate(qubit_names):
            # data
            if type(dml) is list or i == len(y_labels) - 1:
                arr[i][j] = fig.add_subplot(len(y_labels), n, 1 + n * i + j)
            else:
                arr[i][j] = fig.add_subplot(len(y_labels), n, 1 + n * i + j, projection='3d')
            if type(dml) is list:
                if i < len(y_labels) - 1:
                    y_val = [expect(opl[i](), dm.ptrace(j)) for dm in dml]
                else:
                    y_val = [(dm.ptrace(j) ** 2).tr() for dm in dml]
                arr[i][j].plot(x_val, y_val, linewidth=2)
                if type(var) is list and type(var[0]) is str:
                    plt.xticks(range(len(var)), var, size='small', rotation='vertical')
                if vertical_lines_at is not None:
                    for vla in vertical_lines_at:
                        arr[i][j].axvline(x=vla, linewidth=1, color='r')
            else:
                if i < len(y_labels) - 1:
                    b.append(Bloch(fig, arr[i][j]))
                    b[j].add_states(dml.ptrace(j))
                    b[j].render(fig, arr[i][j])
                    arr[i][j].annotate('x: {:.3f}\ny: {:.3f}\nz: {:.3f}'.format(
                        *[expect(opl[ea](), dml.ptrace(j)) for ea in range(3)]),
                                       xy=(1, 0), xycoords='axes fraction', fontsize=16, xytext=(-5, 5),
                                       textcoords='offset points', ha='right', va='bottom', multialignment="left")
                else:
                    arr[i][j].plot(np.linspace(0., 1., 11), [(dm.ptrace(j) ** 2).tr() for dm in [dml] * 11],
                                   linewidth=2)
            if type(dml) is list or i == len(y_labels) - 1:
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
                # for a single qubit the purity has a lower bound of 0.5
                arr[i][j].axhline(y=0.5, linewidth=1, color='r', linestyle='dashed')
            # # names
            if j == 0:
                arr[i][j].set_ylabel('{}'.format(ylabel))
            if i == len(y_labels) - 1:
                arr[i][j].set_xlabel('{}'.format(qubit_names[j]))


    # def test_states(gate, pure=False, **kwargs):
#     dims = gate.dims[0]
#     out = {}
#
#     x = basis(2, 0) + basis(2, 1)
#     xm = basis(2, 0) - basis(2, 1)
#     y = basis(2, 0) + 1j * basis(2, 1)
#     ym = basis(2, 0) - 1j * basis(2, 1)
#     z = basis(2, 0)
#     zm = basis(2, 1)
#     test_states_single = dict([(name, locals()[name]) for name in ['x', 'xm', 'y', 'ym', 'z', 'zm']])
#     if 'names_multi_list' in kwargs:
#         names_multi_list = kwargs.get('names_multi_list')
#     else:
#         names_single = kwargs.get('names_single', ['x', 'xm', 'y', 'ym', 'z', 'zm'])
#         names_multi_list = itertools.product(*[names_single] * len(dims))
#     for names_multi in names_multi_list:
#         name_multi = "".join(names_multi)
#         out[name_multi] = tensor(*[test_states_single[name] for name in names_multi]).unit()
#         if not pure:
#             out[name_multi] = ket2dm(out[name_multi])
#     return out
#
# def propagate_test_states(gate, test_states):
#     out = {}
#     for key, val in test_states.items():
#         out [key] = gate * val * gate.dag()
#     return out
#
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
#
# def get_transition_frequency(h, **kwargs):
#     """
#     energy required for transition from state s0 to state s1
#     IMPORTANT: states are numbered [0,1,2, ..], e.g. when ms= 0, -1 are seen as qubit, states are [0,1], not [1,2]
#     """
#     dims = h.dims[0]
#     m_list = list(itertools.product(*[range(i) for i in dims]))
#     h_diag = h.diag()
#     def t(s0, s1):
#         return h_diag[m_list.index(s1)] - h_diag[m_list.index(s0)]
#     if 's0' in kwargs and 's1' in kwargs:
#         return t(kwargs['s0'], kwargs['s1'])
#     else:
#         return [t(s[0], s[1]) for s in kwargs.states_list]
#
# def state_num2name(state, name_list):
#     return tuple([name_dict[i][val] for i, val in enumerate(state)])
#
# def state_name2num(state, name_list):
#     return tuple([name_list[i].index(val) for i, val in enumerate(state)])
#