from qutip_enhanced import *
import matplotlib.pyplot as plt
import itertools

def purity(dm):
    """should work for qudits also"""
    s = range(len(dm.dims[0]))
    comb = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))[1:]
    return dict((i, ((dm.ptrace(i))**2).tr()) for i in comb)

def plot_state(dml, var=None, qubit_names=None, vertical_lines_at=None):
    dims = dml[0].dims[0] if type(dml) is list else dml.dims[0]
    n = len(dims)
    qubit_names = ['Qubit {}'.format(i) for i in range(n)] if qubit_names is None else qubit_names

    if type(dml) is list:
        if type(var) is not None and len(var) != len(dml):
            raise Exception('list of density matrices must be as long as list of x-variables.')
    if type(dml) is not list and var is not None:
        raise Exception('only one density matrix given, list of x-variables not allowed.')

    if type(dml) is list:
        opl = [sigmax, sigmay, sigmaz]
        y_labels = ['x', 'y', 'z', 'purity']

    else:
        y_labels=['traced out state', 'purity']
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
                    arr[i][j].plot(var, [expect(opl[i](), dm.ptrace(j)) for dm in dml], linewidth=2)
                else:
                    arr[i][j].plot(var, [(dm.ptrace(j) ** 2).tr() for dm in dml], linewidth=2)
                if vertical_lines_at is not None:
                    for vla in vertical_lines_at:
                        arr[i][j].axvline(x=vla, linewidth=1, color='r')
            else:
                if i < len(y_labels) - 1:
                    b.append(Bloch(fig, arr[i][j]))
                    b[j].add_states(dml.ptrace(j))
                    b[j].render(fig, arr[i][j])
                else:
                    arr[i][j].plot(np.linspace(0., 1., 11), [(dm.ptrace(j) ** 2).tr() for dm in [dml]*11], linewidth=2)
            if type(dml) is list or i == len(y_labels) - 1:
                arr[i][j].grid(True)
                arr[i][j].set_ylim(lim[i])
            if i == len(y_labels) - 1:
                arr[i][j].axhline(y=1 / np.product(dims, dtype=float), linewidth=1, color='r', linestyle='dashed')
            # # names
            if j == 0:
                arr[i][j].set_ylabel('{}'.format(ylabel))
            if i == len(y_labels) - 1:
                arr[i][j].set_xlabel('{}'.format(qubit_names[j]))
    plt.show()
