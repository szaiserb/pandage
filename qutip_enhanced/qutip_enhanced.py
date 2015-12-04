import numpy as np
np.set_printoptions(suppress=True, linewidth=500)
import itertools
import coordinates as co
from qutip import *
import copy

np.set_printoptions(linewidth=1e5)  # matrices are displayd much more clearly


class Eigenvector():
    def __init__(self, dims):
        """
        Eigenvalues are not sorted in any way which makes plotting or 
        following special level at anticrossings difficult. This class follows energy levels 
        according to their eigenvectors. 
        For every newly calculated eigenvectors and eigenvalues the sort function must be called.
        -dims must be numpy array of system dimensions, e.g. np.array([3,3]) for electron + 14N
        
        """
        self.dims = np.array(dims)
        self.evecs_old = make_vector_basis(dims)
        self.idx = np.array(range(self.dims.prod()))
        self.evals_sorted_list = None

    def sort(self, evecs_new, evals_new):
        """    
        keep order of eigenvalues by comparing the
        eigenvectors from former and current run
        -evecs_new is a numpy array of qutip eigenvectors in arbitrary order like given by evals, evec = Qobj.eigenstates()
        -evals_new is numpy array of eigenvalues like given by evals, evec = Qobj.eigenstates()   
        """
        self.idx_new = copy.deepcopy(self.idx)
        for i in range(self.dims.prod()):
            for j in range(self.dims.prod()):
                if (self.evecs_old[i].trans() * evecs_new[j]).norm() >= 0.5 and (
                            self.evecs_old[1].trans() * evecs_new[j]).norm() <= 1.5:
                    self.idx_new[i] = j
                    break
                elif j == self.dims.prod() - 1:
                    raise Exception("No two orthonormal EV have been found. Try to make more steps.")
        self.evals_sorted = np.take(evals_new, self.idx_new)
        self.evecs_old = np.take(evecs_new, self.idx_new)
        if self.evals_sorted_list == None:
            self.evals_sorted_list = np.array([self.evals_sorted])
        else:
            self.evals_sorted_list = np.append(self.evals_sorted_list, np.array([self.evals_sorted]), axis=0)

def make_vector_basis(dims):
    """

    :param dims: np.array
        e.g. [2,3] for system of two spins: spin 1/2. and spin 1
    :return: list of dims.prod() many Qobj vectors
        returns the standard basis for the system given by dims
    """
    sd = itertools.product(*[range(dim) for dim in dims])
    return [tensor([basis(dims[i], s) for i, s in enumerate(state)]) for state in sd]

def dim2spin(dims):
    """
    always returns numpy matrix
    """
    if type(dims) == list:
        raise Exception("Insert numpy array, not list!")
    if isinstance(dims, np.ndarray):
        if dims.ndim > 0:
            spins = (dims - 1) / 2.0
    else:
        spins = (dims - 1) / 2.0
    return spins

def spin2dim(spins):
    """
    always returns numpy matrix
    """
    if type(spins) == list:
        raise Exception("Insert numpy array, not list!")
    if isinstance(spins, np.ndarray):
        if spins.ndim > 0:
            dims = (2 * spins + 1).astype(np.int32, copy=False)
    else:
        dims = int(2 * spins + 1)
    return dims

def qsum(ql):
    """

    :param ql: list of Qobj
        all list elements must have same dimensions e.g [sigmax(), sigma(y)]
    :return: Qobj
        ql[0] + ql[1] + .. + ql[-1]
    """
    sum_ql = 0 * ql[0]
    for qobj in ql:
        sum_ql += qobj
    return sum_ql

def get_sub_matrix(op, levels):
    """
    :param op: Qobj operator
        Qobj operator describing a single, non-concatenated system, e.g. the NV electron spin.
        This implies op.dim == op.shape
    :param levels: list of int
        list of  levels of system described by operator 'op' that should be included in the output operator
        maximum list size is range(op.dims[0]). Minimum list size is [] (then an empty operator is returned)

    :return: Qobj operator
        output operator has same characteristics as 'op' but describes a system with size specified in 'levels'
        Example: 'op' describes a spin 1 system, i.e. op.dims = 3. Given levels=[1,2], in the returned operator,
        line and column 0 are removed, i.e. the returned operator has dim = [2,2]
    """
    matrix = op.data.todense()
    levels = numpy.sort(levels)
    for i in range(len(matrix) - 1, -1, -1):
        if not i in levels:
            for j in range(2):
                matrix = np.delete(matrix, i, j)
    op = Qobj(matrix)
    return op

def get_expanded_matrix(op, levels, value=0, make_eye='no'):
    """
    gives back an expanded matrix, that has value on all columns and rows in levels
    -matrix must be numpy matrix or maybe numpy array
    - levels are inserted as [0,2], max(levels) must not be larger than len(levels) + len(matrix) - 1
    example: insert matrix = np.zeros([3,3]), levels = [0,3], value = 5
    """
    matrix = op.data.todense()
    levels = numpy.sort(levels)
    for i in range(len(levels)):
        k = levels[i]
        for j in range(2):
            matrix = np.insert(matrix, k, value, j)
        if make_eye == 'yes':
            matrix[k, k] = 1
    op = Qobj(matrix)
    return op

def get_rot_matrix(dim, transition='all', rotation_axis={'x': 1, 'y': 1}):
    """
    Returns a rotation matrix M  that rotates a single spin. The dimension of the matrix is the multiplicity of the spin.

        Parameters
        ----------
        dim : float
            multiplicity of the spin
            e.g. 2,3,4,...

        rotation_axis : dictionary according to module 'coordinates'
            axis about which the rotation is done
            Example:
                rotation around x_axis:
                    rotation_axis = {'x': 1, 'y': 0, 'z': 0} (equal to {'x': 1})

        transition : list of integers or 'all'
            sublevels of the spin that should be rotated.
            Needs not to be given for spin = 1/2.0
            len(transition) !> 2

            Examples:
            transition = 'all' rotates all levels (equal to transition = range(2*spin + 1)) resulting in a state of basis
            spin = 1, transition = [0,2] gives a rotation matrix M that rotates between the states +1 and -1
                M = [[a,0,b],
                     [0,0,0],
                     [c,0,d]
                     ]

        Returns
        --------
        rot_mat : qutip operator
        """

    rotation_axis = co.Coord().coord_unit(rotation_axis, 'cart')
    if transition == 'all':
        dim_rot = dim
    else:
        dim_rot = 2
    rot_mat = 0*qeye(dim_rot)
    for axis_name in co.Coord().cart_coord:
        rot_mat = rot_mat + rotation_axis[axis_name] * jmat((dim_rot - 1) / 2.0, axis_name)
    if transition != 'all':
        rot_mat = get_expanded_matrix(rot_mat, [x for x in range(dim) if x not in transition])
    return rot_mat

def get_rot_operator(dim, rotation_axis={'x': 1, 'y': 1}, transition='all', angle=np.pi / 2.0):
    rot_mat = get_rot_matrix(dim, transition=transition, rotation_axis=rotation_axis)
    U = (-1j * angle * rot_mat).expm()
    return U

def get_rot_operator_all_spins(dims, rotation_axis={'x': 1, 'y': 1}, rotated_spin=0, transition='all',
                               selective_to={}, angle=np.pi / 2.0):
    """
    - selective_to[0] names a list of spins. selective_to[1] gives the levels of spin i that the rotation should be selective on,
    if a spin is omitted here in this dictionary, rotated_spin will be rotated in all spin states
    example: given a system with three spins, spin 1, 1, 1/2.0, transition = [0,1], rotated spin = 1 and selective_to = {0:[1,2],2:[1]} will rotate the second spins transition 1 < - > 0,
    if spin 0 is in spin states 1 or 2 but not if it is in spin state 0 and if spin 2 is in spin state 1 but not if it is in spin state 0
    """

    def complete_selective_to(selective_to):
        """
        -selective_to is dictionary that includes the spins a that the rotation should be selective to.
        omit a spin, if this spins spin state should not play a role
        """
        return dict(selective_to.items() + [[i, range(dims[i])] for i in range(len(dims)) if not i in selective_to])

    def add_spin(dim, rot_op, side, selectivity):
        dims_rot_op = np.array(rot_op.dims[0])
        # init matrix
        if side == 'front':
            new_dims = np.insert(dims_rot_op, 0, dim)
        else:
            new_dims = np.append(dims_rot_op, dim)
        rot_op_all = 0*qeye(list(new_dims))
        # add subrotations for each substate of the specific spin
        for i in range(dim):
            rot_op_i = qeye(rot_op.dims[0])
            if i in selectivity:
                rot_op_i = rot_op
            if side == 'front':
                rot_op_all = rot_op_all + tensor(fock_dm(dim, i), rot_op_i)
            elif side == 'back':
                rot_op_all = rot_op_all + tensor(rot_op_i, fock_dm(dim, i))
        return rot_op_all

    selective_to = complete_selective_to(selective_to)
    rot_op_all = get_rot_operator(dims[rotated_spin], transition=transition, rotation_axis=rotation_axis,
                                       angle=angle)
    for i in range(rotated_spin - 1, -1, -1):
        rot_op_all = add_spin(dims[i], rot_op_all, side='front', selectivity=selective_to[i])
    for i in range(rotated_spin + 1, len(dims), +1):
        rot_op_all = add_spin(dims[i], rot_op_all, side='back', selectivity=selective_to[i])
    return rot_op_all

def rotate(op, rotation_axis={'x': 1, 'y': 1}, rotated_spin=0, transition='all', selective_to={},
           angle=np.pi / 2.0):
    """
    -selective_to gives the a dictionary of the transitions, that the rotation should be selective to. if omitted, the rotation is
    performed for all states of the other spins in the spin system. example: selective_to = {2: [1,2]}, rotated_spin = 0 would rotate rotated_spin
    for all spin states of spin 1 but not in case spin 2 is in spin state 0
    """
    U = get_rot_operator_all_spins(op.dims[0], rotation_axis=rotation_axis, rotated_spin=rotated_spin,
                                        transition=transition, selective_to=selective_to, angle=angle)
    operator = U * op * U.dag()
    return operator


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rho0 = ket2dm(tensor(basis(2, 0), basis(2, 0)))

    # print qte.get_rot_operator_all_spins(dims=[2], rotation_axis={'y':1}, angle=np.pi/2.)

    def r(phi):
        rho = rotate(rho0, rotation_axis={'y': 1}, rotated_spin=0, angle=np.pi/2.)
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=1, angle=np.pi/2.)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=0, angle=phi)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=1, angle=2*phi)
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=1, angle=-np.pi/2)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=0, angle=-np.pi/2., selective_to={1: [1]})
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=0, angle=-np.pi/2)
        return rho
    #
    cphase = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'z': 1}, rotated_spin=0, angle=np.pi, selective_to={1:[1]})
    # rotx = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'y': 1}, rotated_spin=1, angle=np.pi/2.)
    # cphase = rotx*cphase*rotx.dag()
    # x = (basis(2,0) + basis(2, 1)).unit()
    # xm = (basis(2,0) - basis(2, 1)).unit()
    # y = (basis(2,0) + 1j*basis(2, 1)).unit()
    # ym = (basis(2,0) - 1j*basis(2, 1)).unit()
    # z = basis(2,0)
    # zm = basis(2,1)
    # pl = np.linspace(0, 2*np.pi, 30)
    # z0 = [expect(tensor(sigmaz(), qeye(2)), r(p)) for p in pl]
    # z1 = [expect(tensor(qeye(2), sigmaz()), r(p)) for p in pl]
    # #
    # plt.plot(pl, z0)
    # plt.plot(pl, z1)
    # plt.show()
    # b = ket2dm((basis(2,0) + basis(2,1)).unit())
    # rho = tensor(ket2dm(basis(2,0)), b, b)
    # print qte.get_rot_operator_all_spins([2], rotation_axis={'y':1}, angle=np.pi/2.)
    # ctwopi = qte.get_rot_operator_all_spins([2, 2, 2], rotation_axis={'x':1}, rotated_spin=0, selective_to={1: [1],  2: [1]}, angle=np.pi)
    # print ctwopi.ptrace([1, 2])
    # ctwopi = ctwopi.ptrace([1, 2])

    # class Args(object):
    #     AO = 1
    #     ALPHA = np.pi/2.
    #
    # args = Args()
    #
    # if args.AO == 1:
    #     gate_0 = qte.get_rot_operator(2, rotation_axis={'y': 1}, angle=args.ALPHA).data.todense()
    #     gate_1 = qte.get_rot_operator(2, rotation_axis={'y': 1}, angle=np.pi - args.ALPHA).data.todense()
    # elif args.AO == 0:
    #     gate_1 = qte.get_rot_operator(2, rotation_axis={'y': 1}, angle=args.ALPHA).data.todense()
    #     gate_0 = qte.get_rot_operator(2, rotation_axis={'y': 1}, angle=np.pi - args.ALPHA).data.todense()
    #
    # print gate_0
    # print gate_1
    # a = ket2dm(tensor(basis(2, 0), basis(2, 1)))
    #
    #
    # # o1 = qte.get_rot_operator_all_spins([2, 2], rotation_axis=dict(y=1),rotated_spin=0,selective_to={1: [1]})
    # # o2 = qte.get_rot_operator_all_spins([2, 2], rotation_axis=dict(y=1),rotated_spin=0,selective_to={1: [0]})
    #
    # def evolve(rho, length_mus, H):
    #     options = Odeoptions(nsteps=100000)
    #     if length_mus != 0.0:
    #         # rho = mesolve(h(**kwargs), rho, np.linspace(0, length_mus, 2), [], [], options=options).states[-1]
    #         u = (-2*np.pi*1j * H * length_mus).expm()
    #         rho = u * rho * u.dag()
    #     return rho
    #
    #
    # import qutip_nv_hamilton as nvh
    #
    # H = nvh.NVHam(magnet_field={'z': 0.53756}, electron_levels=[1, 2], nitrogen_levels=[0, 1], n_type='n14')
    # h = H.h_nv
    # tau = 1.* 2*np.pi/ H.hf_para_n
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=1, selective_to={0: [0]}, angle=np.pi / 2.)
    #
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [1]}, angle=np.pi / 2.)
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [0]}, angle=np.pi / 2.)
    #
    # a = qte.rotate(a, rotation_axis=dict(z=1), rotated_spin=0, angle=np.pi/2.)
    #
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [1]}, angle=np.pi / 2.)
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [0]}, angle=np.pi / 2.)
    #
    # a = tensor(Qobj(np.diag(a.ptrace(0).diag())), a.ptrace(0))
    #
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [1]}, angle=np.pi / 2.)
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [0]}, angle=np.pi / 2.)
    #
    # a = qte.rotate(a, rotation_axis=dict(z=1), rotated_spin=0, angle=np.pi)
    #
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [1]}, angle=np.pi / 2.)
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [0]}, angle=np.pi / 2.)
    #
    # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=1, selective_to={0: [0]}, angle=np.pi / 2.)
    # a = qte.rotate(a, rotation_axis=dict(y=-1), rotated_spin=1, selective_to={0: [1]}, angle=np.pi / 2.)
    # print np.around(a.ptrace(1).data.todense(), 1)
    #
    #
    # def y(tau):
    #     a = ket2dm(tensor(basis(2, 0), basis(2, 1)))
    #     a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=1, selective_to={0: [0]}, angle=np.pi / 2.)
    #     # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [1]}, angle=np.pi / 2.)
    #     # a = qte.rotate(a, rotation_axis=dict(y=1), rotated_spin=0, selective_to={1: [0]}, angle=np.pi / 2.)
    #     a = evolve(a, tau, h)
    #     return expect(a.ptrace(1), sigmax())

    # import matplotlib.pyplot as plt
    # t = -1./ H.hf_para_n
    # x = np.linspace(0, 1*t, 50)
    # plt.plot(x, [y(i) for i in x])
    # plt.show()