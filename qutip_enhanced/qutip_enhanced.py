# coding=utf-8
from __future__ import print_function, absolute_import, division
__metaclass__ = type

from qutip import *
import numpy as np
np.set_printoptions(suppress=True, linewidth=500)

from . import coordinates
import copy
import scipy.linalg

from numpy import *
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

np.set_printoptions(linewidth=1e5)  # matrices are displayd much more clearly

class Eigenvector:
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
                    raise Exception("No two orthonormal EV have been found. Try to make more steps. {}".format((self.evecs_old[i].trans() * evecs_new[j]).norm() ))
        self.evals_sorted = np.take(evals_new, self.idx_new)
        self.evecs_old = np.take(evecs_new, self.idx_new)
        if self.evals_sorted_list is None:
            self.evals_sorted_list = np.array([self.evals_sorted])
        else:
            self.evals_sorted_list = np.append(self.evals_sorted_list, np.array([self.evals_sorted]), axis=0)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Bloch(Bloch):

    def draw_arrow(self, rho, elev, azim, length=0.1, shorten_line_factor=0.98, n=100, arrowstyle="-|>",  **kwargs):
        if not self.axes:
            raise Exception('For this functionality axes must be given to Bloch() upon instantiation')
        x, y, z = coordinates.sph2cart(rho, elev, azim)
        ahs = int(np.ceil(n*shorten_line_factor))
        self.axes.plot(x[:ahs], y[:ahs], z[:ahs], label='parametric curve', **kwargs)
        self.axes.add_artist(self.arrowhead(x, y, z, length=length, arrowstyle=arrowstyle, **kwargs))

    def arrowhead(self, x, y, z, length=0.1, arrowstyle="-|>", **kwargs):
        if not self.axes:
            raise Exception('For this functionality axes must be given to Bloch() upon instantiation')
        xyz = np.gradient(np.column_stack([x, y, z]))[0]
        iii = -1
        x1 = xyz[iii, 0]
        y1 = xyz[iii, 1]
        z1 = xyz[iii, 2]
        l = length
        a = Arrow3D([x[iii] - x1 * l, x[iii] + x1 * l], [y[iii] - y1 * l, y[iii] + y1 * l],
                    [z[iii] - z1 * l, z[iii] + z1 * l],
                    mutation_scale=20,
                    lw=3, arrowstyle=arrowstyle, **kwargs)
        return a


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
    if type(dims) is list:
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
    if type(spins) is list:
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

def do_zero(h):
    dims = h.dims
    h = h.data.todense()
    d = np.diag(h) - h[0,0]
    np.fill_diagonal(h, d)
    return Qobj(h, dims=dims)

def get_transition_frequency(h, s0, s1):
    """energy required for transition from state s0 to state s1"""
    dims = h.dims[0]
    m_list = list(itertools.product(*[range(i) for i in dims]))
    h_diag = h.diag()
    return h_diag[m_list.index(s1)] - h_diag[m_list.index(s0)]

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
    levels = np.sort(levels)
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
    levels = np.sort(levels)
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

    rotation_axis = coordinates.Coord().coord_unit(rotation_axis, 'cart')
    if transition == 'all':
        dim_rot = dim
    else:
        dim_rot = 2
    rot_mat = 0*qeye(dim_rot)
    for axis_name in coordinates.Coord().cart_coord:
        rot_mat = rot_mat + rotation_axis[axis_name] * jmat((dim_rot - 1) / 2.0, axis_name)
    if transition != 'all':
        rot_mat = get_expanded_matrix(rot_mat, [x for x in range(dim) if x not in transition])
    return rot_mat


def get_rot_matrix_all_spins(*args, **kwargs):
    o = get_rot_operator_all_spins(*args, **kwargs)
    dims = o.dims
    return Qobj(scipy.linalg.logm(o.data.todense()), dims=dims)/np.pi*1.j

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
        return dict(list(selective_to.items()) + [[i, range(dims[i])] for i in range(len(dims)) if not i in selective_to])

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
    cphase0 = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'z': 1}, rotated_spin=0, angle=np.pi/2., selective_to={1:[0]})
    cphase1 = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'z': 1}, rotated_spin=0, angle=- np.pi/2., selective_to={1:[1]})
    rotx = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'y': 1}, rotated_spin=1, angle=np.pi/2.)
    print(rotx*cphase0*cphase1*rotx.dag())

    a = np.array([[ 0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                  [ 0.+0.j,  0.-1.j,  0.+0.j , 0.+0.j],
                  [ 0.+0.j,  0.+0.j , 0.+1.j,  0.+0.j],
                  [ 0.+0.,  0.+0.j , 0.+0.j,  0.+1.j]])