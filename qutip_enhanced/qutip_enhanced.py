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
import itertools

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

def qmul(ql):
    sum_ql = ql[0]
    for qobj in ql[1:]:
        sum_ql *= qobj
    return sum_ql

def do_zero(h):
    dims = h.dims
    h = h.data.todense()
    d = np.diag(h) - h[0,0]
    np.fill_diagonal(h, d)
    return Qobj(h, dims=dims)

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

def get_rot_matrix(dim, rotation_axis=None, **kwargs):
    """
    :param dim: int
        dimensionality of rotated qubit
        example: 2
    :param rotation_axis: dict
        example: {'x': 1, 'y': 1}
    :param kwargs:
        a) transition, list/np.array
        b) rotated_levels, list/np.array
        example: rotated_levels = [0, 2]
    list/np.array
    :return:
    """
    ## generate rotation matrix for sublevels
    rotation_axis = coordinates.Coord().coord_unit(rotation_axis, 'cart')
    rotated_levels = kwargs.get('transition', kwargs.get('rotated_levels', range(dim)))
    dim_rot = len(rotated_levels)
    rot_mat = qsum([rotation_axis[axis_name] * jmat(dim2spin(dim_rot), axis_name) for axis_name in coordinates.Coord().cart_coord])
    ## expand rotation matrix to full size 'dim' of single qubit
    out = np.zeros([dim, dim], dtype=np.complex128)
    for idxrm0, idxe0 in zip(range(dim_rot), rotated_levels):
        for idxrm1, idxe1 in zip(range(dim_rot), rotated_levels):
            out[idxe0, idxe1] = rot_mat[idxrm0, idxrm1]
    return Qobj(out)

def get_rot_operator(dim, angle=np.pi / 2.0, **kwargs):
    rot_mat = get_rot_matrix(dim, **kwargs)
    U = (-1j * angle * rot_mat).expm()
    return U

def get_rot_operator_all_spins(dims=None, selective_to=None, rotated_spin=None, **kwargs):
    """
    - selective_to[0] names a list of spins. selective_to[1] gives the levels of spin i that the rotation should be selective on,
    if a spin is omitted here in this dictionary, rotated_spin will be rotated in all spin states
    example: given a system with three spins, spin 1, 1, 1/2.0, transition = [0,1], rotated spin = 1 and selective_to = {0:[1,2],2:[1]} will rotate the second spins transition 1 < - > 0,
    if spin 0 is in spin states 1 or 2 but not if it is in spin state 0 and if spin 2 is in spin state 1 but not if it is in spin state 0
    """
    selective_to = {} if selective_to is None else selective_to
    if rotated_spin in selective_to:
        raise Exception('Error: rotated_spin must not be in selective_to!!\n{}, {}'.format(dims, selective_to))
    for idx, val in selective_to.items():
        if len(val) == 0 or len(val) > dims[idx]:
            raise Exception('Error: {}, {}'.format(dims, selective_to))
    out = []
    for stl in [True, False]:
        if stl is False and len(selective_to) == 0:
            continue
        l = []
        for i, dim in enumerate(dims):
            if i == rotated_spin:
                if stl:
                    l.append([kwargs.get('rot_op', get_rot_operator(dim=dims[rotated_spin], **kwargs))])
                else:
                    l.append([qeye(dim)])
            else:
                if stl:
                    if i in selective_to:
                        sl = selective_to[i]
                    else:
                        sl = range(dim)
                else:
                    if i in selective_to:
                        sl = [k for k in range(dim) if k not in selective_to[i]]
                    else:
                        sl = []
                sll = []
                for item in sl:
                    sll.append(fock_dm(dim, item))
                l.append(sll)
        out.append(qsum([tensor(*items) for items in itertools.product(*l)]))
    return qsum(out)

def get_rot_matrix_all_spins(*args, **kwargs):
    o = get_rot_operator_all_spins(*args, **kwargs)
    dims = o.dims
    return Qobj(scipy.linalg.logm(o.data.todense()), dims=dims) / np.pi * 1.j

def rotate(op, **kwargs):
    U = get_rot_operator_all_spins(op.dims[0], **kwargs)
    operator = U * op * U.dag()
    return operator


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rho0 = ket2dm(tensor(basis(2, 0), basis(2, 0)))

    # print qte.get_rot_operator_all_spins(dims=[2], rotation_axis={'y':1}, angle=np.pi/2.)

    def r(phi):
        rho = rotate(rho0, rotation_axis={'y': 1.}, rotated_spin=0, angle=np.pi/2.)
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=1, angle=np.pi/2.)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=0, angle=phi)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=1, angle=2*phi)
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=1, angle=-np.pi/2)
        rho = rotate(rho, rotation_axis={'z': 1}, rotated_spin=0, angle=-np.pi/2., selective_to={1: [1]})
        rho = rotate(rho, rotation_axis={'y': 1}, rotated_spin=0, angle=-np.pi/2)
        return rho

    cphase0 = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'z': 1}, rotated_spin=0, angle=np.pi/2., selective_to={1:[0]})
    cphase1 = get_rot_operator_all_spins(dims = [2, 2], rotation_axis={'z': 1}, rotated_spin=0, angle=- np.pi/2., selective_to={1:[1]})
    rotx = get_rot_operator_all_spins(dims=[2, 2], rotation_axis={'y': 1}, rotated_spin=1, angle=np.pi/2.)
    print(rotx*cphase0*cphase1*rotx.dag())

    a = np.array([[ 0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                  [ 0.+0.j,  0.-1.j,  0.+0.j , 0.+0.j],
                  [ 0.+0.j,  0.+0.j , 0.+1.j,  0.+0.j],
                  [ 0.+0.,  0.+0.j , 0.+0.j,  0.+1.j]])