import numpy as np
from scipy.constants import *
np.set_printoptions(suppress=True, linewidth=500)

from qutip_enhanced import *

import coordinates as co

class NVHam():
    """
    Hamiltonian describing a single NV + nuclear spins.

    ALWAYS USE NUMPY ARRAYS, DONT USE LISTS.
    
    All frequencies are in [MHz]. Magnetic field in [Tesla].

    Example: NV with N14, one 13C at location x = 1e-9 meters; Magnetic field is 0.5 Tesla in z, 1 gauss in y, only electron spin states ms = 0 and ms = -1 are considered
        ham = NVHam(magnet_field = {'z':0.5, 'y':0.0001}, n_type = 'n14', electron_levels = [0,1]) #create base nv hamiltonian
        ham.add_spin(ham.hft_13c_dd(location = {'x': 1e-9}), ham.h_13c()) #add 13C nuclear spin with pure dipolar coupling
        evals, evecs = ham.h_nv.eigenstates() #calculate eigenvalues and eigenstates
    """


    def __init__(self, magnet_field={'z': 0.1}, n_type=None, electron_levels=None, nitrogen_levels=None, D=2870):
        """

        :param magnet_field: dict
            The applied external magnetic field in [Tesla] must be given as dictionary like described in module coordinates. Cartesian and Spherical coordinates possible.
        :param n_type: str
            The type of nitrogen atom : 'n14', 'n15', None
        :param electron_levels: list
            The sublevels of the electron spins are chosen with electron_levels, which can be e.g. [1,2], [0,1,2]
            but not [0], [1] or [2].
        :param nitrogen_levels: list
            Same as electron_levels but for the nitrogen spin.
        :param D: float
            Zero field splitting, default 2870 MHz.
        """
        self.magnet_field_cart = co.Coord().coord(magnet_field, 'cart')  # magnet field in cartesian coordinates
        self.nitrogen_levels = nitrogen_levels
        self.n_type = n_type
        self.electron_levels = [0, 1, 2] if electron_levels is None else electron_levels
        self.D = D
        self.set_h_nv()

    # gyromagnetic ratios given in 1/2pi MHz/T, i.e. f = gamma*B
    gamma_e = - 2.8025e4  # defenition on http://nmrwiki.org/wiki/index.php?title=Gyromagnetic_ratio is wrong
    gamma_13c = 10.705

    def get_nitrogen_params(self):
        """
        sets the parameters for the used nitrogen spin, i.e. for 14N or 15N. If n_type == None, nothing happens
        
        Usable Class Attributes
        --------
        
        j_n : float
            quantum number of current NVs nitrogen isotope
        Q_n : float
            
            
        """
        if self.n_type == 'n14':
            self.j_n = 1
            self.Q_n = -4.945
            self.gamma_n = 3.0766
            self.hf_para_n = -2.165
            self.hf_perp_n = -2.7
            self.nitrogen_levels = [0, 1, 2] if self.nitrogen_levels is None else self.nitrogen_levels
        elif self.n_type == 'n15':
            self.j_n = 1 / 2.0
            self.Q_n = 0
            self.gamma_n = -4.3156
            self.hf_para_n = +3.03
            self.hf_perp_n = +3.65
            self.nitrogen_levels = [0,1] if self.nitrogen_levels is None else self.nitrogen_levels
        else:
            raise Exception("Chosen 'n_type' is not allowed.")
        if self.n_type is not None and type(self.nitrogen_levels) != list:
            raise Exception("Chosen 'nitrogen_levels must be a list, e.g. nitrogen_levels = [1,2] for n_type='n14'.")


    def calc_zeeman(self, gamma, j):
        """

        :param gamma: float
            gyromagnetic ratio of the subject spin
        :param j: float
            total spin quantum number, i.e. 1/2, 1, 3/2, ..
        :return: zeeman splitting. However be careful with gyromagnetic ratios of electron spin. Use their negative
        """
        return - qsum([gamma * self.magnet_field_cart[axis] * jmat(j, axis) for axis in co.Coord().cart_coord])

    def h_electron(self):
        """
        calculate electron hamilton matrix with zerofield and zeeman splitting. 
        Final size of h_electron depends on self.electron_levels. 
        """
        self.h_ezfs = self.D * jmat(1, 'z') ** 2
        self.h_eze = self.calc_zeeman(gamma=self.gamma_e, j=1)
        self.h_e = self.h_ezfs + self.h_eze
        return self.h_e

    def h_nitrogen(self):
        """
        calculate nitrogen hamilton operator with quadrupol, hyperfine and zeeman. 
        Final size of h_nitrogen depends on self.nitrogen_levels
        """
        self.h_nqp = self.Q_n * jmat(self.j_n, 'z') ** 2
        self.h_nze = self.calc_zeeman(self.gamma_n, self.j_n)
        self.h_n = self.h_nqp + self.h_nze
        return self.h_n

    def hft_nitrogen(self):
        """
        Gives the nitrogen hyperfine tensor according to self.get_nitrogen_params()
        """
        return np.diag([self.hf_perp_n, self.hf_perp_n, self.hf_para_n])

    def h_13c(self):
        """
        calculate 13C hamilton matrix with with zeeman splitting. This 
        Final size of h_electron depends on self.electron_levels. 
        """
        return self.calc_zeeman(self.gamma_13c, 1 / 2.0)

    def hft_13c_dd(self, location={'rho': 1e-9, 'elev': np.pi / 2, 'azim': 0}):
        """
        Calculates 13C hyperfine tensor for pure dipolar coupling at position location. 
        location must be inserted according to module coordinates. 
        For very close nuclei this gets wrong as contact interaction then plays a role.
        """
        loc_cart = co.Coord().coord_unit(location, 'cart')
        rho = co.Coord().coord(location, 'sph')['rho']
        x, y, z = [loc_cart[i] for i in ['x', 'y', 'z']]
        prefactor = mu_0 / (4.0 * pi) * h * self.gamma_e * 1e6 * self.gamma_13c * 1e6 / rho ** 3  # given in Hertz
        prefactor_mhz = prefactor * 1e-6  # given in MHz
        mat = numpy.matrix([[1 - 3 * x * x, -3 * x * y, -3 * x * z],
                            [-3 * x * y, 1 - 3 * y * y, -3 * y * z],
                            [-3 * x * z, -3 * y * z, 1 - 3 * z * z]])
        return prefactor_mhz * mat

    def h_hf(self, hft, nsd, nslvl_l):
        """
        Calculates the hyperfine hamiltonian between electron spin and a spin 'nsq'.
        'hft' is the hyperfine tensor for the interaction, e.g. calculated from hft_nitrogen or hft_13c or inserted freely
        'nsq' is the spin that is added, i.e. nsq = 1/2.0 for a 13c
        'nslvl_l' nuclear_spin_lvl_list (i guess that's what this variable name means)
        """
        h_hf = 0*qeye(np.append(self.h_nv.dims[0], len(nslvl_l)))
        eye_mat = qeye(dims=self.h_nv.dims[0][1:])  # unity matrix for spinsnot involved in HF
        for i in range(3):
            electron = get_sub_matrix(jmat(self.spins[0], co.Coord().cart_coord[i]), self.spin_levels[0])
            for j in range(3):
                new = get_sub_matrix(jmat(dim2spin(nsd), co.Coord().cart_coord[j]), nslvl_l)
                tmp = tensor(electron, eye_mat, new) if eye_mat is not None else tensor(electron, new)
                h_hf = h_hf + hft[i, j] * tmp
        return h_hf

    def set_h_nv(self):
        """
        calculates the base hamiltonian with parameters specified at instantiation
        """
        self.spins = np.array([1])
        self.spin_levels = [self.electron_levels]
        self.h_nv = get_sub_matrix(self.h_electron(), self.electron_levels)
        if self.n_type is not None:
            self.get_nitrogen_params()
            self.add_spin(hft=self.hft_nitrogen(), h_ns=self.h_nitrogen(), nslvl_l=self.nitrogen_levels)

    def add_spin(self, hft, h_ns, nslvl_l):
        """

        :param hft: np.array
            hyperfine tensor for the interaction between electron spin and the spin that should be added. might be taken
            from self.hft_nitrogen() or self.hft_13c_dd()
        :param h_ns: Qobj operator
            hamilton operator of the spin that should be added. ns stands for 'nsq' might be taken from
            self.h_nitrogen() or self.h_13c()
        :param nslvl_l: list
            spin levels of 'h_ns' that should be considered. For a 13C this will usually be [0,1].
        :return:
        """

        nsd = h_ns.shape[0]  # multiplicity of new spin
        nsq = qte.dim2spin(nsd)  # quantum number of new spin
        h_nv_ext = tensor(self.h_nv,
                          qeye(len(nslvl_l)))  # nv hamilton operator extended with qeye of dimension of new spin
        h_ns_ext = tensor(qte.qeye(dims=self.h_nv.dims[0]),
                          qte.get_sub_matrix(h_ns, nslvl_l))  # extended hamilton operator of the new spin
        h_hf = self.h_hf(hft, nsd, nslvl_l)  # new spin - electron hyperfine interaction tensor
        self.h_nv = h_nv_ext + h_ns_ext + h_hf
        self.spins = np.append(self.spins, nsq)
        self.spin_levels.append(nslvl_l)
        return self.h_nv

if __name__ == '__main__':
    nvham = NVHam(magnet_field={'z': 0.55}, n_type='n14', nitrogen_levels=[0, 1, 2], electron_levels=[0, 1, 2])
    C13_hyperfine_tensor = nvham.hft_13c_dd(location={'rho': 0.155e-9, 'elev': np.pi/2.})
    print C13_hyperfine_tensor
    # nvham.add_spin(C13_hyperfine_tensor, nvham.h_13c(), [0, 1])
    # print nvham.h_nv
    # a = nvham.h_hf(nvham.hft_nitrogen(), nvham.h_nitrogen().shape[0], nvham.nitrogen_levels)
    # C13_hyperfine_tensor = ham.hft_13c_dd(location={'rho': 0.458e-9, 'elev': pi/ 2., 'azim': 0.0})
    # ham.add_spin(C13_hyperfine_tensor, ham.h_13c(), [0, 1])
    # print ham.h_nv[0,0] - ham.h_nv[1,1]
    # print C13_hyperfine_tensor
    # C13_hamilton = ham.h_13c()
    # ham.add_spin(C13_hyperfine_tensor, C13_hamilton, nslvl_l=[0, 1])


    # print ham.hft_13c_dd(location={'rho': 1e-9, 'elev': np.pi / 2.})
    # # print ham.h_nv
    # print abs(ham.h_nv[0, 0] - ham.h_nv[1, 1]) - abs(ham.h_nv[2, 2] - ham.h_nv[3, 3])
    # print abs(ham.h_nv[4, 4] - ham.h_nv[5, 5]) - abs(ham.h_nv[2, 2] - ham.h_nv[3, 3])


    # ham = NVHam(magnet_field={'z': 0.52373915}, electron_levels=[0, 1, 2], n_type='n14', nitrogen_levels=[0, 1, 2])
    # print ham.h_nv
    # print ham.h_nv[7, 7] - ham.h_nv[6, 6]
    # print ham.h_nv[1, 1] - ham.h_nv[0, 0]
    # ham.add_spin(ham.hft_13c_dd(location = {'x': 1e-9}), ham.h_13c())
    # evals, evecs = ham.h_nv.eigenstates()
    #     print evals
    # ev = qutip_enhanced.Eigenvector(dims=np.array([4,3]))
    # mag_z_l = np.linspace(0, 0.1028, 30)
    # ham = NVHam(magnet_field = {'z':0.55}, n_type='14', electron_levels=[0,1,2])
    # print ham.h_nv
    #    for mag_z in mag_z_l:
    #        ham = NVHam(magnet_field = {'rho': mag_z, 'elev': np.pi/2.}, n_type = 'n14', electron_levels=[1,2], nitrogen_levels=[0, 1, 2])
    # #        ham.add_spin(ham.h_13c(), ham.hft_13c_dd())
    #        evals, evecs = ham.h_nv.eigenstates()
    #        ev.sort(evecs, evals)
    #    figure(1)
    #    colors = ['b', 'r', 'g']  # list of colors for plotting
    #    for n in range(prod(ev.dims)):
    #        plot(mag_z_l, (ev.evals_sorted_list[:, n]), lw=2)
    #    show()