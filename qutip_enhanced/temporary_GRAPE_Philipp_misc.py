# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals, division
__metaclass__ = type

import numpy as np
import itertools

SDx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype='complex_')/2.
SDy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype='complex_')/2.
SDz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype='complex_')/2.
STx = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype='complex_')/np.sqrt(2)
STy = np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype='complex_')/np.sqrt(2)
STz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype='complex_')

def xy2aphi(xy):
    norm = np.array([np.linalg.norm(xy, axis=1)]).transpose()
    phi = np.arctan2(xy[:, 1:2], xy[:,0:1])
    return np.concatenate([norm, phi], axis=1)

def aphi2xy(aphi):
    return np.array([aphi[:, 0] * np.cos(aphi[:, 1]),
                     aphi[:, 0] * np.sin(aphi[:, 1])]).T

def lorentz(x, g, x0):
    """Lorentzian centered at mu, with amplitude a, offset y0 and HWHM g."""
    return np.pi * (abs(g) / ((x - x0) ** 2 + g ** 2))

def f_drift(d, sigma, mu, drift_width, f='lorentz'):
    fun = globals()[f]
    out = []
    for dd in d:
        if -drift_width/2. <= dd <= drift_width/2.:
            out.append(fun(0.0, sigma, mu))
        elif dd < drift_width/2.:
            out.append(fun(dd + drift_width/2., sigma, mu))
        elif dd > drift_width/2.:
            out.append(fun(dd - drift_width / 2., sigma, mu))
    return out

def ensemble(detunings_base, p_scales_base, weight_p_scales_base, weight_detunings_base):
    """

    :param detunings_base: list
        example:
        [[0.1, 0.0, -0.1], [-10, 0., 20]]

    :param p_scales_base: array
        [np.array([[e, e] for e in electron_ps]), np.array([[n,n] for n in nuclear_ps]), [1.0]*3], for electron and nuclear controls (x,y respectively) and three additional controls

    :param weight_base: {'detunings': [[1.0, 1.0, 1.0], [0.25, 1.0, 0.25]],
                         'p_scales': [[gaussian(electron_ps), gaussian(nuclear_ps], [1.0]*3]}
    :return:
    """
    if not all([len(i.shape) == 2 for i in p_scales_base]):
        raise Exception(p_scales_base)

    p_scales_base = [np.hstack(i) for i in list(itertools.product(*p_scales_base))]
    detunings_base = list(itertools.product(*detunings_base))
    detunings = list(itertools.chain(*[(i,) * len(p_scales_base) for i in detunings_base]))

    p_scales = np.tile(p_scales_base, (len(detunings_base), 1))
    weight_p_scales_base = [np.array(i)/sum(i) for i in weight_p_scales_base]
    weight_detunings_base = [np.array(i)/sum(i) for i in weight_detunings_base]

    weight_p_scales_base = [np.hstack(i) for i in list(itertools.product(*weight_p_scales_base))]
    weight_detunings_base = list(itertools.product(*weight_detunings_base))
    weight_detunings = list(itertools.chain(*[(i,) * len(weight_p_scales_base) for i in weight_detunings_base]))
    weight_p_scales = np.tile(weight_p_scales_base, (len(weight_detunings_base), 1))
    weight = np.prod(np.concatenate([weight_detunings, weight_p_scales], axis=1), axis=1)
    weight = weight / np.sum(weight)

    return detunings, p_scales, weight

# if __name__ == '__main__':
#
#     electron_detunings_base = np.linspace(-0.25, 0.25, 5)
#     electron_ps = np.linspace(0.995, 1.005, 3)
#     p_scales_base = [np.array([[e, e] for e in electron_ps]), np.array([[1.0] * 3])]
#     detunings_base = [electron_detunings_base]
#     weight_p_scales_base = [f_drift(electron_ps, sigma=0.01, mu=1.0, drift_width=0.00), np.array([[1.0] * 3])]
#     weight_detunings_base = [f_drift(electron_detunings_base, sigma=0.02, mu=0.0, drift_width=0.02)]
#     weight_p_scales_base = [np.array(i)/sum(i) for i in weight_p_scales_base]
#     weight_detunings_base = [np.array(i)/sum(i) for i in weight_detunings_base]
#
#     p_scales_base = [np.hstack(i) for i in list(itertools.product(*p_scales_base))]
#     detunings_base = list(itertools.product(*detunings_base))
#     detunings = list(itertools.chain(*[(i,) * len(p_scales_base) for i in detunings_base]))
#     p_scales = np.tile(p_scales_base, (len(detunings_base), 1))
#
#     weight_p_scales_base = [np.hstack(i) for i in list(itertools.product(*weight_p_scales_base))]
#     weight_detunings_base = list(itertools.product(*weight_detunings_base))
#
#     weight_detunings = list(itertools.chain(*[(i,) * len(weight_p_scales_base) for i in weight_detunings_base]))
#     weight_p_scales = np.tile(weight_p_scales_base, (len(weight_detunings_base), 1))
#     weight = np.prod(np.concatenate([weight_detunings, weight_p_scales], axis=1), axis=1)
#     weight = weight / np.sum(weight)
