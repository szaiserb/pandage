# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import numpy as np
import itertools
import lmfit.lineshapes
from .qutip_enhanced import *

SDx = jmat(.5, 'x')  # np.array([[0.0, 1.0], [1.0, 0.0]], dtype='complex_') / 2.
SDy = jmat(.5, 'y')  # np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype='complex_') / 2.
SDz = jmat(.5, 'z')  # np.array([[1.0, 0.0], [0.0, -1.0]], dtype='complex_') / 2.
STx = jmat(1., 'x')  # np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype='complex_') / np.sqrt(2)
STy = jmat(1., 'y')  # np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype='complex_') / np.sqrt(2)
STz = jmat(1., 'z')  # np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype='complex_')

def f_drift(x, fwhm, center=0.0, drift_width=0.0, f='lorentzian'):
    if f == 'lorentzian':
        sigma = fwhm / 2.
    elif f == 'gaussian':
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    fun = getattr(lmfit.lineshapes, f)
    out = []
    for xi in x:
        if -drift_width / 2. <= xi <= drift_width / 2.:
            x = 0.0
        elif xi < drift_width / 2.:
            x = xi + drift_width / 2.
        elif xi > drift_width / 2.:
            x = xi - drift_width / 2.
        out.append(fun(x=x, sigma=sigma, center=center))
    return out


def ensemble(base, weight_base=None):
    """
    NOTE: For documentation see qutip_enhanced examples
    
        :param base: dict with keys ['det', 'ps']
    
        - base['det'] = [nparr_0, nparr_1, nparr_2, ...] for controls 0,1,2.. Each numpy array arr_i includes the detunings for one class of controls and has len(shape) = 1, e.g. mw on a specific transition or nuclear spin driving of a specific nuclear transition (e.g. 13C90 mS-1)
            Examples: 
                - [0.0]
                - [-0.2, -0.1, 0.0, 0.1, 0.2]
               
        base['ps'] = [nparr_a, nparr_b, nparr_c, ...], for controls 0, 1, 2 where nparr_i.shape = (number_of_powerscalings, number_of_connected_controls)
            Examples:
                - base['ps'] = [np.array([[1.]])] for a z-control, no powerscaling
                - base['ps'] = [np.array([[1., 1.]])] for x,y controls, no powerscaling
                - base['ps'] = [np.array([[.9, .9], [1., 1.], [1.1,1.1]])] for +- 10% powerscaling
    
    :param weight_base: dict with keys ['det', 'ps'] or None
    
        - weight_base['det'] 
            
            exactly the same form as base['det']
            
            Examples: 
                - base['det'] = [[0.0], [-0.1, 0.0, 0.1]] --> weight_base['det'] = [[1.0], [0.5, 1.0, 0.5]]
            
        - weight_base['ps']
         
            - len(weight_base['ps']) = len(base['ps'])
            - weight_base['ps'][i].shape = base['ps'].shape[0]
        
            Examples:
                - base['ps'] = [np.array([[1.]])] -> weight['ps'] = [np.array([[1.]])]
                - base['det']  = np.array([[1., 1.]]) -> weight['ps'] = [np.array([[1.]])]
                - base['det']  = np.array([[.9, .9], [1., 1.], [1.1,1.1]]) -> weight['ps'] = [np.array([[0.5, 1.0, 0.5]])]

    """
    for key, val in base.items():
        if type(val) != list:
            raise Exception(key, type(val), val)
    if not all(len(i.shape) == 1 for i in base['det']):
        raise Exception(base['det'])
    if not all([len(i.shape) == 2 for i in base['ps']]):
        raise Exception(base['ps'])
    if weight_base is None:
        weight_base = {}
        for key in base:
            weight_base[key] = []
            for idx, item in enumerate(base[key]):

                weight_base[key].append(np.ones(base[key][idx].shape[0]))

    # What this loop does:
    #   - extend weights that are [] by equal weighting
    #   - extends the weights of powerscalings to shape (n,2) such that form of base and weight_base are equal and can then be processed more easily
    #   - normalizes the weights such, that among each subweight (e.g. for electron detunings [-0.1, 0.0, 0.1] the weights are [.123, 1.0, .123])
    for key, val in weight_base.items():
        for i, item in enumerate(val):
            if key == 'det':
                if len(item) == 0:
                    weight_base[key][i] = np.ones(len(base[key][i]))
                elif len(item) != len(base[key][i]):
                    raise Exception("Error: {}, {}".format(base[key], weight_base[key]))
            elif key == 'ps':
                if len(item) == 0:
                    weight_base[key][i] = np.ones(base[key][i].shape[0])
                elif len(item) != len(base[key][i]):
                    raise Exception("Error: {}, {}".format(base[key], weight_base[key]))
                # weight_base[key][i] = np.repeat(np.array(weight_base[key][i]).reshape(-1, 1), base[key][i].shape[1], axis=1)

        weight_base[key] = [np.array(i) / np.max(i) for i in weight_base[key]]

    ps_base = [np.hstack(i) for i in list(itertools.product(*base['ps']))]
    det_base = list(itertools.product(*base['det']))
    det = np.array(list(itertools.chain(*[(i,) * len(ps_base) for i in det_base])))
    ps = np.tile(ps_base, (len(det_base), 1))

    weight_ps_base = [np.hstack(i) for i in list(itertools.product(*weight_base['ps']))]
    weight_det_base = np.array(list(itertools.product(*weight_base['det'])))
    weight_det = np.array(list(itertools.chain(*[(i,) * len(weight_ps_base) for i in weight_det_base])))
    weight_ps = np.tile(weight_ps_base, (len(weight_det_base), 1))
    weight = np.prod(np.concatenate([weight_det, weight_ps], axis=1), axis=1)
    weight = weight / np.sum(weight)

    return det, ps, weight


if __name__ == '__main__':

    # electron_ps = [.995, 1.0, 1.005]
    # nuclear_ps = [.9, 1.0, 1.1]
    # base = {'det': [np.linspace(-0.1, 0.1, 5), np.linspace(-50e-6, 50e-6, 5)],
    #         'ps': [np.array([[e, e] for e in electron_ps]), np.array([[n, n] for n in nuclear_ps]), np.array([[1.0]*4])]}
    # weight_base = {'det': [f_drift(base['det'][0], fwhm=40e-3, drift_width=10e-3), f_drift(base['det'][1], fwhm=50e-6)],
    #                # 'ps': [f_drift(electron_ps, fwhm=0.0025), f_drift(nuclear_ps, fwhm=0.0025), []]}
    #                'ps': [f_drift(electron_ps, fwhm=0.01, center=1.0), f_drift(nuclear_ps, fwhm=0.01, center=1.0), []]}

    base = {'det':  [np.array([-.1, 0.0, .1])],
            'ps': [np.array([[ps, ps] for ps in [.9, 1., 1.1]])]}
    weight_base = None

    print(ensemble(base))

