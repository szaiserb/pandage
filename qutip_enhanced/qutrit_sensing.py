# coding=utf-8

import numpy as np
import lmfit

fdd_t = {'+': {}, '0': {}, '-': {}}

def plus_plus_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 1/9.*(1+2*np.cos(arg))**2

def zero_not_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 4/9.*(2+np.cos(arg))*np.sin(arg/2.)**2

def zero_plus_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 1/9.*(-1+ np.cos(arg) + np.sqrt(3)*np.sin(arg))**2

def zero_minus_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 1/9.*(-1+ np.cos(arg) - np.sqrt(3)*np.sin(arg))**2


def minus_plus_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 4/9.*np.sin(arg/2.)**2*(2+np.cos(arg)+np.sqrt(3)*np.sin(arg))


def minus_minus_t(x, Azz, x0):
    arg = 2*Azz*np.pi*(x-x0)
    return 4 / 9. * np.sin(arg / 2.) ** 2 * (2 + np.cos(arg) - np.sqrt(3) * np.sin(arg))

fdd_t['+']['not'] = plus_plus_t
fdd_t['+']['+'] = plus_plus_t
fdd_t['+']['-'] = plus_plus_t
fdd_t['0']['not'] = zero_not_t
fdd_t['0']['+'] = zero_plus_t
fdd_t['0']['-'] = zero_minus_t
fdd_t['-']['not'] = zero_not_t
fdd_t['-']['+'] = minus_plus_t
fdd_t['-']['-'] = plus_plus_t

def decay_matrix(t, t2):
    a = 1- 2/3.*(1-np.exp(-t/t2))
    b = 1/3.*(1-np.exp(-t/t2))
    return np.array(
        [
            [a, b, b],
            [b, a, b],
            [b, b, a]
        ]
    )

def readout_fidelity_matrix(frpp, frp0, frpm, fr0p, fr00, fr0m, frmp, frm0, frmm):
    return np.array(
        [
            [frpp, frp0, frpm],
            [fr0p, fr00, fr0m],
            [frmp, frm0, frmm],
        ]
    )
        # [
        #     [frp,           (1-fr0)/2., (1-frm)/2.],
        #     [(1-frp)/2.,    fr0,        (1-frm)/2.],
        #     [(1-frp)/2.,    (1-fr0)/2., frm]
        # ])

def gate_fidelity_matrix(fg):
    return np.array(
        [
            [fg,           (1-fg)/2., (1-fg)/2.],
            [(1-fg)/2.,    fg,        (1-fg)/2.],
            [(1-fg)/2.,    (1-fg)/2., fg]
        ])




def f(x, Azz, x0, pnv0, fg, frpp, frp0, frpm, fr0p, fr00, fr0m, frmp, frm0, frm, t2, mn):
    """
    NOTE: ALL variables (including x) must be scalars, can probably rewritten to take vectors
    """
    mfid_read = readout_fidelity_matrix(frpp, frp0, frpm, fr0p, fr00, fr0m, frmp, frm0, frm)
    mt2 = decay_matrix(x, t2)
    mfid_gate = gate_fidelity_matrix(fg)
    p0 = np.array([fdd_t[i][mn](x, Azz, x0) for i in ['+', '0', '-']])
    p1 = np.dot(mfid_gate, p0)
    p2 = np.dot(mt2, p1)
    return np.dot(mfid_read, (1 - pnv0) * p2 + pnv0 * np.array([1, 0, 0]))

class QutritSensingModel(lmfit.Model):
    def __init__(self,  mn,  n14_state, *args, **kwargs):
        n14_sn = {'+1': 0, '0': 1, '-1': 2}[n14_state]
        def the_fun(x, Azz, x0, pnv0, fg, frpp, frp0, frpm, fr0p, fr00, fr0m, frmp, frm0, frm, t2):
            return [f(xi, Azz, x0, pnv0, fg, frpp, frp0, frpm, fr0p, fr00, fr0m, frmp, frm0, frm, t2, mn)[n14_sn] for xi in x]
        super(QutritSensingModel, self).__init__(the_fun, *args, **kwargs)

class QutritSensingModelAll(lmfit.Model):
    def fit_function(params, x=None, dat1=None, dat2=None):
        model1 = params['offset'] + x * params['slope1']
        model2 = params['offset'] + x * params['slope2']

        resid1 = dat1 - model1
        resid2 = dat2 - model2
        return np.concatenate((resid1, resid2))

class Mod(lmfit.Model):
    def fit_function(self, x=None, data=None):
        return np.array([1, 2, 3])

    def residuals(self):
        raise Exception('HI')