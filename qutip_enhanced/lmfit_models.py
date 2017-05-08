# coding=utf-8
from __future__ import print_function, absolute_import, division
from imp import reload
__metaclass__ = type


import numpy as np
import lmfit
import lmfit.models
import itertools


class CosineNoOffsetNoDecayModel(lmfit.Model):
    def __init__(self, *args, **kwargs):
        def cosine(x, amplitude, T, x0):
            return amplitude * np.cos(2 * np.pi * (x - x0) / float(T))

        super(CosineNoOffsetNoDecayModel, self).__init__(cosine, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        def CosineNoOffsetNoDecayEstimator(y=None, x=None):
            a = 2**0.5 * np.sqrt( (y**2).sum() )
            # better to do estimation of period from
            Y = np.fft.fft(y)
            N = len(Y)
            D = float(x[1] - x[0])
            i = abs(Y[1:int(N/2+1)]).argmax()+1
            T = (N * D) / i
            x0 = 0
            return a, T, x0
        amplitude, T, x0 = CosineNoOffsetNoDecayEstimator(y=data, x=x)
        if amplitude < 0:
            amplitude *= -1
            x0 = ((x0 / T + 0.5) % 1) * T
        return lmfit.models.update_param_vals(self.make_params(amplitude=amplitude, T=T, x0=x0), self.prefix, **kwargs)

class CosineNoDecayModel(lmfit.Model):
    def __init__(self, *args, **kwargs):
        def cosine(x, amplitude, T, x0, c):
            return amplitude * np.cos(2 * np.pi * (x - x0) / float(T)) + c

        super(CosineNoDecayModel, self).__init__(cosine, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        c = data.mean()
        y_temp = data - c
        m = CosineNoOffsetNoDecayModel()
        p = m.fit(y_temp, m.guess(data=y_temp, x=x), x=x).result.params
        if p['amplitude'] < 0:
            p['amplitude'].value *= -1
            p['x0'].value = ((p['x0'] / p['T'] + 0.5) % 1) * p['T']
            p = m.fit(y_temp, m.guess(data=y_temp, x=x), x=x).result.params
        return lmfit.models.update_param_vals(self.make_params(amplitude=p['amplitude'].value, T=p['T'].value, x0=p['x0'].value, c=c), self.prefix, **kwargs)

class CosineModel(lmfit.Model):
    def __init__(self, *args, **kwargs):

        def cosine(x, amplitude, T, x0, c, t2):
            return amplitude * np.cos(2 * np.pi * (x - x0) / float(T)) * np.exp(-(x - x0) / t2) + c

        super(CosineModel, self).__init__(cosine, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        p = CosineNoDecayModel().guess(data, x=x)
        if p['amplitude'] < 0:
            p['amplitude'].value *= -1
            p['x0'].value = ((p['x0'] / p['T'] + 0.5) % 1) * p['T']
        return lmfit.models.update_param_vals(self.make_params(amplitude=p['amplitude'].value, T=p['T'].value, x0=p['x0'].value, c=p['c'].value, t2=10 * max(x)), self.prefix, **kwargs)


class CosineMultiDetModel(lmfit.Model):
    def __init__(self, hyperfine_list, *args, **kwargs):

        self.hyperfine_list = hyperfine_list

        def cosine_multi_det_lmfit(x, amplitude, T, x0, c, t2, f0):
            hfsl = [[+self.hyperfine_list[0], 0.0, -self.hyperfine_list[0]]]
            for hf in self.hyperfine_list[1:]:
                hfsl.append([-hf / 2., hf / 2.])
            delta_l = [sum(i) for i in itertools.product(*hfsl)]
            rabi_0 = 1 / float(T)

            def rabi(delta, x):
                rabi_eff = np.sqrt(rabi_0 ** 2 + delta ** 2)
                A = rabi_0 ** 2 / (rabi_0 ** 2 + delta ** 2)
                return A * np.cos(2 * np.pi * rabi_eff * (x - x0)) * np.exp(-(x - x0) / t2)

            return amplitude * sum(rabi(d - f0, x) for d in delta_l) + c

        super(CosineMultiDetModel, self).__init__(cosine_multi_det_lmfit, *args, **kwargs)

if __name__ == '__main__':
    fp = "D:\data/NuclearOPs\CNOT_KDD\cnot_kdd/20170411-h13m43s31_cnot_kdd/20170411-h14m04s25_kdd_data.hdf"

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    import lmfit

    data = pd.read_hdf(fp)

    d = data.groupby(['seq_num', 'x']).agg({'result_0': np.mean}).reset_index([0, 1])
    aggd = d.pivot(columns='seq_num', index='x')
    fig, ax = plt.subplots()
    aggd.plot(ax=ax, legend=False)
    ax.set_title("Rabi_oscillations (x-axis scaling wrong)")

    mod = CosineModel()
    y = np.array(d[d.seq_num == int(0)].result_0)
    x = aggd.index.values
    pars = mod.guess(data=y, x=x)
    pars['T'].value = 2*np.pi
    pars['T'].vary = False
    result = mod.fit(y, pars, x=x)
    plt.plot(x, result.best_fit)
    plt.title("Fit of rabi_oscillations (x-axis scaling right)")
    a = np.array([0, result.params['T'].value, result.params['x0'].value])