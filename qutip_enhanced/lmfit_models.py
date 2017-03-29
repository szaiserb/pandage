# coding=utf-8

from __future__ import print_function, absolute_import, unicode_literals, division
from imp import reload


import numpy as np
import lmfit
import lmfit.models
import itertools

class CosineModel(lmfit.Model):
    def __init__(self, *args, **kwargs):

        def cosine(x, amplitude, T, x0, c, t2):
            return amplitude * np.cos(2 * np.pi * (x - x0) / float(T)) * np.exp(-(x - x0) / t2) + c

        super(CosineModel, self).__init__(cosine, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        def CosinusEstimator(x, y):
            c = y.mean()
            amplitude = 2 ** 0.5 * np.sqrt(((y - c) ** 2).sum())
            # better to do estimation of period from
            Y = np.fft.fft(y)
            N = len(Y)
            D = float(x[1] - x[0])
            i = abs(Y[1:N / 2 + 1]).argmax() + 1
            T = (N * D) / i
            x0 = 0
            return amplitude, T, x0, c
        amplitude, T, x0, c = CosinusEstimator(x, data)
        t2 = 10 * max(x)
        return lmfit.models.update_param_vals(self.make_params(amplitude=amplitude, T=T, x0=x0, c=c, t2=t2), self.prefix, **kwargs)

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

    # def guess(self, data, x=None, **kwargs):
    #     def CosinusEstimator(x, y):
    #         c = y.mean()
    #         a = 2 ** 0.5 * np.sqrt(((y - c) ** 2).sum())
    #         # better to do estimation of period from
    #         Y = np.fft.fft(y)
    #         N = len(Y)
    #         D = float(x[1] - x[0])
    #         i = abs(Y[1:N / 2 + 1]).argmax() + 1
    #         T = (N * D) / i
    #         x0 = 0
    #         return a, T, x0, c
    #
    #     a, T, x0, c = CosinusEstimator(x, data)
    #     t2 = 10 * max(x)
    #     return lmfit.models.update_param_vals(self.make_params(a=a, T=T, x0=x0, c=c, t2=t2), self.prefix, **kwargs)
    #
    # def guess2(self, data, x=None, **kwargs):
    #     y_offset = data.mean()
    #     y = data - y_offset
    #     try:
    #         p = pi3d.Fit.Fit(x, y, pi3d.Fit.CosinusNoOffset, pi3d.Fit.CosinusNoOffsetEstimator)
    #     except:
    #         return None
    #     if p[0] < 0:
    #         p[0] = -p[0]
    #         p[2] = ((p[2] / p[1] + 0.5) % 1) * p[1]
    #         p = pi3d.Fit.Fit(x, y, pi3d.Fit.CosinusNoOffset, p)
    #     p = (p[0], p[1], p[2], y_offset)
    #     p = pi3d.Fit.Fit(x, self.results, pi3d.Fit.Cosinus, p)
    #     while (p[2] > 0.5 * p[1]):
    #         p[2] -= p[1]
    #     p = pi3d.Fit.Fit(x, self.results, pi3d.Fit.Cosinus, p)
    #     p = list(p)
    #     p.append(10 * max(x))
    #     p = pi3d.Fit.Fit(x, self.results, pi3d.Fit.Cosinus_dec, p)
    #     pp = list(p)
    #     pp.append(self.centerfreq)
    #     import Fitmodule
    #     d = dict(zip(['a', 'T', 'x0', 'c', 't2', 'f0'], pp))
    #     fitresult, fitparams = Fitmodule.Fit(x, self.results, pi3d.Fit.CosineMultiDetLmFit, d)

if __name__ == '__main__':
    mod = CosineModel()
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    pars = mod.guess(y, x)
    a = mod.fit(y, pars, x=x)
    import matplotlib.pyplot as plt
    a.plot_fit()