# coding=utf-8
"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""
import numpy as np
import lmfit
import lmfit.models

class CosineModel(lmfit.Model):
    def __init__(self, *args, **kwargs):

        def cosine(x, a, T, x0, c, t2):
            return a * np.cos(2 * np.pi * (x - x0) / float(T)) * np.exp(-(x - x0) / t2) + c

        super(CosineModel, self).__init__(cosine, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        def CosinusEstimator(x, y):
            c = y.mean()
            a = 2 ** 0.5 * np.sqrt(((y - c) ** 2).sum())
            # better to do estimation of period from
            Y = np.fft.fft(y)
            N = len(Y)
            D = float(x[1] - x[0])
            i = abs(Y[1:N / 2 + 1]).argmax() + 1
            T = (N * D) / i
            x0 = 0
            return a, T, x0, c
        a, T, x0, c = CosinusEstimator(x, data)
        t2 = 10 * max(x)
        return lmfit.models.update_param_vals(self.make_params(a=a, T=T, x0=x0, c=c, t2=t2), self.prefix, **kwargs)


    #     def cosine(x, a, T, x0, c):
    #         """Returns a Cosinus function with the given parameters"""
    #         return a * np.cos(2 * np.pi * (x - x0) / float(T)) + c
    #
    #     super(CosineModel, self).__init__(cosine, *args, **kwargs)
    #
    # def guess(self, data, x=None, **kwargs):
    #     c = data.mean()
    #     a = 2 ** 0.5 * np.sqrt(((data - c) ** 2).sum())
    #     # better to do estimation of period from
    #     Y = np.fft.fft(data)
    #     N = len(Y)
    #     D = float(x[1] - x[0])
    #     i = abs(Y[1:N / 2 + 1]).argmax() + 1
    #     T = (N * D) / i
    #     x0 = 0
    #     return lmfit.models.update_param_vals(self.make_params(a=a, T=T, x0=x0, c=c), self.prefix, **kwargs)



if __name__ == '__main__':
    mod = CosineModel()
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    pars = mod.guess(y, x)
    a = mod.fit(y, pars, x=x)
    import matplotlib.pyplot as plt
    a.plot_fit()

#
# def Cosinus_dec_nophase(x, a, T, c, t2):
#     return a * np.cos(2 * np.pi * (x) / float(T)) * np.exp(-(x) / t2) + c
#
# def Cosinus_dec_nophase_estimator(x, y):
#     a, T, x0, c = cosine_estimator(x, y)
#     t2 = 10 * max(x)
#     return a, T, c, t2
#
# def cosinus_nophase(x, amp, per, c):
#     """Returns a Cosinus function with phase = 0."""
#     return amp * np.cos(2 * np.pi * np.array(x) / float(per)) + c
#
#
# def cosinus_nophase_estimator(x, y):
#     x = np.array(x)
#     y = np.array(y)
#     c = y.mean()
#     a = y.max() - y.min()
#     if y[0] <= c: a = -a
#     # better to do estimation of period from
#     Y = np.fft.fft(y)
#     N = len(Y)
#     D = float(x[1] - x[0])
#     i = abs(Y[1:N / 2 + 1]).argmax() + 1
#     T = (N * D) / i
#     return a, T, c
#
#
# def cosine_no_offset(x, a, T, x0):
#     """Returns a Cosinus function with the given parameters"""
#     return a * np.cos(2 * np.pi * (x - x0) / float(T))
#
# def cosine_no_offset_estimator(x, y):
#     a = 2 ** 0.5 * np.sqrt((y ** 2).sum())
#     # better to do estimation of period from
#     Y = np.fft.fft(y)
#     N = len(Y)
#     D = float(x[1] - x[0])
#     i = abs(Y[1:N / 2 + 1]).argmax() + 1
#     T = (N * D) / i
#     x0 = 0
#     return a, T, x0
#
# def cosine_multi_det(x, a, T, x0, c, t2, f0):
#     try:
#         from pi3diamond import pi3d
#         tt = pi3d.tt
#         hfl = []
#         for nuc in ['14N', '13C414', '13C90']:
#             hfl.append(tt.get_f(nuc + '_hf'))
#     except:
#         hfl = [2.16, 0., 0.089]
#     # hfl.append(0.0123)
#
#     hfsl = [[+hfl[0], 0.0, -hfl[0]]]
#     for hf in hfl[1:]:
#         hfsl.append([-hf / 2., hf / 2.])
#     delta_l = [sum(i) for i in itertools.product(*hfsl)]
#     rabi_0 = 1 / float(T)
#
#     def rabi(delta, x):
#         rabi_eff = np.sqrt(rabi_0 ** 2 + delta ** 2)
#         A = rabi_0 ** 2 / (rabi_0 ** 2 + delta ** 2)
#         return A * np.cos(2 * np.pi * rabi_eff * (x - x0)) * np.exp(-(x - x0) / t2)
#
#     return a * sum(rabi(d - f0, x) for d in delta_l) + c
# #
#
# # def cosine_multi_det_lmfit(x, a, T, x0, c, t2, f0):
# #     try:
# #         from pi3diamond import pi3d
# #         tt = pi3d.tt
# #         hfl = []
# #         for nuc in ['14N', '13C414', '13C90']:
# #             hfl.append(tt.get_f(nuc + '_hf'))
# #     except:
# #         hfl = [2.16, 0.413, 0.089]
# #     # hfl.append(0.0123)
# #
# #     hfsl = [[+hfl[0], 0.0, -hfl[0]]]
# #     for hf in hfl[1:]:
# #         hfsl.append([-hf / 2., hf / 2.])
# #     delta_l = [sum(i) for i in itertools.product(*hfsl)]
# #     rabi_0 = 1 / float(T)
# #
# #     def rabi(delta, x):
# #         rabi_eff = np.sqrt(rabi_0 ** 2 + delta ** 2)
# #         A = rabi_0 ** 2 / (rabi_0 ** 2 + delta ** 2)
# #         return A * np.cos(2 * np.pi * rabi_eff * (x - x0)) * np.exp(-(x - x0) / t2)
# #
# #     return a * sum(rabi(d - f0, x) for d in delta_l) + c
#
#
# def brot_transitions_upper(B, D, E, phase):
#     return lambda theta: 3. / 2. * B ** 2 / D * np.sin(theta + phase) ** 2 + (B ** 2 * np.cos(theta + phase) ** 2 + (E + B ** 2 / (2 * D) * np.sin(theta + phase) ** 2) ** 2) ** 0.5 + D
#
#
# def brot_transitions_lower(B, D, E, phase):
#     return lambda theta: 3. / 2. * B ** 2 / D * np.sin(theta + phase) ** 2 - (B ** 2 * np.cos(theta + phase) ** 2 + (E + B ** 2 / (2 * D) * np.sin(theta + phase) ** 2) ** 2) ** 0.5 + D
#
#
# def FCSTranslationRotation(alpha, tau_r, tau_t, N):
#     """Fluorescence Correlation Spectroscopy. g(2) accounting for translational and rotational diffusion."""
#     return lambda t: (1 + alpha * np.exp(-t / tau_r)) / (N * (1 + t / tau_t))
#
# def FCSTranslation(tau, N):
#     """Fluorescence Correlation Spectroscopy. g(2) accounting for translational diffusion."""
#     return lambda t: 1. / (N * (1 + t / tau))
#
# def Antibunching(alpha, c, tau, t0):
#     """Antibunching. g(2) accounting for Poissonian background."""
#     return lambda t: c * (1 - alpha * np.exp(-(t - t0) / tau))
#
# def gaussian(x, mu, sigma, x_area_range=None, area=1):
#
#     """
#     definition https://en.wikipedia.org/wiki/Gaussian_function
#
#     area is area under gaussian curve
#
#     """
#     a = area / (sigma * np.sqrt(2 * np.pi))
#     def f(x):
#         return  a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
#     area_range = scipy.integrate.quad(lambda xr: f(xr), x_area_range[0], x_area_range[1])[0]
#     return f(x)*area/area_range
#
# def gaussian_old(x, c, a, x0, w):
#     """Gaussian function with offset."""
#     return c + a * np.exp(-0.5 * ((x - x0) / w) ** 2)
#
# def double_gaussian(x, a1, a2, x01, x02, w1, w2):
#     """Gaussian function with offset."""
#     return a1 * np.exp(-0.5 * ((x - x01) / w1) ** 2) + a2 * np.exp(-0.5 * ((x - x02) / w2) ** 2)
#
# def double_gaussian_estimator(x, y):
#     center = (x * y).sum() / y.sum()
#     ylow = y[x < center]
#     yhigh = y[x > center]
#     x01 = x[ylow.argmax()]
#     x02 = x[len(ylow) + yhigh.argmax()]
#     a1 = ylow.max()
#     a2 = yhigh.max()
#     w1 = w2 = center ** 0.5
#     return a1, a2, x01, x02, w1, w2
#
#
# # important note: lorentzian can also be parametrized with an a' instead of a,
# # such that a' is directly related to the amplitude (a'=f(x=x0)). In this case a'=a/(pi*g)
# # and f = a * g**2 / ( (x-x0)**2 + g**2 ) + c.
# # However, this results in much poorer fitting success. Probably the g**2 in the numerator
# # causes problems in Levenberg-Marquardt algorithm when derivatives
# # w.r.t the parameters are evaluated. Therefore it is strongly recommended
# # to stick to the parametrization given below.
# def lorentzian(x, x0, g, a, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     return a / np.pi * (g / ((x - x0) ** 2 + g ** 2)) + c
#
#

# def lorentzian_estimator(x, y):
#     c = scipy.stats.mode(y)[0][0]
#     yp = y - c
#     Y = np.sum(yp) * (x[-1] - x[0]) / len(x)
#     ymin = yp.min()
#     ymax = yp.max()
#     if ymax > abs(ymin):
#         y0 = ymax
#     else:
#         y0 = ymin
#     x0 = x[y.argmin()]
#     g = Y / (np.pi * y0)
#     a = y0 * np.pi * g
#     return x0, g, a, c
#
#
# def lorentzian_neg(x, x0, g, a, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     return -abs(a) / np.pi * (abs(g) / ((x - x0) ** 2 + g ** 2)) + c
#
# def lorentzian_estimator_neg(x, y):
#     # c = scipy.stats.mode(y)[0][0]
#     c = scipy.mean(y)
#     yp = np.array(y - c)
#     c = scipy.mean(c + abs(yp))
#     yp = y - c
#     Y = np.sum(yp) * (x[-1] - x[0]) / len(x)
#     y0 = yp.min()
#     x0 = x[y.argmin()]
#     g = Y / (np.pi * y0)
#     a = y0 * np.pi * g
#     return x0, g, a, c
#
#
# def lorentzian_pos(x0, g, a, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     return abs(a) / np.pi * (g / ((x - x0) ** 2 + g ** 2)) + c
#
# def lorentzian_estimator_pos(x, y):
#     """x0 = shift, g = ?, a = amplitude, c = offset """
#     c = scipy.stats.mode(y)[0][0]
#     yp = y - c
#     Y = np.sum(yp) * (x[-1] - x[0]) / len(x)
#     y0 = yp.max()
#     # ymin = yp.min()
#     # ymax = yp.max()
#     # if ymax > abs(ymin):
#     #    y0 = ymax
#     # else:
#     #    y0 = ymin
#     x0 = x[y.argmax()]
#     g = Y / (np.pi * y0)
#     a = y0 * np.pi * g
#     return x0, g, a, c
#
#
# def lorentzian_neg_sum(p, args):
#     """args frequencies relative to fitting frequency, i.e. args = [0, 1, 2]
#     p are fitting params, ordered as p = [freq, width, c, amp1, amp2, ..., amp_n]"""
#
#     def result(x):
#         y = np.zeros(len(x))
#         y += p[2]
#         for i in range(len(args)):
#             y += -abs(p[i + 3]) / np.pi * (p[1] ** 2 / ((x - (p[0] + args[i])) ** 2 + p[1] ** 2))
#         return y
#
#     return result
#
#
# def lorentzian_neg_sum_estimator(x, y, args):
#     dx = abs(x[1] - x[0])
#     index = []
#     for f in args:
#         index.append(int(round(f * 1. / dx)))
#     trip_mean = []
#     for i in range(1, len(y) - index[-1] - 1):
#         m = 0
#         for j in range(len(index)):
#             m += (y[i + index[j] - 1] + y[i + index[j]] + y[i + index[j] + 1]) * 1. / 3
#         m = m * 1. / len(index)
#         trip_mean.append(m)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x0 = x[trip_mean.argmin()]
#     g = 0.01 * 1e6
#     est = [x0, g, c]
#     for i in range(len(args)):
#         est.append((trip_mean.min() - c) * np.pi)
#     return est
#
#
# def Lorentz_sum_13C(x1, x2, g, a, c):
#     """Sum of two negative Lorentz, with same width and amplitude, but arbitrary spilling."""
#     return -abs(a) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) + c
#
#
# def Lorentz_sum_13C_estimator(x, y):
#     y_sorted = np.sort(y)
#     y_max = np.mean(y_sorted[-5:])
#     y_min = np.mean(y_sorted[:5])
#     median = (y_max + y_min) * 1. / 2
#     y_avg = []
#     for i in range(2, len(y) - 2):
#         y_avg.append(np.mean(y[i - 2:i + 2]))
#     x_avg = x[2:-2]
#     below = []
#     for val in y_avg:
#         below.append(int(val < median))
#     for i in range(len(below)):
#         if below[i] and below[i + 1]:
#             x1 = x_avg[i]
#             break
#     for i in range(len(below)):
#         if below[-i] and below[-i - 1]:
#             x2 = x_avg[-i]
#             break
#     c = y_max
#     g = 0.01 * 1e6
#     a = (y_min - c) * np.pi
#     return x1, x2, g, a, c
#
#
# def Lorentz_arbitrary_sum(p, args=None):
#     """Sum of 2*n number of Lorentz (n = number of 13Cs).
#     parameters p are ordered as [f0, a, g, c, hf1, hf2, ...], where
#     f0 = center freq., a = float, the amplitude(s), g = float, the linewidth, c = float offset,
#     and hf are hyperfine splittings."""
#     f0, a, g, c = p[:4]
#     hf = p[4:]
#     n = len(hf)
#     freqs = np.zeros(2 ** n) + f0
#     for i in range(2 ** n):
#         for j in range(n):
#             freqs[i] += (-1) ** int(i / 2 ** j) * hf[j] / 2.
#             # if isinstance(a, float) or isinstance(a, int):
#
#     # a = np.zeros(len(f)) + a
#     # if isinstance(g, float) or isinstance(g, int):
#     #    g = np.zeros(len(f)) + g
#     def result(x):
#         y = np.zeros(len(x))
#         y += c
#         for i in range(len(freqs)):
#             y += -abs(a) / np.pi * (g ** 2 / ((x - freqs[i]) ** 2 + g ** 2))
#         return y
#
#     return result
#
#
# def Lorentz_arbitrary_sum_estimator(x, y, args=None):
#     """n = number of 13C, can work for n = 1, 2.
#     Works better if peaks are given by args = [center_freq, splitting1, splitting2, etc]."""
#     y_sorted = np.sort(y)
#     y_max = np.mean(y_sorted[-5:])
#     c = y_max
#     y_min = np.mean(y_sorted[:5])
#     a = (y_min - c) * np.pi
#     g = 0.01 * 1e6
#     median = (2 * y_max + 1 * y_min) * 1. / 3  # median slightly above middle, in case of unequal amplitudes
#     y_avg = []
#     for i in range(1, len(y) - 1):
#         y_avg.append(np.mean(y[i - 1:i + 1]))
#     x_avg = x[1:-1]
#     below = []
#     for val in y_avg:
#         below.append(int(val < median))
#     # detect edges, e.g. rising edge = [below, below, above, above], and similar for falling edge
#     rising = []
#     falling = []
#     for i in range(len(below) - 4):
#         if below[i:i + 4] == [0, 0, 1, 1]:
#             falling.append(x_avg[i + 2])
#         if below[i:i + 4] == [1, 1, 0, 0]:
#             rising.append(x_avg[i + 2])
#     nf = len(falling)
#     nr = len(rising)
#     falling = np.array(falling)
#     rising = np.array(rising)
#     if args == None:
#         f0 = 0.5 * (np.mean(falling) + np.mean(rising))
#         if (nf == 1 and nr == 1) or (nf == 2 and nr == 2):
#             hf = [np.mean(rising - falling)]
#         elif nf == 4 and nr == 4:
#             freqs = 0.5 * (rising + falling)
#             hf = [0.5 * (freqs[1] - freqs[0] + freqs[3] - freqs[2]), 0.5 * (freqs[3] - freqs[1] + freqs[2] - freqs[0])]
#         elif nf == 3 and nr == 3:
#             if rising[0] > falling[0]:  # first falling, then rising => 2x same hf
#                 hf = [0.25 * (falling[2] - falling[1] + falling[1] - falling[0] + rising[2] - rising[1] + rising[1] - rising[0])]
#                 hf.append(hf[-1])
#             else:  # spectrum not broad enough, ignore larger hf
#                 hf = [np.mean(rising[1:] - falling[:-1])]
#         else:
#             hf = [0]
#     else:
#         f0 = args[0]
#         hf = args[1:]
#     result = [f0, a, g, c]
#     for v in hf:
#         result.append(v)
#     return result
#
#
# def TripleLorentzian(x1, x2, x3, g1, g2, g3, a1, a2, a3, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     return a1 / np.pi * (g1 ** 2 / ((x - x1) ** 2 + g1 ** 2)) + a2 / np.pi * (g2 ** 2 / ((x - x2) ** 2 + g2 ** 2)) + a3 / np.pi * (g3 ** 2 / ((x - x3) ** 2 + g3 ** 2)) + c
#
#
# setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')
#
#
# def trip_lorentz_n14(x1, g, a1, a2, a3, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     n14_split = 2.165 * 1e6
#     x2 = x1 + n14_split
#     x3 = x1 + 2 * n14_split
#     return -abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) - abs(a3) / np.pi * (g ** 2 / ((x - x3) ** 2 + g ** 2)) + c
#
#
# def trip_lorentz_n14_estimator(x, y):
#     n14_split = 2.165 * 1e6  # x in Hz
#     dx = abs(x[1] - x[0])
#     split_index1 = int(round(n14_split / dx, 0))
#     split_index2 = int(round(n14_split * 2. / dx, 0))
#     trip_mean = []
#     for i in range(len(y) - split_index2):
#         trip_mean.append((y[i] + y[i + split_index1] + y[i + split_index2]) / 3.)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x1 = x[trip_mean.argmin()]
#     g = 0.5 * 1e6  # HWHM
#     a1 = a2 = a3 = (trip_mean.min() - c) * np.pi
#     return x1, g, a1, a2, a3, c
#
#
# def trip_lorentz(split):
#     def f(x1, g, a1, a2, a3, c):
#         """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#         n14_split = split
#         x2 = x1 + n14_split
#         x3 = x1 + 2 * n14_split
#         return -abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (
#             g ** 2 / ((x - x2) ** 2 + g ** 2)) - abs(a3) / np.pi * (g ** 2 / ((x - x3) ** 2 + g ** 2)) + c
#
#     return f
#
#
# def trip_lorentz_estimator(split):
#     def f(x, y):
#         n14_split = split  # x in Hz
#         dx = abs(x[1] - x[0])
#         split_index1 = int(round(n14_split / dx, 0))
#         split_index2 = int(round(n14_split * 2. / dx, 0))
#         trip_mean = []
#         for i in range(len(y) - split_index2):
#             trip_mean.append((y[i] + y[i + split_index1] + y[i + split_index2]) / 3.)
#         trip_mean = np.array(trip_mean)
#         c = trip_mean.max()
#         x1 = x[trip_mean.argmin()]
#         g = 0.1 * 1e6  # HWHM
#         a1 = a2 = a3 = (trip_mean.min() - c) * np.pi
#         return x1, g, a1, a2, a3, c
#
#     return f
#
#
# def lorentz_n14_1mhz_c13(x1, g, a1, a2, a3, a4, a5, a6, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     n14_split = 2.16 * 1e6
#     c13_split = 0.75 * 1e6
#     x2 = x1 + c13_split
#     x3 = x1 + n14_split
#     x4 = x3 + c13_split
#     x5 = x3 + n14_split
#     x6 = x5 + c13_split
#     return c - abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) - abs(a3) / np.pi * (g ** 2 / ((x - x3) ** 2 + g ** 2)) \
#                      - abs(a4) / np.pi * (g ** 2 / ((x - x4) ** 2 + g ** 2)) - abs(a5) / np.pi * (g ** 2 / ((x - x5) ** 2 + g ** 2)) - abs(a6) / np.pi * (g ** 2 / ((x - x6) ** 2 + g ** 2))
#
#
# def lorentz_n14_1mhz_c13_estimator(x, y):
#     n14_split = 2.16 * 1e6  # x in Hz
#     c13_split = 0.75 * 1e6
#     dx = abs(x[1] - x[0])
#     split_index1 = int(round(c13_split / dx, 0))
#     split_index2 = int(round(n14_split / dx, 0))
#     split_index3 = int(round((n14_split + c13_split) / dx, 0))
#     split_index4 = int(round((n14_split * 2.) / dx, 0))
#     split_index5 = int(round((n14_split * 2. + c13_split) / dx, 0))
#     trip_mean = []
#     for i in range(len(y) - split_index5):
#         trip_mean.append((y[i] + y[i + split_index1] + y[i + split_index2] + y[i + split_index3] + y[i + split_index4] + y[i + split_index5]) / 6.)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x1 = x[trip_mean.argmin()]
#     g = 0.25 * 1e6  # HWHM
#     a1 = a2 = a3 = a4 = a5 = a6 = (trip_mean.min() - c) * np.pi
#     return x1, g, a1, a2, a3, a4, a5, a6, c
#
#
# def lorentz_n14_1mhz_c13_awg(x1, g, a1, a2, a3, a4, a5, a6, a7, a8, a9, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     n14_split = 2.16 * 1e6
#     c13_split = 0.75 * 1e6
#     x2 = x1 + c13_split
#     x3 = x1 + c13_split * 2.
#     x4 = x1 + n14_split
#     x5 = x4 + c13_split
#     x6 = x4 + c13_split * 2.
#     x7 = x4 + n14_split
#     x8 = x7 + c13_split
#     x9 = x7 + c13_split * 2.
#     return c - abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) - abs(a3) / np.pi * (g ** 2 / ((x - x3) ** 2 + g ** 2)) \
#                      - abs(a4) / np.pi * (g ** 2 / ((x - x4) ** 2 + g ** 2)) - abs(a5) / np.pi * (g ** 2 / ((x - x5) ** 2 + g ** 2)) - abs(a6) / np.pi * (g ** 2 / ((x - x6) ** 2 + g ** 2)) \
#                      - abs(a7) / np.pi * (g ** 2 / ((x - x7) ** 2 + g ** 2)) - abs(a8) / np.pi * (g ** 2 / ((x - x8) ** 2 + g ** 2)) - abs(a9) / np.pi * (g ** 2 / ((x - x9) ** 2 + g ** 2))
#
#
# def lorentz_n14_1mhz_c13_awg_estimator(x, y):
#     n14_split = 2.16 * 1e6  # x in Hz
#     c13_split = 0.75 * 1e6
#     dx = abs(x[1] - x[0])
#     split_index1 = int(round(c13_split / dx, 0))
#     split_index2 = int(round(c13_split * 2. / dx, 0))
#     split_index3 = int(round(n14_split / dx, 0))
#     split_index4 = int(round((n14_split + c13_split) / dx, 0))
#     split_index5 = int(round((n14_split + c13_split * 2.) / dx, 0))
#     split_index6 = int(round((n14_split * 2.) / dx, 0))
#     split_index7 = int(round((n14_split * 2. + c13_split) / dx, 0))
#     split_index8 = int(round((n14_split * 2. + c13_split * 2.) / dx, 0))
#     trip_mean = []
#     sum = np.sum
#     for i in range(1, len(y) - split_index8 - 1):
#         trip_mean.append((sum(y[i - 1:i + 2]) / 3. + 2 * sum(y[i + split_index1 - 1:i + split_index1 + 2]) / 3. + sum(y[i + split_index2 - 1:i + split_index2 + 2]) / 3. + sum(y[i + split_index3 - 1:i + split_index3 + 2]) / 3. + 2 * sum(y[i + split_index4 - 1:i + split_index4 + 2]) / 3.
#                           + sum(y[i + split_index5 - 1:i + split_index5 + 2]) / 3. + sum(y[i + split_index6 - 1:i + split_index6 + 2]) / 3. + 2 * sum(y[i + split_index7 - 2:i + split_index7 + 2]) / 3. + sum(y[i + split_index8 - 1:i + split_index8 + 2]) / 3.) / 12.)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x1 = x[trip_mean.argmin()]
#     g = 0.25 * 1e6  # HWHM
#     a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = (trip_mean.min() - c) * np.pi
#     return x1, g, a1, a2, a3, a4, a5, a6, a7, a8, a9, c
#
#
# def lorentz_n14_1mhz_c13_awg_emp(x1, g, a1, a2, a3, a4, a5, a6, a7, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     x2 = x1 + 0.890e6
#     x3 = x1 + 1.94e6
#     x4 = x1 + 3.031e6
#     x5 = x1 + 4.097e6
#     x6 = x1 + 5.244e6
#     x7 = x1 + 6.16e6
#
#     return c - abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) - abs(a3) / np.pi * (g ** 2 / ((x - x3) ** 2 + g ** 2)) \
#                      - abs(a4) / np.pi * (g ** 2 / ((x - x4) ** 2 + g ** 2)) - abs(a5) / np.pi * (g ** 2 / ((x - x5) ** 2 + g ** 2)) - abs(a6) / np.pi * (g ** 2 / ((x - x6) ** 2 + g ** 2)) \
#                      - abs(a7) / np.pi * (g ** 2 / ((x - x7) ** 2 + g ** 2))
#
#
# def lorentz_n14_1mhz_c13_awg_emp_estimator(x, y):
#     n14_split = 2.16 * 1e6  # x in Hz
#     c13_split = 1.03 * 1e6
#     dx = abs(x[1] - x[0])
#     split_index1 = int(round(0.890e6 / dx, 0))
#     split_index2 = int(round(1.94e6 / dx, 0))
#     split_index3 = int(round(3.031e6 / dx, 0))
#     split_index4 = int(round((4.097e6) / dx, 0))
#     split_index5 = int(round((5.244e6) / dx, 0))
#     split_index6 = int(round((6.16e6) / dx, 0))
#
#     trip_mean = []
#     sum = np.sum
#     for i in range(1, len(y) - split_index6 - 1):
#         trip_mean.append((sum(y[i - 1:i + 2]) / 3. + 2 * sum(y[i + split_index1 - 1:i + split_index1 + 2]) / 3. + sum(y[i + split_index2 - 1:i + split_index2 + 2]) / 3. + 2 * sum(y[i + split_index3 - 1:i + split_index3 + 2]) / 3. + sum(y[i + split_index4 - 1:i + split_index4 + 2]) / 3.
#                           + 2 * sum(y[i + split_index5 - 1:i + split_index5 + 2]) / 3. + sum(y[i + split_index6 - 1:i + split_index6 + 2]) / 3.) / 8.)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x1 = x[trip_mean.argmin()]
#     g = 0.25 * 1e6  # HWHM
#     a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = (trip_mean.min() - c) * np.pi
#     return x1, g, a1, a2, a3, a4, a5, a6, a7, c
#
#
# def trip_lorentz_n15(x1, g, a1, a2, c):
#     """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
#     n15_split = 3.03 * 1e6
#     x2 = x1 + n15_split
#     return -abs(a1) / np.pi * (g ** 2 / ((x - x1) ** 2 + g ** 2)) - abs(a2) / np.pi * (g ** 2 / ((x - x2) ** 2 + g ** 2)) + c
#
#
# def trip_lorentz_n15_estimator(x, y):
#     n15_split = 3.03 * 1e6  # x in Hz
#     dx = abs(x[1] - x[0])
#     split_index = int(round(n15_split / dx, 0))
#     trip_mean = []
#     for i in range(len(y) - split_index):
#         trip_mean.append((y[i] + y[i + split_index]) / 2.)
#     trip_mean = np.array(trip_mean)
#     c = trip_mean.max()
#     x1 = x[trip_mean.argmin()]
#     g = 0.6 * 1e6  # HWHM
#     a1 = a2 = (trip_mean.min() - c) * np.pi
#     return x1, g, a1, a2, c
#
#
# def SumOverFunctions(functions):
#     """Creates a factory that returns a function representing the sum over 'functions'.
#     'functions' is a list of functions.
#     The resulting factory takes as arguments the parameters to all functions,
#     flattened and in the same order as in 'functions'."""
#
#     def function_factory(*args):
#         def f(x):
#             y = np.zeros(x.shape)
#             i = 0
#             for func in functions:
#                 n = func.func_code.co_argcount
#                 y += func(*args[i, i + n])(x)
#                 i += n
#
#         return f
#
#     return function_factory
#
#
# def Fit(x, y, Model, Estimator):
#     """Perform least-squares fit of two dimensional data (x,y) to model 'Model' using Levenberg-Marquardt algorithm.\n
#     'Model' is a callable that takes as an argument the model parameters and returns a function representing the model.\n
#     'Estimator' can either be an N-tuple containing a starting guess of the fit parameters, or a callable that returns a respective N-tuple for given x and y."""
#     if callable(Estimator):
#         return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator(x, y))[0]
#     else:
#         return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator)[0]
#
#
# def Fit2(x, y, Model, Estimator, args=None):
#     """Perform least-squares fit of two dimensional data (x,y) to model 'Model' using Levenberg-Marquardt algorithm.\n
#     'Model' is a callable that takes as an argument the model parameters and returns a function representing the model.\n
#     'Estimator' can either be an N-tuple containing a starting guess of the fit parameters, or a callable that returns a respective N-tuple for given x and y."""
#     if callable(Estimator):
#         return scipy.optimize.leastsq(lambda pp: Model(pp, args)(x) - y, Estimator(x, y, args))[0]
#     else:
#         return scipy.optimize.leastsq(lambda pp: Model(pp, args)(x) - y, Estimator)[0]
#
#
# class Gaussfit2D(object):
#     def __init__(self, data):
#         self.data = data
#
#     def gauss(self, A0, A, x0, y0, wx, wy, theta):
#         wx = np.float(wx)
#         wy = np.float(wy)
#         # def f(x,y):
#         #    x = (x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)
#         #    y = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
#         #    return A0**2+A*A*np.exp(-((x/wx)**2+(y/wy)**2)/2)
#         # return f
#         return lambda x, y: A0 ** 2 + A * A * np.exp(-(((x0 - x) / wx) ** 2 + ((y0 - y) / wy) ** 2) / 2)
#
#     def moments(self, data):
#         total = data.sum()
#         X, Y = np.indices(data.shape)
#         x = (X * data).sum() / total
#         y = (Y * data).sum() / total
#         col = data[:, int(y)]
#         wx = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
#         row = data[int(x), :]
#         wy = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
#         A = np.sqrt(data.max())
#         A0 = np.sqrt(data.min())
#         theta = 0
#         return A0, A, x, y, wx, wy, theta
#
#     def fitgaussian(self, data):
#         params = self.moments(data)
#         errorfunction = lambda p: np.ravel(self.gauss(*p)(*np.indices(data.shape)) - data)
#         p, success = scipy.optimize.leastsq(errorfunction, params)
#         return p
#
#     def Execute(self, data=None):
#         if data is None:
#             data = self.data
#         params = self.fitgaussian(data)
#         (A0, A, y, x, wx, wy, theta) = params
#         # a  = A**2
#         # return x , y , a
#         return A0 ** 2, A ** 2, y, x, wx, wy, theta


    # def Fit(x, y, f, d, guess=None):
    #     params = Parameters()
    #     mod = Model(f)
    #     for key, val in d.items():
    #         v = True
    #         v = False if key == 'f0' else v
    #         params.add(key, value=val, vary=v)
    #     result = []
    #     redchi = []
    #     for method in ['nelder', 'leastsq', 'lbfgsb', 'powell']:
    #         result.append(mod.fit(y, params, x=x, method=method))
    #         redchi.append(result[-1].redchi)
    #     result = result[np.argmin(redchi)]
    #     return result.best_fit, result.best_values