# coding=utf-8
from __future__ import print_function, absolute_import, division
__metaclass__ = type

import numpy as np
import numbers
from . import coordinates
import itertools
import lmfit.lineshapes
import collections

class pd(dict):

    def __init__(self):
        super(pd, self).__init__(self)
        self.ddp = dict()
        self.ddp['fid'] = np.array([])
        self.ddp['hahn'] = np.array([0.0])
        self.ddp['xy4'] = np.array([0.0, np.pi / 2., 0.0, np.pi / 2.])
        self.ddp['xy8'] = np.concatenate([self.ddp['xy4'], self.ddp['xy4'][::-1]])
        self.ddp['xy16'] = np.concatenate([self.ddp['xy8'], self.ddp['xy8'] + np.pi])
        self.ddp['knillpi'] = np.array([np.pi / 6., 0, np.pi / 2., 0, np.pi / 6.])
        self.ddp['kdd4'] = np.concatenate([phasexy + self.ddp['knillpi'] for phasexy in self.ddp['xy4']])
        self.ddp['kdd8'] = np.concatenate([phasexy + self.ddp['knillpi'] for phasexy in self.ddp['xy8']])
        self.ddp['kdd16'] = np.concatenate([phasexy + self.ddp['knillpi'] for phasexy in self.ddp['xy16']])


    def __getitem__(self, i):
        if type(i) is not str:
            raise Exception('Error:{}, {}'.format(type(i), i))
        ii = i.split("_")
        bs = ii[-1]
        if 'cpmg' in bs:
            if len(bs) == 4:
                raise Exception('Error: {} does not tell how many pi-pulses you want. Valid would be 5_cpmg8'.format(i))
            else:
                bp = np.array(int(bs[4:]) * [0])
        else:
            bp = self.ddp[bs]
        if len(ii) == 2:
            bp = np.tile(bp, int(ii[0]))
        return bp


__PHASES_DD__ = pd()


class DDParameters:
    def __init__(self, name, rabi_period, time_digitization=None, **kwargs):
        self.name = name
        self.time_digitization = time_digitization
        self.set_total_tau(**kwargs)
        self.rabi_period = rabi_period

    @property
    def phases(self):
        name = self.name
        if name[-6:] == '_uhrig' in name:
            name = name[:-6]
        return __PHASES_DD__[name]

    @property
    def n_tau(self):
        if self.number_of_pi_pulses == 0:
            return 2
        else:
            return 2 * (self.number_of_pi_pulses)

    def set_total_tau(self, **kwargs):
        if self.name[-6:] == '_uhrig':
            if self.time_digitization is not None:
                raise NotImplementedError('time_digitization has no effect for Urig')
            return kwargs['total_tau']
        else:
            if 'tau' in kwargs:
                tau = kwargs['tau']
            else:
                tau = kwargs['total_tau'] / self.n_tau
        if self.time_digitization is not None:
            tau = np.around(tau / self.time_digitization) * self.time_digitization
        self.total_tau = self.n_tau * tau

    @property
    def number_of_pi_pulses(self):
        return len(self.phases)

    @property
    def number_of_pulses(self):
        return self.number_of_pi_pulses + 2

    @property
    def uhrig_pulse_positions_normalized(self):
        return np.sin(np.pi * np.arange(self.number_of_pulses) / (2 * self.number_of_pulses - 2)) ** 2

    @property
    def uhrig_taus_normalized(self):
        return np.diff(self.uhrig_pulse_positions_normalized)

    @property
    def uhrig_pulse_positions(self):
        return self.total_tau * self.uhrig_pulse_positions_normalized

    @property
    def uhrig_taus(self):
        return self.total_tau * self.uhrig_taus_normalized

    @property
    def tau_list(self):
        name = self.name
        if name[-6:] == '_uhrig' in name:
            return self.uhrig_taus
        else:
            tau = self.total_tau / self.n_tau
            return [tau] + [2 * tau for i in range(self.number_of_pi_pulses - 1)] + [tau]

    @property
    def eff_pulse_dur_waiting_time(self):
        return np.concatenate([[3 / 8. * self.rabi_period], 0.5 * self.rabi_period * np.ones(self.number_of_pi_pulses), [3 / 8. * self.rabi_period]])  # effective duration of pi pulse for each waiting time. The effective phase evolution time during the pi/2 pulse is taken as half of the pulse duration

    @property
    def minimum_total_tau(self):
        return self.eff_pulse_dur_waiting_time[0] / self.uhrig_taus_normalized[0]

    @property
    def effective_durations_dd(self):
        if self.minimum_total_tau > self.total_tau:
            raise Exception('Waiting times smaller than zero are not allowed. '
                            'Total tau must be at least {} (current: {})'.format(self.minimum_total_tau, self.total_tau))
        return self.tau_list - self.eff_pulse_dur_waiting_time

class Arbitrary:

    @property
    def controls(self):
        if hasattr(self, '_controls'):
            return self._controls
        elif type(self.column_dict) is not collections.OrderedDict:
            raise Exception("Error: {}, {}".format(self.column_dict, type(self.column_dict)))
        return self.column_dict.keys()

    @property
    def n_columns(self):
        return int(np.sum([len(i) for i in self.column_dict.keys()]))

    def locations(self, name):
        return [i for i, val in enumerate(self.sequence) if name in val]

    def l_active(self, name):
        return [True if name in val else False for val in self.sequence]

    @property
    def n_bins(self):
        return len(self.sequence)

    def mask(self, name):
        out = np.zeros((len(self.sequence), self.n_columns), dtype=bool)
        for i in self.locations(name):
            out[i, self.column_dict[name]] = np.ones(len(self.column_dict[name]), dtype=bool)
        return out

    @property
    def T(self):
        return sum(self.times())

    def times(self, name=None):
        times = self.times_full
        if name is not None:
            return times[self.locations(name)]
        else:
            return times

    def fields(self, name=None):
        fields = self.fields_full
        if name is not None:
            return fields[self.locations(name)][:, self.column_dict[name]]
        else:
            return fields

    # def fields_aphi_mhz(self, name=None):
    #     l = []
    #     for k in self.columns:
    #         if
    #     if name is None:
    #         pass
    #     else:
    #         if len(self.column_dict) == 1:
    #
    #
    #
    #     return self.xy2aphi(self.fields_xy_mhz(n))

    def split(self, locations, n):
        self._times_full =  np.array(list(itertools.chain(*[[t/n] * n if idx in locations else [t] for idx, t in enumerate(self.times_full)])))
        self._fields_full = np.repeat(self.fields_full, [n if i in locations else 1 for i in range(self.n_bins)], axis=0)
        self._sequence = list(itertools.chain(*[[item] * n if idx in locations else [item] for idx, item in enumerate(self.sequence)]))

    def print_info(self):
        for s, t in zip(self.sequence, self.times()):
            print(s, t)

    def save_times_fields_mhz(self, path):
        np.savetxt(path, np.around(np.concatenate([self.times_full.reshape(-1,1), self.fields_full], axis=1), 9), fmt=str('%+1.9e'))

    @property
    def control_names_dict(self):
        return getattr(self, '_control_names_dict', collections.OrderedDict([(cn, cn) for cn in self.controls]))

    @control_names_dict.setter
    def control_names_dict(self, val):
        """
        NOTE: Not all controls have to be given, if e.g. only 'mw' should be renamed to 'MW' then val = {'mw': 'MW'} is allowed
        """
        out = self.control_names_dict
        for k, v  in val.items():
            if k not in self.controls:
                raise Exception("Error: {}, {}, {}, {}".format(val, self.control_names_dict), k, v)
            out[k] = v
        self._control_names_dict = out




    # #
    # # export_mask_list = [dda.mask(name='rf'),
    # #                     dda.mask(name='mw'),
    # #                     dda.mask(name='wait')]
    # # fields_names_list = ['RF', 'MW', 'WAIT']
    #
    # # @property
    # # def fields_names_list(self):
    # #     field_names_list = self._fields_names_list if hasattr(self, '_fields_names_list') else ["{}".format(i) for i in range(len(self.export_mask_list))]
    # #     return field_names_list
    # #
    # # @fields_names_list.setter
    # # def fields_names_list(self, val):
    # #     self._fields_names_list = val
    #
    # @property
    # def export_mask_list(self):
    #     if not hasattr(self, '_export_mask_list') or self._export_mask_list is None:
    #         return [self.get_mask()[:self.n_bins, :]]
    #     else:
    #         return self._export_mask_list
    #
    # @export_mask_list.setter
    # def export_mask_list(self, val):
    #     if type(val) is not list:
    #         raise Exception('Error, {}'.format(val))
    #     columns = self.get_mask().shape[1] - 1 # last column of the optimize mask are timeslices
    #     for i, v in enumerate(val):
    #         if v.shape != (self.n_bins, columns):
    #             raise Exception('Error: i: {}, v.shape: {}, bins: {}, columns: {}'.format(i, v.shape, self.n_bins, columns))
    #     self._export_mask_list = val
    #
    # @property
    # def sequence(self):
    #     seq = []
    #     for i in range(self.n_bins):
    #         step = []
    #         for j, em in enumerate(self.export_mask_list):
    #             if np.any(em[i]):
    #                 step.append(self.fields_names_list[j])
    #         if len(step) > 0:
    #             seq.append(step)
    #     return seq
    #
    # def export_mask_columns(self, n):
    #     return np.where(~np.all(self.export_mask_list[n] == 0, axis=0))[0]
    #
    # def export_mask_rows(self, n):
    #     if not self.export_mask_list[n].any():
    #         return np.array([])
    #     else:
    #         return np.where(self.export_mask_list[n][:, np.where(~np.all(self.export_mask_list[n] == 0, axis=0))[0][0]])[0]
    #
    # @property
    # def times_fields_mhz(self):
    #     out = np.array(self.eng.eval("dyn.export"))
    #     out[:, 1:] *= 2 * np.pi
    #     return out
    #
    # def xy2aphi(self, xy):
    #     norm = np.array([np.linalg.norm(xy, axis=1)]).transpose()
    #     phi = np.arctan2(xy[:, 1:2], xy[:, 0:1])
    #     return np.concatenate([norm, phi], axis=1)

class Wait(Arbitrary):
    def __init__(self, n_bins_wait, t_wait):
        self.n_bins_wait = n_bins_wait
        self.t_wait = t_wait

    column_dict = {'wait': [0]}

    @property
    def sequence(self):
        return getattr(self, '_sequence', [['wait']] * self.n_bins_wait)

    @property
    def fields_full(self):
        return getattr(self, '_fields_full', np.zeros((len(self.sequence), self.n_columns)))

    @property
    def times_full(self):
        return getattr(self, '_times_full', np.array([self.t_wait / float(self.n_bins_wait)] * self.n_bins_wait))


class Rabi(Arbitrary):
    def __init__(self, n_bins_rabi, t_rabi, omega, phase=0.0, control_field='mw'):
        self.n_bins_rabi = n_bins_rabi
        self.t_rabi = t_rabi
        self.phase = phase
        self.omega = omega
        self.control_field = control_field
        self.column_dict = collections.OrderedDict([(self.control_field, [0, 1])])

    @property
    def sequence(self):
        return getattr(self, '_sequence', [[self.control_field]] * self.n_bins_rabi)

    def omega_list(self):
        return np.ones([self.n_bins_rabi]) * self.omega

    @property
    def fields_full(self):
        ol = self.omega_list()
        return getattr(self, '_fields_full', np.array([ol * np.cos(self.phase), ol * np.sin(self.phase)]).T)

    @property
    def times_full(self):
        return getattr(self, '_times_full', np.array([self.t_rabi / self.n_bins_rabi] * self.n_bins_rabi))


class RabiGaussian(Rabi):
    def __init__(self, gaussian_parameters=None, *args, **kwargs):
        super(RabiGaussian, self).__init__(*args, **kwargs)
        self.gaussian_parameters = gaussian_parameters

    @property
    def gaussian_parameters(self):
        return self._gaussian_parameters

    @gaussian_parameters.setter
    def gaussian_parameters(self, val):
        val['mu'] = float(val.get('mu', self.t_rabi / 2.))
        if val['mu'] < 0. or val['mu'] > self.t_rabi:
            raise Exception('Error: val')
        if val['sigma'] > self.t_rabi:
            raise Exception('Error: val')
        val['area'] = float(val.get('area', self.t_rabi * self.omega))
        val['x_area_range'] = [0.0, self.t_rabi]
        self._gaussian_parameters = val

    def omega_list(self):
        return lmfit.lineshapes.gaussian(self.times_center(self.control_field), **self.gaussian_parameters)

class DDAlpha(Arbitrary):
    def __init__(self, pi_phases=None, spt=None,
                 wait=None, target_t_rf=None, rabi_period=None,
                 target_Azz=None, omega=None,
                 time_digitization=None):
        self.time_digitization = time_digitization
        self.pi_phases = pi_phases
        self.spt = spt
        self.wait = wait
        self.target_t_rf = target_t_rf
        self.rabi_period = rabi_period
        self.target_Azz = target_Azz
        self.omega = omega

    column_dict = collections.OrderedDict([('mw', [0, 1]), ('rf', [2, 3]), ('wait', [4])])

    @property
    def n_pi(self):
        return len(self.pi_phases)

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        if val in [None, 'pi', '2pi'] or isinstance(val, numbers.Number):
            if val in [None, 'pi']:
                o = 0.5 / self.t_rf
            elif val == '2pi':
                o = 1 / self.t_rf
            elif isinstance(val, numbers.Number):
                o = val
            self._omega = o * np.ones(2 * self.n_pi * self.spt)
        elif len(val) == 2 * self.n_pi * self.spt:
            self._omega = val
        else:
            raise Exception('Error: {}'.format(val))

    @property
    def t_rf(self):
        return 2 * self.n_pi * self.spt * self.tau()

    @property
    def sequence(self):
        wait = [['wait']] if self.wait is not None else []
        base = [['rf']] * self.spt + wait + [['mw']] + wait + [['rf']] * self.spt
        return getattr(self, '_sequence', list(itertools.chain(*[base for i in range(self.n_pi)])))

    def alpha(self, n):
        delta_alpha = 2 * np.pi * self.target_Azz * (self.tau() * self.spt + self.wait + 0.25 * self.rabi_period)
        return int(np.ceil(n / 2.)) * (delta_alpha + np.pi) % (2 * np.pi)

    def rf_array_aphi(self):
        u_phi = np.repeat([self.alpha(n) for n in range(2 * self.n_pi)], self.spt)
        return np.array([self.omega, u_phi]).T

    def rf_array_xy(self):
        rho, azim = tuple(np.hsplit(self.rf_array_aphi(), 2))
        x, y, z = coordinates.sph2cart(rho, 0, azim)
        return np.concatenate([x, y], axis=1)

    def mw_array_xy(self):
        return np.array([np.array([np.cos(phi), np.sin(phi)]) / self.rabi_period for phi in self.pi_phases])

    def mw_array_aphi(self):
        x, y = tuple(np.hsplit(self.mw_array_xy(), 2))
        rho, elev, azim = coordinates.cart2sph(x, y, 0)
        return np.concatenate([rho, azim], axis=1)

    @property
    def fields_full(self):
        out = np.zeros((len(self.sequence), self.n_columns))
        mw_array = self.mw_array_xy()
        rf_array_xy = self.rf_array_xy()
        idxmw = 0
        idxrf = 0
        for i, val in enumerate(self.sequence):
            if val == 'mw':
                out[i, self.column_dict[val]] = mw_array[idxmw]
                idxmw += 1
            elif val == 'rf':
                out[i, self.column_dict[val]] = rf_array_xy[idxrf]
                idxrf += 1
        return getattr(self, '_fields_full', out)

    def tau(self):
        out = self.target_t_rf / (2 * self.n_pi * self.spt)
        if self.time_digitization is not None:
            out = np.around(out / float(self.time_digitization)) * float(self.time_digitization)
        return out

    @property
    def times_full(self):
        td = dict(rf=self.tau(),
                  mw=0.5 * self.rabi_period,
                  wait=self.wait)
        out = np.array([td[i[0]] for i in self.sequence if td[i[0]] is not None])
        out[self.locations('wait')[0]] /= 2
        out[self.locations('wait')[-1]] /= 2
        return getattr(self, '_times_full', out)

class DD(Arbitrary):
    def __init__(self, dd_type=None, rabi_period=None, time_digitization=None, **kwargs):
        self.dd_type = dd_type
        self.rabi_period = rabi_period
        self.time_digitization = time_digitization
        self.set_total_tau(**kwargs)

    column_dict = collections.OrderedDict([('mw', [0, 1]), ('wait', [2])])

    @property
    def sequence(self):
        if self.number_of_pi_pulses == 0:
            out = ['wait'] + ['wait']
        else:
            out = list(np.insert(['wait'] * self.n_tau, range(1, self.n_tau), [['mw'] * self.number_of_pi_pulses]))

        return getattr(self, '_sequence', [[i] for i in out])

    def mw_array_xy(self):
        rho, azim = tuple(np.hsplit(self.mw_array_aphi(), 2))
        x, y, z = coordinates.sph2cart(rho, 0, azim)
        return np.concatenate([x, y], axis=1)

    def mw_array_aphi(self):
        rho = np.ones([self.number_of_pi_pulses]) / self.rabi_period
        return np.vstack([rho, self.phases]).T

    @property
    def fields_full(self):
        out = np.zeros((len(self.sequence), self.n_columns))
        mw_array = self.mw_array_xy()
        idxmw = 0
        for i, val in enumerate(self.sequence):
            if val == 'mw':
                out[i, self.column_dict[val]] = mw_array[idxmw]
                idxmw += 1
        return getattr(self, '_fields_full', out)

    def tau_list(self):
        name = self.dd_type
        if name[-6:] == '_uhrig' in name:
            return self.uhrig_taus
        else:
            tau = self.total_tau / self.n_tau
            return np.array([tau] + [2 * tau for i in range(self.number_of_pi_pulses - 1)] + [tau])

    @property
    def times_full(self):
        tl = self.tau_list()
        if self.number_of_pi_pulses == 0:
            out = self.tau_list()
        else:
            out = np.insert(tl, range(1, len(tl)), [[0.5 * self.rabi_period] * self.number_of_pi_pulses])
        return getattr(self, '_times_full', out)

    @property
    def phases(self):
        name = self.dd_type
        if name[-6:] == '_uhrig' in name:
            name = name[:-6]
        return __PHASES_DD__[name]

    @property
    def n_tau(self):
        if self.number_of_pi_pulses == 0:
            return 2
        else:
            return self.number_of_pi_pulses + 1

    def set_total_tau(self, **kwargs):
        if self.dd_type[-6:] == '_uhrig':
            return kwargs['total_tau']
        else:
            if 'tau' in kwargs and kwargs['tau'] is not None:
                tau = kwargs['tau']
            else:
                tau = kwargs['total_tau'] / self.n_tau
        if self.time_digitization is not None:
            tau = np.around(tau / self.time_digitization) * self.time_digitization
        self.total_tau = self.n_tau * tau

    @property
    def number_of_pi_pulses(self):
        return len(self.phases)

    @property
    def number_of_pulses(self):
        return self.number_of_pi_pulses + 2

    @property
    def uhrig_pulse_positions_normalized(self):
        return np.sin(np.pi * np.arange(self.number_of_pulses) / (2 * self.number_of_pulses - 2)) ** 2

    @property
    def uhrig_taus_normalized(self):
        return np.diff(self.uhrig_pulse_positions_normalized)

    @property
    def uhrig_pulse_positions(self):
        return self.total_tau * self.uhrig_pulse_positions_normalized

    @property
    def uhrig_taus(self):
        out = self.total_tau * self.uhrig_taus_normalized
        if self.time_digitization is not None:
            out = np.around(out / self.time_digitization) * self.time_digitization
        return out

    @property
    def eff_pulse_dur_waiting_time(self):
        return np.concatenate([[3 / 8. * self.rabi_period], 0.5 * self.rabi_period * np.ones(self.number_of_pi_pulses), [3 / 8. * self.rabi_period]])  # effective duration of pi pulse for each waiting time. The effective phase evolution time during the pi/2 pulse is taken as half of the pulse duration

    @property
    def minimum_total_tau(self):
        return self.eff_pulse_dur_waiting_time[0] / self.uhrig_taus_normalized[0]

    @property
    def effective_durations_dd(self):
        if self.minimum_total_tau > self.total_tau:
            raise Exception('Waiting times smaller than zero are not allowed. '
                            'Total tau must be at least {} (current: {})'.format(self.minimum_total_tau, self.total_tau))
        return self.tau_list - self.eff_pulse_dur_waiting_time

class Z(Arbitrary):
    def __init__(self, fields_l, times_l):
        self.column_dict = collections.OrderedDict([("z{}".format(i), [i]) for i in range(len(fields_l))])
        self.fields_full = np.diag(fields_l)
        self.times_full = np.array(times_l)
        self.sequence = [[i] for i in self.controls]

class Concatenated(Arbitrary):
    def __init__(self, p_list, controls):
        for p in p_list:
            if not issubclass(type(p), Arbitrary):
                raise Exception('Error: type of p is {}'.type(p))
        self.p_list = p_list
        self._controls = controls
        self.check_control_length()

    def set_p_list(self, times_full, fields_full):
        nbcs = np.cumsum([0] + [p.n_bins for p in self.p_list])
        cd = self.column_dict
        for idx in range(len(nbcs)-1):
            self.p_list[idx]._fields_full = fields_full[nbcs[idx]:nbcs[idx+1], list(itertools.chain(*[cd[c] for c in self.p_list[idx].controls]))]
            self.p_list[idx]._times_full = times_full[nbcs[idx]:nbcs[idx+1]]

    @property
    def sequence(self):
        return getattr(self, '_sequence', list(itertools.chain(*[i.sequence for i in self.p_list])))

    def check_control_length(self):
        for c in self.controls:
            if len(set([len(p.column_dict[c]) for p in self.p_list if c in p.column_dict])) != 1:
                raise Exception("Error. column_dict: {}".format(self.column_dict))

    @property
    def column_dict(self):
        if hasattr(self, '_column_dict'):
            return self._column_dict
        else:
            n_cc = {}
            for c in self.controls:
                for p in self.p_list:
                    if c in p.column_dict:
                        if c in n_cc and n_cc[c] != len(p.column_dict[c]):
                            raise Exception('Error: Control {} has different column length on different items of p_list'.format(c))
                        else:
                            n_cc[c] = len(p.column_dict[c])
            out = {}
            idx = 0
            for c in self.controls:
                out[c] = list(np.arange(n_cc[c]) + idx)
                idx += n_cc[c]
            return out

    @property
    def n_columns(self):
        return len(list(itertools.chain(*self.column_dict.values())))

    def extend_fields_full(self, n):
        fields_full = np.zeros([self.p_list[n].n_bins, self.n_columns])
        for c in self.controls:
            if c in self.p_list[n].column_dict:
                fields_full[:, self.column_dict[c]] = self.p_list[n].fields_full[:, self.p_list[n].column_dict[c]]
        return fields_full

    @property
    def fields_full(self):
        return getattr(self, '_fields_full', np.concatenate([self.extend_fields_full(i) for i in range(len(self.p_list))]))

    @property
    def times_full(self):
        return getattr(self, '_times_full', np.concatenate([i.times_full for i in self.p_list]))

if __name__ == '__main__':
    pass
