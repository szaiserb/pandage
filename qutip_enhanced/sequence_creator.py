import numpy as np
import itertools
import numbers
import scipy.optimize
import scipy.integrate
import coordinates
import itertools
import lmfit_models

class pd(dict):

    ddp = dict()
    ddp['fid'] = np.array([])
    ddp['hahn'] = np.array([0.0])
    ddp['xy4'] = np.array([0.0, np.pi / 2., 0.0, np.pi / 2.])
    ddp['xy8'] = np.concatenate([ddp['xy4'], ddp['xy4'][::-1]])
    ddp['xy16'] = np.concatenate([ddp['xy8'], ddp['xy8'] + np.pi])
    ddp['knillpi'] = np.array([np.pi / 6., 0, np.pi / 2., 0, np.pi / 6.])
    ddp['kdd4'] = np.concatenate([phasexy + ddp['knillpi'] for phasexy in ddp['xy4']])
    ddp['kdd8'] = np.concatenate([phasexy + ddp['knillpi'] for phasexy in ddp['xy8']])
    ddp['kdd16'] = np.concatenate([phasexy + ddp['knillpi'] for phasexy in ddp['xy16']])

    def __getitem__(self, i):
        if type(i) is not str:
            raise Exception('Error')
        ii = i.split("_")
        bs = ii[-1]
        if 'cpmg' in bs:
            if len(bs) == 4:
                raise Exception('Error: {} does not tell how many pi-pulses you want. Valid would be 5_cpmg8'.format(i))
            else:
                bp = np.array(int(bs[4:])*[0])
        else:
            bp = self.ddp[bs]
        if len(ii) == 2:
            bp = np.tile(bp, int(ii[0]))
        return bp

__PHASES_DD__ = pd()

class DDParameters():

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
            return 2*(self.number_of_pi_pulses)

    def set_total_tau(self, **kwargs):
        if self.name[-6:] == '_uhrig':
            if self.time_digitization is not None:
                raise NotImplementedError('time_digitization has no effect for Urig')
            return kwargs['total_tau']
        else:
            if 'tau' in kwargs:
                tau = kwargs['tau']
            else:
                tau = kwargs['total_tau']/self.n_tau
        if self.time_digitization is not None:
            tau = np.around(tau/self.time_digitization)*self.time_digitization
        self.total_tau = self.n_tau*tau

    @property
    def number_of_pi_pulses(self):
        return len(self.phases)

    @property
    def number_of_pulses(self):
        return self.number_of_pi_pulses + 2

    @property
    def uhrig_pulse_positions_normalized(self):
        return np.sin(np.pi*np.arange(self.number_of_pulses)/(2*self.number_of_pulses - 2))**2

    @property
    def uhrig_taus_normalized(self):
        return np.diff(self.uhrig_pulse_positions_normalized)

    @property
    def uhrig_pulse_positions(self):
        return self.total_tau*self.uhrig_pulse_positions_normalized

    @property
    def uhrig_taus(self):
      return self.total_tau*self.uhrig_taus_normalized

    @property
    def tau_list(self):
        name = self.name
        if name[-6:] == '_uhrig' in name:
            return self.uhrig_taus
        else:
            tau = self.total_tau/self.n_tau
            return [tau] + [2*tau for i in range(self.number_of_pi_pulses - 1)] + [tau]

    @property
    def eff_pulse_dur_waiting_time(self):
        return np.concatenate([[3/8.*self.rabi_period], 0.5*self.rabi_period*np.ones(self.number_of_pi_pulses), [3/8.*self.rabi_period]]) #effective duration of pi pulse for each waiting time. The effective phase evolution time during the pi/2 pulse is taken as half of the pulse duration

    @property
    def minimum_total_tau(self):
        return self.eff_pulse_dur_waiting_time[0]/self.uhrig_taus_normalized[0]

    @property
    def effective_durations_dd(self):
        if self.minimum_total_tau > self.total_tau:
            raise Exception('Waiting times smaller than zero are not allowed. '
                            'Total tau must be at least {} (current: {})'.format(self.minimum_total_tau, self.total_tau))
        return self.tau_list - self.eff_pulse_dur_waiting_time

class Arbitrary(object):

    def locations(self, name):
        return [i for i, val in enumerate(self.sequence()) if val == name]

    def l_active(self, name):
        return [True if val == name else False for val in self.sequence()]

    @property
    def n_bins(self):
        return len(self.sequence())

    def mask(self, name):
        out = np.zeros((len(self.sequence()), self.n_columns), dtype=bool)
        for i in self.locations(name):
            out[i, self.column_dict[name]] = np.ones(len(self.column_dict[name]), dtype=bool)
        return out

    @property
    def T(self):
        return sum(self.times())

    @property
    def vary_times(self):
        return self._vary_times

    @vary_times.setter
    def vary_times(self, val):
        if val is None:
            self._vary_times = {}
        else:
            for k, v in val.items():
                if k not in self.column_dict or len(v[0]) != len(v[1]):
                    raise Exception('Error. {}, val: {}'.format(self.column_dict, val))
            self._vary_times = val

    def times(self, name=None):
        times = self.times_raw()
        for c in self.controls:
            t = self.vary_times.get(c, [[], []])
            for i, r in enumerate(t[0]):
                times[self.locations(c)[r]] += t[1][i]
        if any(times <0):
            raise Exception('Error: There are times smaller than zero: {}'.format(times))
        if name is not None:
            return times[self.locations(name)]
        else:
            return times

    # def times_center(self, name=None):
    #     return np.array([0.0] + list(np.cumsum(self.times(name))))[:-1] + 0.5 * self.times(name)

    def fields_varied_individual(self):
        fields = self.fields_raw()
        for c in self.controls:
            t = self.vary_times.get(c, [[], []])
            for i, r in enumerate(t[0]):
                fields[self.locations(c)[r], self.column_dict[c]] /= self.times()[self.locations(c)[r]]/ self.times_raw()[self.locations(c)[r]]
        return fields

    def fields_varied_combined(self):
        fields = self.fields_raw()
        for c in self.controls:
            if (sum(self.times()[self.locations(c)]) > 0) and (sum(self.times_raw()[self.locations(c)])):
                fields[np.array(self.locations(c))[:, None], self.column_dict[c]] /= sum(self.times()[self.locations(c)])/ sum(self.times_raw()[self.locations(c)])
            # fields[self.locations(c)][:, self.column_dict[c]] /= sum(self.times()[self.locations(c)])/ sum(self.times_raw()[self.locations(c)])
        return fields

    def print_info(self):
        for s, t in zip(self.sequence(), self.times()):
            print s, t

class Wait(Arbitrary):

    def __init__(self, n_bins_wait, t_wait, vary_times=None):
        self.n_bins_wait = n_bins_wait
        self.t_wait = t_wait
        self.vary_times = vary_times
        self.fields = self.fields_varied_individual

    column_dict = {'wait': [0]}
    controls = ['wait']
    n_columns = 1

    def sequence(self):
        return ['wait'] * self.n_bins_wait

    def fields_raw(self):
        return np.zeros((len(self.sequence()), self.n_columns))

    def times_raw(self):
        return np.array([self.t_wait/float(self.n_bins_wait)]*self.n_bins_wait)

class Rabi(Arbitrary):

    def __init__(self, n_bins_rabi, t_rabi, omega, phase=0.0, control_field='mw', vary_times=None):
        self.n_bins_rabi = n_bins_rabi
        self.t_rabi = t_rabi
        self.phase = phase
        self.omega = omega
        self.control_field = control_field
        self.vary_times = vary_times
        self.fields = self.fields_varied_individual
        self.column_dict = {self.control_field: [0, 1]}
        self.controls = [self.control_field]

    n_columns = 2

    def sequence(self):
        return [self.control_field]*self.n_bins_rabi

    def omega_list(self):
        return np.ones([self.n_bins_rabi]) * self.omega

    def fields_raw(self):
        ol = self.omega_list()
        return np.array([ol * np.cos(self.phase), ol*np.sin(self.phase)]).T

    def times_raw(self):
        return np.array([self.t_rabi/self.n_bins_rabi]*self.n_bins_rabi)

class RabiGaussian(Rabi):

    def __init__(self, gaussian_parameters=None, *args, **kwargs):
        super(RabiGaussian, self).__init__(*args, **kwargs)
        self.gaussian_parameters = gaussian_parameters

    @property
    def gaussian_parameters(self):
        return self._gaussian_parameters

    @gaussian_parameters.setter
    def gaussian_parameters(self, val):
        val['mu'] = float(val.get('mu', self.t_rabi/2.))
        if val['mu'] < 0. or val['mu'] > self.t_rabi:
            raise Exception('Error: val')
        if val['sigma'] > self.t_rabi:
            raise Exception('Error: val')
        val['area'] = float(val.get('area', self.t_rabi*self.omega))
        val['x_area_range'] = [0.0, self.t_rabi]
        self._gaussian_parameters = val

    def omega_list(self):
        return lmfit_models.gaussian(self.times_center(self.control_field), **self.gaussian_parameters)

class DDAlpha(Arbitrary):

    def __init__(self, pi_phases=None, spt=None, vary_times=None,
                 wait=None, target_t_rf=None, rabi_period=None,
                 target_Azz=None, omega=None, rf_frequency=None,
                 time_digitization=None):
        self.time_digitization = time_digitization
        self.rf_frequency = rf_frequency
        self.pi_phases = pi_phases
        self.spt = spt
        self.wait = wait
        self.target_t_rf = target_t_rf
        self.rabi_period = rabi_period
        self.target_Azz = target_Azz
        self.vary_times = vary_times
        self.fields = self.fields_varied_combined
        self.omega = omega

    column_dict = {'mw': [0, 1], 'rf': [2, 3], 'wait': [4]}
    controls = ['mw', 'rf', 'wait']
    n_columns = 5

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
                o = 0.5 / self.effective_t_rf
            elif val == '2pi':
                o = 1 / self.effective_t_rf
            elif isinstance(val, numbers.Number):
                o = val
            self._omega = o * np.ones(2 * self.n_pi * self.spt)
        elif len(val) == 2 * self.n_pi * self.spt:
            self._omega = val
        else:
            raise Exception('Error: {}'.format(val))

    @property
    def t_rf(self):
        return 2*self.n_pi*self.spt*self.tau()

    @property
    def effective_t_rf(self):
        return self.t_rf - sum(self.vary_times.get('rf', [[], []])[1])

    def sequence(self):
        wait = ['wait'] if self.wait is not None else []
        base = ['rf'] * self.spt + wait + ['mw'] + wait + ['rf'] * self.spt
        return list(itertools.chain(*[base for i in range(self.n_pi)]))

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

    def fields_raw(self):
        out = np.zeros((len(self.sequence()), self.n_columns))
        mw_array = self.mw_array_xy()
        rf_array_xy = self.rf_array_xy()
        idxmw = 0
        idxrf = 0
        for i, val in enumerate(self.sequence()):
            if val == 'mw':
                out[i, self.column_dict[val]] = mw_array[idxmw]
                idxmw += 1
            elif val == 'rf':
                out[i, self.column_dict[val]] = rf_array_xy[idxrf]
                idxrf += 1
        return out

    def tau(self):
        out = self.target_t_rf / (2 * self.n_pi * self.spt)
        if self.rf_frequency is not None:
            out = np.around(out * float(self.rf_frequency)) / float(self.rf_frequency)
        if self.time_digitization is not None:
            out = np.around(out / float(self.time_digitization)) * float(self.time_digitization)
        return out

    def times_raw(self):
        td = dict(rf=self.tau(),
                  mw=0.5 * self.rabi_period,
                  wait=self.wait)
        return np.array([td[i] for i in self.sequence() if td[i] is not None])

class DD(Arbitrary):

    def __init__(self, dd_type=None, rabi_period=None, vary_times=None, time_digitization=None, **kwargs):
        self.dd_type = dd_type
        self.time_digitization = time_digitization
        self.set_total_tau(**kwargs)
        self.rabi_period = rabi_period
        self.vary_times = vary_times
        self.fields = self.fields_varied_individual

    column_dict = {'mw': [0, 1], 'wait': [2]}
    controls = ['mw', 'wait']
    n_columns = 3

    def sequence(self):
        if self.number_of_pi_pulses == 0:
            return ['wait'] + ['wait']
        if self.number_of_pi_pulses == 0:
            return self.tau_list()
        else:
            return np.insert(['wait']*self.n_tau, range(1, self.n_tau), [['mw']*self.number_of_pi_pulses])

    def mw_array_xy(self):
        rho, azim = tuple(np.hsplit(self.mw_array_aphi(), 2))
        x, y, z = coordinates.sph2cart(rho, 0, azim)
        return np.concatenate([x, y], axis=1)

    def mw_array_aphi(self):
        rho = np.ones([self.number_of_pi_pulses])/self.rabi_period
        return np.vstack([rho, self.phases]).T

    def fields_raw(self):
        out = np.zeros((len(self.sequence()), self.n_columns))
        mw_array = self.mw_array_xy()
        idxmw = 0
        for i, val in enumerate(self.sequence()):
            if val == 'mw':
                out[i, self.column_dict[val]] = mw_array[idxmw]
                idxmw += 1
        return out

    def tau_list(self):
        name = self.dd_type
        if name[-6:] == '_uhrig' in name:
            return self.uhrig_taus
        else:
            tau = self.total_tau/self.n_tau
            return np.array([tau] + [2*tau for i in range(self.number_of_pi_pulses - 1)] + [tau])

    def times_raw(self):
        tl = self.tau_list()
        if self.number_of_pi_pulses == 0:
            return self.tau_list()
        else:
            return np.insert(tl, range(1, len(tl)), [[0.5 * self.rabi_period]*self.number_of_pi_pulses])

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

    def set_total_tau(self, **kwargs ):
        if self.dd_type[-6:] == '_uhrig':
            if self.time_digitization is not None:
                raise NotImplementedError('time_digitization has no effect for Urig')
            return kwargs['total_tau']
        else:
            if 'tau' in kwargs and kwargs['tau'] is not None:
                tau = kwargs['tau']
            else:
                tau = kwargs['total_tau']/self.n_tau
        if self.time_digitization is not None:
            tau = np.around(tau/self.time_digitization)*self.time_digitization
        self.total_tau = self.n_tau*tau

    @property
    def number_of_pi_pulses(self):
        return len(self.phases)

    @property
    def number_of_pulses(self):
        return self.number_of_pi_pulses + 2

    @property
    def uhrig_pulse_positions_normalized(self):
        return np.sin(np.pi*np.arange(self.number_of_pulses)/(2*self.number_of_pulses - 2))**2

    @property
    def uhrig_taus_normalized(self):
        return np.diff(self.uhrig_pulse_positions_normalized)

    @property
    def uhrig_pulse_positions(self):
        return self.total_tau*self.uhrig_pulse_positions_normalized

    @property
    def uhrig_taus(self):
      return self.total_tau*self.uhrig_taus_normalized

    @property
    def eff_pulse_dur_waiting_time(self):
        return np.concatenate([[3/8.*self.rabi_period], 0.5*self.rabi_period*np.ones(self.number_of_pi_pulses), [3/8.*self.rabi_period]]) #effective duration of pi pulse for each waiting time. The effective phase evolution time during the pi/2 pulse is taken as half of the pulse duration

    @property
    def minimum_total_tau(self):
        return self.eff_pulse_dur_waiting_time[0]/self.uhrig_taus_normalized[0]

    @property
    def effective_durations_dd(self):
        if self.minimum_total_tau > self.total_tau:
            raise Exception('Waiting times smaller than zero are not allowed. '
                            'Total tau must be at least {} (current: {})'.format(self.minimum_total_tau, self.total_tau))
        return self.tau_list - self.eff_pulse_dur_waiting_time

# class DDAlpha2(DD):
#
#     def __init__(self, omega, target_Azz, wait=None, rf_frequency=None, spt=1, **kwargs):
#         DD.__init__(self, **kwargs)
#         self.wait = wait
#         self.rf_frequency = rf_frequency
#         self.spt = spt
#         # self.target_Azz = target_Azz
#         # self.fields = self.fields_varied_combined
#         # self.omega = omega
#
#     column_dict = {'mw': [0, 1], 'rf': [2, 3], 'wait': [4]}
#     controls = ['mw', 'rf', 'wait']
#     n_columns = 5
#
#     def sequence(self):
#         out = list(itertools.chain.from_iterable([['wait', 'rf', 'wait'] if i == 'wait' else [i] for i in super(DDAlpha2, self).sequence()]))[1:-1]
#         return ['rf']*self.spt + list(itertools.chain.from_iterable( [['rf']*2*self.spt if i == 'rf' else [i] for i in out[1:-1]])) + ['rf']*self.spt
#
#     # def times_raw(self):
#     #     tl = super(DDAlpha2, self).tau_list()
#     #     out = [(tl[0] - self.wait)/self.spt]*self.spt + [self.wait]
#     #     for
#     # def wait_list
#     #
#     # def tau_list(self):
#     #
#     #     out = [tl[0] - self.wait] + list(itertools.chain.from_iterable([[i - 2*self.wait] for i in tl]))[1:-1] + [tl[-1] - self.wait]
#     #     return [out[0]/float(self.spt)]*self.spt + [self.wait] + list(itertools.chain.from_iterable([[self.wait] +  [i/float(2*self.spt)]*2*self.spt + [self.wait] for i in out[1:-1]])) + [self.wait] + [out[0]/float(self.spt)]*self.spt
#
#     @property
#     def t_rf(self):
#         return 2*self.number_of_pi_pulses*self.spt*self.tau()
#
#     def tau(self):
#         out = self.target_t_rf / (2 * self.n_pi * self.spt)
#         if self.rf_frequency is not None:
#             out = np.around(out * float(self.rf_frequency)) / float(self.rf_frequency)
#         if self.time_digitization is not None:
#             out = np.around(out / float(self.time_digitization)) * float(self.time_digitization)
#         return out
#
#     def times_raw(self):
#         td = dict(rf=self.tau(),
#                   mw=0.5 * self.rabi_period,
#                   wait=self.wait)
#         return np.array([td[i] for i in self.sequence() if td[i] is not None])
#
#     @property
#     def effective_t_rf(self):
#         return self.t_rf - sum(self.vary_times.get('rf', [[], []])[1])
#
#     @property
#     def omega(self):
#         return self._omega
#
#     @omega.setter
#     def omega(self, val):
#         if val in [None, 'pi', '2pi'] or isinstance(val, numbers.Number):
#             if val in [None, 'pi']:
#                 o = 0.5 / self.effective_t_rf
#             elif val == '2pi':
#                 o = 1 / self.effective_t_rf
#             elif isinstance(val, numbers.Number):
#                 o = val
#             self._omega = o * np.ones(2 * self.n_pi * self.spt)
#         elif len(val) == 2 * self.n_pi * self.spt:
#             self._omega = val
#         else:
#             raise Exception('Error: {}'.format(val))
#
#     def alpha(self, n):
#         delta_alpha = 2 * np.pi * self.target_Azz * (self.tau() * self.spt + self.wait + 0.25 * self.rabi_period)
#         return int(np.ceil(n / 2.)) * (delta_alpha + np.pi) % (2 * np.pi)
#
#     def rf_array_aphi(self):
#         u_phi = np.repeat([self.alpha(n) for n in range(2 * self.n_pi)], self.spt)
#         return np.array([self.omega, u_phi]).T
#
#     def rf_array_xy(self):
#         rho, azim = tuple(np.hsplit(self.rf_array_aphi(), 2))
#         x, y, z = coordinates.sph2cart(rho, 0, azim)
#         return np.concatenate([x, y], axis=1)
#
#     def mw_array_xy(self):
#         return np.array([np.array([np.cos(phi), np.sin(phi)]) / self.rabi_period for phi in self.pi_phases])
#
#     def mw_array_aphi(self):
#         x, y = tuple(np.hsplit(self.mw_array_xy(), 2))
#         rho, elev, azim = coordinates.cart2sph(x, y, 0)
#         return np.concatenate([rho, azim], axis=1)
#
#     def fields_raw(self):
#         out = np.zeros((len(self.sequence()), self.n_columns))
#         mw_array = self.mw_array_xy()
#         rf_array_xy = self.rf_array_xy()
#         idxmw = 0
#         idxrf = 0
#         for i, val in enumerate(self.sequence()):
#             if val == 'mw':
#                 out[i, self.column_dict[val]] = mw_array[idxmw]
#                 idxmw += 1
#             elif val == 'rf':
#                 out[i, self.column_dict[val]] = rf_array_xy[idxrf]
#                 idxrf += 1
#         return out

class Concatenated(Arbitrary):

    def __init__(self, p_list, controls):
        for p in p_list:
            if not issubclass(type(p), Arbitrary):
                raise Exception('Error: type of p is {}'.type(p))
        self.p_list = p_list
        self.controls = controls
        self.check_control_length()

    def sequence(self):
        return list(itertools.chain(*[i.sequence() for i in self.p_list]))

    def check_control_length(self):
        for c in self.controls:
            if len(set([len(p.column_dict[c]) for p in self.p_list if c in p.column_dict])) != 1:
                raise Exception("Error. column_dict: {}".format(self.column_dict))
    @property
    def column_dict(self):
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
    def vary_times(self):
        out = {}
        for i, c in enumerate(self.controls):
            l = [1 for p in self.p_list if c in p.column_dict]
            if len(l) > 0:
                out[c] = [[], []]
            n_bins = 0
            for p in self.p_list:
                if c in p.vary_times:
                    out[c][0].extend(list(np.array(p.vary_times[c][0]) + n_bins))
                    out[c][1].extend(p.vary_times[c][1])
                n_bins += len(p.locations(c))
        return out

    @property
    def n_columns(self):
        return len(list(itertools.chain(*self.column_dict.values())))

    def extend_fields_raw(self, n):
        fields_raw = np.zeros([self.p_list[n].n_bins, self.n_columns])
        for c in self.controls:
            if c in self.p_list[n].column_dict:
                fields_raw[:, self.column_dict[c]] = self.p_list[n].fields_raw()[:, self.p_list[n].column_dict[c]]
        return fields_raw

    def extend_fields(self, n):
        fields = np.zeros([self.p_list[n].n_bins, self.n_columns])
        for c in self.controls:
            if c in self.p_list[n].column_dict:
                fields[:, self.column_dict[c]] = self.p_list[n].fields()[:, self.p_list[n].column_dict[c]]
        return fields

    def fields_raw(self):
        return np.concatenate([self.extend_fields_raw(i) for i in range(len(self.p_list))])

    def fields(self):
        return np.concatenate([self.extend_fields(i) for i in range(len(self.p_list))])

    def times_raw(self):
        return np.concatenate([i.times_raw() for i in self.p_list])

    def times(self, name=None):
        return np.concatenate([i.times(name=name) for i in self.p_list])
#
# if __name__ == '__main__':
#     kdd = DDAlpha2(omega=0.0, target_Azz=0.0, wait=0.5, dd_type='xy4', total_tau=100, spt=3)