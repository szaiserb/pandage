from __future__ import print_function, absolute_import, division
__metaclass__ = type

import sys
if sys.version_info.major == 2:
    from imp import reload
else:
    from importlib import reload

import errno
import shutil
import traceback
import datetime
from . import data_handling
import qutip_enhanced.analyze as qta; reload(qta)
import os
import itertools


class DataGeneration:

    current_idx_str = data_handling.ret_property_typecheck('current_idx_str', str) #######
    current_parameter_str = data_handling.ret_property_typecheck('current_parameter_str', str) #######

    # Save Data:
    file_path = data_handling.ret_property_typecheck('file_path', str)
    file_name = data_handling.ret_property_typecheck('file_name', str)
    file_notes = data_handling.ret_property_typecheck('file_notes', str)
    meas_code = data_handling.ret_property_typecheck('meas_code', str)
    state = data_handling.ret_property_list_element('state', ['idle', 'run'])

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        self._parameters = itertools.OrderedDict((data_handling.check_type(k, 'key', str), data_handling.check_array_like(v, 'val')) for k, v in data_handling.check_type(val, 'parameters', OrderedDict).items())
        self.plot_type = 'scatter'
        if len(self._parameters.items()[-1][1]) > 2:
            self.plot_type = 'line'

    @property
    def observation_names(self):
        raise NotImplementedError

    @property
    def number_of_simultaneous_measurements(self):
        if hasattr(self, '_number_of_simultaneous_measurements'):
            return self._number_of_simultaneous_measurements
        else:
            return 1

    @number_of_simultaneous_measurements.setter
    def number_of_simultaneous_measurements(self, val):
        if type(val) is int:
            self._number_of_simultaneous_measurements = val
        else:
            raise Exception('Error: {}'.format(val))

    def set_iterator_list_done(self):
        itld = self.data.df[self.parameters.keys()].values
        itidxld = self.data.df[[self.parameters.keys()[i]+'_idx' for i in range(len(self.parameters.keys()))]].values
        self.iterator_list_done = [tuple(itld[i]) for i in range((len(itld)))]
        self.iterator_idx_list_done = [tuple(itidxld[i]) for i in range((len(itidxld)))]

    def set_iterator_list(self):
        self.iterator_list = list(itertools.product(*self.parameters.values()))
        self.iterator_idx_list = list(itertools.product(*[range(len(i)) for i in self.parameters.values()]))
        for v, idx in zip(self.iterator_list_done, self.iterator_idx_list_done):
            try:
                self.iterator_list.remove(v)
                self.iterator_idx_list.remove(idx)
            except:
                pass

    def init_data(self, init_from_file=None, iff=None):
        init_from_file = iff if iff is not None else init_from_file
        self.data = data_handling.Data(
            parameter_names=self.parameters.keys() + [self.parameters.keys()[i]+'_idx' for i in range(len(self.parameters.keys()))],
            observation_names=self.observation_names,
            dtypes=self.dtypes
        )
        self.data.init(init_from_file=init_from_file)
        if hasattr(self, 'pld'):
            self.pld.data = self.data
        if init_from_file is not None:
            folder = os.path.split(init_from_file)[0]
            os.rename(folder, folder+'_tbc')
            shutil.move(folder+'_tbc', self.save_dir)

    def init_gui(self, title):
        self.pld = data_handling.PlotData(title)
        if hasattr(self, 'data'):
            self.pld.data = self.data
        self.pld.show()

    def update_current_str(self):
        self.current_idx_str = "\n".join(["{}: {} ({})".format(key, int(self.current_indices_dict_list[0]["{}_idx".format(key)]), len(val)) for key, val in self.parameters.items()])
        self.current_parameter_str = "\n".join(["{}: {}".format(key, self.current_parameters_dict_list[0][key]) for key in self.parameters.keys()])
        self.pld.info.setPlainText('State: ' + self.state + '\n\n' + 'Current parameters:\n' + self.current_parameter_str + '\n\n' + 'Current indices\n' + self.current_idx_str)

    def init_run(self, init_from_file=None, iff=None):
        self.state = 'run'
        self.reinit()
        self.init_data(init_from_file=init_from_file, iff=iff)
        self.set_iterator_list_done()
        self.set_iterator_list()

    def iterator(self):
        if hasattr(self, 'pv_l'):
            self.iterator_list_done.extend(self.pv_l)
        if hasattr(self, 'pidx_l'):
            self.iterator_idx_list_done.extend(self.pidx_l)
        while len(self.iterator_list) > 0:
            self.pv_l = [self.iterator_list.pop(0) for _ in range(min(self.number_of_simultaneous_measurements, len(self.iterator_list)))]
            self.pidx_l = [self.iterator_idx_list.pop(0) for _ in range(min(self.number_of_simultaneous_measurements, len(self.iterator_idx_list)))]
            self.current_parameters_dict_list = [itertools.OrderedDict([(key, pv[i]) for i, key in enumerate(self.parameters.keys())]) for pv in self.pv_l]
            self.current_indices_dict_list = [itertools.OrderedDict([("{}_idx".format(key), pidx[i]) for i, key in enumerate(self.parameters.keys())]) for pidx in self.pidx_l]
            l = [itertools.OrderedDict(i.items() + j.items()) for i, j in zip(self.current_parameters_dict_list, self.current_indices_dict_list)]
            self.data.append(l)
            self.update_current_str()
            yield l

    def reinit(self):
        self.start_time = datetime.datetime.now()

    @property
    def save_dir(self):
        sts = self.start_time.strftime('%Y%m%d-h%Hm%Ms%S')
        save_dir = "{}/{}_{}".format(self.file_path, sts, self.file_name)
        try:
            os.makedirs(save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        except Exception:
            print(save_dir)
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        return save_dir

    @property
    def save_dir_tmp(self):
        save_dir_tmp = "{}/tmp".format(self.save_dir)
        try:
            os.makedirs(save_dir_tmp)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return save_dir_tmp

    def save(self):
        fpp = "{}/".format(self.save_dir)
        self.pld.fig.savefig("{}plot.png".format(fpp))
        if self.file_notes != '':
            with open("{}notes.dat".format(fpp), "w") as text_file:
                text_file.write(self.file_notes)
        if self.meas_code != '':
            with open("{}meas_code.py".format(fpp), "w") as text_file:
                text_file.write(self.meas_code)
        self.data.save("{}/data.csv".format(self.save_dir))
        self.data.save("{}/data.hdf".format(self.save_dir))
        print("saved nuclear to '{}'".format(fpp))

# import time
# from pi3diamond import pi3d
# import misc
# from numbers import Number
# import zipfile
#
# class NuclearOPs(DataGeneration):
#
#     # Tracking stuff:
#     refocus_interval = misc.ret_property_typecheck('refocus_interval', int)
#     odmr_interval = misc.ret_property_typecheck('odmr_interval', Number)
#     additional_recalibration_interval = misc.ret_property_typecheck('additional_recalibration_interval', int)
#     initial_confocal_odmr_refocus = misc.ret_property_typecheck('initial_confocal_odmr_refocus', bool)
#
#     @property
#     def analyze_type(self):
#         return self.ana_trace.analyze_type
#
#     @analyze_type.setter
#     def analyze_type(self, val):
#         self.ana_trace.analyze_type = val
#
#     @property
#     def observation_names(self):
#         return ['result_{}'.format(i) for i in range(self.number_of_results)] + ['trace', 'events', 'thresholds', 'start_time', 'end_time', 'local_oscillator_freq']
#
#     @property
#     def dtypes(self):
#         if not hasattr(self, '_dtypes'):
#             self._dtypes = dict(trace='object', events='int', start_time='str', end_time='str', local_oscillator_freq='float', thresholds='object')
#         return self._dtypes
#
#     @dtypes.setter
#     def dtypes(self, val):
#         if isinstance(val, dict):
#             for k, v in val.items():
#                 if not (k in self.parameters.keys() or isinstance(v, str)):
#                     raise Exception("Error: {}".format(val))
#             self._dtypes = val
#
#     @property
#     def data_observation_names(self):
#         return ['result_{}'.format(i) for i in range(self.number_of_results)] + ['trace', 'events', 'thresholds', 'start_time', 'end_time', 'local_oscillator_freq']
#
#     @property
#     def number_of_results(self):
#         if self.analyze_type == 'tomography':
#             return 2**len([[i[0]] for i in self.ana_trace.analyze_sequence if i[0] == 'result'])
#         if self.analyze_type == 'multifreq':
#             return self.ana_trace.analyze_sequence[-1][-1]
#         if getattr(self, 'ana_trace', None) is not None:
#             return len([[i[0]] for i in self.ana_trace.analyze_sequence if i[0] == 'result'])
#         else:
#             return 1
#
#     def run(self, abort, init_from_file=None, iff=None):
#         self.init_run(init_from_file=init_from_file, iff=iff)
#         try:
#             pi3d.microwave.On()
#             for l in self.iterator():
#                 if abort.is_set(): break
#                 self.do_refocusodmr(abort, l)
#                 if abort.is_set(): break
#                 self.setup_rf(l)
#                 if abort.is_set(): break
#                 self.data.set_observations([OrderedDict(local_oscillator_freq=pi3d.tt.current_local_oscillator_freq)]*len(self.pv_l))
#                 self.data.set_observations([OrderedDict(start_time=datetime.datetime.now().strftime('%Y%m%d-h%Hm%Ms%S'))]*len(self.pv_l))
#                 trace = self.get_trace(abort)
#                 if abort.is_set(): break
#                 self.data.set_observations([OrderedDict(end_time=datetime.datetime.now().strftime('%Y%m%d-h%Hm%Ms%S'))]*len(self.pv_l))
#                 self.data.set_observations([OrderedDict(trace=trace)]*len(self.pv_l))
#                 if abort.is_set(): break
#                 self.analyze(trace=trace)
#                 if abort.is_set(): break
#                 self.save()
#                 if abort.is_set(): break
#                 self.iterator_list_done.extend(self.pv_l)
#                 self.iterator_idx_list_done.extend(self.pidx_l)
#                 self.set_iterator_list()
#         except Exception:
#             abort.set()
#             exc_type, exc_value, exc_tb = sys.exc_info()
#             traceback.print_exception(exc_type, exc_value, exc_tb)
#         finally:
#             pi3d.multi_channel_awg_sequence.stop_awgs()
#             self.state = 'idle'
#             self.update_current_str()
#
#     def reinit(self):
#         super(NuclearOPs, self).reinit()
#         self.odmr_count = 0
#         self.additional_recalibration_interval_count = 0
#         self.initial_confocal_odmr_refocus = False
#         self.last_odmr = {True: -np.inf, False: time.time()}[self.initial_confocal_odmr_refocus]
#         self.last_rabi_refocus = time.time()
#
#     def do_refocusodmr(self, abort, ls):
#         delta_t = time.time() - self.last_odmr
#         if self.odmr_interval != 0 and (delta_t >= self.odmr_interval):
#             if self.refocus_interval != 0 and self.odmr_count % self.refocus_interval == 0:
#                 pi3d.confocal.run_refocus()
#             pi3d.odmr.external_stop_request = abort
#             pi3d.odmr.do_frequency_refocus()
#             self.odmr_count += 1
#             self.last_odmr = time.time()
#         if self.additional_recalibration_interval != 0 and self.additional_recalibration_interval_count % self.additional_recalibration_interval == 0:
#             self.additional_recalibration_fun(ls)
#         self.additional_recalibration_interval_count += 1
#         pi3d.odmr.set_odmr_f(pi3d.tt.current_local_oscillator_freq)
#         pi3d.save_values_to_file([pi3d.odmr.odmr_contrast], 'odmr_contrast')
#
#     def get_trace(self, abort):
#         self.mcas.initialize_sequence(abort=abort)
#         pi3d.gated_counter.count(abort, ch_dict=self.mcas.ch_dict)
#         return np.array(pi3d.gated_counter.trace.copy(), dtype=np.int)
#
#     def setup_rf(self, ls):
#         self.mcas = self.ret_mcas(ls)
#         pi3d.mcas_dict[self.mcas.seq_name] = self.mcas
#         self.mcas.write_seq()
#
#     def analyze(self, trace):
#         if self.analyze_type is not None:
#             pi3d.trace = trace
#             self.ana_trace.raw_trace = trace
#             analysis = self.ana_trace.analyze(analyze_type=self.analyze_type, number_of_simultaneous_measurements=self.number_of_simultaneous_measurements)
#             thresholds = self.ana_trace.thresholds
#             self.data.set_observations([OrderedDict(('result_{}'.format(i), result) for i, result in enumerate(results)) for results in analysis['results']])
#             self.data.set_observations([OrderedDict(events=events) for events in analysis['events']])
#             self.data.set_observations([OrderedDict(thresholds=thresholds)]*self.number_of_simultaneous_measurements)
#             self.pld.data = self.data
#             for results, events in zip(analysis['results'], analysis['events']):
#                 print("result: {results}, events: {events}, thresholds: {thresholds}".format(thresholds=thresholds, results=results, events=events))
#
#     def save(self):
#         super(NuclearOPs, self).save()
#         self.save_pi3diamond(destination_dir=self.save_dir)
#
#     def save_pi3diamond(self, destination_dir):
#         f = '{}/pi3diamond.zip'.format(destination_dir)
#         if not os.path.isfile(f):
#             zf = zipfile.ZipFile(f, 'a')
#             for fp in os.listdir(os.getcwd()):
#                 if fp.endswith('.py'):
#                     zf.write(fp)
#             zf.close()
#
#     def reset_settings(self):
#         """
#         Here only settings are changed that are not automatically changed during run()
#         :return:
#         """
#         self.additional_recalibration_interval = 0
#         self.ret_mcas = None
#         self.mcas = None
#         self.refocus_interval = 2
#         self.odmr_interval = 15
#         self.file_path = 'D:/data/NuclearOPs/'
#         self.file_name = ''
#         self.file_notes = ''
#         self.meas_code = ''
#         self.thread = None