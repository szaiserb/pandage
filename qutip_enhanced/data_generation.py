from __future__ import print_function, absolute_import, division
__metaclass__ = type

import sys
if sys.version_info.major == 2:
    from imp import reload
else:
    from importlib import reload

import numpy as np
import errno
import shutil
import traceback
import datetime
import time
import zipfile
from . import data_handling
import qutip_enhanced.analyze as qta; reload(qta)
import os
import itertools
import collections
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QObject
import datetime

class DataGeneration(QObject):

    def __init__(self, parent=None):
        super(DataGeneration, self).__init__(parent)
        self.update_current_str_signal.connect(self.update_current_str)
        self.date_of_creation = datetime.datetime.now()
        self.remeasure_items = None
        self.remeasure_indices = None

    current_idx_str = data_handling.ret_property_typecheck('current_idx_str', str) #######
    current_parameter_str = data_handling.ret_property_typecheck('current_parameter_str', str) #######

    # Save Data:
    file_path = data_handling.ret_property_typecheck('file_path', str)
    file_name = data_handling.ret_property_typecheck('file_name', str)
    file_notes = data_handling.ret_property_typecheck('file_notes', str)
    meas_code = data_handling.ret_property_typecheck('meas_code', str)
    state = data_handling.ret_property_list_element('state', ['idle', 'run'])

    update_current_str_signal = pyqtSignal()
    show_gui_signal = pyqtSignal()
    close_gui_signal = pyqtSignal()

    @property
    def parameters(self):
        try:
            return self._parameters
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)


    @parameters.setter
    def parameters(self, val):
        self._parameters = collections.OrderedDict((data_handling.check_type(k, 'key', str), data_handling.check_array_like(v, 'val')) for k, v in data_handling.check_type(val, 'parameters', collections.OrderedDict).items())
        self.plot_type = 'scatter'
        if len(self._parameters.items()[-1][1]) > 2:
            self.plot_type = 'line'

    @property
    def observation_names(self):
        try:
            raise NotImplementedError
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def number_of_simultaneous_measurements(self):
        try:
            if hasattr(self, '_number_of_simultaneous_measurements'):
                return self._number_of_simultaneous_measurements
            else:
                return 1
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

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

    @property
    def data(self):
        try:
            if hasattr(self, '_pld'):
                return self.pld.data
            else:
                return self._data
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @data.setter
    def data(self, val):
        if hasattr(self, '_pld'):
            self.pld.data = val
        else:
            self._data = val

    @property
    def pld(self):
        try:
            return self._pld
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @pld.setter
    def pld(self, val):
        if hasattr(val, '_data'):
            raise Exception('Error!')
        self._pld = val
        if hasattr(self, '_data'):
            self.pld.data = self._data
            del self._data
        self.show_gui_signal.connect(self.show_gui)
        self.close_gui_signal.connect(self.close_gui)


    def set_iterator_list(self):
        self.iterator_list = list(itertools.product(*self.parameters.values()))
        self.iterator_idx_list = list(itertools.product(*[range(len(i)) for i in self.parameters.values()]))
        for v, idx in zip(self.iterator_list_done, self.iterator_idx_list_done):
            try:
                self.iterator_list.remove(v)
                self.iterator_idx_list.remove(idx)
            except:
                pass

    @property
    def progress(self):
        try:
            return getattr(self, '_progress', 0)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def init_data(self, init_from_file=None, iff=None, move_folder=True):
        init_from_file = iff if iff is not None else init_from_file
        self.init_from_file = init_from_file
        self.data = data_handling.Data(
            parameter_names=self.parameters.keys() + [self.parameters.keys()[i]+'_idx' for i in range(len(self.parameters.keys()))],
            observation_names=self.observation_names,
            dtypes=self.dtypes
        )
        self.data.init(init_from_file=init_from_file)
        if init_from_file is not None and move_folder:
            # TODO: might be useful to use shutil.copy2 followed by shutil.rmtree to copy metadata (e.g. creation date)
            folder = os.path.split(init_from_file)[0]
            os.rename(folder, folder+'_tbc')
            shutil.move(folder+'_tbc', self.save_dir)

    def move_init_from_file_folder_back(self):
        # TODO: might be useful to use shutil.copy2 followed by shutil.rmtree to copy metadata (e.g. creation date)
        if self.init_from_file is not None:
            initial_folder = os.path.split(self.init_from_file)[0]
            current_folder = os.path.join(self.save_dir, "{}_tbc".format(os.path.split(initial_folder)[-1]))
            shutil.move(current_folder, initial_folder)

    def init_gui(self, title):
        self.pld = data_handling.PlotData(title)
        self.pld.show()

    def update_current_str(self):
        self.current_idx_str = "\n".join(["{}: {} ({})".format(key, int(self.current_indices_dict_list[0]["{}_idx".format(key)]), len(val)) for key, val in self.parameters.items()])
        self.current_parameter_str = "\n".join(["{}: {}".format(key, self.current_parameters_dict_list[0][key]) for key in self.parameters.keys()])
        self.pld.info.setPlainText('State: ' + self.state + '\n\n' + 'Current parameters:\n' + self.current_parameter_str + '\n\n' + 'Current indices\n' + self.current_idx_str)

    def init_run(self, **kwargs):
        self.state = 'run'
        self.reinit()
        self.init_data(**kwargs)
        self.set_iterator_list_done()
        self.set_iterator_list()

    def show_gui(self):
        self.pld.show()

    def close_gui(self):
        self.pld.close()

    def iterator(self):
        while len(self.iterator_list) > 0:

            if hasattr(self, 'pv_l'):
                self.iterator_list_done.extend(self.pv_l)
            if hasattr(self, 'pidx_l'):
                self.iterator_idx_list_done.extend(self.pidx_l)
            self.process_remeasure_items()
            self._progress = len(self.iterator_list_done) / np.prod([len(i) for i in self.parameters.values()])
            self.pv_l = [self.iterator_list.pop(0) for _ in range(min(self.number_of_simultaneous_measurements, len(self.iterator_list)))]
            self.pidx_l = [self.iterator_idx_list.pop(0) for _ in range(min(self.number_of_simultaneous_measurements, len(self.iterator_idx_list)))]
            self.current_parameters_dict_list = [collections.OrderedDict([(key, pv[i]) for i, key in enumerate(self.parameters.keys())]) for pv in self.pv_l]
            self.current_indices_dict_list = [collections.OrderedDict([("{}_idx".format(key), pidx[i]) for i, key in enumerate(self.parameters.keys())]) for pidx in self.pidx_l]
            l = [collections.OrderedDict(i.items() + j.items()) for i, j in zip(self.current_parameters_dict_list, self.current_indices_dict_list)]
            self.data.append(l)
            import time
            time.sleep(1)
            self.update_current_str_signal.emit()
            yield l
        self._progress = len(self.iterator_list_done) / np.prod([len(i) for i in self.parameters.values()])

    def reinit(self):
        self.start_time = datetime.datetime.now()

    @property
    def save_dir(self):
        try:
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
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def save_dir_tmp(self):
        try:
            save_dir_tmp = "{}/tmp".format(self.save_dir)
            try:
                os.makedirs(save_dir_tmp)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            return save_dir_tmp
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def save_qutip_enhanced(self, destination_dir):
        src = r'D:\Python\qutip_enhanced\qutip_enhanced'
        f = r'{}/qutip_enhanced.zip'.format(destination_dir)
        if not os.path.isfile(f):
            zf = zipfile.ZipFile(f, 'a')
            for root, dirs, files in os.walk(src):
                if not '__pycach__' in root:
                    for file in files:
                        if not any([file.endswith(i) for i in ['.pyc', '.orig']]):
                            zf.write(os.path.join(root, file), os.path.join(root.replace(os.path.commonprefix([root, src]), ""), file))
            zf.close()

    def save(self):
        fpp = "{}/".format(self.save_dir)
        try:
            self.pld.save_plot("{}plot.png".format(fpp))
        except:
            pass
        if self.file_notes != '':
            with open("{}notes.dat".format(fpp), "w") as text_file:
                text_file.write(self.file_notes)
        if self.meas_code != '':
            with open("{}meas_code.py".format(fpp), "w") as text_file:
                text_file.write(self.meas_code)
        self.data.save("{}/data.csv".format(self.save_dir))
        self.data.save("{}/data.hdf".format(self.save_dir))
        print("saved nuclear to '{}'".format(fpp))

    def remeasure(self, l):
        if type(l) is not list:
            raise Exception('Error: l must be a list of items in iterator_list_done')
        for idx, item in enumerate(l):
            if type(item) in [list, tuple]:
                l[idx] = collections.OrderedDict([(key, val) for key, val in zip([i for i in self.data.parameter_names if not '_idx' in i], item)])
        remeasure_indices = self.data.dict_access(l).index
        if len(remeasure_indices) != len(l):
            raise Exception('Error: Not every item in l could be found in the dataframe!\n{}, {}, {}'.format(len(l), len(self.remeasure_indices), l))
        if self.number_of_simultaneous_measurements != 1:
            if len(l)%self.number_of_simultaneous_measurements != 0:
                raise Exception('Error: length of indices to be remeasured must be an integer multiple of number_of_simultaneous_measurements!\n{} {} {}'.format(l, remeasure_indices, self.number_of_simultaneous_measurements))
            if remeasure_indices[0]%self.number_of_simultaneous_measurements != 0:
                raise Exception('Error: {} {}'.format(remeasure_indices, self.number_of_simultaneous_measurements))
            if not all(y==x+1 for x, y in zip(remeasure_indices, remeasure_indices[1:])):
                raise Exception("Error: Only connected packets of indices are allowed, i.e. monotonously increasing with stepsize 1".format(remeasure_indices))
        self.remeasure_indices = remeasure_indices
        self.remeasure_items = l

    def process_remeasure_items(self):
        if self.remeasure_items is not None:
            if self.remeasure_indices is None:
                print('Something went horribly wrong! {}'.format(self.remeasure_items))
                return
            self.iterator_list.extend([i for idx, i in enumerate(self.iterator_list) if idx in self.remeasure_indices ])
            self.iterator_idx_list.extend([i for idx, i in enumerate(self.iterator_list) if idx in self.remeasure_indices])
            self.iterator_list_done = [self.iterator_list_done[i] for i in xrange(len(self.iterator_list_done )) if i not in self.remeasure_indices ]
            self.iterator_idx_list_ = [self.iterator_idx_list_done[i] for i in xrange(len(self.iterator_idx_list_done )) if i not in self.remeasure_indices ]
            self.data.dict_delete(self.remeasure_items)
            self.remeasure_items = None
            self.remeasure_indices = None
