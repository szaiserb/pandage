from __future__ import print_function, absolute_import, division
__metaclass__ = type

from . import data_handling

import numpy as np
import pandas as pd
import errno, shutil, traceback, zipfile, itertools, collections, datetime, sys, os

class DataGeneration:

    def __init__(self):
        super(DataGeneration, self).__init__()
        self.date_of_creation = datetime.datetime.now()
        self.remeasure_df = None

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
        if type(val) != collections.OrderedDict:
            raise Exception('Error: {}, {}'.format(type(val), val))
        for k, v in val.items():
            _ = data_handling.check_type(k, 'key', str)
            _ = data_handling.check_array_like(v, 'value_in_{}'.format(k))
            if len(v) == 0:
                raise Exception("Error: parameter {} has length zero {}".format(k))
            items_occuring_more_than_once = [item_val for item_val, number_of_occurences in collections.Counter(v).iteritems() if number_of_occurences > 1]
            if len(items_occuring_more_than_once) > 0:
                raise Exception("Error: parameter {} has duplicate items {}".format(k, items_occuring_more_than_once))
        self._parameters = val

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

    @property
    def data(self):
        return self.pld.data

    @data.setter
    def data(self, val):
        self.pld.data = val

    @property
    def pld(self):
        return self._pld

    @pld.setter
    def pld(self, val):
        self._pld = val

    def dropnan_data(self):
        ldf = len(self.data.df)
        self.data.df.dropna(axis=0, how='any', inplace=True)
        if ldf - len(self.data.df) > self.number_of_simultaneous_measurements:
            raise Exception('Error: there seem to be more nan values than can be expected from unfinished measurements ({}, {}). Whats wrong?'.format(ldf, len(self.data.df)))

    def init_iterator_df_done(self):
        if len(self.data.df) > 0:
            self.iterator_df_done = self.data.df.iloc[:, :len(self.parameters.keys())]
        else:
            self.iterator_df_done = self.iterator_df.drop(self.iterator_df.index, inplace=False)

    def iterator_df_drop_done(self):
        self.iterator_df = data_handling.df_drop_duplicate_rows(self.iterator_df, self.iterator_df_done, keep=False)

    def set_iterator_df(self):
        self.iterator_df = pd.DataFrame(
            list(itertools.product(*self.parameters.values())),
            columns=self.parameters.keys()
        )
        for cn in self.iterator_df.columns:
            setattr(self.iterator_df, cn, getattr(self.iterator_df, cn).astype(type(self.parameters[cn][0])))

    @property
    def progress(self):
        return getattr(self, '_progress', 0)

    def init_data(self, init_from_file=None, iff=None, move_folder=True):
        init_from_file = iff if iff is not None else init_from_file
        self.init_from_file = init_from_file
        self.move_folder = move_folder
        self.pld._data = data_handling.Data(
            parameter_names=self.parameters.keys(),
            observation_names=self.observation_names,
            dtypes=self.dtypes,
            init_from_file=init_from_file
        )
        self.dropnan_data()

    # def move_init_from_file_folder_back(self):
    #     # TODO: might be useful to use shutil.copy2 followed by shutil.rmtree to copy metadata (e.g. creation date)
    #     if self.init_from_file is not None:
    #         initial_folder = os.path.split(self.init_from_file)[0]
    #         current_folder = os.path.join(self.save_dir, "{}_tbc".format(os.path.split(initial_folder)[-1]))
    #         shutil.move(current_folder, initial_folder)

    def update_current_str(self):
        if hasattr(self, 'current_iterator_df') and len(self.current_iterator_df) > 0:
            cid = self.current_iterator_df.iloc[-1, :].to_dict()
            cps = ""
            cis = ""
            for key, val in cid.items():
                cps += "{}: {}\n".format(key, val)
                try:
                    cis += "{}: {} ({})\n".format(key, list(self.parameters[key]).index(val), len(self.parameters[key]))
                except:
                    pass
            self.pld.update_info_text('State: ' + self.state + '\n\n' + 'Current parameters:\n' + cps + '\n\n' + 'Current indices\n' + cis)

    def init_run(self, **kwargs):
        self.state = 'run'
        self.reinit()
        self.set_iterator_df()
        self.init_data(**kwargs)
        self.init_iterator_df_done()
        self.iterator_df_drop_done()

    def iterator_df_pop(self, n):
        out = self.iterator_df.head(n)
        self.iterator_df = self.iterator_df.iloc[n:, :]
        return out

    def update_progress(self):
        self._progress = len(self.iterator_df_done) / np.prod([len(i) for i in self.parameters.values()])

    def iterator(self):
        while len(self.iterator_df) > 0:
            if hasattr(self, 'current_iterator_df'):
                self.iterator_df_done = self.iterator_df_done.append(self.current_iterator_df)
            self.set_iterator_df()
            self.iterator_df_drop_done()
            self.process_remeasure_items()
            self.update_progress()
            self.current_iterator_df = self.iterator_df_pop(min(self.number_of_simultaneous_measurements, len(self.iterator_df)))
            self.data.append(self.current_iterator_df)
            self.update_current_str()
            yield self.current_iterator_df
        if hasattr(self, 'current_iterator_df'):
            self.iterator_df_done = self.iterator_df_done.append(self.current_iterator_df)
            del self.current_iterator_df
        self.update_progress()

    def reinit(self):
        self.start_time = datetime.datetime.now()
        if hasattr(self, 'current_iterator_df'):
            del self.current_iterator_df
        if hasattr(self, '_file_path') and hasattr(self, '_file_name'):
            self.pld.data_path = "{}/data.hdf".format(self.save_dir)

    @property
    def session_meas_count(self):
        if len(self.data.df) == 0 or len(self.iterator_df_done) == 0:
            return 0
        else:
            return len(self.iterator_df_done) - self.data.df[self.data.df.start_time > self.start_time].index[0]

    @property
    def save_dir(self):
        return "{}/{}_{}".format(self.file_path, self.start_time.strftime('%Y%m%d-h%Hm%Ms%S'), self.file_name)

    def make_save_dir(self):
        try:
            os.makedirs(self.save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        except Exception:
            print(self.save_dir)
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def save_dir_tmp(self):
        return "{}/tmp".format(self.save_dir)

    def make_save_dir_tmp(self):
        try:
            os.makedirs(self.save_dir_tmp)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        except Exception:
            print(self.save_dir_tmp)
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, val):
        self._file_name = val

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, val):
        self._file_path = val

    def make_save_location_params(self, script_path, **kwargs):
        script_path = os.path.abspath(script_path)
        self.file_name = str(os.path.splitext(os.path.basename(script_path))[0])
        folder = kwargs['folder'] if 'folder' in kwargs else os.path.dirname(os.path.dirname(script_path))
        if 'sub_folder_kw' in kwargs:
            fnl = script_path.split('\\')
            fnl = fnl[fnl.index(kwargs['sub_folder_kw']) + 1:]
            fnl[-1] = fnl[-1].split('.')[0]
            subfolder = "/".join(fnl)
            self.file_path = str(os.path.join(folder, subfolder))
        else:
            self.file_path = str(folder)

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

    def save(self, name='', notify=False):
        if not os.path.exists(self.save_dir):
            self.make_save_dir()
        if self.init_from_file is not None and self.move_folder:
            new_iff_path = os.path.join(self.save_dir, os.path.basename(os.path.dirname(self.init_from_file)) + '_tbc', os.path.basename(self.init_from_file))
            if not os.path.exists(new_iff_path):
                # TODO: might be useful to use shutil.copy2 followed by shutil.rmtree to copy metadata (e.g. creation date)
                folder = os.path.dirname(self.init_from_file)
                os.rename(folder, folder+'_tbc')
                shutil.move(folder+'_tbc', self.save_dir)
        if len(self.iterator_df_done) >= 0:
            import time
            t0 = time.time()
            if hasattr(self, 'file_notes'):
                with open("{}/notes.dat".format(self.save_dir), "w") as text_file:
                    text_file.write(self.file_notes)
            t1 = time.time()-t0
            if hasattr(self, '_meas_code'):
                with open("{}/meas_code.py".format(self.save_dir), "w") as text_file:
                    text_file.write(self.meas_code)
            t2 = time.time() - t0 - t1
            # self.data.save("{}/data.csv".format(self.save_dir)) #this takes forever
            t3 = time.time() - t0 - t1 - t2
            self.data.save("{}/data.hdf".format(self.save_dir))
            t4 = time.time() - t0 - t1 - t2 - t3
            self.pld.save_plot("{}/plot.png".format(self.save_dir))
            t5 = time.time() - t0 - t1 - t2 - t3 - t4
            if notify:
                print("saved {} to '{} ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(name, self.save_dir, t1, t2, t3, t4, t5))

    def remeasure(self, df):
        indices = self.data.df[self.data.df.iloc[:, :len(self.parameters.keys())].isin(df).all(axis=1)].index #get indices in self.df.data to be replaced
        if len(indices) != len(df):
            raise Exception('Not all rows in df could be found in self.data: (missing: {})'.format(len(df) - len(indices)))
        elif indices[0]% self.number_of_simultaneous_measurements != 0:
            raise Exception('Error: The first index to be remeasured must be an integer multiple of number_of_simultaneous_measurements!\n{} {} {}'.format(df, indices, self.number_of_simultaneous_measurements))
        elif len(indices) % self.number_of_simultaneous_measurements != 0:
            raise Exception('Error: length of indices to be remeasured must be an integer multiple of number_of_simultaneous_measurements!\n{} {} {}'.format(df, indices, self.number_of_simultaneous_measurements))
        elif not ((indices[1:] - indices[:-1]) == 1).all():
            raise Exception("Error: Only connected packets of indices are allowed, i.e. monotonously increasing with stepsize 1".format(indices))
        self.remeasure_df = df

    def process_remeasure_items(self):
        if self.remeasure_df is not None:
            self.iterator_df.append(self.remeasure_df)
            self.iterator_df_done.append(self.remeasure_df)
            self.iterator_df_done.drop_duplicates(keep=False, inplace=True)
            self.data.df.drop_duplicates(keep=False, inplace=True, subset=self.data.parameter_names)
            self.remeasure_df = None

