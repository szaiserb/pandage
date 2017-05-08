# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals, division

__metaclass__ = type

import sys, os
if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

import matlab.engine

from qutip_enhanced import *
import shutil
import matlab.engine
import itertools
import datetime
import time
from . import temporary_GRAPE_Philipp_misc as misc
import threading

class DynPython:

    def __init__(self, dynamo_path, initial, final, dims=None, sample_frequency=12e3, use_engine=None, print_flag=True):
        self.dims = dims
        self.eng = self.get_eng(use_engine=use_engine)
        self.sample_frequency = sample_frequency
        self.dynamo_path = dynamo_path
        self.out = StringIO()
        self.err = StringIO()
        self.already_printed = ''
        self.initial = initial
        self.final = final
        self.print_flag = print_flag

    def get_eng(self, use_engine=None):
        if type(use_engine) == matlab.engine.matlabengine.MatlabEngine:
            print('using existing engine {}'.format(use_engine.eval('matlab.engine.engineName')))
            return use_engine
        m = matlab.engine.find_matlab()
        if use_engine is not None and use_engine in m:
            eng = matlab.engine.connect_matlab(use_engine)
            print('connected to {}'.format(use_engine))
            return eng
        mdp = [float(i[3:]) for i in m if 'dpe' in i]
        il = itertools.count()
        while True:
            n = next(il)
            if not n in mdp:
                break
        name = "dpe{}".format(n) if use_engine is None else use_engine
        eng = matlab.engine.start_matlab()
        eng.eval("matlab.engine.shareEngine('{}')".format(name), nargout=0)
        print('started matlab engine {}'.format(name))
        return eng

    @property
    def fields_names_list(self):
        field_names_list = self._fields_names if hasattr(self, '_fields_names') else ["{}".format(i) for i in range(len(self.export_mask_list))]
        return field_names_list
        # if hasattr(self, '_fields_names'):
        #     return self._fields_names
        # else:
        #     return ["{}".format(i) for i in range(len(self.export_mask_list))]

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, val):
        if not(type(val) is list or (type(val) is np.ndarray and len(val.shape) == 1)):
            raise Exception("Error: {}".format(val))
        self._dims = val

    @fields_names_list.setter
    def fields_names_list(self, val):
        self._fields_names = val

    def print_out(self):
        if self.print_flag:
            to_print = self.out.getvalue()
            print(to_print.replace(self.already_printed, '')) #dont use strip())
            self.already_printed = to_print

    @property
    def dyn(self):
        return self.eng.workspace[str("dyn")]

    def set_dyn(self, task, H_drift_list, H_ctrl_list, weight):
        self.eng.addpath(self.eng.genpath(self.dynamo_path))
        initial = matlab.double(self.initial.data.todense().tolist(), is_complex=True)
        final = matlab.double(self.final.data.todense().tolist(), is_complex=True)
        self._dyn = self.eng.dynamo(task, initial, final, H_drift_list, H_ctrl_list, weight, stdout=self.out, stderr=self.err)
        self.eng.workspace[str("dyn")] = self._dyn
        self.print_out()

    @property
    def export_mask_list(self):
        if not hasattr(self, '_export_mask_list') or self._export_mask_list is None:
            return [self.get_mask()[:self.n_bins, :]]
        else:
            return self._export_mask_list

    @export_mask_list.setter
    def export_mask_list(self, val):
        if type(val) is not list:
            raise Exception('Error, {}'.format(val))
        columns = self.get_mask().shape[1] - 1 # last column of the optimize mask are timeslices
        for i, v in enumerate(val):
            if v.shape != (self.n_bins, columns):
                raise Exception('Error: i: {}, v.shape: {}, bins: {}, columns: {}'.format(i, v.shape, self.n_bins, columns))
        self._export_mask_list = val

    @property
    def sequence(self):
        seq = []
        for i in range(self.n_bins):
            step = []
            for j, em in enumerate(self.export_mask_list):
                if np.any(em[i]):
                    step.append(self.fields_names_list[j])
            seq.append(step)
        return seq

    def export_mask_columns(self, n):
        return np.where(~np.all(self.export_mask_list[n] == 0, axis=0))[0]

    def export_mask_rows(self, n):
        if not self.export_mask_list[n].any():
            return np.array([])
        else:
            return np.where(self.export_mask_list[n][:, np.where(~np.all(self.export_mask_list[n] == 0, axis=0))[0][0]])[0]

    def set_labels(self, title, c_labels):
        self.eng.eval(str("dyn.system.set_labels('{}', {}, {{{}}})").format(title, self.dims, ", ".join("'{}'".format(i) for i in c_labels)), stdout=self.out, stderr=self.err, nargout=0)

    def get_mask(self, optimize_times=False):
        # if optimize_times is None:
        #     return np.array(self.eng.eval("dyn.opt.control_mask"))
        # else:
        return np.array(self.eng.full_mask(self.dyn, int(optimize_times)))

    def seq_init(self, n_bins, TL, control_type, control_par):
        self.eng.seq_init(self.dyn, n_bins, TL, control_type, control_par, stdout=self.out, stderr=self.err, nargout=0)
        self.print_out()

    def set_controls(self, fields):
        self.eng.set_controls(self.dyn, fields, stdout=self.out, stderr=self.err, nargout=0)
        self.print_out()

    def open_ui(self):
        self.eng.ui_open(self.dyn, nargout=0)

    def search_thread(self, mask, options=None, dt=2., dt_walltime=120., stop_too_bad_fun=None, abort=None, kill=None):
        options = {} if options is None else options
        mask = matlab.logical(mask.tolist())
        def run():
            if 'max_walltime' in options:
                n = range(int(options['max_walltime']/dt))
                options['max_walltime'] = dt
                t0 = time.time()
                for _ in n:
                    if abort is not None and abort.is_set(): break
                    if kill is not None and kill.is_set(): break
                    self.eng.search(self.dyn, mask, options, stdout=self.out, stderr=self.err, nargout=0)
                    frob_norm = np.sqrt(2 * self.eng.compute_error(self.dyn) * self.eng.eval("dyn.system.norm2"))
                    print(self.eng.eval('dyn.stats{end}.term_reason') )
                    elapsed_time = time.time() - t0
                    if stop_too_bad_fun is not None and elapsed_time > dt_walltime:
                        if frob_norm > stop_too_bad_fun(elapsed_time):
                            s1 = 'Final'
                        else:
                            s1 = 'Current'
                    else:
                        s1 = 'Current'
                    print("{} Frobenius norm: {} ({}), t={}s".format(s1, frob_norm, stop_too_bad_fun(elapsed_time), elapsed_time))
                    if s1 == 'Final' or self.eng.eval('dyn.stats{end}.term_reason') != "Wall time limit reached": break
        t = threading.Thread(target=run)
        t.start()
        return t

    def search(self, mask, options=None):
        options = {} if options is None else options
        mask = matlab.logical(mask.tolist())
        self.eng.search(self.dyn, mask, options, nargout=0)

    @property
    def n_bins(self):
        return len(self.get_mask())

    @property
    def times_fields_mhz(self):
        out = np.array(self.eng.eval("dyn.export"))
        out[:, 1:] *= 2 * np.pi
        return out

    def times(self, n):
        return self.times_fields_mhz[list(self.export_mask_rows(n)), 0]

    @property
    def times_bins(self):
        return self.times_fields_mhz[:self.n_bins, 0]

    def fields_xy_mhz(self, n):
        out = self.times_fields_mhz[list(self.export_mask_rows(n)), 1:][:, ~np.all(self.export_mask_list[n]==0, axis=0)]
        if out.shape[1] == 1:
            out = np.column_stack([out, 0*out])
        return out

    def fields_aphi_mhz(self, n):
        return misc.xy2aphi(self.fields_xy_mhz(n))

    # @property
    # def fields_add_slices_mhz(self):
    #     out = self.times_fields_mhz[self.n_bins:, -self.add_slices:]
    #     if self.add_slices > 1:
    #         out = np.diag(out)
    #     return out
    #
    # @property
    # def add_slices_angles(self):
    #     if self.add_slices > 0:
    #         return (2 * np.pi * self.times_add_slices*self.fields_add_slices_mhz) % (2 * np.pi)

    def X(self, n_ens, include_initial=True, include_final=True):
        out = []
        for i in range(1, n_ens + 1):
            out.append([])
            if include_initial:
                out[-1].append(Qobj(self.initial))
                if out[-1][-1].isket:
                    out[-1][-1] = ket2dm(out[-1][-1])
                out[-1][-1].dims = [self.dims, self.dims]
            for j in range(1, self.n_bins + 1):
                out[-1].append(Qobj(np.array(self.eng.eval('dyn.X({}, {})'.format(j, i)))))
                if out[-1][-1].isket:
                    out[-1][-1] = ket2dm(out[-1][-1])
                out[-1][-1].dims = [self.dims, self.dims]
            if include_final:
                out[-1].append(Qobj(self.final))
                if out[-1][-1].isket:
                    out[-1][-1] = ket2dm(out[-1][-1])
                out[-1][-1].dims = [self.dims, self.dims]
        return out

    def save_times_fields_mhz(self, path):
        np.savetxt(path, np.around(self.times_fields_mhz, 9), fmt=str('%+1.9e'))

    def save_times_fields_xy(self, n=None, path=None):
        t = self.round2float(self.times(n), 1/self.sample_frequency)
        txy = np.column_stack([t, self.fields_mhz(n)])
        np.savetxt(path, np.around(txy, 9), fmt=str('%+1.9e'))

    def save_times_fields_aphi(self, n=None, path=None):
        t = self.round2float(self.times(n), 1/self.sample_frequency)
        taphi = np.column_stack([t, self.fields_aphi_mhz(n)])
        np.savetxt(path, np.around(taphi, 9), fmt=str('%+1.9e'))

    def save_add_slices_angles(self, path):
        pass
        # asa = self.add_slices_angles
        # if asa is not None:
        #     np.savetxt(path, asa, fmt=str('%+1.4e'))
        # print("add_slices_angles saved.")

    def round2float(self, arr, val):
        return np.around(arr/val) * val

    def save_dynamo_fields(self, base_path):
        self.save_times_fields_mhz("{}\\fields.dat".format(base_path))
        for n, name in enumerate(self.fields_names_list):
            self.save_times_fields_aphi(n=n, path="{}\\{}.dat".format(base_path, name))
            print("Fields {} saved.".format(name))
        # self.save_add_slices_angles(path="{}\\add_slices_angles.dat".format(base_path))

    def save_export_mask(self, path):
        ems = sum(self.export_mask_list)
        np.savetxt(path, ems, fmt=str('%i'))
        print("Export mask saved.")

    def save_sequence_steps(self, path):
        """
        saves the sequence ['RF, WAIT', 'WAIT', 'MW', ..] along with the line number in the respective file
        :param path:
        :return:
        """
        fsn = np.empty((self.n_bins, len(self.export_mask_list)))
        nl = np.zeros(len(self.export_mask_list))
        out = []
        for i, step in enumerate(self.sequence):
            osln = []
            for j, name in enumerate(self.fields_names_list):
                if name in step:
                    nl[j] += 1
                    osln.append(nl[j])
            fsn[i] = nl
            out.append([', '.join(step), ', '.join(['{:d}'.format(int(i)) for i in osln])])
        np.savetxt(path,  out, delimiter="\t", fmt=str("%s"))
        print("Sequence_steps saved.")

    def save_matlab_output(self, path):
        with open(path, "w") as f:
            f.write(self.out.getvalue())
        print("matlab_output saved.")

    def save_script_code(self, path, script_code):
        if script_code != None:
            with open(path, "w") as text_file:
                text_file.write(script_code)
            print("script_code saved.")

    def save_notebook(self, path, script_path):
        sp, file_extension = os.path.splitext(script_path)
        dp = path# + os.path.splitext(os.path.basename(script_path))[0]
        if file_extension == '.ipynb':
            shutil.copy(sp + '.ipynb', dp + '.ipynb')
            shutil.move(sp + '.html', dp + '.html')
            shutil.move(sp + '.py', dp + '.py')


    def save_matlab_workspace(self, path):
        self.eng.save(path, nargout=0)

    def save_gates(self, path):

        gates = [Qobj(np.array(self.eng.eval('dyn.X({}, 1)'.format(i))), dims=[self.dims]*2) for i in range(1, self.n_bins + 1)]
        qsave(gates, path)
        print("gates saved as *.qu -file.")

    def save(self, script_path=None, dns=None, script_code=None, substr=None, save_notebook=False):
        if dns is None:
            filename, file_extension = os.path.splitext(script_path)
            if file_extension == '.ipynb':
                script_path = script_path.split('.')[0] + '.ipynb'
            else:
                script_path = script_path.split('.')[0] + '.py'
            dn = os.path.dirname(script_path)
            dns = dn + '\\' + datetime.datetime.now().strftime('%Y%m%d-h%Hm%Ms%S') + substr
        if not os.path.isdir(dns):
            os.makedirs(dns)
        self.save_script_code(dns + '\\' + os.path.basename(script_path), script_code)
        if save_notebook:
            self.save_notebook(dns, script_path)
        self.save_matlab_workspace(dns + "\\workspace.mat")
        self.save_matlab_output(path=dns+"\\matlab_output.dat")
        self.save_gates(path=dns+"\\gates")
        import qutip_enhanced
        shutil.copytree(os.path.dirname(qutip_enhanced.__file__), dns + '\\qutip_enhanced')
        print("qutip_enhanced saved.")
        self.save_dynamo_fields(base_path=dns)
        # self.save_export_mask(path=dns+"\\export_mask.dat")
        self.save_sequence_steps(path=dns + "\\sequence_steps.dat")
        return dns

def rotop(*args, **kwargs):
    return 2*np.pi*get_rot_matrix_all_spins(*args, **kwargs).data.todense()

def load(eng, directory):
    """
    :param directory: directory in which mat file and script and so on lies
    :return:
    """
    eng.addpath(eng.genpath(directory))