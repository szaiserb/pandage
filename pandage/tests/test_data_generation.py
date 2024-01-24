import unittest
from datetime import datetime
from collections import OrderedDict
from pandas import DataFrame, concat
from pandas.testing import assert_frame_equal

from pandage.data_generation import DataGeneration
from pandage.pdplotddata import PlotData
from pathlib import Path
import re
from itertools import product
import sys
import traceback



__PARAMETERS__                          = OrderedDict([('day', [1,2,3]), ('city', ['Schwieberdingen', 'Feuerbach'])])
__OBSERVATION_NAMES__                   = ['string_from_parameters']
__DTYPES__                              = OrderedDict([('string_from_parameters', 'str')])
__NUMBER_OF_SIMULTANEOUS_MEASUREMENTS__ = 2

def get_iterator_df(parameters):
    iterator_df = DataFrame(
        list(product(*parameters.values())),
        columns=list(parameters.keys())
    )
    for cn in iterator_df.columns:
        iterator_df[cn] = iterator_df[cn].astype(type(parameters[cn][0]))    
    return iterator_df

class DataGenerationUser(DataGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.parameters        = kwargs.pop('parameters')
        self.dtypes            = kwargs.pop('dtypes')
        self.gui               = kwargs.pop('gui', False)
        self.progress_bar      = kwargs.pop('progress_bar', None)
        self.pld = PlotData('pld_test', gui=self.gui)
        self.pld.x_axis_parameter = __OBSERVATION_NAMES__[0]

    @property
    def observation_names(self):
        return __OBSERVATION_NAMES__

    def run(self, abort=None):
        try:
            for idxo, _ in enumerate(self.iterator()):
                if abort is not None and abort.is_set(): break
                observation_dict_list = []
                for idx, _I_ in self.current_iterator_df.iterrows():
                    if abort is not None and abort.is_set(): break
                    observation_dict_list.append(OrderedDict([('string_from_parameters', str(_I_['day']) + _I_['city'])]))
                self.data.set_observations(observation_dict_list)
                if self.gui:
                    self.pld.new_data_arrived()
            self.pld.new_data_arrived()
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            self.state = 'idle'
            self.update_current_str()

class TestDataGeneration(unittest.TestCase):

    def getDgu(self):
        return DataGenerationUser(
            parameters        = __PARAMETERS__,
            dtypes            = __DTYPES__
        )

    def test_init(self):
        dgu = self.getDgu()
        self.assertIsInstance(dgu.date_of_creation, datetime)
        self.assertEqual(dgu.parameters, __PARAMETERS__)
        self.assertEqual(dgu.dtypes, __DTYPES__)
        self.assertEqual(dgu.pld.x_axis_parameter, __OBSERVATION_NAMES__[0])

    def test_set_iterator_df(self):
        dgu = self.getDgu()
        dgu.set_iterator_df()
        assert_frame_equal(dgu.iterator_df, get_iterator_df(__PARAMETERS__))


        # for d in __PARAMETERS__['day']:
        #     for c in __PARAMETERS__['city']:
        #         self.assertIn((d, c), dgu.iterator_df.values)
        # for i, cn in enumerate(dgu.iterator_df.columns):
        #     self.assertEqual(dgu.iterator_df[cn].dtype, ['int', 'O'][i])

    def test_make_save_location_params(self):
        dgu = self.getDgu()
        dgu.make_save_location_params(__file__)
        self.assertEqual(dgu.file_path, str(Path(__file__).parent))
        self.assertEqual(dgu.file_name, str(Path(__file__).stem))

    def test_init_run(self):
        dgu = self.getDgu()
        dgu.make_save_location_params(__file__)
        dgu.current_iterator_df = 'THISCOULDBEANYTHING'
        self.assertEqual(dgu.current_iterator_df, 'THISCOULDBEANYTHING')
        dgu.init_run()
        self.assertFalse(hasattr(dgu, 'current_iterator_df'))
        pattern = r"{file_path}/\d{{8}}-h\d{{2}}m\d{{2}}s\d{{2}}_{file_name}/data.hdf".format(file_path=re.escape(dgu.file_path), file_name=re.escape(dgu.file_name))
        self.assertTrue(re.search(pattern, dgu.pld.data_path))
        self.assertEqual(dgu.pld.data.parameter_names,list(dgu.parameters.keys()))
        self.assertEqual(dgu.pld.data.observation_names, __OBSERVATION_NAMES__)
        self.assertEqual(dgu.pld.data.dtypes, __DTYPES__)
        self.assertTrue(dgu.pld.data.df.empty)

    def test_iterator_df_drop_done(self):
        dgu = self.getDgu()
        dgu.make_save_location_params(__file__)
        dgu.init_run()
        dgu.set_iterator_df()
        assert_frame_equal(dgu.iterator_df, get_iterator_df(__PARAMETERS__))
        self.assertTrue(dgu.pld.data.df.empty)
        dgu.pld.data.init()
        dgu.pld.data.append(dgu.iterator_df.iloc[:4])
        dgu.set_iterator_df_done()
        dgu.iterator_df_drop_done()
        assert_frame_equal(dgu.iterator_df, get_iterator_df(__PARAMETERS__).iloc[4:].reset_index(drop=True))

    def test_iterator(self):
        dgu = self.getDgu()
        dgu.init_run()
        dgu.number_of_simultaneous_measurements = __NUMBER_OF_SIMULTANEOUS_MEASUREMENTS__
        for current_iterator_df, current_iterator_df_changes in dgu.iterator():
            assert_frame_equal(current_iterator_df, get_iterator_df(__PARAMETERS__).iloc[:__NUMBER_OF_SIMULTANEOUS_MEASUREMENTS__].reset_index(drop=True))
            self.assertEqual(dgu.progress, 0)
            assert_frame_equal(current_iterator_df_changes, DataFrame({'day': [True, False], 'city': [True, True]}))
            break

    def test_run(self):
        dgu = self.getDgu()
        dgu.init_run()
        dgu.run()
        obs_l = [(str(r[0]) + r[1]) for r in product(*__PARAMETERS__.values())]
        compare_df = concat([get_iterator_df(__PARAMETERS__), DataFrame({'string_from_parameters': obs_l})], axis=1)
        assert_frame_equal(dgu.pld.data.df, compare_df)

if __name__ == '__main__':
    unittest.main()