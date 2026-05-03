import os.path
import unittest
import warnings

import numpy as np

from skyllh.core.config import (
    Config,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
    DatasetOrigin,
    DatasetTransferError,
    RSYNCDatasetTransfer,
    WGETDatasetTransfer,
    get_data_subset,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.datasets import (
    create_datasets,
    data_samples,
)
from skyllh.datasets.i3 import (
    PublicData_10y_ps,
    TestData,
)
from skyllh.datasets.i3.PublicData_10y_ps import (
    create_dataset_collection,
)


class TestRSYNCDatasetTransfer(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.ds = TestData.create_dataset_collection(
            cfg=self.cfg, base_path=os.path.join(os.getcwd(), '.repository')
        ).get_dataset('TestData')

        # Remove the dataset if it already exists.
        if self.ds.exists:
            self.ds.remove_data()

        # Define the origin and transfer method of this dataset.
        self.ds.origin = DatasetOrigin(
            base_path='/data/user/mwolf/skyllh',
            sub_path='testdata',
            host='cobalt',
            transfer_func=RSYNCDatasetTransfer().transfer,
        )

    def test_transfer(self):
        try:
            if not self.ds.make_data_available():
                raise RuntimeError(f'The data of dataset {self.ds.name} could not be made available!')
        except DatasetTransferError:
            self.skipTest(f'The data of dataset {self.ds.name} could not be transfered.')

        # Check that there are no missing files.
        missing_files = self.ds.get_missing_files()
        self.assertEqual(len(missing_files), 0)


class TestWGETDatasetTransfer(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.ds = TestData.create_dataset_collection(
            cfg=self.cfg, base_path=os.path.join(os.getcwd(), '.repository')
        ).get_dataset('TestData')

        # Remove the dataset if it already exists.
        if self.ds.exists:
            self.ds.remove_data()

        # Define the origin and transfer method of this dataset.
        self.ds.origin = DatasetOrigin(
            base_path='/data/user/mwolf/skyllh',
            sub_path='testdata',
            host='convey.icecube.wisc.edu',
            username='icecube',
            transfer_func=WGETDatasetTransfer(protocol='https').transfer,
        )

    def test_transfer(self):
        password = os.environ.get('ICECUBE_PASSWORD', None)
        if password is None:
            self.skipTest(f'No password for username "{self.ds.origin.username}" provided via the environment!')

        if not self.ds.make_data_available(
            password=password,
        ):
            raise RuntimeError(f'The data of dataset {self.ds.name} could not be made available!')

        # Check that there are no missing files.
        missing_files = self.ds.get_missing_files()
        self.assertEqual(len(missing_files), 0)


class TestDatasetFunctions(unittest.TestCase):
    def setUp(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.exp_data = DataFieldRecordArray(np.load(os.path.join(path, 'testdata/exp_testdata.npy')))
        self.mc_data = DataFieldRecordArray(np.load(os.path.join(path, 'testdata/mc_testdata.npy')))
        self.livetime_datafile = np.load(os.path.join(path, 'testdata/livetime_testdata.npy'))
        self.livetime = 100

    def test_get_data_subset(self):
        # Whole interval.
        t_start = 58442.0
        t_end = 58445.0
        dataset_data = DatasetData(self.exp_data, self.mc_data, self.livetime)
        livetime_data = Livetime(self.livetime_datafile)
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data, livetime_data, t_start, t_end)

        self.assertEqual(len(dataset_data_subset.exp), 4)
        self.assertEqual(len(dataset_data_subset.mc), 4)
        self.assertAlmostEqual(livetime_subset.livetime, 1)

        # Sub interval without cutting livetime.
        t_start = 58443.3
        t_end = 58444.3
        dataset_data = DatasetData(self.exp_data, self.mc_data, self.livetime)
        livetime_data = Livetime(self.livetime_datafile)
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data, livetime_data, t_start, t_end)

        self.assertEqual(len(dataset_data_subset.exp), 2)
        self.assertEqual(len(dataset_data_subset.mc), 2)
        self.assertAlmostEqual(livetime_subset.livetime, 0.5)

        # Cutting first livetime interval.
        t_start = 58443.1
        t_end = 58444.75
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data, livetime_data, t_start, t_end)

        self.assertEqual(len(dataset_data_subset.exp), 3)
        self.assertEqual(len(dataset_data_subset.mc), 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.9)

        # Cutting last livetime interval.
        t_start = 58443.0
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data, livetime_data, t_start, t_end)

        self.assertEqual(len(dataset_data_subset.exp), 4)
        self.assertEqual(len(dataset_data_subset.mc), 4)
        self.assertAlmostEqual(livetime_subset.livetime, 0.85)

        # Cutting first and last livetime interval.
        t_start = 58443.1
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data, livetime_data, t_start, t_end)

        self.assertEqual(len(dataset_data_subset.exp), 3)
        self.assertEqual(len(dataset_data_subset.mc), 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.75)


class TestDatasetCollection(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.dsc = create_dataset_collection(cfg=self.cfg)

    def test__getitem__single(self):
        ds = self.dsc['IC40']
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.name, 'IC40')

    def test__getitem__multi(self):
        ds_list = self.dsc['IC59', 'IC40']
        self.assertIsInstance(ds_list, list)
        self.assertEqual(len(ds_list), 2)
        self.assertIsInstance(ds_list[0], Dataset)
        self.assertIsInstance(ds_list[1], Dataset)
        self.assertEqual(ds_list[0].name, 'IC59')
        self.assertEqual(ds_list[1].name, 'IC40')


class TestCreateDatasets(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_default_names_match_DATASET_NAMES(self):
        from skyllh.datasets.i3.TestData import DATASET_NAMES

        ds_list = create_datasets('TestData', cfg=self.cfg)
        self.assertIsInstance(ds_list, list)
        self.assertEqual([ds.name for ds in ds_list], list(DATASET_NAMES))

    def test_single_name_string_coercion(self):
        ds_list = create_datasets('TestData', cfg=self.cfg, names='TestData')
        self.assertIsInstance(ds_list, list)
        self.assertEqual(len(ds_list), 1)
        self.assertEqual(ds_list[0].name, 'TestData')

    def test_sequence_of_names(self):
        ds_list = create_datasets('IceTracks-DR1', cfg=self.cfg, names=('IC40', 'IC59'))
        self.assertIsInstance(ds_list, list)
        self.assertEqual(len(ds_list), 2)
        self.assertEqual([ds.name for ds in ds_list], ['IC40', 'IC59'])

    def test_unknown_sample_raises_KeyError(self):
        with self.assertRaises(KeyError) as ctx:
            create_datasets('nonexistent', cfg=self.cfg)
        self.assertIn('nonexistent', str(ctx.exception))

    def test_legacy_name_warns_and_returns_datasets(self):
        from skyllh.datasets.i3.PublicData_10y_ps import DATASET_NAMES

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ds_list = create_datasets('PublicData_10y_ps', cfg=self.cfg)
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertIn('IceTracks-DR1', str(w[0].message))
        # Warning must point at the caller's line, not at skyllh internals.
        # A wrong stacklevel would make this fail.
        self.assertIn('test_dataset', w[0].filename)
        self.assertEqual([ds.name for ds in ds_list], list(DATASET_NAMES))


class TestDataSamplesDict(unittest.TestCase):
    def test_getitem_deprecated_key_warns_and_resolves(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = data_samples['PublicData_10y_ps']
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertIn('IceTracks-DR1', str(w[0].message))
        self.assertIn('test_dataset', w[0].filename)
        self.assertIs(result, PublicData_10y_ps)

    def test_get_deprecated_key_warns_and_resolves(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = data_samples.get('PublicData_10y_ps')
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertIn('IceTracks-DR1', str(w[0].message))
        self.assertIn('test_dataset', w[0].filename)
        self.assertIs(result, PublicData_10y_ps)

    def test_get_missing_key_returns_default(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = data_samples.get('nonexistent', 'sentinel')
        self.assertEqual(len(w), 0)
        self.assertEqual(result, 'sentinel')

    def test_getitem_new_key_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = data_samples['IceTracks-DR1']
        self.assertEqual(len(w), 0)
        self.assertIs(result, PublicData_10y_ps)

    def test_get_new_key_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = data_samples.get('IceTracks-DR1')
        self.assertEqual(len(w), 0)
        self.assertIs(result, PublicData_10y_ps)

    def test_contains_deprecated_key(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = 'PublicData_10y_ps' in data_samples
        self.assertTrue(result)
        self.assertEqual(len(w), 0)

    def test_contains_new_key(self):
        self.assertIn('IceTracks-DR1', data_samples)

    def test_contains_unknown_key(self):
        self.assertNotIn('nonexistent', data_samples)


if __name__ == '__main__':
    unittest.main()
