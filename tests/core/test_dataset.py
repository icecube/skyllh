# -*- coding: utf-8 -*-

import numpy as np
import os.path
import unittest

from skyllh.core.config import (
    Config,
)
from skyllh.core.dataset import (
    get_data_subset,
    DatasetData,
    DatasetOrigin,
    DatasetTransferError,
    RSYNCDatasetTransfer,
    WGETDatasetTransfer,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.datasets.i3 import (
    TestData,
)


class TestRSYNCDatasetTransfer(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.ds = TestData.create_dataset_collection(
            cfg=self.cfg,
            base_path=os.path.join(os.getcwd(), '.repository')).get_dataset(
                'TestData')

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
                raise RuntimeError(
                    f'The data of dataset {self.ds.name} could not be made '
                    'available!')
        except DatasetTransferError:
            self.skipTest(
                f'The data of dataset {self.ds.name} could not be transfered.')

        # Check that there are no missing files.
        missing_files = self.ds.get_missing_files()
        self.assertEqual(len(missing_files), 0)


class TestWGETDatasetTransfer(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.ds = TestData.create_dataset_collection(
            cfg=self.cfg,
            base_path=os.path.join(os.getcwd(), '.repository')).get_dataset(
                'TestData')

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
            self.skipTest(
                f'No password for username "{self.ds.origin.username}" '
                'provided via the environment!')

        if not self.ds.make_data_available(
            password=password,
        ):
            raise RuntimeError(
                f'The data of dataset {self.ds.name} could not be made '
                'available!')

        # Check that there are no missing files.
        missing_files = self.ds.get_missing_files()
        self.assertEqual(len(missing_files), 0)


class TestDatasetFunctions(
    unittest.TestCase,
):
    def setUp(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.exp_data = DataFieldRecordArray(
            np.load(os.path.join(path, 'testdata/exp_testdata.npy')))
        self.mc_data = DataFieldRecordArray(
            np.load(os.path.join(path, 'testdata/mc_testdata.npy')))
        self.livetime_datafile = np.load(
            os.path.join(path, 'testdata/livetime_testdata.npy'))
        self.livetime = 100

    def tearDown(self):
        # self.exp_data.close()
        # self.mc_data.close()
        # self.livetime_datafile.close()
        pass

    def test_get_data_subset(self):
        # Whole interval.
        t_start = 58442.0
        t_end = 58445.0
        dataset_data = DatasetData(
            self.exp_data, self.mc_data, self.livetime)
        livetime_data = Livetime(self.livetime_datafile)
        (dataset_data_subset, livetime_subset) = get_data_subset(
            dataset_data,
            livetime_data,
            t_start,
            t_end)

        self.assertEqual(len(dataset_data_subset.exp), 4)
        self.assertEqual(len(dataset_data_subset.mc), 4)
        self.assertAlmostEqual(livetime_subset.livetime, 1)

        # Sub interval without cutting livetime.
        t_start = 58443.3
        t_end = 58444.3
        dataset_data = DatasetData(
            self.exp_data, self.mc_data, self.livetime)
        livetime_data = Livetime(self.livetime_datafile)
        (dataset_data_subset, livetime_subset) = get_data_subset(
            dataset_data,
            livetime_data,
            t_start,
            t_end)

        self.assertEqual(len(dataset_data_subset.exp), 2)
        self.assertEqual(len(dataset_data_subset.mc), 2)
        self.assertAlmostEqual(livetime_subset.livetime, 0.5)

        # Cutting first livetime interval.
        t_start = 58443.1
        t_end = 58444.75
        (dataset_data_subset, livetime_subset) = get_data_subset(
            dataset_data,
            livetime_data,
            t_start,
            t_end)

        self.assertEqual(len(dataset_data_subset.exp), 3)
        self.assertEqual(len(dataset_data_subset.mc), 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.9)

        # Cutting last livetime interval.
        t_start = 58443.0
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(
            dataset_data,
            livetime_data,
            t_start,
            t_end)

        self.assertEqual(len(dataset_data_subset.exp), 4)
        self.assertEqual(len(dataset_data_subset.mc), 4)
        self.assertAlmostEqual(livetime_subset.livetime, 0.85)

        # Cutting first and last livetime interval.
        t_start = 58443.1
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(
            dataset_data,
            livetime_data,
            t_start,
            t_end)

        self.assertEqual(len(dataset_data_subset.exp), 3)
        self.assertEqual(len(dataset_data_subset.mc), 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.75)


if __name__ == '__main__':
    unittest.main()
