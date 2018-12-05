# -*- coding: utf-8 -*-

from __future__ import division

import os.path
import unittest
import numpy as np

from skyllh.core.dataset import (
    get_data_subset,
    DatasetData
)
from skyllh.core.livetime import Livetime

from skyllh.core.py import float_cast


class TestDatasetFunctions(unittest.TestCase):
    def setUp(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.exp_data = np.load(os.path.join(path, '../data_files/exp_data.npy'))
        self.mc_data = np.load(os.path.join(path, '../data_files/mc_data.npy'))
        self.livetime_data = np.load(os.path.join(path, '../data_files/livetime_data.npy'))

    def tearDown(self):
        # self.exp_data.close()
        # self.mc_data.close()
        # self.livetime_data.close()
        pass

    def test_get_data_subset(self):
        # Whole interval.
        t_start = 58442.0
        t_end = 58445.0
        dataset_data = DatasetData(self.exp_data, self.mc_data)
        livetime_data = Livetime(self.livetime_data)
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data,
                                                                 livetime_data,
                                                                 t_start,
                                                                 t_end)

        self.assertEqual(dataset_data_subset.exp.size, 4)
        self.assertEqual(dataset_data_subset.mc.size, 4)
        self.assertAlmostEqual(livetime_subset.livetime, 1)

        # Sub interval without cutting livetime.
        t_start = 58443.3
        t_end = 58444.3
        dataset_data = DatasetData(self.exp_data, self.mc_data)
        livetime_data = Livetime(self.livetime_data)
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data,
                                                                 livetime_data,
                                                                 t_start,
                                                                 t_end)

        self.assertEqual(dataset_data_subset.exp.size, 2)
        self.assertEqual(dataset_data_subset.mc.size, 2)
        self.assertAlmostEqual(livetime_subset.livetime, 0.5)

        # Cutting first livetime interval.
        t_start = 58443.1
        t_end = 58444.75
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data,
                                                                 livetime_data,
                                                                 t_start,
                                                                 t_end)

        self.assertEqual(dataset_data_subset.exp.size, 3)
        self.assertEqual(dataset_data_subset.mc.size, 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.9)

        # Cutting last livetime interval.
        t_start = 58443.0   
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data,
                                                                 livetime_data,
                                                                 t_start,
                                                                 t_end)

        self.assertEqual(dataset_data_subset.exp.size, 4)
        self.assertEqual(dataset_data_subset.mc.size, 4)
        self.assertAlmostEqual(livetime_subset.livetime, 0.85)

        # Cutting first and last livetime interval.
        t_start = 58443.1   
        t_end = 58444.6
        (dataset_data_subset, livetime_subset) = get_data_subset(dataset_data,
                                                                 livetime_data,
                                                                 t_start,
                                                                 t_end)

        self.assertEqual(dataset_data_subset.exp.size, 3)
        self.assertEqual(dataset_data_subset.mc.size, 3)
        self.assertAlmostEqual(livetime_subset.livetime, 0.75)

if(__name__ == '__main__'):
    unittest.main()
