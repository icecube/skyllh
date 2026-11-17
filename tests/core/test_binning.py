import unittest

import numpy as np

from skyllh.core.binning import (
    BinningDefinition,
    get_bin_indices_from_binedges,
)


class GetBinIndicesFromBinedgesTestCase(unittest.TestCase):
    def setUp(self):
        self.edges = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.nbins = len(self.edges) - 1

    def test_matches_digitize_inside_range(self):
        values = np.linspace(self.edges[0], self.edges[-1], 1001)[:-1]
        np.testing.assert_array_equal(
            get_bin_indices_from_binedges(self.edges, values),
            np.digitize(values, self.edges) - 1,
        )

    def test_upper_edge_is_last_bin(self):
        self.assertEqual(int(get_bin_indices_from_binedges(self.edges, 1.0)), self.nbins - 1)
        np.testing.assert_array_equal(
            get_bin_indices_from_binedges(self.edges, np.array([-1.0, 1.0])),
            [0, self.nbins - 1],
        )

    def test_out_of_range_is_unchanged(self):
        np.testing.assert_array_equal(
            get_bin_indices_from_binedges(self.edges, np.array([-1.5, 1.5])),
            [-1, self.nbins],
        )

    def test_scalar_returns_0d(self):
        self.assertEqual(np.ndim(get_bin_indices_from_binedges(self.edges, 0.2)), 0)


class BinningDefinitionTestCase(unittest.TestCase):
    def test_get_binwidth_from_value_at_upper_edge(self):
        binning = BinningDefinition('x', np.array([0.0, 1.0, 3.0]))
        self.assertEqual(binning.get_binwidth_from_value(3.0), 2.0)
        self.assertEqual(binning.get_binwidth_from_value(0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
