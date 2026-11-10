import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.smearing_matrix import PDSmearingMatrix


class GetTrueDecIdxTestCase(unittest.TestCase):
    """Tests for PDSmearingMatrix.get_true_dec_idx, in particular the handling
    of the declination poles (-90 and +90 degrees).
    """

    def setUp(self):
        # Declination bin edges spanning the full sky, in radians, as the
        # public data smearing matrices have.
        self.edges = np.radians([-90.0, -30.0, -5.0, 0.0, 5.0, 30.0, 90.0])
        self.sm = object.__new__(PDSmearingMatrix)
        self.sm._true_dec_bin_edges = self.edges
        self.n_bins = len(self.edges) - 1

    def test_south_pole_is_first_bin(self):
        self.assertEqual(self.sm.get_true_dec_idx(-np.pi / 2), 0)
        self.assertEqual(self.sm.get_true_dec_idx(np.deg2rad(-90.0)), 0)

    def test_north_pole_is_last_bin(self):
        self.assertEqual(self.sm.get_true_dec_idx(np.pi / 2), self.n_bins - 1)
        self.assertEqual(self.sm.get_true_dec_idx(np.deg2rad(90.0)), self.n_bins - 1)

    def test_poles_index_is_usable(self):
        # The returned index must be usable to index the declination axis.
        dec_axis = np.arange(self.n_bins)
        for dec in (-np.pi / 2, np.pi / 2):
            idx = self.sm.get_true_dec_idx(dec)
            self.assertEqual(dec_axis[idx], idx)

    def test_interior_values_fall_in_containing_bin(self):
        for dec in np.linspace(self.edges[0], self.edges[-1], 1001)[:-1]:
            idx = self.sm.get_true_dec_idx(dec)
            self.assertTrue(self.edges[idx] <= dec < self.edges[idx + 1])

    def test_outside_range_raises(self):
        with self.assertRaises(ValueError):
            self.sm.get_true_dec_idx(np.pi / 2 + 1e-6)
        with self.assertRaises(ValueError):
            self.sm.get_true_dec_idx(-np.pi / 2 - 1e-6)


class GetLog10TrueEIdxTestCase(unittest.TestCase):
    """Tests for PDSmearingMatrix.get_log10_true_e_idx at the range edges."""

    def setUp(self):
        self.edges = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.sm = object.__new__(PDSmearingMatrix)
        self.sm._true_e_bin_edges = self.edges
        self.n_bins = len(self.edges) - 1

    def test_lowest_edge_is_first_bin(self):
        self.assertEqual(self.sm.get_log10_true_e_idx(self.edges[0]), 0)

    def test_highest_edge_is_last_bin(self):
        self.assertEqual(self.sm.get_log10_true_e_idx(self.edges[-1]), self.n_bins - 1)

    def test_upper_edge_mode_is_unchanged(self):
        self.assertEqual(self.sm.get_log10_true_e_idx(self.edges[-1], upper_edge=True), self.n_bins)
        self.assertEqual(self.sm.get_log10_true_e_idx(4.5, upper_edge=True), 3)

    def test_outside_range_raises(self):
        with self.assertRaises(ValueError):
            self.sm.get_log10_true_e_idx(self.edges[-1] + 1e-6)
        with self.assertRaises(ValueError):
            self.sm.get_log10_true_e_idx(self.edges[0] - 1e-6)


if __name__ == '__main__':
    unittest.main()
