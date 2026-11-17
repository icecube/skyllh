import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff


class GetAeffForDecnuTestCase(unittest.TestCase):
    """Tests for PDAeff.get_aeff_for_decnu, in particular the handling of the
    declination poles (-90 and +90 degrees).
    """

    def setUp(self):
        self.edges = np.radians([-90.0, -30.0, 0.0, 30.0, 90.0])
        self.n_bins = len(self.edges) - 1
        self.aeff = object.__new__(PDAeff)
        self.aeff._decnu_binedges = self.edges
        # Each row holds its own declination bin index.
        self.aeff._aeff_decnu_log10enu = np.repeat(np.arange(self.n_bins, dtype=np.double)[:, np.newaxis], 3, axis=1)

    def test_south_pole_is_first_bin(self):
        np.testing.assert_array_equal(self.aeff.get_aeff_for_decnu(-np.pi / 2), 0)

    def test_north_pole_is_last_bin(self):
        np.testing.assert_array_equal(self.aeff.get_aeff_for_decnu(np.pi / 2), self.n_bins - 1)

    def test_interior_values_fall_in_containing_bin(self):
        for dec in np.linspace(self.edges[0], self.edges[-1], 101)[:-1]:
            idx = int(self.aeff.get_aeff_for_decnu(dec)[0])
            self.assertTrue(self.edges[idx] <= dec < self.edges[idx + 1])

    def test_outside_range_raises(self):
        with self.assertRaises(ValueError):
            self.aeff.get_aeff_for_decnu(np.pi / 2 + 1e-6)
        with self.assertRaises(ValueError):
            self.aeff.get_aeff_for_decnu(-np.pi / 2 - 1e-6)


class GetDecnuIdxTestCase(unittest.TestCase):
    """Tests for PDAeff.get_decnu_idx at the declination poles and edges."""

    def setUp(self):
        self.edges = np.radians([-90.0, -30.0, 0.0, 30.0, 90.0])
        self.n_bins = len(self.edges) - 1
        self.aeff = object.__new__(PDAeff)
        self.aeff._decnu_binedges = self.edges

    def test_returns_python_int(self):
        self.assertIs(type(self.aeff.get_decnu_idx(0.1)), int)

    def test_south_pole_is_first_bin(self):
        self.assertEqual(self.aeff.get_decnu_idx(-np.pi / 2), 0)

    def test_north_pole_is_last_bin(self):
        self.assertEqual(self.aeff.get_decnu_idx(np.pi / 2), self.n_bins - 1)

    def test_inner_edges_belong_to_upper_bin(self):
        for i, edge in enumerate(self.edges[1:-1], start=1):
            self.assertEqual(self.aeff.get_decnu_idx(edge), i)

    def test_outside_range_raises(self):
        with self.assertRaises(ValueError):
            self.aeff.get_decnu_idx(np.pi / 2 + 1e-6)
        with self.assertRaises(ValueError):
            self.aeff.get_decnu_idx(-np.pi / 2 - 1e-6)


if __name__ == '__main__':
    unittest.main()
