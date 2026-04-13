import logging
import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.signal_generator import PDDatasetSignalGenerator
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import PDSmearingMatrix


class EnergyRangeBinAlignmentTestCase(unittest.TestCase):
    @staticmethod
    def _make_sm(edges):
        sm = object.__new__(PDSmearingMatrix)
        sm._true_e_bin_edges = np.asarray(edges, dtype=np.float64)
        return sm

    @staticmethod
    def _make_generator(edges):
        gen = object.__new__(PDDatasetSignalGenerator)
        gen.sm = EnergyRangeBinAlignmentTestCase._make_sm(edges)
        gen._logger = logging.getLogger(__name__)
        gen._input_range = None
        gen._energy_range_log10 = None
        # Keep the unit test isolated from dataset/source-dependent setup.
        gen._create_source_dependent_data_structures = lambda: None
        return gen

    def test_upper_edge_index_exact_last_sm_edge(self):
        sm = self._make_sm([2.0, 2.5, 3.0, 3.5])

        idx = sm.get_log10_true_e_idx(3.5, upper_edge=True)

        self.assertEqual(idx, 3)

    def test_energy_range_exact_sm_edges_no_extra_bins(self):
        gen = self._make_generator([2.0, 2.5, 3.0, 3.5])

        gen.energy_range = (10**2.0, 10**3.0)

        self.assertEqual(gen._log10_energy_range, (2.0, 3.0))

    def test_energy_range_exact_upper_edge_no_extra_bin(self):
        gen = self._make_generator([2.0, 2.5, 3.0, 3.5])

        gen.energy_range = (10**2.2, 10**3.0)

        self.assertEqual(gen._log10_energy_range, (2.0, 3.0))

    def test_energy_range_exact_lower_edge_keeps_that_edge(self):
        gen = self._make_generator([2.0, 2.5, 3.0, 3.5])

        gen.energy_range = (10**2.5, 10**2.7)

        self.assertEqual(gen._log10_energy_range, (2.5, 3.0))


if __name__ == '__main__':
    unittest.main()
