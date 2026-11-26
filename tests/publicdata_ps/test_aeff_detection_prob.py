import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff


class DetectionProbTestCase(unittest.TestCase):
    LOG10_EDGES = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    AEFF = np.array([1.0, 4.0, 9.0, 12.0, 14.0])

    def _make_aeff(self):
        aeff = object.__new__(PDAeff)
        aeff._log10_enu_binedges = self.LOG10_EDGES
        aeff._log10_enu_binedges_lower = self.LOG10_EDGES[:-1]
        aeff._log10_enu_binedges_upper = self.LOG10_EDGES[1:]
        aeff.get_aeff_for_decnu = lambda decnu: self.AEFF
        return aeff

    def test_whole_bins_get_their_share_of_the_effective_area(self):
        edges = 10**self.LOG10_EDGES

        det_prob = self._make_aeff().get_detection_prob_for_decnu(0.0, edges[:-1], edges[1:], edges[0], edges[-1])

        np.testing.assert_allclose(det_prob, self.AEFF / self.AEFF.sum(), rtol=1e-12)

    def test_a_range_ending_inside_a_bin_keeps_that_part_of_the_bin(self):
        edges = 10**self.LOG10_EDGES
        range_max = 10**2.9

        det_prob = self._make_aeff().get_detection_prob_for_decnu(
            0.0, np.array([edges[3], edges[4]]), np.array([edges[4], range_max]), edges[0], range_max
        )

        part = (range_max - edges[4]) / (edges[5] - edges[4])
        norm = self.AEFF[:4].sum() + part * self.AEFF[4]
        np.testing.assert_allclose(det_prob, [self.AEFF[3] / norm, part * self.AEFF[4] / norm], rtol=1e-12)

    def test_bin_mask_includes_bins_the_range_cuts(self):
        mask = self._make_aeff().get_log10_enu_bin_mask(2.1, 2.7)

        np.testing.assert_array_equal(mask, [True, True, True, True, False])


if __name__ == '__main__':
    unittest.main()
