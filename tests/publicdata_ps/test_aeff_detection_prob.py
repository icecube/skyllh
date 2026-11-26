import unittest
from unittest import mock

import numpy as np
from astropy import units

from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff
from skyllh.core.config import Config
from skyllh.core.flux_model import PowerLawEnergyFluxProfile


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

    def test_zero_effective_area_over_the_range_raises(self):
        aeff = self._make_aeff()
        aeff.get_aeff_for_decnu = lambda decnu: np.zeros_like(self.AEFF)
        edges = 10**self.LOG10_EDGES

        with self.assertRaisesRegex(ValueError, 'is zero over the entire energy range'):
            aeff.get_detection_prob_for_decnu(0.0, edges[:-1], edges[1:], edges[0], edges[-1])

    def test_energy_range_outside_the_binning_raises(self):
        aeff_arr = self.AEFF[np.newaxis, :]
        loaded = (aeff_arr, np.array([-0.1]), np.array([0.1]), self.LOG10_EDGES[:-1], self.LOG10_EDGES[1:])
        with (
            mock.patch('skyllh.analyses.i3.publicdata_ps.aeff.load_effective_area_array', return_value=loaded),
            self.assertRaisesRegex(ValueError, 'does not overlap the energy binning'),
        ):
            PDAeff(pathfilenames='unused', src_dec=0.0, min_log10enu=3.5, max_log10enu=4.0)


class DetectableFluxTestCase(unittest.TestCase):
    LOG10_EDGES = DetectionProbTestCase.LOG10_EDGES
    AEFF = DetectionProbTestCase.AEFF

    def _detectable_flux_reference(self, profile, e1, e2):
        """Integrates A(E) * Phi(E) numerically on a fine grid within each bin,
        so the jumps of A(E) at the bin edges do not enter the integration.
        """
        trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz  # pyright: ignore[reportAttributeAccessIssue]
        total = 0.0
        for lo, hi, a in zip(self.LOG10_EDGES[:-1], self.LOG10_EDGES[1:], self.AEFF, strict=True):
            lo, hi = max(lo, np.log10(e1)), min(hi, np.log10(e2))
            if hi <= lo:
                continue
            log10_e = np.linspace(lo, hi, 20_001)
            E = 10**log10_e
            total += a * trapezoid(profile(E) * E * np.log(10), log10_e)
        return total

    def test_ranges_across_and_within_bins_are_exact(self):
        aeff = DetectionProbTestCase._make_aeff(self)  # pyright: ignore[reportArgumentType]
        profile = PowerLawEnergyFluxProfile(E0=1e3, gamma=3.5, energy_unit=units.GeV, cfg=Config())
        # A range covering several bins and ending inside one, a range within
        # a single bin, and a range reaching beyond the binning.
        e_min = 10 ** np.array([2.1, 2.45, 2.9])
        e_max = 10 ** np.array([2.7, 2.55, 3.3])

        flux = aeff.get_detectable_flux_for_decnu(0.0, e_min, e_max, profile)

        expected = [self._detectable_flux_reference(profile, a, b) for a, b in zip(e_min, e_max, strict=True)]
        np.testing.assert_allclose(flux, expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
