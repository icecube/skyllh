import types
import unittest

import numpy as np
from astropy import units

from skyllh.analyses.i3.publicdata_ps.signal_generator import PDDatasetSignalGenerator
from skyllh.core.config import Config
from skyllh.core.flux_model import PowerLawEnergyFluxProfile, UnityEnergyFluxProfile


class TrueEnergyInverseCDFTestCase(unittest.TestCase):
    EDGES = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    DET_PROB = np.array([0.40, 0.25, 0.17, 0.11, 0.07])

    def _make_distribution(self, log_e_min=EDGES[0], log_e_max=EDGES[-1], det_prob=None, energy_profile=None):
        if det_prob is None:
            det_prob = self.DET_PROB
        if energy_profile is None:
            # A flux that is flat in energy, so a bin's flux integral is its width.
            energy_profile = UnityEnergyFluxProfile(energy_unit=units.GeV, cfg=Config())
        lower, upper = self.EDGES[:-1], self.EDGES[1:]
        mask = (upper > log_e_min) & (lower < log_e_max)
        effA = types.SimpleNamespace(
            log10_enu_binedges_lower=lower,
            log10_enu_binedges_upper=upper,
            get_log10_enu_bin_mask=lambda lo, hi: (upper > lo) & (lower < hi),
            det_prob=det_prob[mask],
        )
        gen = object.__new__(PDDatasetSignalGenerator)
        gen._effA_arr = np.array([effA], dtype=np.object_)
        fluxmodel = types.SimpleNamespace(energy_profile=energy_profile)
        return gen._create_true_e_distribution(src_idx=0, fluxmodel=fluxmodel, log_e_min=log_e_min, log_e_max=log_e_max)

    def _bin_prob(self):
        prob = np.diff(10**self.EDGES) * self.DET_PROB
        return prob / prob.sum()

    def test_energy_range_inside_bins_is_drawn_exactly(self):
        dist = self._make_distribution(log_e_min=2.1, log_e_max=2.7)
        rng = np.random.default_rng(2)

        log10_e = dist.get_log10_e(rng.uniform(size=200_000))

        self.assertAlmostEqual(log10_e.min(), 2.1, delta=1e-3)
        self.assertAlmostEqual(log10_e.max(), 2.7, delta=1e-3)
        # A cut bin keeps its whole-bin detection probability and the flux of its remaining part.
        edges = np.array([2.1, 2.2, 2.4, 2.6, 2.7])
        prob = np.diff(10**edges) * self.DET_PROB[:4]
        frac = np.histogram(log10_e, bins=edges)[0] / log10_e.size
        np.testing.assert_allclose(frac, prob / prob.sum(), atol=5e-3)

    def test_cumulative_probability_is_reached_at_upper_bin_edges(self):
        dist = self._make_distribution()
        cum = np.concatenate(([0.0], np.cumsum(self._bin_prob())))

        log10_e = dist.get_log10_e(cum)

        np.testing.assert_allclose(log10_e, self.EDGES, atol=1e-12)

    def test_drawn_energies_fill_each_bin_with_its_probability(self):
        dist = self._make_distribution()
        rng = np.random.default_rng(1)

        log10_e = dist.get_log10_e(rng.uniform(size=200_000))

        self.assertGreaterEqual(log10_e.min(), self.EDGES[0])
        self.assertLessEqual(log10_e.max(), self.EDGES[-1])
        # The upper half of the highest bin must be reachable.
        self.assertGreater(log10_e.max(), 0.5 * (self.EDGES[-2] + self.EDGES[-1]))
        frac = np.histogram(log10_e, bins=self.EDGES)[0] / log10_e.size
        np.testing.assert_allclose(frac, self._bin_prob(), atol=5e-3)

    def test_dropped_bins_receive_no_energies(self):
        # Bins whose probability is negligible (<= 1e-15) are dropped. The
        # neighbouring kept bins must not spread into their energy range.
        det_prob = self.DET_PROB.copy()
        det_prob[[0, 2]] = 1e-20
        dist = self._make_distribution(det_prob=det_prob)
        rng = np.random.default_rng(3)

        log10_e = dist.get_log10_e(rng.uniform(size=200_000))

        frac = np.histogram(log10_e, bins=self.EDGES)[0] / log10_e.size
        prob = np.diff(10**self.EDGES) * det_prob
        prob[[0, 2]] = 0
        np.testing.assert_allclose(frac, prob / prob.sum(), atol=5e-3)
        self.assertEqual(frac[0], 0)
        self.assertEqual(frac[2], 0)

    def test_energies_follow_the_flux_within_bins(self):
        # With a steep power law, the energies within each bin must follow
        # E^-gamma, not a uniform distribution in log10(E).
        gamma = 3.5
        energy_profile = PowerLawEnergyFluxProfile(E0=1e3, gamma=gamma, energy_unit=units.GeV, cfg=Config())
        dist = self._make_distribution(energy_profile=energy_profile)
        rng = np.random.default_rng(4)

        log10_e = dist.get_log10_e(rng.uniform(size=400_000))

        # Compare the fractions of events in the lower half of each bin.
        a = 1 - gamma
        for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:], strict=True):
            mid = 0.5 * (lo + hi)
            in_bin = log10_e[(log10_e >= lo) & (log10_e < hi)]
            expected = (10 ** (a * mid) - 10 ** (a * lo)) / (10 ** (a * hi) - 10 ** (a * lo))
            sigma = np.sqrt(expected * (1 - expected) / in_bin.size)
            self.assertAlmostEqual(np.mean(in_bin < mid), expected, delta=5 * sigma)
            # A uniform distribution in log10(E) would put half of them there.
            self.assertGreater(expected, 0.6)


if __name__ == '__main__':
    unittest.main()
