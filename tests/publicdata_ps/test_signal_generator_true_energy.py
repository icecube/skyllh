import types
import unittest

import numpy as np
from scipy import interpolate

from skyllh.analyses.i3.publicdata_ps.signal_generator import PDDatasetSignalGenerator


class _FlatEnergyProfile:
    """A flux that is flat in energy, so a bin's flux integral is its width."""

    @staticmethod
    def get_integral(E1, E2):
        return np.asarray(E2, dtype=np.float64) - np.asarray(E1, dtype=np.float64)


class TrueEnergyInverseCDFTestCase(unittest.TestCase):
    EDGES = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    DET_PROB = np.array([0.40, 0.25, 0.17, 0.11, 0.07])

    def _make_spline(self):
        effA = types.SimpleNamespace(
            log10_enu_bincenters=0.5 * (self.EDGES[:-1] + self.EDGES[1:]),
            log10_enu_binedges_lower=self.EDGES[:-1],
            log10_enu_binedges_upper=self.EDGES[1:],
            det_prob=self.DET_PROB,
        )
        gen = object.__new__(PDDatasetSignalGenerator)
        gen._effA_arr = [effA]
        fluxmodel = types.SimpleNamespace(energy_profile=_FlatEnergyProfile())
        return gen._create_inv_cdf_spline(
            src_idx=0, fluxmodel=fluxmodel, log_e_min=self.EDGES[0], log_e_max=self.EDGES[-1]
        )

    def _bin_prob(self):
        prob = np.diff(10**self.EDGES) * self.DET_PROB
        return prob / prob.sum()

    def test_cumulative_probability_is_reached_at_upper_bin_edges(self):
        spl = self._make_spline()
        cum = np.concatenate(([0.0], np.cumsum(self._bin_prob())))

        log10_e = interpolate.splev(cum, spl)

        np.testing.assert_allclose(log10_e, self.EDGES, atol=1e-12)

    def test_drawn_energies_fill_each_bin_with_its_probability(self):
        spl = self._make_spline()
        rng = np.random.default_rng(1)

        log10_e = interpolate.splev(rng.uniform(size=200_000), spl)

        self.assertGreaterEqual(log10_e.min(), self.EDGES[0])
        self.assertLessEqual(log10_e.max(), self.EDGES[-1])
        # The upper half of the highest bin must be reachable.
        self.assertGreater(log10_e.max(), 0.5 * (self.EDGES[-2] + self.EDGES[-1]))
        frac = np.histogram(log10_e, bins=self.EDGES)[0] / log10_e.size
        np.testing.assert_allclose(frac, self._bin_prob(), atol=5e-3)


if __name__ == '__main__':
    unittest.main()
