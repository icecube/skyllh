import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.backgroundpdf import PDMCBackgroundI3EnergyPDF
from skyllh.core.binning import BinningDefinition
from skyllh.core.config import Config


class PDMCBackgroundI3EnergyPDFGetPdTestCase(unittest.TestCase):
    """Tests the bin lookup of PDMCBackgroundI3EnergyPDF.get_pd at the
    edges of its binning. Out-of-range data is rejected beforehand by
    assert_is_valid_for_trial_data.
    """

    def setUp(self):
        self.log10emu_edges = np.array([1.0, 2.0, 3.0, 4.0])
        self.sindecmu_edges = np.array([-1.0, 0.0, 1.0])
        # Unique value per bin: 10*log10emu_idx + sindecmu_idx + 1.
        hist = (
            10 * np.arange(len(self.log10emu_edges) - 1)[:, np.newaxis]
            + np.arange(len(self.sindecmu_edges) - 1)[np.newaxis, :]
            + 1.0
        )
        self.pdf = PDMCBackgroundI3EnergyPDF(
            pdf_log10emu_sindecmu=hist,
            log10emu_binning=BinningDefinition('log_energy', self.log10emu_edges),
            sindecmu_binning=BinningDefinition('sin_dec', self.sindecmu_edges),
            cfg=Config(),
        )

    def _get_pd(self, log_energy, sin_dec):
        tdm = {'log_energy': np.asarray(log_energy), 'sin_dec': np.asarray(sin_dec)}
        (pd, grads) = self.pdf.get_pd(tdm=tdm)  # pyright: ignore[reportArgumentType]
        self.assertEqual(grads, {})
        return pd

    def test_upper_edges_are_last_bins(self):
        pd = self._get_pd([4.0, 1.0, 4.0], [1.0, 1.0, -1.0])
        np.testing.assert_array_equal(pd, [22.0, 2.0, 21.0])

    def test_interior_values(self):
        pd = self._get_pd([1.5, 3.5, 2.0], [-0.5, 0.5, 0.0])
        np.testing.assert_array_equal(pd, [1.0, 22.0, 12.0])


if __name__ == '__main__':
    unittest.main()
