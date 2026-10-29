import os
import tempfile
import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.aeff import load_effective_area_array
from skyllh.analyses.i3.publicdata_ps.detsigyield import PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder
from skyllh.core.config import Config
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.flux_model import PowerLawEnergyFluxProfile, SteadyPointlikeFFM
from skyllh.core.parameters import Parameter
from skyllh.core.services import DetSigYieldService, SrcDetSigYieldWeightsService
from skyllh.core.source_hypo_grouping import SourceHypoGroup, SourceHypoGroupManager
from skyllh.core.source_model import PointLikeSource

# Deliberately non-uniform declination binning in degrees, covering the full sky, so that the difference between a
# declination axis in radians and in sin(dec) is large away from the equator.
DEC_BINEDGES_DEG = np.array([-90.0, -60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0, 90.0])
LOG10_ENU_BINEDGES = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

LIVETIME_DAYS = 100.0
PHI0 = 1.0
E0 = 1e3
GAMMA = 2.0


def _aeff_value(dec_idx, log10_enu_idx):
    """Returns the synthetic effective area in cm^2, which depends strongly on the declination bin."""
    return (1.0 + dec_idx) ** 3 * 10.0 ** (0.5 * log10_enu_idx)


def _write_aeff_file(pathfilename):
    """Writes a synthetic effective area file in the public data format."""
    with open(pathfilename, 'w') as f:
        f.write('# log10(E_nu/GeV)_min log10(E_nu/GeV)_max Dec_nu_min[deg] Dec_nu_max[deg] A_Eff[cm^2]\n')
        f.writelines(
            f'{LOG10_ENU_BINEDGES[e_idx]} {LOG10_ENU_BINEDGES[e_idx + 1]} '
            f'{DEC_BINEDGES_DEG[dec_idx]} {DEC_BINEDGES_DEG[dec_idx + 1]} '
            f'{_aeff_value(dec_idx, e_idx)}\n'
            for dec_idx in range(len(DEC_BINEDGES_DEG) - 1)
            for e_idx in range(len(LOG10_ENU_BINEDGES) - 1)
        )


def _expected_detsigyield(dec_idx, gamma):
    """Calculates the expected detector signal yield for a power-law flux and the given declination bin."""
    e1 = np.power(10, LOG10_ENU_BINEDGES[:-1])
    e2 = np.power(10, LOG10_ENU_BINEDGES[1:])
    flux_integral = PHI0 * E0 / (1 - gamma) * ((e2 / E0) ** (1 - gamma) - (e1 / E0) ** (1 - gamma))
    aeff = np.array([_aeff_value(dec_idx, e_idx) for e_idx in range(len(e1))])
    return np.sum(aeff * flux_integral) * LIVETIME_DAYS * 86400.0


def _sin_dec_bincenter_as_dec(dec_idx):
    """Returns the declination in radians, whose sin is the center of the given declination bin in sin(dec)."""
    sin_edges = np.sin(np.radians(DEC_BINEDGES_DEG))
    return float(np.arcsin(0.5 * (sin_edges[dec_idx] + sin_edges[dec_idx + 1])))


class PDDetSigYieldTestCase(unittest.TestCase):
    """Tests that the public data detector signal yield is a function of the
    source declination, consistent with the effective area declination binning
    (see GitHub issue #337).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        _write_aeff_file(os.path.join(cls.tmpdir.name, 'aeff.csv'))

        cls.cfg = Config()

        cls.ds = Dataset(
            cfg=cls.cfg,
            name='synthetic',
            exp_pathfilenames=None,
            mc_pathfilenames=None,
            livetime=LIVETIME_DAYS,
            default_sub_path_fmt='',
            version=1,
            base_path=cls.tmpdir.name,
        )
        cls.ds.add_aux_data_definition('eff_area_datafile', 'aeff.csv')
        cls.data = DatasetData(data_exp=None, data_mc=None, livetime=LIVETIME_DAYS)

        cls.fluxmodel = SteadyPointlikeFFM(
            Phi0=PHI0,
            energy_profile=PowerLawEnergyFluxProfile(E0=E0, gamma=GAMMA, cfg=cls.cfg),
            cfg=cls.cfg,
        )
        cls.param_gamma = Parameter(name='gamma', initial=GAMMA, valmin=1.0, valmax=4.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _create_shg_mgr(self, dec):
        builder = PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
            cfg=self.cfg, param_grid=self.param_gamma.as_linear_grid(delta=0.1), ncpu=1
        )
        return SourceHypoGroupManager(
            SourceHypoGroup(
                sources=PointLikeSource(ra=0.0, dec=dec),
                fluxmodel=self.fluxmodel,
                detsigyield_builders=builder,
            )
        )

    @staticmethod
    def _create_src_params_recarray(gamma=GAMMA):
        return np.array([(gamma, 1)], dtype=[('gamma', np.float64), ('gamma:gpidx', np.int32)])

    def _construct_detsigyield(self):
        shg_mgr = self._create_shg_mgr(dec=0.0)
        builder = shg_mgr.shg_list[0].detsigyield_builder_list[0]
        return builder.construct_detsigyield(dataset=self.ds, data=self.data, shg=shg_mgr.shg_list[0])

    def test_load_effective_area_array_returns_radians(self):
        (aeff, decnu_lower, decnu_upper, _, _) = load_effective_area_array(os.path.join(self.tmpdir.name, 'aeff.csv'))

        np.testing.assert_allclose(decnu_lower, np.radians(DEC_BINEDGES_DEG[:-1]))
        np.testing.assert_allclose(decnu_upper, np.radians(DEC_BINEDGES_DEG[1:]))
        self.assertEqual(aeff.shape, (len(DEC_BINEDGES_DEG) - 1, len(LOG10_ENU_BINEDGES) - 1))

    def test_sin_dec_binning_is_in_sin_dec(self):
        detsigyield = self._construct_detsigyield()

        np.testing.assert_allclose(detsigyield.sin_dec_binning.binedges, np.sin(np.radians(DEC_BINEDGES_DEG)))

        knots = detsigyield.log_spl_sinDec_param.get_knots()[0]
        self.assertGreaterEqual(knots.min(), -1.0)
        self.assertLessEqual(knots.max(), 1.0)

    def test_detsigyield_at_declination_bin_centers(self):
        """The spline interpolates exactly through the sin(dec) bin centers,
        so the detector signal yield must match the analytic expectation there.
        """
        detsigyield = self._construct_detsigyield()

        n_dec_bins = len(DEC_BINEDGES_DEG) - 1
        src_recarray = np.array(
            [(_sin_dec_bincenter_as_dec(i),) for i in range(n_dec_bins)], dtype=[('dec', np.float64)]
        )
        src_params_recarray = np.array(
            [(GAMMA, 1)] * n_dec_bins, dtype=[('gamma', np.float64), ('gamma:gpidx', np.int32)]
        )

        (values, _) = detsigyield(src_recarray=src_recarray, src_params_recarray=src_params_recarray)

        expected = np.array([_expected_detsigyield(i, GAMMA) for i in range(n_dec_bins)])
        np.testing.assert_allclose(values, expected, rtol=1e-10)

    def test_detsigyield_follows_source_without_rebuilding_detsigyield_service(self):
        """Changing the source declination and updating only the source
        detector signal yield weights service (as ``update_detsigyield_service=False``
        does) must yield the same values as re-building the detector signal
        yield service, because the detector signal yield is a function of the
        source declination.
        """
        dec_initial = _sin_dec_bincenter_as_dec(4)
        dec_new = _sin_dec_bincenter_as_dec(6)
        src_params_recarray = self._create_src_params_recarray()

        shg_mgr = self._create_shg_mgr(dec=dec_initial)
        detsigyield_service = DetSigYieldService(shg_mgr=shg_mgr, dataset_list=[self.ds], data_list=[self.data])
        weights_service = SrcDetSigYieldWeightsService(detsigyield_service=detsigyield_service)

        weights_service.calculate(src_params_recarray)
        (a_jk_initial, _) = weights_service.get_weights()
        assert a_jk_initial is not None
        np.testing.assert_allclose(a_jk_initial[0, 0], _expected_detsigyield(4, GAMMA), rtol=1e-10)

        # Change the source in-place, like Analysis.change_source does.
        shg_mgr.shg_list[0].source_list[0] = PointLikeSource(ra=0.0, dec=dec_new)
        weights_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.calculate(src_params_recarray)
        (a_jk_new, _) = weights_service.get_weights()
        assert a_jk_new is not None

        np.testing.assert_allclose(a_jk_new[0, 0], _expected_detsigyield(6, GAMMA), rtol=1e-10)

        # Re-building the detector signal yield service must give the same.
        detsigyield_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.calculate(src_params_recarray)
        (a_jk_rebuilt, _) = weights_service.get_weights()
        assert a_jk_rebuilt is not None

        np.testing.assert_allclose(a_jk_new, a_jk_rebuilt, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
