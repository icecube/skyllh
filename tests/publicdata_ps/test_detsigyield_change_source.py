import os
import tempfile
import unittest

import numpy as np

from skyllh.analyses.i3.publicdata_ps.detsigyield import PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder
from skyllh.core.config import Config
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.flux_model import PowerLawEnergyFluxProfile, SteadyPointlikeFFM
from skyllh.core.parameters import Parameter
from skyllh.core.services import DetSigYieldService, SrcDetSigYieldWeightsService
from skyllh.core.source_hypo_grouping import SourceHypoGroup, SourceHypoGroupManager
from skyllh.core.source_model import PointLikeSource

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


class ChangeSourceDetSigYieldTestCase(unittest.TestCase):
    """Tests that the source detector signal yield weights follow a change of
    the source, as done by ``Analysis.change_source``, with and without
    re-building the detector signal yield service.
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

    def test_detsigyield_follows_source_without_rebuilding_detsigyield_service(self):
        """Changing the source declination and updating only the source
        detector signal yield weights service (as ``update_detsigyield_service=False``
        does) must yield the same values as re-building the detector signal
        yield service, because the detector signal yield is a function of the
        source declination.
        """
        src_params_recarray = np.array([(GAMMA, 1)], dtype=[('gamma', np.float64), ('gamma:gpidx', np.int32)])

        shg_mgr = self._create_shg_mgr(dec=np.radians(5.0))
        detsigyield_service = DetSigYieldService(shg_mgr=shg_mgr, dataset_list=[self.ds], data_list=[self.data])
        weights_service = SrcDetSigYieldWeightsService(detsigyield_service=detsigyield_service)

        weights_service.calculate(src_params_recarray)
        (a_jk_initial, _) = weights_service.get_weights()
        assert a_jk_initial is not None

        # Change the source in-place and update only the weights service, like
        # Analysis.change_source(update_detsigyield_service=False) does.
        shg_mgr.shg_list[0].source_list[0] = PointLikeSource(ra=0.0, dec=np.radians(45.0))
        weights_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.calculate(src_params_recarray)
        (a_jk_new, _) = weights_service.get_weights()
        assert a_jk_new is not None

        # The effective area depends strongly on the declination.
        self.assertFalse(np.allclose(a_jk_new, a_jk_initial))

        # Re-building the detector signal yield service must give the same.
        detsigyield_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.change_shg_mgr(shg_mgr=shg_mgr)
        weights_service.calculate(src_params_recarray)
        (a_jk_rebuilt, _) = weights_service.get_weights()
        assert a_jk_rebuilt is not None

        np.testing.assert_allclose(a_jk_new, a_jk_rebuilt, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
