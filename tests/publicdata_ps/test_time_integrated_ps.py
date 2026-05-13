import os
import unittest

import numpy as np
from scipy import integrate

from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.core.config import Config
from skyllh.core.logging import setup_logging
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource
from skyllh.core.timing import TimeLord
from skyllh.datasets.i3 import PublicData_10y_ps

# Setup the logger for this Python module, which has to be done only once
# (on import of this module).
logger = setup_logging(
    cfg=Config(),
    name=__name__,
)


class AnalysisTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.cfg['repository']['base_path'] = os.path.join(os.getcwd(), '.repository')

        dsc = PublicData_10y_ps.create_dataset_collection(cfg=cls.cfg)
        cls.datasets = dsc[
            'IC40',
            'IC59',
            'IC79',
            'IC86_I',
            'IC86_II-VII',
        ]

        for ds in cls.datasets:
            if not ds.make_data_available():
                raise RuntimeError(f'The data of dataset {ds.name} could not be made available!')
            logger.info(f'{ds}')

        cls.source = PointLikeSource(name='point-source', ra=np.radians(77.358), dec=np.radians(5.693), weight=1)

        tl = TimeLord()
        with tl.task_timer('Creating analysis.'):
            cls.ana = create_analysis(
                cfg=cls.cfg,
                datasets=cls.datasets,
                source=cls.source,
                tl=tl,
            )
        logger.info(f'{tl}')

    def setUp(self):
        self.tl = TimeLord()

    def test_change_source_and_unblind(self):
        source = PointLikeSource(name='NGC 1068', ra=np.radians(40.67), dec=np.radians(-0.01), weight=1)
        self.ana.change_source(source)
        try:
            rss = RandomStateService(seed=1)
            with self.tl.task_timer('Call unblind for NGC 1068.'):
                (TS, params_dict, status) = self.ana.unblind(minimizer_rss=rss, tl=self.tl)

            logger.info(f'{self.tl}')
            logger.info(f'TS = {TS:.7f}')
            logger.info(f'ns_fit = {params_dict["ns"]:.7f}')
            logger.info(f'gamma_fit = {params_dict["gamma"]:.7f}')
            logger.info(f'minimizer status = {status}')

            np.testing.assert_allclose(TS, 19.5117469, rtol=1e-5)
            np.testing.assert_allclose(params_dict['ns'], 54.9606574, rtol=1e-5)
            np.testing.assert_allclose(params_dict['gamma'], 3.0884301, rtol=1e-5)
        finally:
            self.ana.change_source(self.__class__.source)

    def test_unblind(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call unblind.'):
            (TS, params_dict, status) = self.ana.unblind(minimizer_rss=rss, tl=self.tl)

        logger.info(f'{self.tl}')
        logger.info(f'TS = {TS:.7f}')
        logger.info(f'ns_fit = {params_dict["ns"]:.7f}')
        logger.info(f'gamma_fit = {params_dict["gamma"]:.7f}')
        logger.info(f'minimizer status = {status}')

        np.testing.assert_allclose(TS, 13.408184, rtol=1e-5)
        np.testing.assert_allclose(params_dict['ns'], 12.879558, rtol=1e-5)
        np.testing.assert_allclose(params_dict['gamma'], 2.122526, rtol=1e-5)

    def test_signal_energy_pdf_integral_normalization(self):
        """Each PDSignalEnergyPDF spline integrates to 1 over its energy range."""
        for pdfratio in self.ana._pdfratio_list:
            for pdf in pdfratio.pdfratio2.sig_pdf_set.values():
                integral = (
                    integrate.quad(
                        pdf.f_e_spl.evaluate,
                        pdf.log10_reco_e_min,
                        pdf.log10_reco_e_max,
                        limit=200,
                        full_output=1,
                    )[0]
                    / pdf.f_e_spl.norm
                )
                np.testing.assert_allclose(integral, 1.0, rtol=1e-6)

    def test_do_trial(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call do_trial.'):
            res = self.ana.do_trial(
                rss=rss,
                mean_n_sig=40,
                tl=self.tl,
            )[0]

        logger.info(f'{self.tl}')
        logger.info(f'n_sig = {res["n_sig"]}')
        logger.info(f'TS = {res["ts"]}')
        logger.info(f'ns = {res["ns"]}')
        logger.info(f'gamma = {res["gamma"]}')

        np.testing.assert_equal(res['n_sig'], 37)
        np.testing.assert_allclose(res['ts'], 112.744284, rtol=1e-5)
        np.testing.assert_allclose(res['ns'], 27.681108, rtol=1e-5)
        np.testing.assert_allclose(res['gamma'], 1.987327, rtol=1e-5)


class AnalysisWithEnergyRangeTestCase(unittest.TestCase):
    ENERGY_RANGE = (1e3, 1e6)
    ENERGY_RANGE_ALT = (1e4, 1e7)

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.cfg['repository']['base_path'] = os.path.join(os.getcwd(), '.repository')

        dsc = PublicData_10y_ps.create_dataset_collection(cfg=cls.cfg)
        cls.datasets = dsc[
            'IC40',
            'IC59',
            'IC79',
            'IC86_I',
            'IC86_II-VII',
        ]

        for ds in cls.datasets:
            if not ds.make_data_available():
                raise RuntimeError(f'The data of dataset {ds.name} could not be made available!')
            logger.info(f'{ds}')

        cls.source = PointLikeSource(name='point-source', ra=np.radians(77.358), dec=np.radians(5.693), weight=1)

        # Analysis 1: energy_range passed directly to create_analysis.
        cls.ana = create_analysis(
            cfg=cls.cfg,
            datasets=cls.datasets,
            source=cls.source,
            energy_range=cls.ENERGY_RANGE,
        )
        # Analysis 2: created without energy_range, then set via property.
        cls.ana_post_set = create_analysis(
            cfg=cls.cfg,
            datasets=cls.datasets,
            source=cls.source,
        )
        cls.ana_post_set.energy_range = cls.ENERGY_RANGE

        # Analysis 3: dedicated instance for the energy_range change-and-restore test.
        cls.ana_mutable = create_analysis(
            cfg=cls.cfg,
            datasets=cls.datasets,
            source=cls.source,
            energy_range=cls.ENERGY_RANGE,
        )

    def setUp(self):
        self.tl = TimeLord()

    def test_do_trial(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call do_trial with energy_range.'):
            res = self.ana.do_trial(
                rss=rss,
                mean_n_sig=40,
                tl=self.tl,
            )[0]

        logger.info(f'{self.tl}')
        logger.info(f'n_sig = {res["n_sig"]}')
        logger.info(f'TS = {res["ts"]}')
        logger.info(f'ns = {res["ns"]}')
        logger.info(f'gamma = {res["gamma"]}')

        np.testing.assert_equal(res['n_sig'], 37)
        np.testing.assert_allclose(res['ts'], 30.877059342233714, rtol=1e-5)
        np.testing.assert_allclose(res['ns'], 34.889627240419706, rtol=1e-5)
        np.testing.assert_allclose(res['gamma'], 2.2829856920449307, rtol=1e-5)

    def test_energy_range_consistency(self):
        """Passing energy_range to create_analysis gives the same result as setting it via the
        energy_range property after construction."""
        np.testing.assert_allclose(
            self.ana.calculate_fluxmodel_scaling_factor(),
            self.ana_post_set.calculate_fluxmodel_scaling_factor(),
            rtol=1e-10,
        )

        rss1 = RandomStateService(seed=1)
        res1 = self.ana.do_trial(rss=rss1, mean_n_sig=40)[0]

        rss2 = RandomStateService(seed=1)
        res2 = self.ana_post_set.do_trial(rss=rss2, mean_n_sig=40)[0]

        np.testing.assert_equal(res1['n_sig'], res2['n_sig'])
        np.testing.assert_allclose(res1['ts'], res2['ts'], rtol=1e-10)
        np.testing.assert_allclose(res1['ns'], res2['ns'], rtol=1e-10)
        np.testing.assert_allclose(res1['gamma'], res2['gamma'], rtol=1e-10)

    def test_energy_range_change_after_trial(self):
        """Changing energy_range mid-use and restoring it reproduces the original do_trial result."""
        # First trial with initial energy_range.
        rss1 = RandomStateService(seed=1)
        res1 = self.ana_mutable.do_trial(rss=rss1, mean_n_sig=40)[0]

        # Change energy_range and run a trial with the new range.
        self.ana_mutable.energy_range = self.ENERGY_RANGE_ALT
        rss_mid = RandomStateService(seed=2)
        res_mid = self.ana_mutable.do_trial(rss=rss_mid, mean_n_sig=40)[0]

        np.testing.assert_equal(res_mid['n_sig'], 32)
        np.testing.assert_allclose(res_mid['ts'], 123.047538, rtol=1e-5)
        np.testing.assert_allclose(res_mid['ns'], 40.036228, rtol=1e-5)
        np.testing.assert_allclose(res_mid['gamma'], 2.027787, rtol=1e-5)

        # Restore original energy_range and re-run with the same seed.
        self.ana_mutable.energy_range = self.ENERGY_RANGE
        rss3 = RandomStateService(seed=1)
        res3 = self.ana_mutable.do_trial(rss=rss3, mean_n_sig=40)[0]

        np.testing.assert_equal(res1['n_sig'], res3['n_sig'])
        np.testing.assert_allclose(res1['ts'], res3['ts'], rtol=1e-10)
        np.testing.assert_allclose(res1['ns'], res3['ns'], rtol=1e-10)
        np.testing.assert_allclose(res1['gamma'], res3['gamma'], rtol=1e-10)

    def test_mu2flux_flux2mu_consistency(self):
        """mu2flux and flux2mu are mutual inverses, and both analyses agree on all flux values."""
        mu_in = 10.0
        flux_norm = self.ana.mu2flux(mu_in)

        # Round-trip consistency
        np.testing.assert_allclose(self.ana.flux2mu(flux_norm), mu_in, rtol=1e-10)

        # Both analyses must agree on mu2flux and flux2mu
        np.testing.assert_allclose(self.ana.mu2flux(mu_in), self.ana_post_set.mu2flux(mu_in), rtol=1e-10)
        np.testing.assert_allclose(self.ana.flux2mu(flux_norm), self.ana_post_set.flux2mu(flux_norm), rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
