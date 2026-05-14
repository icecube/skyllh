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

    def test_fitparam_values_with_energy_range_raises(self):
        """Combining fitparam_values with a configured energy_range raises NotImplementedError because
        the energy range correction factors are precomputed at reference parameters and would be stale."""
        fitparam_values = np.zeros(self.ana._pmm.n_global_floating_params, dtype=np.float64)
        fitparam_values[self.ana._pmm.get_gflp_idx('gamma')] = 3.0

        # The energy_range is set as create_analysis parameter.
        with self.assertRaises(NotImplementedError):
            self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=fitparam_values)

        # The energy_range is set as ana.energy_range property.
        with self.assertRaises(NotImplementedError):
            self.ana_post_set.calculate_fluxmodel_scaling_factor(fitparam_values=fitparam_values)

    def test_mu2flux_flux2mu_consistency(self):
        """mu2flux and flux2mu are mutual inverses, and both analyses agree on all flux values."""
        mu_in = 10.0
        flux_norm = self.ana.mu2flux(mu_in)

        # Round-trip consistency
        np.testing.assert_allclose(self.ana.flux2mu(flux_norm), mu_in, rtol=1e-10)

        # Both analyses must agree on mu2flux and flux2mu
        np.testing.assert_allclose(self.ana.mu2flux(mu_in), self.ana_post_set.mu2flux(mu_in), rtol=1e-10)
        np.testing.assert_allclose(self.ana.flux2mu(flux_norm), self.ana_post_set.flux2mu(flux_norm), rtol=1e-10)


class FitparamValuesTestCase(unittest.TestCase):
    """Integration tests for the fitparam_values parameter of calculate_fluxmodel_scaling_factor,
    mu2flux, and flux2mu on SingleSourceMultiDatasetLLHRatioAnalysis.

    The publicdata_ps analysis has two global floating parameters in this order:
      index 0: ns   (mapped to detector model — does not affect the flux scaling factor)
      index 1: gamma (mapped to source — determines the detector signal yield)
    The reference gamma embedded in the flux model is 2.0 (refplflux_gamma default).
    """

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

        cls.source = PointLikeSource(name='point-source', ra=np.radians(77.358), dec=np.radians(5.693), weight=1)

        cls.ana = create_analysis(
            cfg=cls.cfg,
            datasets=cls.datasets,
            source=cls.source,
        )

        # The reference gamma is 2.122526, which is the best-fit gamma obtained in test_unblind() with the default analysis configuration.
        cls.ana_ref = create_analysis(
            cfg=cls.cfg,
            datasets=cls.datasets,
            source=cls.source,
            refplflux_gamma=2.122526,
        )

        # PMM order: [ns, gamma]. The flux model reference gamma is 2.0.
        cls.GAMMA_IDX = cls.ana._pmm.get_gflp_idx(name='gamma')
        cls.N_PARAMS = cls.ana._pmm.n_global_floating_params
        cls.REF_GAMMA = 2.0

    def _fitparam_values(self, gamma):
        """Build a fitparam_values array with the given gamma; ns is set to zero."""
        values = np.zeros(self.N_PARAMS, dtype=np.float64)
        values[self.GAMMA_IDX] = gamma
        return values

    def test_fitparam_values_at_reference_gamma_matches_default(self):
        """calculate_fluxmodel_scaling_factor() and the same call with fitparam_values at the
        reference gamma must return equal results."""
        scaling_default = self.ana.calculate_fluxmodel_scaling_factor()
        scaling_ref = self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=self._fitparam_values(self.REF_GAMMA))
        np.testing.assert_allclose(scaling_ref, scaling_default, rtol=1e-10)

    def test_fitparam_values_different_gamma_changes_scaling_factor(self):
        """Providing a gamma different from the reference must return a different scaling factor."""
        scaling_default = self.ana.calculate_fluxmodel_scaling_factor()
        scaling_alt = self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=self._fitparam_values(3.0))
        self.assertFalse(
            np.isclose(scaling_default, scaling_alt, atol=0, rtol=1e-10),
            msg=f'Expected scaling factors to differ but got {scaling_default} and {scaling_alt}',
        )

    def test_default_cache_unchanged_after_fitparam_call(self):
        """Calling with fitparam_values must not alter the signal generator's cached reference state.
        A subsequent default call must reproduce the original scaling factor."""
        scaling_before = self.ana.calculate_fluxmodel_scaling_factor()

        # Call with a non-reference gamma to exercise the custom-params path.
        self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=self._fitparam_values(3.0))

        scaling_after = self.ana.calculate_fluxmodel_scaling_factor()
        np.testing.assert_allclose(scaling_after, scaling_before, rtol=1e-10)

    def test_mu2flux_with_fitparam_values_consistent(self):
        """mu2flux(mu, fitparam_values=...) equals calculate_fluxmodel_scaling_factor(fitparam_values=...) * mu."""
        mu = 25.0
        fv = self._fitparam_values(3.0)
        expected = self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=fv) * mu
        result = self.ana.mu2flux(mu, fitparam_values=fv)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_flux2mu_with_fitparam_values_consistent(self):
        """flux2mu(flux_norm, fitparam_values=...) equals flux_norm / calculate_fluxmodel_scaling_factor(fitparam_values=...)."""
        flux_norm = 1e-12
        fv = self._fitparam_values(3.0)
        expected = flux_norm / self.ana.calculate_fluxmodel_scaling_factor(fitparam_values=fv)
        result = self.ana.flux2mu(flux_norm, fitparam_values=fv)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_mu2flux_flux2mu_roundtrip_with_fitparam_values(self):
        """flux2mu(mu2flux(mu, fitparam_values=...), fitparam_values=...) recovers mu."""
        mu = 15.0
        fv = self._fitparam_values(3.0)
        flux_norm = self.ana.mu2flux(mu, fitparam_values=fv)
        mu_recovered = self.ana.flux2mu(flux_norm, fitparam_values=fv)
        np.testing.assert_allclose(mu_recovered, mu, rtol=1e-10)

    def test_ana_with_fitparam_values_matches_ana_ref(self):
        """Using fitparam_values with ana should match the reference analysis without fitparam_values."""
        print(self.ana.sig_gen_energy_range)
        print(self.ana.sig_gen_energy_range_is_set)
        flux1 = self.ana.mu2flux(10, fitparam_values=self._fitparam_values(2.122526))
        flux2 = self.ana_ref.mu2flux(10)
        np.testing.assert_allclose(flux1, flux2, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
