import os
import unittest

import numpy as np
from scipy import integrate

import skyllh
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.core.config import Config
from skyllh.core.logging import setup_logging
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource
from skyllh.core.timing import TimeLord

# Setup the logger for this Python module, which has to be done only once
# (on import of this module).
logger = setup_logging(
    cfg=Config(),
    name=__name__,
)


class NGC1068UnblindingDR2TestCase(unittest.TestCase):
    """Unblinds NGC 1068 using the IceCube 14-year public point-source data
    release (IceTracks-DR2), initialized via the ``skyllh.create_datasets``
    dataset initialization API.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.cfg['repository']['base_path'] = os.path.join(os.getcwd(), '.repository')

        cls.datasets = skyllh.create_datasets(
            'IceTracks-DR2',
            cfg=cls.cfg,
        )

        for ds in cls.datasets:
            if not ds.make_data_available():
                raise RuntimeError(f'The data of dataset {ds.name} could not be made available!')
            logger.info(f'{ds}')

        cls.source = PointLikeSource(name='NGC 1068', ra=np.radians(40.67), dec=np.radians(-0.01))

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

    def test_unblind(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call unblind for NGC 1068.'):
            (TS, params_dict, status) = self.ana.unblind(minimizer_rss=rss, tl=self.tl)

        logger.info(f'{self.tl}')
        logger.info(f'TS = {TS:.7f}')
        logger.info(f'ns_fit = {params_dict["ns"]:.7f}')
        logger.info(f'gamma_fit = {params_dict["gamma"]:.7f}')
        logger.info(f'minimizer status = {status}')

        np.testing.assert_allclose(TS, 29.1535172, rtol=1e-5)
        np.testing.assert_allclose(params_dict['ns'], 80.0938193, rtol=1e-5)
        np.testing.assert_allclose(params_dict['gamma'], 3.2088480, rtol=1e-5)

    def test_do_trial_background(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call do_trial with mean_n_sig=0.'):
            res = self.ana.do_trial(
                rss=rss,
                mean_n_sig=0,
                tl=self.tl,
            )[0]

        logger.info(f'{self.tl}')
        logger.info(f'n_sig = {res["n_sig"]}')
        logger.info(f'TS = {res["ts"]}')
        logger.info(f'ns = {res["ns"]}')
        logger.info(f'gamma = {res["gamma"]}')

        np.testing.assert_equal(res['n_sig'], 0)
        np.testing.assert_allclose(res['ts'], 0.5616519, rtol=1e-5)
        np.testing.assert_allclose(res['ns'], 9.0537408, rtol=1e-5)
        np.testing.assert_allclose(res['gamma'], 3.1219808, rtol=1e-5)

    def test_do_trial_signal(self):
        rss = RandomStateService(seed=1)
        with self.tl.task_timer('Call do_trial with mean_n_sig=40.'):
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

        np.testing.assert_equal(res['n_sig'], 41)
        np.testing.assert_allclose(res['ts'], 189.0684401, rtol=1e-5)
        np.testing.assert_allclose(res['ns'], 41.9057837, rtol=1e-5)
        np.testing.assert_allclose(res['gamma'], 1.9499991, rtol=1e-5)

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
                        full_output=True,
                    )[0]
                    / pdf.f_e_spl.norm
                )
                np.testing.assert_allclose(integral, 1.0, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
