# -*- coding: utf-8 -*-

import unittest

import numpy as np

from skyllh.core import (
    tool,
)
from skyllh.core.config import (
    Config,
)
from skyllh.core.flux_model import (
    PowerLawEnergyFluxProfile,
    SteadyPointlikeFFM,
)
from skyllh.core.parameters import (
    ParameterGrid,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.i3.detector_model import (
    IceCubeDetectorModel,
)
from skyllh.i3.detsigyield import (
    SingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)

DATA_SAMPLES_IMPORTED = tool.is_available('i3skyllh')


class SingleParamFluxPointLikeSourceI3DetSigYield_TestCase(
        unittest.TestCase,
):
    @classmethod
    def setUpClass(cls):
        if not DATA_SAMPLES_IMPORTED:
            return

        cls.cfg = Config()

        repository = tool.get('i3skyllh.datasets.repository')
        repository.setup_repository(cls.cfg)

        datasets = tool.get('i3skyllh.datasets')

        dataset_name = 'PointSourceTracks_v004p00'
        dsc = datasets.data_samples[dataset_name].create_dataset_collection(
            cfg=cls.cfg)
        ds = dsc['IC86, 2019']
        data = ds.load_and_prepare_data()

        param_grid = ParameterGrid(
            name='gamma',
            grid=np.linspace(1, 4, num=int((4-1)/0.2)+1))
        builder = SingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
            cfg=cls.cfg,
            param_grid=param_grid,
            sin_dec_binning=ds.get_binning_definition('sin_dec'),
        )
        fluxmodel = SteadyPointlikeFFM(
            cfg=cls.cfg,
            Phi0=1,
            energy_profile=PowerLawEnergyFluxProfile(
                cfg=cls.cfg,
                E0=1000,
                gamma=2,
            )
        )
        cls.sources = [
            PointLikeSource(
                name='TXS',
                ra=np.deg2rad(77),
                dec=np.deg2rad(-5),
            )
        ]
        shg = SourceHypoGroup(
            sources=cls.sources,
            fluxmodel=fluxmodel,
            detsigyield_builders=builder,
        )

        cls.detsigyield = builder.construct_detsigyield(
            detector_model=IceCubeDetectorModel(),
            dataset=ds,
            data=data,
            shg=shg,
        )

    @unittest.skipIf(not DATA_SAMPLES_IMPORTED, 'Data samples not imported!')
    def test__call__(self):
        src_params_recarray = np.array(
            [(2, 1)],
            dtype=[
                ('gamma', np.float64),
                ('gamma:gpidx', np.int32)
            ])
        (Y, Ygrad) = self.detsigyield(
            src_recarray=self.detsigyield.sources_to_recarray(self.sources),
            src_params_recarray=src_params_recarray,
        )

        np.testing.assert_allclose(Y, [1.08676002e+15], rtol=0.01)
        self.assertIsInstance(Ygrad, dict)
        self.assertEqual(len(Ygrad), 1)
        self.assertTrue(0 in Ygrad)
        np.testing.assert_allclose(Ygrad[0], [-6.0136921e+15], rtol=0.01)


if __name__ == '__main__':
    unittest.main()
