# -*- coding: utf-8 -*-

import unittest

import numpy as np

from astropy import (
    units,
)
from astropy.coordinates import (
    EarthLocation,
)

from skyllh.core import (
    tool,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.config import (
    Config,
)
from skyllh.core.datafields import (
    DataFieldStages,
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
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.detsigyield import (
    SingleParamFluxPointLikeSourceDetSigYieldBuilder,
)
from skyllh.i3.livetime import (
    I3Livetime,
)

DATA_SAMPLES_IMPORTED = tool.is_available('i3skyllh')


class SingleParamFluxPointLikeSourceDetSigYield_TestCase(
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

        # Add true_zen data field to the dataset.
        def add_true_zen(data):
            data.mc['true_zen'] = np.pi/2 + data.mc['true_dec']
        ds.add_data_preparation(add_true_zen)
        cls.cfg['datafields']['true_zen'] = DataFieldStages.ANALYSIS_MC

        data = ds.load_and_prepare_data()

        # Create a Livetime instance.
        livetime = I3Livetime.from_grl_data(data.grl)
        param_grid = ParameterGrid(
            name='gamma',
            grid=np.linspace(1, 4, num=int((4-1)/0.2)+1))
        # Transform sin(true_dec) into cos(true_zen) for a detector at
        # South Pole and sort the binning in ascending order.
        sin_dec_binning = ds.get_binning_definition('sin_dec')
        true_dec = np.arcsin(sin_dec_binning.binedges)
        true_zen = np.pi/2 + true_dec
        cos_true_zen = np.sort(np.cos(true_zen))
        cos_true_zen_binning = BinningDefinition(
            name='cos_true_zen',
            binedges=cos_true_zen,
        )
        builder = SingleParamFluxPointLikeSourceDetSigYieldBuilder(
            cfg=cls.cfg,
            livetime=livetime,
            param_grid=param_grid,
            cos_true_zen_binning=cos_true_zen_binning,
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
        detector_model = DetectorModel(
            name='Detector at true South Pole',
            location=EarthLocation.from_geodetic(
                lon=0*units.deg,
                lat=-90*units.deg,
                height=0*units.m,
            )
        )
        cls.detsigyield = builder.construct_detsigyield(
            detector_model=detector_model,
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

        np.testing.assert_allclose(Y, [1.09241272e+15], rtol=0.01)
        self.assertIsInstance(Ygrad, dict)
        self.assertEqual(len(Ygrad), 1)
        self.assertTrue(0 in Ygrad)
        np.testing.assert_allclose(Ygrad[0], [-6.03707843e+15], rtol=0.01)


if __name__ == '__main__':
    unittest.main()
