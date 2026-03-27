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
from skyllh.core.services import (
    DatasetSignalWeightFactorsService,
    DetSigYieldService,
    SrcDetSigYieldWeightsService,
)
from skyllh.core.signal_generator import (
    MCMultiDatasetSignalGenerator,
    MultiDatasetSignalGenerator,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.i3.config import (
    add_icecube_specific_analysis_required_data_fields,
)
from skyllh.i3.detsigyield import (
    SingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)
from skyllh.i3.signal_generation import (
    PointLikeSourceI3SignalGenerationMethod,
)

DATA_SAMPLES_IMPORTED = tool.is_available('i3skyllh')


def create_signal_generator(
    cfg,
    sig_generator_cls,
    sig_gen_method,
):
    """Creates a SignalGenerator instance of the given class
    ``sig_generator_cls`` using the signal generation method instance
    ``sig_gen_method``.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    sig_generator_cls : class of SignalGenerator
        The class object of the signal generator.
    sig_gen_method : instance of SignalGenerationMethod
        The SignalGenerationMethod instance that should be used for the
        signal generation.

    Returns
    -------
    sig_gen : instance of ``sig_generator_cls``
        The created instance of ``sig_generator_cls`` which is derived from
        SignalGenerator.
    """
    add_icecube_specific_analysis_required_data_fields(cfg)

    datasets = tool.get('i3skyllh.datasets')

    dataset_name = 'PointSourceTracks_v004p00'
    dsc = datasets.data_samples[dataset_name].create_dataset_collection(cfg=cfg)
    ds_list = dsc['IC86, 2018', 'IC86, 2019']

    data_list = [ds.load_and_prepare_data() for ds in ds_list]

    sources = [
        PointLikeSource(ra=np.deg2rad(0), dec=np.deg2rad(10)),
        PointLikeSource(ra=np.deg2rad(30), dec=np.deg2rad(2)),
    ]

    fluxmodel = SteadyPointlikeFFM(
        Phi0=1,
        energy_profile=PowerLawEnergyFluxProfile(E0=1000, gamma=2, cfg=cfg),
        cfg=cfg,
    )

    gamma_grid = ParameterGrid(name='gamma', grid=np.arange(1, 4.1, 0.1))
    detsigyield_builder = SingleParamFluxPointLikeSourceI3DetSigYieldBuilder(param_grid=gamma_grid, cfg=cfg)

    shg_mgr = SourceHypoGroupManager(
        SourceHypoGroup(
            sources=sources,
            fluxmodel=fluxmodel,
            detsigyield_builders=detsigyield_builder,
            sig_gen_method=sig_gen_method,
        )
    )

    detsigyield_service = DetSigYieldService(
        shg_mgr=shg_mgr,
        dataset_list=ds_list,
        data_list=data_list,
    )

    src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
        detsigyield_service=detsigyield_service,
    )

    ds_sig_weight_factors_service = DatasetSignalWeightFactorsService(
        src_detsigyield_weights_service=src_detsigyield_weights_service,
    )

    sig_gen = sig_generator_cls(
        cfg=cfg,
        shg_mgr=shg_mgr,
        dataset_list=ds_list,
        data_list=data_list,
        ds_sig_weight_factors_service=ds_sig_weight_factors_service,
    )

    return sig_gen


class TestSignalGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This class method will run only once for this TestCase."""
        if not DATA_SAMPLES_IMPORTED:
            return

        cls.cfg = Config()

        repository = tool.get('i3skyllh.datasets.repository')
        repository.setup_repository(cls.cfg)

        cls._sig_gen = create_signal_generator(
            cfg=cls.cfg,
            sig_generator_cls=MCMultiDatasetSignalGenerator,
            sig_gen_method=PointLikeSourceI3SignalGenerationMethod(),
        )

    @unittest.skipIf(not DATA_SAMPLES_IMPORTED, 'Data samples not imported!')
    def testSigCandidatesArray(self):

        arr = type(self)._sig_gen._sig_candidates

        # Check data type.
        self.assertTrue(isinstance(arr, np.ndarray))

        # Check field names.
        field_names = arr.dtype.fields.keys()
        self.assertTrue(
            ('ds_idx' in field_names)
            and ('ev_idx' in field_names)
            and ('shg_idx' in field_names)
            and ('shg_src_idx' in field_names)
            and ('weight' in field_names)
        )

        # Check the length of the array.
        self.assertTrue(len(arr) == 894736, 'array length')
        # Check ds_idx values.
        ds_idxs = np.unique(arr['ds_idx'])
        self.assertTrue(ds_idxs.shape == (2,), 'ds_idx shape')
        self.assertTrue(np.all(np.equal(ds_idxs, np.array([0, 1]))), 'ds_idx values')

        # Check shg_idx values.
        shg_idxs = np.unique(arr['shg_idx'])
        self.assertTrue(shg_idxs.shape == (1,), 'shg_idx shape')
        self.assertTrue(np.all(np.equal(shg_idxs, np.array([0]))), 'shg_idx values')

        # Check shg_src_idx values.
        shg_src_idxs = np.unique(arr['shg_src_idx'])
        self.assertTrue(shg_src_idxs.shape == (2,), 'shg_src_idxs shape')
        self.assertTrue(np.all(np.equal(shg_src_idxs, np.array([0, 1]))), 'shg_idx values')

    @unittest.skipIf(not DATA_SAMPLES_IMPORTED, 'Data samples not imported!')
    def testSigCandidatesWeightSum(self):
        weight_sum = type(self)._sig_gen._sig_candidates_weight_sum
        self.assertTrue(np.isclose(weight_sum, 7884630181096259), f'weight sum is {weight_sum}')


class _DummySrcDetSigYieldWeightsService:
    def __init__(self, a_jk):
        self._a_jk = np.asarray(a_jk, dtype=np.float64)

    def calculate(self, src_params_recarray):
        return None

    def get_weights(self):
        return (self._a_jk, None)


class _DummyDatasetSignalWeightFactorsService:
    def __init__(self, src_service, ds_weights=None):
        self.src_detsigyield_weights_service = src_service
        if ds_weights is None:
            ds_weights = np.array([0.5, 0.5], dtype=np.float64)
        self._ds_weights = np.asarray(ds_weights, dtype=np.float64)

    def calculate(self):
        return None

    def get_weights(self):
        return (self._ds_weights, None)


class _DummyShgMgr:
    def __init__(self, n_sources):
        self.n_sources = n_sources


class _DummyDatasetSigGenerator:
    def __init__(self, configured_energy_range, correction_factors):
        self._input_range = configured_energy_range
        self._energy_range = configured_energy_range
        self._correction_factors = np.asarray(correction_factors, dtype=np.float64)
        self.last_generate_kwargs = None

    def get_energy_range_correction_factors(self):
        return self._correction_factors

    def generate_signal_events(self, rss, mean, poisson, src_detsigyield_weights_service, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        return (0, {})


class TestMultiDatasetSignalGeneratorEnergyRangeConsistency(unittest.TestCase):
    @staticmethod
    def _make_sig_gen(configured_ranges):
        sig_gen = MultiDatasetSignalGenerator.__new__(MultiDatasetSignalGenerator)
        sig_gen._shg_mgr = _DummyShgMgr(n_sources=2)
        sig_gen._src_params_recarray = np.zeros((2,), dtype=[('gamma', np.float64), ('gamma:gpidx', np.int32)])
        src_service = _DummySrcDetSigYieldWeightsService(a_jk=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        sig_gen._ds_sig_weight_factors_service = _DummyDatasetSignalWeightFactorsService(src_service=src_service)
        sig_gen._sig_generator_list = [
            _DummyDatasetSigGenerator(configured_energy_range=r, correction_factors=[1.0, 1.0])
            for r in configured_ranges
        ]
        return sig_gen

    def test_no_configured_cut_in_all_datasets_is_allowed(self):
        sig_gen = self._make_sig_gen(configured_ranges=[None, None])

        scaling_factor = sig_gen.fluxmodel_scaling_factor()

        self.assertTrue(np.isclose(scaling_factor, 0.1))

    def test_mixed_none_and_configured_range_raises(self):
        sig_gen = self._make_sig_gen(configured_ranges=[None, (2.0, 8.0)])

        with self.assertRaisesRegex(ValueError, 'single shared energy_range'):
            sig_gen.fluxmodel_scaling_factor()

    def test_different_configured_ranges_raise(self):
        sig_gen = self._make_sig_gen(configured_ranges=[(2.0, 8.0), (3.0, 8.0)])

        with self.assertRaisesRegex(ValueError, 'single shared energy_range'):
            sig_gen.fluxmodel_scaling_factor()

    def test_same_configured_ranges_are_allowed(self):
        sig_gen = self._make_sig_gen(configured_ranges=[(2.0, 8.0), (2.0, 8.0)])

        scaling_factor = sig_gen.fluxmodel_scaling_factor()

        self.assertTrue(np.isclose(scaling_factor, 0.1))


class TestMultiDatasetSignalGeneratorKwargPolicy(unittest.TestCase):
    def test_generate_signal_events_rejects_energy_range_kwarg(self):
        sig_gen = MultiDatasetSignalGenerator.__new__(MultiDatasetSignalGenerator)
        sig_gen._src_params_recarray = np.zeros((1,), dtype=[('gamma', np.float64), ('gamma:gpidx', np.int32)])

        src_service = _DummySrcDetSigYieldWeightsService(a_jk=np.array([[1.0], [1.0]], dtype=np.float64))
        sig_gen._ds_sig_weight_factors_service = _DummyDatasetSignalWeightFactorsService(
            src_service=src_service,
            ds_weights=np.array([0.5, 0.5], dtype=np.float64),
        )

        ds_gen_0 = _DummyDatasetSigGenerator(configured_energy_range=None, correction_factors=[1.0])
        ds_gen_1 = _DummyDatasetSigGenerator(configured_energy_range=None, correction_factors=[1.0])
        sig_gen._sig_generator_list = [ds_gen_0, ds_gen_1]

        class _DummyRandom:
            def poisson(self, mean):
                return int(mean)

            def choice(self, a, size, p):
                return np.zeros((size,), dtype=np.int64)

        class _DummyRSS:
            random = _DummyRandom()

        rss = _DummyRSS()

        with self.assertRaises(TypeError):
            sig_gen.generate_signal_events(rss=rss, mean=10, poisson=False, energy_range=(1e2, 1e3))

        self.assertIsNone(ds_gen_0.last_generate_kwargs)
        self.assertIsNone(ds_gen_1.last_generate_kwargs)


if __name__ == '__main__':
    unittest.main()
