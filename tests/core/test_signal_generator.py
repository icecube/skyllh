# -*- coding: utf-8 -*-

import numpy as np
import os.path
import unittest


from skyllh.core.parameters import (
    ParameterGrid,
)
from skyllh.core.signal_generator import (
    SignalGenerator,
)
from skyllh.core.source_hypothesis import (
    SourceHypoGroupManager,
)
from skyllh.core.source_hypo_group import (
    SourceHypoGroup,
)
from skyllh.i3.detsigyield import (
    PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod,
)
from skyllh.i3.signal_generation import (
    PointLikeSourceI3SignalGenerationMethod,
)
from skyllh.physics.source import (
    PointLikeSource,
)
from skyllh.physics.flux import (
    PowerLawFlux,
)

DATA_SAMPLES_IMPORTED = True
try:
    from i3skyllh.datasets import repository
    from i3skyllh.datasets import data_samples
except:
    DATA_SAMPLES_IMPORTED = False


class TestSignalGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This class method will run only once for this TestCase.
        """
        if not DATA_SAMPLES_IMPORTED:
            return

        dataset_name = 'PointSourceTracks_v004p00'
        dsc = data_samples[dataset_name].create_dataset_collection()
        ds_list = dsc.get_datasets(['IC86, 2018', 'IC86, 2019'])

        data_list = [ds.load_and_prepare_data() for ds in ds_list]

        sources = [
            PointLikeSource(ra=np.deg2rad(0), dec=np.deg2rad(10)),
            PointLikeSource(ra=np.deg2rad(30), dec=np.deg2rad(2)),
        ]

        fluxmodel = PowerLawFlux(
            Phi0=1,
            E0=1000,
            gamma=2)

        gamma_grid = ParameterGrid(name='gamma', grid=np.arange(1, 4.1, 0.1))
        detsigyield_builder = PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod(
            gamma_grid=gamma_grid)

        sig_gen_method = PointLikeSourceI3SignalGenerationMethod()

        shg_mgr = SourceHypoGroupManager(
            SourceHypoGroup(
                sources=sources,
                fluxmodel=fluxmodel,
                detsigyield_implmethods=detsigyield_builder,
                sig_gen_method=sig_gen_method))

        cls._sig_gen = SignalGenerator(
            src_hypo_group_manager=shg_mgr,
            dataset_list=ds_list,
            data_list=data_list)

    @unittest.skipIf(not DATA_SAMPLES_IMPORTED, 'Data samples not imported!')
    def testSigCandidatesArray(self):

        arr = type(self)._sig_gen._sig_candidates

        # Check data type.
        self.assertTrue(
            isinstance(arr, np.ndarray)
        )

        # Check field names.
        field_names = arr.dtype.fields.keys()
        self.assertTrue(
            ('ds_idx' in field_names) and
            ('ev_idx' in field_names) and
            ('shg_idx' in field_names) and
            ('shg_src_idx' in field_names) and
            ('weight' in field_names)
        )

        # Check the length of the array.
        self.assertTrue(
            len(arr) == 894736,
            'array length'
        )
        # Check ds_idx values.
        ds_idxs = np.unique(arr['ds_idx'])
        self.assertTrue(
            ds_idxs.shape == (2,),
            'ds_idx shape'
        )
        self.assertTrue(
            np.all(np.equal(ds_idxs, np.array([0, 1]))),
            'ds_idx values'
        )

        # Check shg_idx values.
        shg_idxs = np.unique(arr['shg_idx'])
        self.assertTrue(
            shg_idxs.shape == (1,),
            'shg_idx shape'
        )
        self.assertTrue(
            np.all(np.equal(shg_idxs, np.array([0]))),
            'shg_idx values'
        )

        # Check shg_src_idx values.
        shg_src_idxs = np.unique(arr['shg_src_idx'])
        self.assertTrue(
            shg_src_idxs.shape == (2,),
            'shg_src_idxs shape'
        )
        self.assertTrue(
            np.all(np.equal(shg_src_idxs, np.array([0,1]))),
            'shg_idx values'
        )

    @unittest.skipIf(not DATA_SAMPLES_IMPORTED, 'Data samples not imported!')
    def testSigCandidatesWeightSum(self):
        self.assertTrue(
            type(self)._sig_gen._sig_candidates_weight_sum == 7884630181096259,
        )


if(__name__ == '__main__'):
    unittest.main()
