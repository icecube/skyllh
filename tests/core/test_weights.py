# -*- coding: utf-8 -*-

"""The unit tests in this module test classes of the skyllh.core.weights module.
"""

import numpy as np
import unittest

from skyllh.core.detsigyield import (
    DetSigYield,
    DetSigYieldBuilder,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterModelMapper,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.weights import (
    SourceDetectorWeights,
)
from skyllh.physics.flux_model import (
    SteadyPointlikeFFM,
)
from skyllh.physics.source_model import (
    PointLikeSource,
)

# Define a DetSigYield class that is a simple function of the source declination
# position.
class SimpleDetSigYieldWithoutGrads(DetSigYield):
    def __init__(self, **kwargs):
        pass

    def sources_to_recarray(self, sources):
        recarr = np.empty((len(sources),), dtype=[('dec', np.double),])
        for (i, src) in enumerate(sources):
            recarr[i]['dec'] = src.dec
        return recarr

    def __call__(self, src_recarray, src_params_recarray):
        """
        Parameters
        ----------
        src_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record array containing the information of the sources.
            The required fields of this record array are implementation
            dependent. In the most generic case for a point-like source, it
            must contain the following three fields: ra, dec.
        src_params_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record ndarray containing the parameter values of the
            sources. The parameter values can be different for the different
            sources.
            The record array needs to contain two fields for each source
            parameter, one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        detsigyield : (N_sources,)-shaped 1D ndarray of float
            The array with the mean number of signal in the detector for each
            given source.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter. The key is the global fit parameter index and the value
            is the (N_sources,)-shaped numpy ndarray holding the gradient value
            dY_k/dp_s.
        """
        detsigyield = np.rad2deg(src_recarray['dec'])
        grads = dict()

        return (detsigyield, grads)

class SimpleDetSigYieldWithGrads(SimpleDetSigYieldWithoutGrads):
    def __init__(self, pname, **kwargs):
        self.param_names = (pname,)

    def __call__(self, src_recarray, src_params_recarray):
        """
        Parameters
        ----------
        src_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record array containing the information of the sources.
            The required fields of this record array are implementation
            dependent. In the most generic case for a point-like source, it
            must contain the following three fields: ra, dec.
        src_params_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record ndarray containing the parameter values of the
            sources. The parameter values can be different for the different
            sources.
            The record array needs to contain two fields for each source
            parameter, one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        values : (N_sources,)-shaped 1D ndarray of float
            The array with the mean number of signal in the detector for each
            given source.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter. The key is the global fit parameter index and the value
            is the (N_sources,)-shaped numpy ndarray holding the gradient value
            dY_k/dp_s.
        """
        local_param_name = self.param_names[0]

        src_dec = np.atleast_1d(src_recarray['dec'])
        src_param = src_params_recarray[local_param_name]
        src_param_gp_idxs = src_params_recarray[f'{local_param_name}:gpidx']

        n_sources = len(src_dec)

        # Check for correct input format.
        if not (len(src_param) == n_sources and
                len(src_param_gp_idxs) == n_sources):
            raise RuntimeError(
                f'The length ({len(src_param)}) of the array for the '
                f'source parameter "{local_param_name}" does not match the '
                f'number of sources ({n_sources})!')

        values = np.rad2deg(src_dec)

        # Determine the number of global parameters the local parameter is
        # made of.
        gfp_idxs = np.unique(src_param_gp_idxs)
        gfp_idxs = gfp_idxs[gfp_idxs > 0] - 1

        grads = dict()
        for gfp_idx in gfp_idxs:
            grads[gfp_idx] = np.zeros((n_sources,), dtype=np.double)

            # Create a mask to select the sources that depend on the global
            # fit parameter with index gfp_idx.
            m = (src_param_gp_idxs == gfp_idx+1)

            grads[gfp_idx][m] = src_param[m]

        return (values, grads)

# Define placeholder class to satisfy type checks.
class NoDetSigYieldBuilder(DetSigYieldBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def construct_detsigyield(self, **kwargs):
        pass


class TestSourceDetectorWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This class method will run only once for this TestCase.
        """
        sources = [
            PointLikeSource(
                name='PS1', ra=0, dec=np.deg2rad(10), weight=1),
            PointLikeSource(
                name='PS2', ra=0, dec=np.deg2rad(20), weight=2),
            PointLikeSource(
                name='PS3', ra=0, dec=np.deg2rad(30), weight=3),
        ]
        fluxmodel = SteadyPointlikeFFM(Phi0=1, energy_profile=None)

        cls._shg_mgr = SourceHypoGroupManager(
            SourceHypoGroup(
                sources=sources,
                fluxmodel=fluxmodel,
                detsigyield_builders=NoDetSigYieldBuilder(),
                sig_gen_method=None))

        p1 = Parameter('p1', 141, 99, 199)
        p2 = Parameter('p2', 142, 100, 200)

        pmm = ParameterModelMapper(models=sources)
        pmm.def_param(p1).def_param(p2)
        cls._pmm = pmm

    def test_without_grads(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance without any gradients.
        """
        # detsigyields is (N_datasets, N_shgs)-shaped.
        detsigyields = [
            [SimpleDetSigYieldWithoutGrads()],
            [SimpleDetSigYieldWithoutGrads()]
        ]

        src_det_weights = SourceDetectorWeights(
            shg_mgr=type(self)._shg_mgr,
            pmm=type(self)._pmm,
            detsigyields=detsigyields)

        gflp_values = np.array([120.0, 177.7])
        (a_jk, a_jk_grads) = src_det_weights(gflp_values)

        self.assertEqual(
            a_jk.shape, (2,3),
            'a_jk.shape')

        self.assertEqual(
            a_jk_grads.shape, (2,2,3),
            'a_jk_grads.shape')

        self.assertTrue(
            np.all(np.isclose(a_jk, np.array(
                [[1*10,2*20,3*30],
                 [1*10,2*20,3*30]
                ]))),
            'a_jk values')

        self.assertTrue(
            np.all(np.isclose(a_jk_grads, np.zeros((2,2,3)))),
            'a_jk_grads values')

    def test_with_grads_p1(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance with gradients for the parameter p1.
        """
        # detsigyields is (N_datasets, N_shgs)-shaped.
        detsigyields = [
            [SimpleDetSigYieldWithGrads(pname='p1')],
            [SimpleDetSigYieldWithGrads(pname='p1')]
        ]

        src_det_weights = SourceDetectorWeights(
            shg_mgr=type(self)._shg_mgr,
            pmm=type(self)._pmm,
            detsigyields=detsigyields)

        gflp_values = np.array([120.0, 177.7])
        (a_jk, a_jk_grads) = src_det_weights(gflp_values)

        self.assertEqual(
            a_jk.shape, (2,3),
            'a_jk.shape')

        self.assertEqual(
            a_jk_grads.shape, (2,2,3),
            'a_jk_grads.shape')

        self.assertTrue(
            np.all(np.isclose(a_jk, np.array(
                [[1*10,2*20,3*30],
                 [1*10,2*20,3*30]
                ]))),
            'a_jk values')

        self.assertTrue(
            np.all(np.isclose(a_jk_grads, np.array(
                [[[1*120., 2*120., 3*120.],
                  [1*120., 2*120., 3*120.]],
                 [[    0.,     0.,     0.],
                  [    0.,     0.,     0.]]
                ]))),
            'a_jk_grads values')

    def test_with_grads_p2(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance with gradients for the parameter p2.
        """
        # detsigyields is (N_datasets, N_shgs)-shaped.
        detsigyields = [
            [SimpleDetSigYieldWithGrads(pname='p2')],
            [SimpleDetSigYieldWithGrads(pname='p2')]
        ]

        src_det_weights = SourceDetectorWeights(
            shg_mgr=type(self)._shg_mgr,
            pmm=type(self)._pmm,
            detsigyields=detsigyields)

        gflp_values = np.array([120., 177.])
        (a_jk, a_jk_grads) = src_det_weights(gflp_values)

        self.assertEqual(
            a_jk.shape, (2,3),
            'a_jk.shape')

        self.assertEqual(
            a_jk_grads.shape, (2,2,3),
            'a_jk_grads.shape')

        self.assertTrue(
            np.all(np.isclose(a_jk, np.array(
                [[1*10,2*20,3*30],
                 [1*10,2*20,3*30]
                ]))),
            'a_jk values')

        self.assertTrue(
            np.all(np.isclose(a_jk_grads, np.array(
                [[[  0.,   0.,   0.],
                  [  0.,   0.,   0.]],
                 [[1*177., 2*177., 3*177.],
                  [1*177., 2*177., 3*177.]]
                ]))),
            'a_jk_grads values')

if(__name__ == '__main__'):
    unittest.main()
