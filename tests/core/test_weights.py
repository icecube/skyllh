# -*- coding: utf-8 -*-

"""The unit tests in this module test classes of the skyllh.core.weights module.
"""

import unittest
from unittest.mock import (
    Mock,
)

import numpy as np

from skyllh.core.detsigyield import (
    DetSigYield,
    DetSigYieldBuilder,
)
from skyllh.core.flux_model import (
    SteadyPointlikeFFM,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterModelMapper,
)
from skyllh.core.services import (
    DatasetSignalWeightFactorsService,
    DetSigYieldService,
    SrcDetSigYieldWeightsService,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    PointLikeSource,
)


# Define a DetSigYield class that is a simple function of the source declination
# position.
class SimpleDetSigYieldWithoutGrads(DetSigYield):
    def __init__(self, scale=1, **kwargs):
        self._scale = scale

    def sources_to_recarray(self, sources):
        recarr = np.empty((len(sources),), dtype=[("dec", np.double)])
        for i, src in enumerate(sources):
            recarr[i]["dec"] = src.dec
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
        values : (N_sources,)-shaped 1D ndarray of float
            The array with the mean number of signal in the detector for each
            given source.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter. The key is the global fit parameter index and the value
            is the (N_sources,)-shaped numpy ndarray holding the gradient value
            dY_k/dp_s.
        """
        values = self._scale * np.rad2deg(src_recarray["dec"])
        grads = dict()

        return (values, grads)


class SimpleDetSigYieldWithGrads(SimpleDetSigYieldWithoutGrads):
    def __init__(self, pname, **kwargs):
        super().__init__(**kwargs)

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

        src_dec = np.atleast_1d(src_recarray["dec"])
        src_param = src_params_recarray[local_param_name]
        src_param_gp_idxs = src_params_recarray[f"{local_param_name}:gpidx"]

        n_sources = len(src_dec)

        # Check for correct input format.
        if not (
            len(src_param) == n_sources and len(src_param_gp_idxs) == n_sources
        ):
            raise RuntimeError(
                f"The length ({len(src_param)}) of the array for the "
                f'source parameter "{local_param_name}" does not match the '
                f"number of sources ({n_sources})!"
            )

        values = self._scale * np.rad2deg(src_dec)

        # Determine the number of global parameters the local parameter is
        # made of.
        gfp_idxs = np.unique(src_param_gp_idxs)
        gfp_idxs = gfp_idxs[gfp_idxs > 0] - 1

        grads = dict()
        for gfp_idx in gfp_idxs:
            grads[gfp_idx] = np.zeros((n_sources,), dtype=np.double)

            # Create a mask to select the sources that depend on the global
            # fit parameter with index gfp_idx.
            m = src_param_gp_idxs == gfp_idx + 1

            grads[gfp_idx][m] = src_param[m]

        return (values, grads)


# Define placeholder class to satisfy type checks.
class NoDetSigYieldBuilder(DetSigYieldBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct_detsigyield(self, **kwargs):
        pass


def create_shg_mgr_and_pmm():
    """Creates a SourceHypoGroupManager and a ParameterModelMapper instance."""
    sources = [
        PointLikeSource(name="PS1", ra=0, dec=np.deg2rad(10), weight=1),
        PointLikeSource(name="PS2", ra=0, dec=np.deg2rad(20), weight=2),
        PointLikeSource(name="PS3", ra=0, dec=np.deg2rad(30), weight=3),
    ]
    fluxmodel = SteadyPointlikeFFM(Phi0=1, energy_profile=None)

    shg_mgr = SourceHypoGroupManager(
        SourceHypoGroup(
            sources=sources,
            fluxmodel=fluxmodel,
            detsigyield_builders=NoDetSigYieldBuilder(),
            sig_gen_method=None,
        )
    )

    p1 = Parameter("p1", 141, 99, 199)
    p2 = Parameter("p2", 142, 100, 200)

    pmm = ParameterModelMapper(models=sources)
    pmm.def_param(p1).def_param(p2)

    return (shg_mgr, pmm)


def create_DetSigYieldService(shg_mgr, detsigyield_arr):
    """Creates a Mock instance mimicing a DetSigYieldService instance with a
    given detsigyield array.
    """
    detsigyield_service = Mock(
        spec_set=[
            "__class__",
            "arr",
            "shg_mgr",
            "n_datasets",
            "n_shgs",
        ]
    )

    detsigyield_service.__class__ = DetSigYieldService
    detsigyield_service.arr = detsigyield_arr
    detsigyield_service.shg_mgr = shg_mgr
    detsigyield_service.n_datasets = detsigyield_arr.shape[0]
    detsigyield_service.n_shgs = detsigyield_arr.shape[1]

    return detsigyield_service


class TestSourceDetectorWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This class method will run only once for this TestCase."""
        (cls._shg_mgr, cls._pmm) = create_shg_mgr_and_pmm()

    def test_without_grads(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance without any gradients.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithoutGrads()],
                [SimpleDetSigYieldWithoutGrads()],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        gflp_values = np.array([120.0, 177.7])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        (a_jk, a_jk_grads) = src_detsigyield_weights_service.get_weights()

        self.assertIsInstance(a_jk, np.ndarray, "instance of a_jk")

        self.assertEqual(a_jk.shape, (2, 3), "a_jk.shape")

        np.testing.assert_allclose(
            a_jk,
            np.array([[1 * 10, 2 * 20, 3 * 30], [1 * 10, 2 * 20, 3 * 30]]),
            err_msg="a_jk values",
        )

        self.assertIsInstance(a_jk_grads, dict, "instance of a_jk_grads")

        self.assertEqual(len(a_jk_grads), 0, "length of a_jk_grads")

    def test_with_grads_p1(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance with gradients for the parameter p1.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithGrads(pname="p1")],
                [SimpleDetSigYieldWithGrads(pname="p1")],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        gflp_values = np.array([120.0, 177.7])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        (a_jk, a_jk_grads) = src_detsigyield_weights_service.get_weights()

        self.assertIsInstance(a_jk, np.ndarray, "instance of a_jk")

        self.assertEqual(a_jk.shape, (2, 3), "a_jk.shape")

        np.testing.assert_allclose(
            a_jk,
            [[1 * 10, 2 * 20, 3 * 30], [1 * 10, 2 * 20, 3 * 30]],
            err_msg="a_jk values",
        )

        self.assertIsInstance(a_jk_grads, dict, "instance of a_jk_grads")

        self.assertEqual(len(a_jk_grads), 1, "length of a_jk_grads")

        self.assertIn(0, a_jk_grads, "0 in a_jk_grads")

        np.testing.assert_allclose(
            a_jk_grads[0],
            [
                [1 * 120.0, 2 * 120.0, 3 * 120.0],
                [1 * 120.0, 2 * 120.0, 3 * 120.0],
            ],
            err_msg="a_jk_grads[0] values",
        )

    def test_with_grads_p2(self):
        """Tests the __call__ method of the SourceDetectorWeights class
        using a DetSigYield instance with gradients for the parameter p2.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithGrads(pname="p2")],
                [SimpleDetSigYieldWithGrads(pname="p2")],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        gflp_values = np.array([120.0, 177.0])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        (a_jk, a_jk_grads) = src_detsigyield_weights_service.get_weights()

        self.assertIsInstance(a_jk, np.ndarray, "instance of a_jk")

        self.assertEqual(a_jk.shape, (2, 3), "a_jk.shape")

        np.testing.assert_allclose(
            a_jk,
            [[1 * 10, 2 * 20, 3 * 30], [1 * 10, 2 * 20, 3 * 30]],
            err_msg="a_jk values",
        )

        self.assertIsInstance(a_jk_grads, dict, "instance of a_jk_grads")

        self.assertEqual(len(a_jk_grads), 1, "length of a_jk_grads")

        self.assertIn(1, a_jk_grads, "1 in a_jk_grads")

        np.testing.assert_allclose(
            a_jk_grads[1],
            [
                [1 * 177.0, 2 * 177.0, 3 * 177.0],
                [1 * 177.0, 2 * 177.0, 3 * 177.0],
            ],
            err_msg="a_jk_grads[1] values",
        )


class TestDatasetSignalWeightFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This class method will run only once for this TestCase."""
        (cls._shg_mgr, cls._pmm) = create_shg_mgr_and_pmm()

    def test_without_grads(self):
        """Tests the __call__ method of the DatasetSignalWeightFactors class
        using a DetSigYield instance without any gradients.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithoutGrads(scale=1)],
                [SimpleDetSigYieldWithoutGrads(scale=2)],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        ds_sig_weight_factors_service = DatasetSignalWeightFactorsService(
            src_detsigyield_weights_service=src_detsigyield_weights_service
        )

        gflp_values = np.array([120.0, 177.7])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        ds_sig_weight_factors_service.calculate()
        (f_j, f_j_grads) = ds_sig_weight_factors_service.get_weights()

        self.assertIsInstance(f_j, np.ndarray, "instance of f_j")

        self.assertEqual(f_j.shape, (2,), "f_j.shape")

        np.testing.assert_allclose(f_j, [1 / 3, 2 / 3], err_msg="f_j values")

        self.assertIsInstance(f_j_grads, dict, "instance of f_j_grads")

        self.assertEqual(len(f_j_grads), 0, "length of f_j_grads")

    def test_with_grads_p1(self):
        """Tests the __call__ method of the DatasetSignalWeightFactors class
        using a DetSigYield instance with gradients for the parameter p1.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithGrads(pname="p1", scale=1)],
                [SimpleDetSigYieldWithGrads(pname="p1", scale=2)],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        ds_sig_weight_factors_service = DatasetSignalWeightFactorsService(
            src_detsigyield_weights_service=src_detsigyield_weights_service
        )

        gflp_values = np.array([120.0, 177.7])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        ds_sig_weight_factors_service.calculate()
        (f_j, f_j_grads) = ds_sig_weight_factors_service.get_weights()

        self.assertIsInstance(f_j, np.ndarray, "instance of f_j")

        self.assertEqual(f_j.shape, (2,), "f_j.shape")

        np.testing.assert_allclose(f_j, [1 / 3, 2 / 3], err_msg="f_j values")

        self.assertIsInstance(f_j_grads, dict, "instance of f_j_grads")

        self.assertEqual(len(f_j_grads), 1, "length of f_j_grads")

        self.assertIn(0, f_j_grads, "0 in f_j_grads")

        np.testing.assert_allclose(
            f_j_grads[0],
            [0.57142857, -0.57142857],
            err_msg="f_j_grads[0] values",
        )

    def test_with_grads_p2(self):
        """Tests the __call__ method of the DatasetSignalWeightFactors class
        using a DetSigYield instance with gradients for the parameter p2.
        """
        # detsigyield_arr is (N_datasets, N_shgs)-shaped.
        detsigyield_arr = np.array(
            [
                [SimpleDetSigYieldWithGrads(pname="p2", scale=1)],
                [SimpleDetSigYieldWithGrads(pname="p2", scale=2)],
            ]
        )

        detsigyield_service = create_DetSigYieldService(
            shg_mgr=type(self)._shg_mgr, detsigyield_arr=detsigyield_arr
        )

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            detsigyield_service=detsigyield_service
        )

        ds_sig_weight_factors_service = DatasetSignalWeightFactorsService(
            src_detsigyield_weights_service=src_detsigyield_weights_service
        )

        gflp_values = np.array([120.0, 177.7])
        src_params_recarray = type(self)._pmm.create_src_params_recarray(
            gflp_values
        )
        src_detsigyield_weights_service.calculate(src_params_recarray)
        ds_sig_weight_factors_service.calculate()
        (f_j, f_j_grads) = ds_sig_weight_factors_service.get_weights()

        self.assertIsInstance(f_j, np.ndarray, "instance of f_j")

        self.assertEqual(f_j.shape, (2,), "f_j.shape")

        np.testing.assert_allclose(f_j, [1 / 3, 2 / 3], err_msg="f_j values")

        self.assertIsInstance(f_j_grads, dict, "instance of f_j_grads")

        self.assertEqual(len(f_j_grads), 1, "length of f_j_grads")

        self.assertIn(1, f_j_grads, "1 in f_j_grads")

        np.testing.assert_allclose(
            f_j_grads[1],
            [0.84619048, -0.84619048],
            err_msg="f_j_grads[1] values",
        )


if __name__ == "__main__":
    unittest.main()
