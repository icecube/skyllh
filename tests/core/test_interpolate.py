# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the
skyllh.core.interpolate module.
"""

import numpy as np

import unittest
from unittest.mock import Mock

from skyllh.core.interpolate import (
    Linear1DGridManifoldInterpolationMethod,
    NullGridManifoldInterpolationMethod,
    Parabola1DGridManifoldInterpolationMethod,
)
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)


def line_manifold_func(
        tdm,
        eventdata,
        gridparams_recarray,
        n_values):
    """This function will calculate the line value of f=2p+1 for the parameter
    p. The values will be the same for each event.
    """
    def line(m, p, b):
        return m*p + b

    # Check for special case, when only one set of parameters is provided for
    # all sources.
    if len(gridparams_recarray) == 1:
        gridparams_recarray = np.tile(gridparams_recarray, tdm.n_sources)

    p = gridparams_recarray['p']

    n_selected_events = eventdata.shape[0]

    values = np.repeat(line(m=2, p=p, b=1), n_selected_events)

    assert len(values) == len(gridparams_recarray)*n_selected_events

    return values


def param_product_func(
        tdm,
        eventdata,
        gridparams_recarray,
        n_values):
    """This function calculates the product of two parameter values p1 and p2.
    The result will be the same for each event.
    """
    def product(p1, p2):
        return p1 * p2

    # Check for special case, when only one set of parameters is provided for
    # all sources.
    if len(gridparams_recarray) == 1:
        gridparams_recarray = np.tile(gridparams_recarray, tdm.n_sources)

    p1 = gridparams_recarray['p1']
    p2 = gridparams_recarray['p2']

    n_selected_events = eventdata.shape[0]

    values = np.repeat(product(p1, p2), n_selected_events)

    assert len(values) == len(gridparams_recarray)*n_selected_events

    return values


def create_tdm(n_sources, n_selected_events):
    """Creates a Mock instance mimicing a TrialDataManager instance with a
    given number of sources and selected events.
    """
    tdm = Mock(spec_set=[
        '__class__',
        'trial_data_state_id',
        'get_n_values',
        'src_evt_idxs',
        'n_sources',
        'n_selected_events',
        'broadcast_params_recarray_to_values_array',
        'broadcast_arrays_to_values_array',
        'broadcast_sources_array_to_values_array',
        'broadcast_sources_arrays_to_values_arrays',
    ])

    def tdm_broadcast_params_recarray_to_values_array(params_recarray):
        return TrialDataManager.broadcast_params_recarray_to_values_array(
            tdm, params_recarray)

    def tdm_broadcast_arrays_to_values_array(arrays):
        return TrialDataManager.broadcast_arrays_to_values_array(
            tdm, arrays)

    def tdm_broadcast_sources_array_to_values_array(*args, **kwargs):
        return TrialDataManager.broadcast_sources_array_to_values_array(
            tdm, *args, **kwargs)

    def tdm_broadcast_sources_arrays_to_values_arrays(*args, **kwargs):
        return TrialDataManager.broadcast_sources_arrays_to_values_arrays(
            tdm, *args, **kwargs)

    tdm.__class__ = TrialDataManager
    tdm.trial_data_state_id = 1
    tdm.get_n_values = lambda: n_sources*n_selected_events
    tdm.src_evt_idxs = (
        np.repeat(np.arange(n_sources), n_selected_events),
        np.tile(np.arange(n_selected_events), n_sources)
    )
    tdm.n_sources = n_sources
    tdm.n_selected_events = n_selected_events
    tdm.broadcast_params_recarray_to_values_array =\
        tdm_broadcast_params_recarray_to_values_array
    tdm.broadcast_arrays_to_values_array =\
        tdm_broadcast_arrays_to_values_array
    tdm.broadcast_sources_array_to_values_array =\
        tdm_broadcast_sources_array_to_values_array
    tdm.broadcast_sources_arrays_to_values_arrays =\
        tdm_broadcast_sources_arrays_to_values_arrays

    return tdm


class NullGridManifoldInterpolationMethod_TestCase(unittest.TestCase):
    def setUp(self):
        param1_grid = ParameterGrid.from_range('p1', -3, 3, 0.1)
        param2_grid = ParameterGrid.from_range('p2', -1.5, 2.3, 0.1)

        self.interpolmethod = NullGridManifoldInterpolationMethod(
            func=param_product_func,
            param_grid_set=ParameterGridSet((param1_grid, param2_grid)))

        self.tdm = create_tdm(n_sources=3, n_selected_events=2)

        self.eventdata = np.zeros(
            (self.tdm.n_selected_events, 1), dtype=np.float64)

    def test__call__with_different_source_values(self):
        """Test for when the interpolation parameters have different values for
        different sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,),
            dtype=[('p1', np.float64), ('p2', np.float64)])
        params_recarray['p1'] = [-2.12, 1.36, 2.4]
        params_recarray['p2'] = [-1.06, 2.1, 1.33]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [2.31, 2.31, 2.94, 2.94, 3.12, 3.12])
        np.testing.assert_almost_equal(
            grads,
            [[0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.]])

    def test__call__with_same_source_values(self):
        """Test for when the interpolation parameter has the same values for all
        sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,),
            dtype=[('p1', np.float64), ('p2', np.float64)])
        params_recarray['p1'] = [2.12, 2.12, 2.12]
        params_recarray['p2'] = [-1.06, -1.06, -1.06]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [-2.31, -2.31, -2.31, -2.31, -2.31, -2.31])
        np.testing.assert_almost_equal(
            grads,
            [[0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.]])

    def test__call__with_single_value(self):
        """Test for when the interpolation parameters have the same values for
        all sources and is provided as a single set.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,),
            dtype=[('p1', np.float64), ('p2', np.float64)])
        params_recarray['p1'] = [2.12]
        params_recarray['p2'] = [-1.06]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [-2.31, -2.31, -2.31, -2.31, -2.31, -2.31])
        np.testing.assert_almost_equal(
            grads,
            [[0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.]])

    def test__call__with_grid_edge_values(self):
        """Test for when the interpolation parameters fall on the grid edges.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,),
            dtype=[('p1', np.float64), ('p2', np.float64)])
        params_recarray['p1'] = [-2.1, 1.4, 2.4]
        params_recarray['p2'] = [-1.1, 2.1, 1.3]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [2.31, 2.31, 2.94, 2.94, 3.12, 3.12])
        np.testing.assert_almost_equal(
            grads,
            [[0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.]])


class Linear1DGridManifoldInterpolationMethod_TestCase(unittest.TestCase):
    def setUp(self):
        param_grid = ParameterGrid.from_range('p', -3, 3, 0.1)

        self.interpolmethod = Linear1DGridManifoldInterpolationMethod(
            func=line_manifold_func,
            param_grid_set=ParameterGridSet((param_grid,)))

        self.tdm = create_tdm(n_sources=3, n_selected_events=2)

        self.eventdata = np.zeros(
            (self.tdm.n_selected_events, 1), dtype=np.float64)

    def test__call__with_different_source_values(self):
        """Test for when the interpolation parameter has different values for
        different sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [-2.12, 1.36, 2.4]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [-3.24, -3.24, 3.72, 3.72, 5.8, 5.8])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_same_source_values(self):
        """Test for when the interpolation parameter has the same values for all
        sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [1.36, 1.36, 1.36]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [3.72, 3.72, 3.72, 3.72, 3.72, 3.72])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_single_value(self):
        """Test for when the interpolation parameter has the same values for all
        sources and is provided as a single value.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [1.36]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [3.72, 3.72, 3.72, 3.72, 3.72, 3.72])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_grid_edge_values(self):
        """Test for when the interpolation parameters fall on the grid edges.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [-3., 0, 3]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)
        np.testing.assert_almost_equal(
            values,
            [-5., -5., 1., 1., 7., 7.])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])


class Parabola1DGridManifoldInterpolationMethod_TestCase(unittest.TestCase):
    def setUp(self):
        param_grid = ParameterGrid.from_range('p', -3, 3, 0.1)

        self.interpolmethod = Parabola1DGridManifoldInterpolationMethod(
            func=line_manifold_func,
            param_grid_set=ParameterGridSet((param_grid,)))

        self.tdm = create_tdm(n_sources=3, n_selected_events=2)

        self.eventdata = np.zeros(
            (self.tdm.n_selected_events, 1), dtype=np.float64)

    def test__call__with_different_source_values(self):
        """Test for when the interpolation parameter has different values for
        different sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [-2.12, 1.36, 2.4]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        # A parabola approximation of a line will be a line again.
        np.testing.assert_almost_equal(
            values,
            [-3.24, -3.24, 3.72, 3.72, 5.8, 5.8])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_same_source_values(self):
        """Test for when the interpolation parameter has the same values for all
        sources.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [1.36, 1.36, 1.36]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [3.72, 3.72, 3.72, 3.72, 3.72, 3.72])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_single_value(self):
        """Test for when the interpolation parameter has the same values for all
        sources and is provided as a single value.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [1.36]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)

        np.testing.assert_almost_equal(
            values,
            [3.72, 3.72, 3.72, 3.72, 3.72, 3.72])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])

    def test__call__with_grid_edge_values(self):
        """Test for when the interpolation parameters fall on the grid edges.
        """
        params_recarray = np.empty(
            (self.tdm.n_sources,), dtype=[('p', np.float64)])
        params_recarray['p'] = [-3., 0, 3]

        (values, grads) = self.interpolmethod(
            tdm=self.tdm,
            eventdata=self.eventdata,
            params_recarray=params_recarray)
        np.testing.assert_almost_equal(
            values,
            [-5., -5., 1., 1., 7., 7.])
        np.testing.assert_almost_equal(
            grads,
            [[2., 2., 2., 2., 2., 2.]])


if __name__ == '__main__':
    unittest.main()
