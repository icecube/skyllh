# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the
skyllh.core.interpolate module.
"""

import numpy as np

import unittest
from unittest.mock import Mock

from skyllh.core.interpolate import (
    Linear1DGridManifoldInterpolationMethod,
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

    n_selected_events = eventdata.shape[0]

    p = gridparams_recarray['p']

    values = np.repeat(line(m=2, p=p, b=1), n_selected_events)

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
        'n_selected_events'])
    tdm.__class__ = TrialDataManager
    tdm.trial_data_state_id = 1
    tdm.get_n_values = lambda: n_sources*n_selected_events
    tdm.src_evt_idxs = (
        np.repeat(np.arange(n_sources), n_selected_events),
        np.tile(np.arange(n_selected_events), n_sources)
    )
    tdm.n_sources = n_sources
    tdm.n_selected_events = n_selected_events

    return tdm


class Linear1DGridManifoldInterpolationMethod_TestCase(unittest.TestCase):
    def setUp(self):
        p_min = -3
        p_max = 3
        dp = 0.1
        p_grid = np.arange(p_min, p_max+dp, dp)

        param_grid = ParameterGrid(
            name='p',
            grid=p_grid)

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
        p_min = -3
        p_max = 3
        dp = 0.1
        p_grid = np.arange(p_min, p_max+dp, dp)

        param_grid = ParameterGrid(
            name='p',
            grid=p_grid)

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
