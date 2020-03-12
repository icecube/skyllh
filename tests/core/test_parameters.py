# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the core/parameters
module.
"""

import numpy as np
import os.path
import sys
import unittest

from skyllh.core.binning import BinningDefinition
from skyllh.core.model import Model
from skyllh.core.parameters import (
    HypoParameterDefinition,
    Parameter,
    ParameterSet,
    ParameterSetArray,
    ParameterGrid,
    ParameterGridSet,
    SingleModelParameterMapper,
    MultiModelParameterMapper
)
from skyllh.core.py import const
from skyllh.core.random import RandomStateService

sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
from utils import isAlmostEqual

GAMMA_GRID = [
    1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.
]

ECUT_GRID = [
    9., 9.1
]


class Parameter_TestCase(unittest.TestCase):
    """This test case tests the Parameter class.
    """
    def setUp(self):
        self.fixed_param_initial = 2.37
        self.floating_param_initial = 7.32
        self.floating_param_valmin = 7.1
        self.floating_param_valmax = 8
        self.floating_param_grid = np.array([
            7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.])

        self.fixed_param = Parameter(
            'fixed_param', self.fixed_param_initial)
        self.floating_param = Parameter(
            'floating_param', self.floating_param_initial,
            valmin=self.floating_param_valmin,
            valmax=self.floating_param_valmax)

    def test__eq__(self):
        fixed_param = Parameter(
            'fixed_param', self.fixed_param_initial)
        floating_param = Parameter(
            'floating_param', self.floating_param_initial,
            valmin=self.floating_param_valmin,
            valmax=self.floating_param_valmax)

        self.assertTrue(self.fixed_param == fixed_param)
        self.assertTrue(self.floating_param == floating_param)

        # Change the parameter name.
        fixed_param = Parameter(
            'fixed_param1', self.fixed_param_initial)
        self.assertFalse(self.fixed_param == fixed_param)

        # Change the initial value.
        fixed_param = Parameter(
            'fixed_param', self.fixed_param_initial+1)
        self.assertFalse(self.fixed_param == fixed_param)

        floating_param = Parameter(
            'floating_param', self.floating_param_initial+0.2,
            valmin=self.floating_param_valmin,
            valmax=self.floating_param_valmax)
        self.assertFalse(self.floating_param == floating_param)

        # Change the valmin.
        floating_param = Parameter(
            'floating_param', self.floating_param_initial,
            valmin=self.floating_param_valmin-1,
            valmax=self.floating_param_valmax)
        self.assertFalse(self.floating_param == floating_param)

        # Change the valmax.
        floating_param = Parameter(
            'floating_param', self.floating_param_initial,
            valmin=self.floating_param_valmin,
            valmax=self.floating_param_valmax+1)
        self.assertFalse(self.floating_param == floating_param)

    def test_name(self):
        self.assertEqual(self.fixed_param.name, 'fixed_param')
        self.assertEqual(self.floating_param.name, 'floating_param')

    def test_initial(self):
        self.assertTrue(isAlmostEqual(
            self.fixed_param.initial, self.fixed_param_initial))
        self.assertTrue(isAlmostEqual(
            self.floating_param.initial, self.floating_param_initial))

    def test_isfixed(self):
        self.assertTrue(self.fixed_param.isfixed)
        self.assertFalse(self.floating_param.isfixed)

    def test_valmin(self):
        self.assertEqual(self.fixed_param.valmin, None)
        self.assertTrue(isAlmostEqual(
            self.floating_param.valmin, self.floating_param_valmin))

    def test_valmax(self):
        self.assertEqual(self.fixed_param.valmax, None)
        self.assertTrue(isAlmostEqual(
            self.floating_param.valmax, self.floating_param_valmax))

    def test_value(self):
        self.assertTrue(isAlmostEqual(
            self.fixed_param.value, self.fixed_param_initial))
        self.assertTrue(isAlmostEqual(
            self.floating_param.value, self.floating_param_initial))

        # Try to change the value of a fixed parameter.
        with self.assertRaises(ValueError):
            self.fixed_param.value = self.floating_param_initial

        # Try to set the value of a floating parameter to a value outside its
        # value range.
        with self.assertRaises(ValueError):
            self.floating_param.value = self.fixed_param_initial

    def test_str(self):
        try:
            str(self.fixed_param)
            str(self.floating_param)
        except:
            self.fail('The __str__ method raised an exception!')

    def test_as_linear_grid(self):
        grid_delta = 0.1
        with self.assertRaises(ValueError):
            self.fixed_param.as_linear_grid(grid_delta)

        param_grid = self.floating_param.as_linear_grid(grid_delta)
        self.assertTrue(np.all(isAlmostEqual(
            param_grid.grid, self.floating_param_grid)))

    def test_change_fixed_value(self):
        with self.assertRaises(ValueError):
            self.floating_param.change_fixed_value(self.fixed_param_initial)

        self.fixed_param.change_fixed_value(self.floating_param_initial)
        self.assertTrue(isAlmostEqual(
            self.fixed_param.initial, self.floating_param_initial))
        self.assertTrue(isAlmostEqual(
            self.fixed_param.value, self.floating_param_initial))

    def test_make_fixed(self):
        self.floating_param.make_fixed(self.fixed_param_initial)
        self.assertTrue(isAlmostEqual(
            self.floating_param.initial, self.fixed_param_initial))
        self.assertTrue(isAlmostEqual(
            self.floating_param.value, self.fixed_param_initial))

    def test_make_floating(self):
        with self.assertRaises(ValueError):
            self.fixed_param.make_floating()
        with self.assertRaises(ValueError):
            self.fixed_param.make_floating(valmin=self.floating_param_valmin)

        # The current value of fixed_param is outside the valmin and valmax
        # range of the floating_param. This should raise an exception when no
        # new initial value is specified.
        with self.assertRaises(ValueError):
            self.fixed_param.make_floating(
                valmin=self.floating_param_valmin,
                valmax=self.floating_param_valmax)

        self.fixed_param.make_floating(
            initial=self.floating_param_initial,
            valmin=self.floating_param_valmin,
            valmax=self.floating_param_valmax)
        self.assertTrue(isAlmostEqual(
            self.fixed_param.initial, self.floating_param_initial))
        self.assertTrue(isAlmostEqual(
            self.fixed_param.value, self.floating_param_initial))
        self.assertTrue(isAlmostEqual(
            self.fixed_param.valmin, self.floating_param_valmin))
        self.assertTrue(isAlmostEqual(
            self.fixed_param.valmax, self.floating_param_valmax))


class ParameterSet_TestCase(unittest.TestCase):
    """This test case tests the ParameterSet class.
    """
    def setUp(self):
        self.fixed_param = Parameter('p0', 2.3)
        self.floating_param = Parameter('p1', 1.1, valmin=0.5, valmax=1.6)
        self.paramset = ParameterSet((self.fixed_param, self.floating_param))

    def test_union(self):
        p0 = Parameter('p0', 2.3)
        p1 = Parameter('p1', 1.1, valmin=0.5, valmax=1.6)
        p2 = Parameter('p2', 3.2, valmin=2.3, valmax=4.7)
        paramset0 = ParameterSet((p0,p2))
        paramset1 = ParameterSet((p1,p2))
        paramset_union = ParameterSet.union(paramset0, paramset1)
        params = paramset_union.params
        self.assertEqual(len(params), 3)
        self.assertEqual(params[0], p0)
        self.assertEqual(params[1], p2)
        self.assertEqual(params[2], p1)

    def test_params(self, paramset=None):
        if(paramset is None):
            paramset = self.paramset
        params = self.paramset.params
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0], self.fixed_param)
        self.assertEqual(params[1], self.floating_param)

    def test_fixed_params(self):
        params = self.paramset.fixed_params
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0], self.fixed_param)

    def test_fixed_params_mask(self):
        mask = self.paramset.fixed_params_mask
        self.assertTrue(len(mask), 2)
        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], False)

    def test_floating_params(self):
        params = self.paramset.floating_params
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0], self.floating_param)

    def test_floating_params_mask(self):
        mask = self.paramset.floating_params_mask
        self.assertTrue(len(mask), 2)
        self.assertEqual(mask[0], False)
        self.assertEqual(mask[1], True)

    def test_n_params(self):
        self.assertEqual(self.paramset.n_params, 2)

    def test_n_fixed_params(self):
        self.assertEqual(self.paramset.n_fixed_params, 1)

    def test_n_floating_params(self):
        self.assertEqual(self.paramset.n_floating_params, 1)

    def test_fixed_param_name_list(self):
        names = self.paramset.fixed_param_name_list
        self.assertEqual(len(names), 1)
        self.assertEqual(names, ['p0'])

    def test_floating_param_name_list(self):
        names = self.paramset.floating_param_name_list
        self.assertEqual(len(names), 1)
        self.assertEqual(names, ['p1'])

    def test_fixed_param_values(self):
        values = self.paramset.fixed_param_values
        self.assertTrue(isAlmostEqual(values, [2.3]))

    def test_floating_param_initials(self):
        initials = self.paramset.floating_param_initials
        self.assertTrue(isAlmostEqual(initials, [1.1]))

    def test_floating_param_bounds(self):
        bounds = self.paramset.floating_param_bounds
        self.assertTrue(isAlmostEqual(bounds[0], [0.5, 1.6]))

    def test_len(self):
        self.assertEqual(len(self.paramset), 2)

    def test_iter(self):
        for (i,param) in enumerate(self.paramset):
            self.assertEqual(param.name, 'p%d'%(i))

    def test_str(self):
        try:
            str(self.paramset)
        except:
            self.fail('The __str__ method raised exception!')

    def test_get_fixed_pidx(self):
        self.assertEqual(self.paramset.get_fixed_pidx('p0'), 0)

        with self.assertRaises(KeyError):
            self.paramset.get_fixed_pidx('p1')

    def test_get_floating_pidx(self):
        self.assertEqual(self.paramset.get_floating_pidx('p1'), 0)

        with self.assertRaises(KeyError):
            self.paramset.get_floating_pidx('p0')

    def test_has_fixed_param(self):
        self.assertTrue(self.paramset.has_fixed_param('p0'))
        self.assertFalse(self.paramset.has_fixed_param('p1'))

    def test_has_floating_param(self):
        self.assertTrue(self.paramset.has_floating_param('p1'))
        self.assertFalse(self.paramset.has_floating_param('p0'))

    def test_make_params_fixed(self):
        # Already fixed parameters cannot be fixed.
        with self.assertRaises(ValueError):
            self.paramset.make_params_fixed({'p0': 42})

        # Fix the floating parameter outside its current range.
        self.paramset.make_params_fixed({'p1': 0.4})
        self.assertTrue(self.paramset.has_fixed_param('p1'))
        self.assertEqual(self.paramset.n_fixed_params, 2)
        self.assertEqual(self.paramset.n_floating_params, 0)
        self.test_params()
        values = self.paramset.fixed_param_values
        self.assertTrue(isAlmostEqual(values, [2.3, 0.4]))
        self.assertEqual(self.paramset.params[1].valmin, None)
        self.assertEqual(self.paramset.params[1].valmax, None)

        self.setUp()

        # Fix the floating parameter to its current value.
        self.paramset.make_params_fixed({'p1': None})
        self.assertTrue(self.paramset.has_fixed_param('p1'))
        self.assertEqual(self.paramset.n_fixed_params, 2)
        self.assertEqual(self.paramset.n_floating_params, 0)
        self.test_params()
        values = self.paramset.fixed_param_values
        self.assertTrue(isAlmostEqual(values, [2.3, 1.1]))
        self.assertTrue(isAlmostEqual(self.paramset.params[1].valmin, 0.5))
        self.assertTrue(isAlmostEqual(self.paramset.params[1].valmax, 1.6))

    def test_make_params_floating(self):
        # Already floating parameters cannot be made floating.
        with self.assertRaises(ValueError):
            self.paramset.make_params_floating({'p1': None})

        # Make the fixed parameter floating.
        with self.assertRaises(ValueError):
            self.paramset.make_params_floating({'p0': None})
        with self.assertRaises(ValueError):
            self.paramset.make_params_floating({'p0': 1.2})
        self.paramset.make_params_floating({'p0': (1.2, 1.0, 1.3)})
        self.assertTrue(self.paramset.has_floating_param('p0'))
        self.assertTrue(isAlmostEqual(self.paramset.params[0].initial, 1.2))
        self.assertTrue(isAlmostEqual(self.paramset.params[0].valmin, 1.0))
        self.assertTrue(isAlmostEqual(self.paramset.params[0].valmax, 1.3))

    def test_update_fixed_param_value_cache(self):
        self.assertAlmostEqual(self.paramset.params[0].value, 2.3)
        self.fixed_param.change_fixed_value(3.1)
        self.assertAlmostEqual(self.paramset.params[0].value, 3.1)
        self.assertTrue(isAlmostEqual(self.paramset.fixed_param_values, [2.3]))
        self.paramset.update_fixed_param_value_cache()
        self.assertTrue(isAlmostEqual(self.paramset.fixed_param_values, [3.1]))

    def test_copy(self):
        new_paramset = self.paramset.copy()
        self.assertFalse(new_paramset == self.paramset)
        self.test_params(new_paramset)

    def test_add_param(self):
        with self.assertRaises(TypeError):
            self.paramset.add_param('p2')
        with self.assertRaises(KeyError):
            param = Parameter('p0', 42.)
            self.paramset.add_param(param)

        # Add parameter at front.
        param = Parameter('p2', 42.)
        self.paramset.add_param(param, atfront=True)
        self.assertEqual(self.paramset.params[0], param)
        self.assertEqual(self.paramset.params[1], self.fixed_param)
        self.assertEqual(self.paramset.params[2], self.floating_param)

        self.setUp()

        # Add parameter at end.
        param = Parameter('p2', 42.)
        self.paramset.add_param(param)
        self.assertEqual(self.paramset.params[0], self.fixed_param)
        self.assertEqual(self.paramset.params[1], self.floating_param)
        self.assertEqual(self.paramset.params[2], param)

    def test_has_param(self):
        self.assertTrue(self.paramset.has_param(self.fixed_param))
        self.assertTrue(self.paramset.has_param(self.floating_param))
        self.assertFalse(self.paramset.has_param(Parameter('p', 0.0)))

    def test_floating_param_values_to_dict(self):
        param_dict = self.paramset.floating_param_values_to_dict(np.array([1.3]))
        self.assertAlmostEqual(param_dict['p0'], 2.3)
        self.assertAlmostEqual(param_dict['p1'], 1.3)


class ParameterSetArray_TestCase(unittest.TestCase):
    """This test case tests the ParameterSetArray class.
    """
    def setUp(self):
        self.fixed_param0 = Parameter(
            'fixed_param0', 2.3)
        self.fixed_param1 = Parameter(
            'fixed_param1', 0.1)
        self.floating_param0 = Parameter(
            'floating_param0', 1.1, valmin=0.5, valmax=1.6)
        self.floating_param1 = Parameter(
            'floating_param1', 13.5, valmin=10.5, valmax=15)
        self.paramset0 = ParameterSet((self.fixed_param0, self.floating_param0))
        self.paramset1 = ParameterSet((self.floating_param1, self.fixed_param1))

        self.paramsetarr = ParameterSetArray(
            (const(self.paramset0), const(self.paramset1)))

    def test__init__(self):
        paramset0 = ParameterSet((self.fixed_param0, self.floating_param0))
        paramset1 = ParameterSet((self.floating_param1, self.fixed_param1))
        with self.assertRaises(TypeError):
            paramsetarr = ParameterSetArray(
                (paramset0, paramset1))

    def test_paramset_list(self):
        paramset_list = self.paramsetarr.paramset_list
        self.assertEqual(len(paramset_list), 2)
        self.assertEqual(paramset_list[0], self.paramset0)
        self.assertEqual(paramset_list[1], self.paramset1)

    def test_n_params(self):
        self.assertEqual(self.paramsetarr.n_params, 4)

    def test_n_fixed_params(self):
        self.assertEqual(self.paramsetarr.n_fixed_params, 2)

    def test_n_floating_params(self):
        self.assertEqual(self.paramsetarr.n_floating_params, 2)

    def test_floating_param_initials(self):
        initials = self.paramsetarr.floating_param_initials
        self.assertEqual(initials.shape, (2,))
        self.assertAlmostEqual(initials[0], 1.1)
        self.assertAlmostEqual(initials[1], 13.5)

    def test_floating_param_bounds(self):
        bounds = self.paramsetarr.floating_param_bounds
        self.assertEqual(bounds.shape, (2,2))
        np.testing.assert_almost_equal(bounds[0], (0.5, 1.6))
        np.testing.assert_almost_equal(bounds[1], (10.5, 15))

    def test__str__(self):
        try:
            str(self.paramsetarr)
        except:
            self.fail('The __str__ method raised exception!')

    def test_generate_random_initials(self):
        rss_ref = RandomStateService(42)
        rn = rss_ref.random.uniform(size=2)

        rss = RandomStateService(42)
        initials = self.paramsetarr.generate_random_initials(rss)
        np.testing.assert_almost_equal(initials,
            (0.5+rn[0]*(1.6-0.5),
            10.5+rn[1]*(15-10.5)))

    def test_split_floating_param_values(self):
        fl_param_values = np.array([0.9, 14.2])
        fl_param_values_list = self.paramsetarr.split_floating_param_values(
            fl_param_values)
        self.assertEqual(len(fl_param_values_list), 2)
        self.assertEqual(len(fl_param_values_list[0]), 1)
        self.assertEqual(len(fl_param_values_list[1]), 1)
        np.testing.assert_almost_equal(fl_param_values_list[0], [0.9])
        np.testing.assert_almost_equal(fl_param_values_list[1], [14.2])


class ParameterGrid_TestCase(unittest.TestCase):
    """This test case tests the ParameterGrid class.
    """
    def setUp(self):
        self.paramgrid_gamma1 = ParameterGrid('gamma1', [ 1.5, 2., 2.5, 3., 3.5])
        self.paramgrid_gamma2 = ParameterGrid('gamma2', GAMMA_GRID)
        self.paramgrid_gamma3 = ParameterGrid('gamma3', [ 1.05, 1.15, 1.25, 1.35])

    def test_from_BinningDefinition(self):
        binning = BinningDefinition(name='gamma', binedges=GAMMA_GRID)
        param_grid = ParameterGrid.from_BinningDefinition(binning)

        self.assertEqual(param_grid.name, binning.name)
        self.assertTrue(isAlmostEqual(param_grid.grid, GAMMA_GRID))

    def test_delta(self):
        self.assertTrue(isAlmostEqual(self.paramgrid_gamma1.delta, 0.5))
        self.assertTrue(isAlmostEqual(self.paramgrid_gamma2.delta, 0.1))
        self.assertTrue(isAlmostEqual(self.paramgrid_gamma3.delta, 0.1))

    def test_decimals(self):
        self.assertTrue(self.paramgrid_gamma1.decimals >= 1)
        self.assertTrue(self.paramgrid_gamma2.decimals >= 1)
        self.assertTrue(self.paramgrid_gamma3.decimals >= 2)

    def test_round_to_nearest_grid_point(self):
        # Test values outside the grid range.
        x = 1.49999999999
        with self.assertRaises(ValueError):
            gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)
        x = 3.50000000001
        with self.assertRaises(ValueError):
            gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)

        # Test a value between two grid points.
        x = [2.1, 2.4, 2.2, 2.3]
        gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [2.0, 2.5, 2., 2.5])

        x = [1.051, 1.14]
        gp = self.paramgrid_gamma3.round_to_nearest_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, [1.05, 1.15]))

        # Test a value on a grid point.
        x = [1.05, 1.35]
        gp = self.paramgrid_gamma3.round_to_nearest_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, [1.05, 1.35]))

    def test_round_to_lower_grid_point(self):
        # Test a value between two grid points.
        x = 2.4
        gp = self.paramgrid_gamma1.round_to_lower_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 2.))

        # Test a value at a grid point.
        x = 2.
        gp = self.paramgrid_gamma1.round_to_lower_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 2.))

        x = 1.6
        gp = self.paramgrid_gamma2.round_to_lower_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 1.6))

        x = [1.05, 1.15, 1.25, 1.35]
        gp = self.paramgrid_gamma3.round_to_lower_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.05, 1.15, 1.25, 1.35])

    def test_round_to_upper_grid_point(self):
        # Test a value between two grid points.
        x = 2.4
        gp = self.paramgrid_gamma1.round_to_upper_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 2.5))

        # Test a value at a grid point.
        x = 2.
        gp = self.paramgrid_gamma1.round_to_upper_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 2.5))

        x = 1.6
        gp = self.paramgrid_gamma2.round_to_upper_grid_point(x)
        self.assertTrue(isAlmostEqual(gp, 1.7))

        x = [1.05, 1.15, 1.25, 1.35]
        gp = self.paramgrid_gamma3.round_to_upper_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.15, 1.25, 1.35, 1.45])


class ParameterGridSet_TestCase(unittest.TestCase):
    """This test case tests the ParameterGridSet class.
    """
    def setUp(self):
        """Setups this test case.
        """
        self.paramgrid_gamma = ParameterGrid('gamma', GAMMA_GRID)
        self.paramgrid_Ecut = ParameterGrid('Ecut', ECUT_GRID)

        self.paramgridset = ParameterGridSet(
            (self.paramgrid_gamma, self.paramgrid_Ecut))

    def test_ndim(self):
        self.assertEqual(
            self.paramgridset.ndim, 2)

    def test_parameter_names(self):
        self.assertEqual(
            self.paramgridset.parameter_names, ['gamma', 'Ecut'])

    def test_parameter_permutation_dict_list(self):
        perm_dict_list = self.paramgridset.parameter_permutation_dict_list

        self.assertTrue(isAlmostEqual(
            [ d['gamma'] for d in perm_dict_list ],
            np.repeat(np.array(GAMMA_GRID), len(ECUT_GRID))
        ))
        self.assertTrue(isAlmostEqual(
            [ d['Ecut'] for d in perm_dict_list ],
            list(ECUT_GRID)*len(GAMMA_GRID)
        ))

    def test_index(self):
        self.assertEqual(
            self.paramgridset.index(self.paramgrid_gamma), 0)

        self.assertEqual(
            self.paramgridset.index(self.paramgrid_Ecut), 1)

    def test_index_by_name(self):
        self.assertEqual(
            self.paramgridset.index_by_name('gamma'), 0)

        self.assertEqual(
            self.paramgridset.index_by_name('Ecut'), 1)

    def test_pop_and_add(self):
        paramgrid_gamma = self.paramgridset.pop('gamma')
        self.assertEqual(paramgrid_gamma.name, 'gamma')

        paramgrid_Ecut = self.paramgridset.pop()
        self.assertEqual(paramgrid_Ecut.name, 'Ecut')

        self.paramgridset.add(paramgrid_gamma)
        self.paramgridset.add(paramgrid_Ecut)

        # The altered ParameterGridSet instance should be the same as the
        # initial ParameterGridSet instance. So just run all the tests on that
        # altered one.
        self.test_ndim()
        self.test_parameter_names()
        self.test_parameter_permutation_dict_list()
        self.test_index()
        self.test_index_by_name()


class SingleModelParameterMapperTestCase(unittest.TestCase):
    def setUp(self):
        self.fixed_param = Parameter('p1', 42)
        self.floating_param = Parameter('p2', 4, 1, 6)
        self.model = Model('the_src')
        self.mpm = SingleModelParameterMapper(
            'signal', self.model)

    def test_name(self):
        self.assertEqual(self.mpm.name, 'signal')

    def test_models(self):
        self.assertEqual(len(self.mpm.models), 1)
        self.assertEqual(self.mpm.models[0], self.model)

    def test_n_models(self):
        self.assertEqual(self.mpm.n_models, 1)

    def test_str(self):
        # Add some parameters to the model parameter mapper.
        self.test_def_param()
        try:
            str(self.mpm)
        except:
            self.fail('The __str__ method raised an exception!')

    def test_def_param(self):
        self.mpm.def_param(self.fixed_param)
        self.mpm.def_param(self.floating_param, 'fp')
        self.assertEqual(self.mpm.n_global_params, 2)
        self.assertEqual(self.mpm.n_global_fixed_params, 1)
        self.assertEqual(self.mpm.n_global_floating_params, 1)

    def test_get_model_param_dict(self):
        # Add some parameters to the model parameter mapper.
        self.test_def_param()
        model_param_dict = self.mpm.get_model_param_dict(np.array([2.4]))
        self.assertEqual(len(model_param_dict), 2)
        self.assertTrue('p1' in model_param_dict)
        self.assertTrue('fp' in model_param_dict)
        self.assertAlmostEqual(model_param_dict['p1'], 42)
        self.assertAlmostEqual(model_param_dict['fp'], 2.4)


class MultiModelParameterMapperTestCase(unittest.TestCase):
    def setUp(self):
        self.fixed_param0 = Parameter('p0', 42)
        self.floating_param0 = Parameter('p1', 4, 1, 6)
        self.floating_param1 = Parameter('p2', 13, 10, 15)
        self.model0 = Model('m0')
        self.model1 = Model('m1')
        self.mpm = MultiModelParameterMapper(
            'signal', (self.model0, self.model1))

    def test_name(self):
        self.assertEqual(self.mpm.name, 'signal')

    def test_models(self):
        self.assertEqual(len(self.mpm.models), 2)
        self.assertEqual(self.mpm.models[0], self.model0)
        self.assertEqual(self.mpm.models[1], self.model1)

    def test_n_models(self):
        self.assertEqual(self.mpm.n_models, 2)

    def test_str(self):
        # Add some parameters.
        self.test_def_param()
        try:
            str(self.mpm)
        except:
            self.fail('The __str__ method raised an exception!')

    def test_def_param(self):
        self.mpm.def_param(self.fixed_param0, models=(self.model1,))
        self.mpm.def_param(self.floating_param0, 'fp', models=(self.model0,self.model1))
        self.mpm.def_param(self.floating_param1, models=(self.model1))
        self.assertEqual(self.mpm.n_global_params, 3)
        self.assertEqual(self.mpm.n_global_fixed_params, 1)
        self.assertEqual(self.mpm.n_global_floating_params, 2)

        # The models cannot be an empty set.
        with self.assertRaises(ValueError):
            self.mpm.def_param(self.fixed_param0, 'fp', models=())
        # A model parameter can only be defined once for a given model.
        with self.assertRaises(KeyError):
            self.mpm.def_param(self.fixed_param0, 'fp', models=(self.model0))

    def test_get_model_param_dict(self):
        # Add some parameters to the model parameter mapper.
        self.test_def_param()

        m0_param_dict = self.mpm.get_model_param_dict(
            np.array([2.4, 11.1]), model_idx=0)
        self.assertEqual(len(m0_param_dict), 1)
        self.assertTrue('fp' in m0_param_dict)
        self.assertAlmostEqual(m0_param_dict['fp'], 2.4)

        m1_param_dict = self.mpm.get_model_param_dict(
            np.array([2.4, 11.1]), model_idx=1)
        self.assertEqual(len(m1_param_dict), 3)
        self.assertTrue('p0' in m1_param_dict)
        self.assertTrue('fp' in m1_param_dict)
        self.assertTrue('p2' in m1_param_dict)
        self.assertAlmostEqual(m1_param_dict['fp'], 2.4)
        self.assertAlmostEqual(m1_param_dict['p2'], 11.1)
        self.assertAlmostEqual(m1_param_dict['p0'], 42)


class HypoParameterDefinitionTestCase(unittest.TestCase):
    def setUp(self):
        self.fixed_param0 = Parameter('fixed_param0', 42)
        self.fixed_param1 = Parameter('fixed_param1', 11)
        self.floating_param0 = Parameter('floating_param0', 4, 1, 6)
        self.floating_param1 = Parameter('floating_param1', 0.3, 0.1, 1)

        self.model0 = Model('m0')
        self.model1 = Model('m1')

        self.mpm0 = SingleModelParameterMapper(
            'mpm0', self.model0)
        self.mpm0.def_param(self.fixed_param0)
        self.mpm0.def_param(self.floating_param0, 'fp0')

        self.mpm1 = SingleModelParameterMapper(
            'mpm1', self.model1)
        self.mpm1.def_param(self.fixed_param1)
        self.mpm1.def_param(self.floating_param1, 'fp1')

        self.hpdef = HypoParameterDefinition((self.mpm0, self.mpm1))

    def test_model_param_mapper_list(self):
        mpm_list = self.hpdef.model_param_mapper_list
        self.assertEqual(len(mpm_list), 2)
        self.assertEqual(mpm_list[0], self.mpm0)
        self.assertEqual(mpm_list[1], self.mpm1)

    def test__str__(self):
        try:
            str(self.hpdef)
        except:
            self.fail('The __str__ method raised an exception!')

    def test_copy(self):
        # Check to raise on bad input type.
        with self.assertRaises(TypeError):
            hpdef = self.hpdef.copy('not of type dict')

        # Create an exact copy.
        hpdef = self.hpdef.copy()
        self.assertTrue(isinstance(hpdef, HypoParameterDefinition))
        mpm0 = hpdef['mpm0']
        mpm1 = hpdef['mpm1']
        self.assertTrue((mpm0 is not self.mpm0) and (mpm0 is not self.mpm1))
        self.assertTrue((mpm1 is not self.mpm0) and (mpm1 is not self.mpm1))

        self.assertEqual(mpm0.global_paramset.n_params, 2)
        self.assertEqual(mpm0.global_paramset.n_fixed_params, 1)
        self.assertEqual(mpm0.global_paramset.n_floating_params, 1)
        self.assertTrue(mpm0.global_paramset.params[0] == self.fixed_param0)
        self.assertTrue(mpm0.global_paramset.params[1] == self.floating_param0)

        self.assertEqual(mpm1.global_paramset.n_params, 2)
        self.assertEqual(mpm1.global_paramset.n_fixed_params, 1)
        self.assertEqual(mpm1.global_paramset.n_floating_params, 1)
        self.assertTrue(mpm1.global_paramset.params[0] == self.fixed_param1)
        self.assertTrue(mpm1.global_paramset.params[1] == self.floating_param1)

        self.setUp()

        # Create a copy where we fix one of the floating parameters.
        hpdef = self.hpdef.copy({'floating_param1':2.3})
        self.assertTrue(isinstance(hpdef, HypoParameterDefinition))
        mpm0 = hpdef['mpm0']
        mpm1 = hpdef['mpm1']
        self.assertTrue((mpm0 is not self.mpm0) and (mpm0 is not self.mpm1))
        self.assertTrue((mpm1 is not self.mpm0) and (mpm1 is not self.mpm1))

        self.assertEqual(mpm0.global_paramset.n_params, 2)
        self.assertEqual(mpm0.global_paramset.n_fixed_params, 1)
        self.assertEqual(mpm0.global_paramset.n_floating_params, 1)
        self.assertTrue(mpm0.global_paramset.params[0] == self.fixed_param0)
        self.assertTrue(mpm0.global_paramset.params[1] == self.floating_param0)

        self.assertEqual(mpm1.global_paramset.n_params, 2)
        self.assertEqual(mpm1.global_paramset.n_fixed_params, 2)
        self.assertEqual(mpm1.global_paramset.n_floating_params, 0)
        self.assertTrue(mpm1.global_paramset.params[0] == self.fixed_param1)
        self.assertFalse(mpm1.global_paramset.params[1] == self.floating_param1)
        self.assertTrue(mpm1.global_paramset.params[1].isfixed)
        self.assertAlmostEqual(mpm1.global_paramset.params[1].initial, 2.3)
        self.assertAlmostEqual(mpm1.global_paramset.params[1].value, 2.3)

    def test_create_ParameterSetArray(self):
        paramsetarr = self.hpdef.create_ParameterSetArray()
        self.assertTrue(isinstance(paramsetarr, ParameterSetArray))


class TestParameters(unittest.TestCase):
    def test_MultiSourceFitParameterMapper(self):
        from skyllh.physics.source import PointLikeSource
        from skyllh.core.parameters import (
            FitParameter,
            MultiSourceFitParameterMapper
        )

        # Define a list of point-like sources.
        sources = [
            PointLikeSource(np.deg2rad(120), np.deg2rad(-23)),
            PointLikeSource(np.deg2rad(266), np.deg2rad(61)),
        ]

        # Define the fit parameters 'gamma1' and 'gamma2' which map to the
        # 'gamma' source parameter of the first and second source, respectively.
        sfpm = MultiSourceFitParameterMapper(sources)
        sfpm.def_fit_parameter(FitParameter('gamma1', 1, 4, 2.0), 'gamma', sources[0])
        sfpm.def_fit_parameter(FitParameter('gamma2', 1, 4, 2.1), 'gamma', sources[1])

        # Check the initial values.
        self.assertTrue(np.all(sfpm.fitparamset.initials == np.array([2.0, 2.1])))

        # Get the source parameters for the first source (feed it with the
        # initials).
        fitparams = sfpm.get_src_fitparams(sfpm.fitparamset.initials, 0)
        self.assertEqual(fitparams, {'gamma': 2.0})

        # Get the source parameters for the second source (feed it with the
        # initials).
        fitparams = sfpm.get_src_fitparams(sfpm.fitparamset.initials, 1)
        self.assertEqual(fitparams, {'gamma': 2.1})

if(__name__ == '__main__'):
    unittest.main()
