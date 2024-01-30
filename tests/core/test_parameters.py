# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the core/parameters
module.
"""

import numpy as np
import unittest

from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.model import (
    Model,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterGrid,
    ParameterGridSet,
    ParameterModelMapper,
    ParameterSet,
)


GAMMA_GRID = [
    1.,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.
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
        np.testing.assert_almost_equal(
            self.fixed_param.initial, self.fixed_param_initial)
        np.testing.assert_almost_equal(
            self.floating_param.initial, self.floating_param_initial)

    def test_isfixed(self):
        self.assertTrue(self.fixed_param.isfixed)
        self.assertFalse(self.floating_param.isfixed)

    def test_valmin(self):
        self.assertEqual(self.fixed_param.valmin, None)
        np.testing.assert_almost_equal(
            self.floating_param.valmin, self.floating_param_valmin)

    def test_valmax(self):
        self.assertEqual(self.fixed_param.valmax, None)
        np.testing.assert_almost_equal(
            self.floating_param.valmax, self.floating_param_valmax)

    def test_value(self):
        np.testing.assert_almost_equal(
            self.fixed_param.value, self.fixed_param_initial)
        np.testing.assert_almost_equal(
            self.floating_param.value, self.floating_param_initial)

        # Try to change the value of a fixed parameter.
        with self.assertRaises(ValueError):
            self.fixed_param.value = self.floating_param_initial

        # Try to set the value of a floating parameter to a value outside its
        # value range.
        with self.assertRaises(ValueError):
            self.floating_param.value = self.fixed_param_initial

    def test_str(self):
        # Make sure the __str__ methods don't raise exceptions.
        str(self.fixed_param)
        str(self.floating_param)

    def test_as_linear_grid(self):
        grid_delta = 0.1
        param_grid_fixed = self.fixed_param.as_linear_grid(grid_delta)
        np.testing.assert_array_almost_equal(
            param_grid_fixed.grid, np.array([self.fixed_param_initial])) 
        param_grid = self.floating_param.as_linear_grid(grid_delta)
        np.testing.assert_almost_equal(
            param_grid.grid, self.floating_param_grid)
        
    def test_change_fixed_value(self):
        with self.assertRaises(ValueError):
            self.floating_param.change_fixed_value(self.fixed_param_initial)

        self.fixed_param.change_fixed_value(self.floating_param_initial)
        np.testing.assert_almost_equal(
            self.fixed_param.initial, self.floating_param_initial)
        np.testing.assert_almost_equal(
            self.fixed_param.value, self.floating_param_initial)

    def test_make_fixed(self):
        self.floating_param.make_fixed(self.fixed_param_initial)
        np.testing.assert_almost_equal(
            self.floating_param.initial, self.fixed_param_initial)
        np.testing.assert_almost_equal(
            self.floating_param.value, self.fixed_param_initial)

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
        np.testing.assert_almost_equal(
            self.fixed_param.initial, self.floating_param_initial)
        np.testing.assert_almost_equal(
            self.fixed_param.value, self.floating_param_initial)
        np.testing.assert_almost_equal(
            self.fixed_param.valmin, self.floating_param_valmin)
        np.testing.assert_almost_equal(
            self.fixed_param.valmax, self.floating_param_valmax)


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
        paramset0 = ParameterSet((p0, p2))
        paramset1 = ParameterSet((p1, p2))
        paramset_union = ParameterSet.union(paramset0, paramset1)
        params = paramset_union.params
        self.assertEqual(len(params), 3)
        self.assertEqual(params[0], p0)
        self.assertEqual(params[1], p2)
        self.assertEqual(params[2], p1)

    def test_params(self, paramset=None):
        if paramset is None:
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

    def test_fixed_params_idxs(self):
        idxs = self.paramset.fixed_params_idxs
        self.assertEqual(len(idxs), 1)
        np.testing.assert_equal(idxs, [0])

    def test_fixed_params_name_list(self):
        names = self.paramset.fixed_params_name_list
        self.assertEqual(len(names), 1)
        self.assertEqual(names, ['p0'])

    def test_floating_params_name_list(self):
        names = self.paramset.floating_params_name_list
        self.assertEqual(len(names), 1)
        self.assertEqual(names, ['p1'])

    def test_floating_params_idxs(self):
        idxs = self.paramset.floating_params_idxs
        self.assertEqual(len(idxs), 1)
        np.testing.assert_equal(idxs, [1])

    def test_fixed_param_values(self):
        values = self.paramset.fixed_param_values
        np.testing.assert_almost_equal(values, [2.3])

    def test_floating_param_initials(self):
        initials = self.paramset.floating_param_initials
        np.testing.assert_almost_equal(initials, [1.1])

    def test_floating_param_bounds(self):
        bounds = self.paramset.floating_param_bounds
        np.testing.assert_almost_equal(bounds[0], [0.5, 1.6])

    def test_len(self):
        self.assertEqual(len(self.paramset), 2)

    def test_iter(self):
        for (i, param) in enumerate(self.paramset):
            self.assertEqual(param.name, f'p{i}')

    def test_str(self):
        # Ensure that __str__ method does not raise an exception.
        str(self.paramset)

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
        np.testing.assert_almost_equal(values, [2.3, 0.4])
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
        np.testing.assert_almost_equal(values, [2.3, 1.1])
        np.testing.assert_almost_equal(self.paramset.params[1].valmin, 0.5)
        np.testing.assert_almost_equal(self.paramset.params[1].valmax, 1.6)

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
        np.testing.assert_almost_equal(self.paramset.params[0].initial, 1.2)
        np.testing.assert_almost_equal(self.paramset.params[0].valmin, 1.0)
        np.testing.assert_almost_equal(self.paramset.params[0].valmax, 1.3)

    def test_update_fixed_param_value_cache(self):
        self.assertAlmostEqual(self.paramset.params[0].value, 2.3)
        self.fixed_param.change_fixed_value(3.1)
        self.assertAlmostEqual(self.paramset.params[0].value, 3.1)
        np.testing.assert_almost_equal(self.paramset.fixed_param_values, [2.3])
        self.paramset.update_fixed_param_value_cache()
        np.testing.assert_almost_equal(self.paramset.fixed_param_values, [3.1])

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
        param_dict = self.paramset.get_floating_params_dict(np.array([1.3]))
        self.assertTrue(len(param_dict), 1)
        self.assertAlmostEqual(param_dict['p1'], 1.3)


class ParameterGrid_TestCase(unittest.TestCase):
    """This test case tests the ParameterGrid class.
    """
    def setUp(self):
        self.paramgrid_gamma1 = ParameterGrid('gamma1', [1.5, 2., 2.5, 3., 3.5])
        self.paramgrid_gamma2 = ParameterGrid('gamma2', GAMMA_GRID)
        self.paramgrid_gamma3 = ParameterGrid('gamma3', [1.05, 1.15, 1.25, 1.35])

    def test_from_BinningDefinition(self):
        binning = BinningDefinition(name='gamma', binedges=GAMMA_GRID)
        param_grid = ParameterGrid.from_BinningDefinition(binning)

        self.assertEqual(param_grid.name, binning.name)
        np.testing.assert_almost_equal(param_grid.grid, GAMMA_GRID)

    def test_delta(self):
        np.testing.assert_almost_equal(self.paramgrid_gamma1.delta, 0.5)
        np.testing.assert_almost_equal(self.paramgrid_gamma2.delta, 0.1)
        np.testing.assert_almost_equal(self.paramgrid_gamma3.delta, 0.1)

    def test_decimals(self):
        self.assertTrue(self.paramgrid_gamma1.decimals >= 1)
        self.assertTrue(self.paramgrid_gamma2.decimals >= 1)
        self.assertTrue(self.paramgrid_gamma3.decimals >= 2)

    def test_round_to_nearest_grid_point(self):
        # Test values outside the grid range.
        x = 1.49999999999
        gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.5])

        x = 3.50000000001
        gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [3.5])

        # Test a value between two grid points.
        x = [2.1, 2.4, 2.2, 2.3]
        gp = self.paramgrid_gamma1.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [2.0, 2.5, 2., 2.5])

        x = [1.051, 1.14]
        gp = self.paramgrid_gamma3.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.05, 1.15])

        # Test a value on a grid point.
        x = [1.05, 1.35]
        gp = self.paramgrid_gamma3.round_to_nearest_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.05, 1.35])

    def test_round_to_lower_grid_point(self):
        # Test a value between two grid points.
        x = 2.4
        gp = self.paramgrid_gamma1.round_to_lower_grid_point(x)
        np.testing.assert_almost_equal(gp, 2.)

        # Test a value at a grid point.
        x = 2.
        gp = self.paramgrid_gamma1.round_to_lower_grid_point(x)
        np.testing.assert_almost_equal(gp, 2.)

        x = 1.6
        gp = self.paramgrid_gamma2.round_to_lower_grid_point(x)
        np.testing.assert_almost_equal(gp, 1.6)

        x = [1.05, 1.15, 1.25, 1.35]
        gp = self.paramgrid_gamma3.round_to_lower_grid_point(x)
        np.testing.assert_almost_equal(gp, [1.05, 1.15, 1.25, 1.35])

    def test_round_to_upper_grid_point(self):
        # Test a value between two grid points.
        x = 2.4
        gp = self.paramgrid_gamma1.round_to_upper_grid_point(x)
        np.testing.assert_almost_equal(gp, 2.5)

        # Test a value at a grid point.
        x = 2.
        gp = self.paramgrid_gamma1.round_to_upper_grid_point(x)
        np.testing.assert_almost_equal(gp, 2.5)

        x = 1.6
        gp = self.paramgrid_gamma2.round_to_upper_grid_point(x)
        np.testing.assert_almost_equal(gp, 1.7)

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

    def test_param_names(self):
        self.assertEqual(
            self.paramgridset.params_name_list, ['gamma', 'Ecut'])

    def test_parameter_permutation_dict_list(self):
        perm_dict_list = self.paramgridset.parameter_permutation_dict_list

        np.testing.assert_almost_equal(
            [d['gamma'] for d in perm_dict_list],
            np.repeat(np.array(GAMMA_GRID), len(ECUT_GRID))
        )
        np.testing.assert_almost_equal(
            [d['Ecut'] for d in perm_dict_list],
            list(ECUT_GRID)*len(GAMMA_GRID)
        )


class ParameterModelMapperTestCase(unittest.TestCase):
    def setUp(self):
        self.fixed_param0 = Parameter('p0', 42)
        self.floating_param0 = Parameter('p1', 4, 1, 6)
        self.floating_param1 = Parameter('p2', 13, 10, 15)
        self.model0 = Model('m0')
        self.model1 = Model('m1')
        self.pmm = ParameterModelMapper(
            models=(self.model0, self.model1))

    def test_models(self):
        self.assertEqual(len(self.pmm.models), 2)
        self.assertEqual(self.pmm.models[0], self.model0)
        self.assertEqual(self.pmm.models[1], self.model1)

    def test_n_models(self):
        self.assertEqual(self.pmm.n_models, 2)

    def test_str(self):
        # Add some parameters.
        self.test_map_param()

        # Ensure that __str__ does not raise an exception.
        str(self.pmm)

    def test_unique_model_param_names(self):
        self.pmm.map_param(
            param=self.fixed_param0,
            models=(self.model0,),
            model_param_names='p')
        self.pmm.map_param(
            param=self.floating_param0,
            models=(self.model1,),
            model_param_names='p')
        self.pmm.map_param(
            param=self.floating_param1)
        names = self.pmm.unique_model_param_names
        self.assertEqual(len(names), 2)
        np.testing.assert_equal(names, ['p', 'p2'])

    def test_map_param(self):
        self.pmm.map_param(
            param=self.fixed_param0,
            models=(self.model1,))
        self.pmm.map_param(
            param=self.floating_param0,
            models=(self.model0, self.model1),
            model_param_names='fp')
        self.pmm.map_param(
            param=self.floating_param1,
            models=(self.model1,))
        self.assertEqual(self.pmm.n_global_params, 3)
        self.assertEqual(self.pmm.n_global_fixed_params, 1)
        self.assertEqual(self.pmm.n_global_floating_params, 2)

        # The models cannot be an empty set.
        with self.assertRaises(ValueError):
            self.pmm.map_param(
                param=self.fixed_param0,
                models=(),
                model_param_names='fp')
        # A model parameter can only be defined once for a given model.
        with self.assertRaises(KeyError):
            self.pmm.map_param(
                param=self.fixed_param0,
                models=(self.model0,),
                model_param_names='fp')

    def test_create_model_params_dict(self):
        # Add some parameters to the model parameter mapper.
        self.test_map_param()

        m0_param_dict = self.pmm.create_model_params_dict(
            np.array([2.4, 11.1]), model=0)
        self.assertEqual(len(m0_param_dict), 1)
        self.assertTrue('fp' in m0_param_dict)
        self.assertAlmostEqual(m0_param_dict['fp'], 2.4)

        m1_param_dict = self.pmm.create_model_params_dict(
            np.array([2.4, 11.1]), model=1)
        self.assertEqual(len(m1_param_dict), 3)
        self.assertTrue('p0' in m1_param_dict)
        self.assertTrue('fp' in m1_param_dict)
        self.assertTrue('p2' in m1_param_dict)
        self.assertAlmostEqual(m1_param_dict['fp'], 2.4)
        self.assertAlmostEqual(m1_param_dict['p2'], 11.1)
        self.assertAlmostEqual(m1_param_dict['p0'], 42)

    def test_get_local_param_is_global_floating_param_mask(self):
        # Add some parameters to the model parameter mapper.
        self.test_map_param()

        mask = self.pmm.get_local_param_is_global_floating_param_mask(
            ['p0', 'fp', 'p2'])
        np.testing.assert_equal(mask, [False, True, True])


if __name__ == '__main__':
    unittest.main()
