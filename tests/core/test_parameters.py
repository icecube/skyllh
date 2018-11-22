# -*- coding: utf-8 -*-

import numpy as np
import unittest

def isAlmostEqual(a, b, decimals=9):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return np.all(np.around(np.abs(a - b), decimals) == 0)


class TestParameters(unittest.TestCase):
    def test_ParameterGridSet_parameter_permutation_dict_list(self):
        """Test the arameter_permutation_dict_list method of the
        ParameterGridSet class.
        """
        from skyllh.core.parameters import make_linear_parameter_grid_1d, ParameterGridSet

        GAMMA_GRID = [ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ]
        ECUT_GRID = [ 9., 9.1 ]

        paramgridset = ParameterGridSet()

        paramgrid = make_linear_parameter_grid_1d(name='gamma', low=1., high=2., delta=0.1)
        self.assertTrue(isAlmostEqual(paramgrid.grid, GAMMA_GRID))

        paramgridset += paramgrid

        perm_dict_list = paramgridset.parameter_permutation_dict_list
        self.assertTrue(isAlmostEqual([ d['gamma'] for d in perm_dict_list ], GAMMA_GRID))

        paramgrid = make_linear_parameter_grid_1d(name='Ecut', low=9., high=9.1, delta=0.1)
        self.assertTrue(isAlmostEqual(paramgrid.grid, ECUT_GRID))

        paramgridset += paramgrid

        perm_dict_list = paramgridset.parameter_permutation_dict_list
        self.assertTrue(isAlmostEqual([ d['Ecut'] for d in perm_dict_list ], ECUT_GRID*len(GAMMA_GRID)))

    def test_MultiSourceFitParameterMapper(self):
        from skyllh.core.random import RandomStateService
        from skyllh.physics.source import PointLikeSource
        from skyllh.core.parameters import MultiSourceFitParameterMapper, FitParameter

        # Define a random state service.
        rss = RandomStateService(seed=0)

        # Define a list of point-like sources.
        sources = [
            PointLikeSource(np.deg2rad(120), np.deg2rad(-23)),
            PointLikeSource(np.deg2rad(266), np.deg2rad(61)),
        ]

        # Define the fit parameters 'gamma1' and 'gamma2' which map to the
        # 'gamma' source parameter of the first and second source, respectively.
        sfpm = MultiSourceFitParameterMapper(rss, sources)
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
