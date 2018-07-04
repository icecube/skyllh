# -*- coding: utf-8 -*-

import numpy as np
import unittest

from skylab.core.parameters import make_linear_parameter_grid_1d, ParameterGridSet

def isAlmostEqual(a, b, decimals=9):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return np.all(np.around(np.abs(a - b), decimals) == 0)

class TestParameters(unittest.TestCase):
    def test_ParameterGridSet_parameter_permutation_dict_list(self):
        """Test the arameter_permutation_dict_list method of the
        ParameterGridSet class.
        """
        GAMMA_GRID = [ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ]
        ECUT_GRID = [ 9., 9.1 ]

        paramgridset = ParameterGridSet()

        paramgrid = make_linear_parameter_grid_1d(name='gamma', low=1., high=2., delta=0.1)
        self.assertTrue(isAlmostEqual(paramgrid.grid, GAMMA_GRID))

        paramgridset += paramgrid

        perm_dict_list = paramgridset.parameter_permutation_dict_list
        self.assertTrue(isAlmostEqual( [ d['gamma'] for d in perm_dict_list ], GAMMA_GRID))

        paramgrid = make_linear_parameter_grid_1d(name='Ecut', low=9., high=9.1, delta=0.1)
        self.assertTrue(isAlmostEqual(paramgrid.grid, ECUT_GRID))

        paramgridset += paramgrid

        perm_dict_list = paramgridset.parameter_permutation_dict_list
        self.assertTrue(isAlmostEqual( [ d['Ecut'] for d in perm_dict_list ], ECUT_GRID*len(GAMMA_GRID)))



if(__name__ == '__main__'):
    unittest.main()
