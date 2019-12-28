# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the core/parameters
module.
"""

import numpy as np
import sys
import unittest

from skyllh.core.binning import BinningDefinition
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet
)

sys.path.append('..')
from utils import isAlmostEqual

GAMMA_GRID = [
    1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.
]

ECUT_GRID = [
    9., 9.1
]


class ParameterGrid_TestCase(unittest.TestCase):
    """This test case tests the ParameterGrid class.
    """
    def test_from_BinningDefinition(self):
        binning = BinningDefinition(name='gamma', binedges=GAMMA_GRID)
        param_grid = ParameterGrid.from_BinningDefinition(binning)

        self.assertEqual(param_grid.name, binning.name)
        self.assertTrue(isAlmostEqual(param_grid.grid, GAMMA_GRID))


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
