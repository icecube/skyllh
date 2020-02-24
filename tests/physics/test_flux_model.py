# -*- coding: utf-8 -*-

from __future__ import division

import unittest
import numpy as np

from astropy import units

from skyllh.physics.flux_model import (
    UnitySpatialFluxProfile,
    PointSpatialFluxProfile,
    EnergyFluxProfile,
    UnityEnergyFluxProfile,
    PowerLawEnergyFluxProfile,
    TimeFluxProfile,
    UnityTimeFluxProfile,
    BoxTimeFluxProfile,
    FactorizedFluxModel,
    PointlikeSourceFFM,
    SteadyPointlikeSourceFFM
)

from skyllh.core.py import float_cast


class UnitySpatialFluxProfileTestCase(unittest.TestCase):
    def setUp(self):
        self.fluxprofile = UnitySpatialFluxProfile()

    def test_math_function_str(self):
        try:
            math_func_str = self.fluxprofile.math_function_str
        except:
            self.fail('The math_function_str property raised an exception!')

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta),
                                      np.array([1]))

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta),
                                      np.array([1, 1]))

class PointSpatialFluxProfileTestCase(unittest.TestCase):
    def setUp(self):
        self.alpha_s = 1.5
        self.delta_s = 2.5
        self.fluxprofile = PointSpatialFluxProfile(
            self.alpha_s, self.delta_s, angle_unit=units.radian)

    def test_init(self):
        self.assertEqual(self.fluxprofile.alpha_s, self.alpha_s)
        self.assertEqual(self.fluxprofile.delta_s, self.delta_s)

    def test_param_names(self):
        param_names = self.fluxprofile.param_names
        print(param_names)
        self.assertEqual(len(param_names), 2)
        self.assertEqual(param_names[0], 'alpha_s')
        self.assertEqual(param_names[1], 'delta_s')

    def test_angle_unit(self):
        self.assertEqual(self.fluxprofile.angle_unit, units.radian)

    def test_alpha_s(self):
        alpha_s = 5.5
        self.fluxprofile.alpha_s = alpha_s
        self.assertAlmostEqual(self.fluxprofile.alpha_s, alpha_s)

    def test_delta_s(self):
        delta_s = -1.4
        self.fluxprofile.delta_s = delta_s
        self.assertAlmostEqual(self.fluxprofile.delta_s, delta_s)

    def test_math_function_str(self):
        try:
            math_func_str = self.fluxprofile.math_function_str
        except:
            self.fail('The math_function_str property raised an exception!')

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta),
                                      np.array([1]))

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta),
                                      np.array([1, 0]))

@unittest.skip("Skip")
class TestFactorizedFluxModel(unittest.TestCase):
    def setUp(self):
        self.flux_model = FactorizedFluxModel()

    def test_init(self):
        self.assertEqual(self.flux_model.angle_unit, units.radian)
        self.assertEqual(self.flux_model.energy_unit, units.GeV)
        self.assertEqual(self.flux_model.length_unit, units.cm)
        self.assertEqual(self.flux_model.time_unit, units.s)

    def test_angle_unit(self):
        self.flux_model.angle_unit = units.deg

        self.assertEqual(self.flux_model.angle_unit, units.deg)

    def test_energy_unit(self):
        self.flux_model.energy_unit = units.eV

        self.assertEqual(self.flux_model.angle_unit, units.eV)

    def test_length_unit(self):
        self.flux_model.length_unit = units.m

        self.assertEqual(self.flux_model.length_unit, units.m)

    def test_time_unit(self):
        self.flux_model.time_unit = units.min

        self.assertEqual(self.flux_model.time_unit, units.min)

    def test_unit_str(self):
        print(self.flux_model.unit_str)


if(__name__ == '__main__'):
    unittest.main()
