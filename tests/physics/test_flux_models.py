# -*- coding: utf-8 -*-

from __future__ import division

import unittest
import numpy as np

from astropy import units

from skyllh.physics.flux_models import (
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


class TestUnitySpatialFluxProfile(unittest.TestCase):
    def setUp(self):
        self.instance = UnitySpatialFluxProfile()

    def test_math_function_str(self):
        self.assertEqual(self.instance.math_function_str(), '1')

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(self.instance(alpha, delta),
                                      np.array([1]))

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(self.instance(alpha, delta),
                                      np.array([1, 1]))

class TestPointSpatialFluxProfile(unittest.TestCase):
    def setUp(self):
        self.alpha_s = 1.5
        self.delta_s = 2.5
        self.instance = PointSpatialFluxProfile(self.alpha_s, self.delta_s)

    def test_init(self):
        self.assertEqual(self.instance.alpha_s, self.alpha_s)
        self.assertEqual(self.instance.delta_s, self.delta_s)

    def test_alpha_s(self):
        alpha_s_new = 5.5
        delta_s_new = 6.5
        self.instance.alpha_s = alpha_s_new
        self.instance.delta_s_new = delta_s_new

        self.assertEqual(self.instance.alpha_s, alpha_s_new)
        self.assertEqual(self.instance.delta_s, delta_s_new)

    def test_math_function_str(self):
        self.assertEqual(self.instance.math_function_str(),
                         'delta(alpha-%.2e)*delta(delta-%.2e)'%(self._alpha_s, self._delta_s))

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(self.instance(alpha, delta),
                                      np.array([1]))

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(self.instance(alpha, delta),
                                      np.array([1, 0]))

@unittest.skip("Skip")
class TestFactorizedFluxModel(unittest.TestCase):
    def setUp(self):
        self.flux_model = FluxModel()

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
