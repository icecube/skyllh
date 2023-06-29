# -*- coding: utf-8 -*-

import unittest

import numpy as np
from astropy import (
    units,
)

from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    CutoffPowerLawEnergyFluxProfile,
    LogParabolaPowerLawEnergyFluxProfile,
    PointSpatialFluxProfile,
    PowerLawEnergyFluxProfile,
    UnitySpatialFluxProfile,
)


class UnitySpatialFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.fluxprofile = UnitySpatialFluxProfile()

    def test_math_function_str(self):
        self.assertEqual(self.fluxprofile.math_function_str, "1")

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(
            self.fluxprofile(alpha, delta), np.array([1])
        )

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(
            self.fluxprofile(alpha, delta), np.array([1, 1])
        )


class PointSpatialFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.ra = 1.5
        self.dec = 2.5
        self.angle_unit = units.radian
        self.fluxprofile = PointSpatialFluxProfile(
            self.ra, self.dec, angle_unit=units.radian
        )

    def test_init(self):
        self.assertEqual(self.fluxprofile.ra, self.ra)
        self.assertEqual(self.fluxprofile.dec, self.dec)

    def test_param_names(self):
        param_names = self.fluxprofile.param_names
        self.assertEqual(len(param_names), 2)
        self.assertEqual(param_names[0], "ra")
        self.assertEqual(param_names[1], "dec")

    def test_angle_unit(self):
        self.assertEqual(self.fluxprofile.angle_unit, units.radian)

    def test_ra(self):
        ra = 5.5
        self.fluxprofile.ra = ra
        self.assertAlmostEqual(self.fluxprofile.ra, ra)

    def test_dec(self):
        dec = -1.4
        self.fluxprofile.dec = dec
        self.assertAlmostEqual(self.fluxprofile.dec, dec)

    def test_math_function_str(self):
        self.assertEqual(
            self.fluxprofile.math_function_str,
            (
                f"delta(ra-{self.ra:g}{self.angle_unit})*"
                f"delta(dec-{self.dec:g}{self.angle_unit})"
            ),
        )

    def test_call(self):
        ra = 1.5
        dec = 2.5

        np.testing.assert_array_equal(self.fluxprofile(ra, dec), np.array([1]))

        ra = np.array([1.5, 2])
        dec = np.array([2.5, 3])

        np.testing.assert_array_equal(
            self.fluxprofile(ra, dec), np.array([1, 0])
        )


class PowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.E0 = 2.5
        self.gamma = 2.7
        self.energy_unit = units.GeV
        self.fluxprofile = PowerLawEnergyFluxProfile(
            E0=self.E0, gamma=self.gamma, energy_unit=self.energy_unit
        )

    def test_gamma(self):
        self.assertEqual(self.fluxprofile.gamma, self.gamma)

        gamma = 2.6
        self.fluxprofile.gamma = gamma
        self.assertEqual(self.fluxprofile.gamma, gamma)

    def test_math_function_str(self):
        self.assertEqual(
            self.fluxprofile.math_function_str,
            f"(E / ({self.E0:g} {self.energy_unit}))^-{self.gamma:g}",
        )

    def test_call(self):
        E = np.array([5])
        values = np.power(E / self.E0, -self.gamma)

        self.assertEqual(self.fluxprofile(E=E, unit=self.energy_unit), values)


class CutoffPowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.E0 = 2.5
        self.gamma = 2.7
        self.Ecut = 2
        self.energy_unit = units.GeV
        self.fluxprofile = CutoffPowerLawEnergyFluxProfile(
            E0=self.E0,
            gamma=self.gamma,
            Ecut=self.Ecut,
            energy_unit=self.energy_unit,
        )

    def test_Ecut(self):
        self.assertEqual(self.fluxprofile.Ecut, self.Ecut)

        Ecut = 2.5
        self.fluxprofile.Ecut = Ecut
        self.assertEqual(self.fluxprofile.Ecut, Ecut)

    def test_math_function_str(self):
        self.assertEqual(
            self.fluxprofile.math_function_str,
            (
                f"(E / ({self.E0:g} {self.energy_unit}))^-{self.gamma:g} "
                f"exp(-E / ({self.Ecut:g} {self.energy_unit}))"
            ),
        )

    def test_call(self):
        E = np.array([5])
        values = np.power(E / self.E0, -self.gamma) * np.exp(-E / self.Ecut)

        self.assertEqual(self.fluxprofile(E=E, unit=self.energy_unit), values)


class LogParabolaPowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.E0 = 2.5
        self.alpha = 1
        self.beta = 2
        self.energy_unit = units.GeV
        self.fluxprofile = LogParabolaPowerLawEnergyFluxProfile(
            E0=self.E0,
            alpha=self.alpha,
            beta=self.beta,
            energy_unit=self.energy_unit,
        )

    def test_alpha(self):
        self.assertEqual(self.fluxprofile.alpha, self.alpha)

        alpha = 1.5
        self.fluxprofile.alpha = alpha
        self.assertEqual(self.fluxprofile.alpha, alpha)

    def test_beta(self):
        self.assertEqual(self.fluxprofile.beta, self.beta)

        beta = 2.5
        self.fluxprofile.beta = beta
        self.assertEqual(self.fluxprofile.beta, beta)

    def test_math_funciton_str(self):
        s_E0 = f"{self.E0:g} {self.energy_unit}"
        test_string = (
            f"(E / {s_E0})"
            f"^(-({self.alpha:g} + {self.beta:g} log(E / {s_E0})))"
        )
        self.assertEqual(self.fluxprofile.math_function_str, test_string)


class BoxTimeFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.t0 = 58430  # MJD time 2018.11.08
        self.tw = 2  # 2 day width of the box profile
        self.time_unit = units.day
        self.profile = BoxTimeFluxProfile(
            t0=self.t0, tw=self.tw, time_unit=self.time_unit
        )

    def test_move(self):
        dt = 5
        self.profile.move(dt=dt)
        self.assertEqual(self.profile.t0, self.t0 + dt)
        self.assertEqual(self.profile.tw, self.tw)

    def test_t0(self):
        self.assertEqual(self.profile.t0, self.t0)

        t0 = 0
        self.profile.t0 = t0
        self.assertEqual(self.profile.t0, t0)

    def test_tw(self):
        self.assertEqual(self.profile.tw, self.tw)

        tw = 2.5
        self.profile.tw = tw
        self.assertEqual(self.profile.tw, tw)

    def test_get_integral(self):
        t1 = self.t0
        t2 = self.t0 + self.tw / 2
        times1 = np.array([self.t0 - self.tw, self.t0 - self.tw / 2, self.t0])
        times2 = np.array(
            [self.t0 + self.tw, self.t0 + self.tw / 2, self.t0 + self.tw / 2]
        )

        self.assertEqual(self.profile.get_integral(t1, t1), 0)
        self.assertEqual(self.profile.get_integral(t1, t2), 1.0)
        np.testing.assert_array_equal(
            self.profile.get_integral(times1, times2), np.array([2, 2, 1])
        )

        # Test cases when t1 > t2.
        self.assertEqual(self.profile.get_integral(t2, t1), -1)
        np.testing.assert_array_equal(
            self.profile.get_integral(times2, times1), np.array([0, -2, -1])
        )

    def test_get_total_integral(self):
        self.assertEqual(self.profile.get_total_integral(), 2)

    def test_call(self):
        self.assertEqual(self.profile(t=self.t0 - self.tw), 0)
        self.assertEqual(self.profile(t=self.t0), 1)
        self.assertEqual(self.profile(t=self.t0 + self.tw), 0)

        times = np.array([self.t0 - self.tw, self.t0, self.t0 + self.tw])
        np.testing.assert_array_equal(
            self.profile(t=times), np.array([0, 1, 0])
        )


if __name__ == "__main__":
    unittest.main()
