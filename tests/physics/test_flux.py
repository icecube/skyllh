# -*- coding: utf-8 -*-

from __future__ import division

import unittest
import numpy as np

from skyllh.physics.flux import (
    PowerLawFlux,
    CutoffPowerLawFlux,
    LogParabolaPowerLawFlux
)

from skyllh.core.py import float_cast


class TestPowerLawFlux(unittest.TestCase):
    def setUp(self):
        self.Phi0 = 1.5
        self.E0 = 2.5
        self.gamma = 2.7
        self.energy_unit = "GeV"
        self.power_law_flux = PowerLawFlux(self.Phi0, self.E0, self.gamma)

    def test_gamma(self):
        gamma = 2.6

        self.assertEqual(self.power_law_flux.gamma, self.gamma)

        self.power_law_flux.gamma = gamma

        self.assertEqual(self.power_law_flux.gamma, gamma)

    def test_math_function_str(self):
        self.assertEqual(self.power_law_flux.math_function_str, "dN/dE = {:.2e} * (E / {:.2e} {})^-{:.2f}".format(self.Phi0, self.E0, self.energy_unit, self.gamma))

    def test_call(self):
        E = 5
        flux = self.Phi0 * np.power(E / self.E0, -self.gamma)

        self.assertEqual(self.power_law_flux(E), flux)


class TestCutoffPowerLawFlux(unittest.TestCase):
    def setUp(self):
        self.Phi0 = 1.5
        self.E0 = 2.5
        self.gamma = 2.7
        self.Ecut = 2
        self.energy_unit = "GeV"
        self.cutoff_power_law_flux = CutoffPowerLawFlux(self.Phi0, self.E0, self.gamma, self.Ecut)

    def test_Ecut(self):
        Ecut = 2.5

        self.assertEqual(self.cutoff_power_law_flux.Ecut, self.Ecut)

        self.cutoff_power_law_flux.Ecut = Ecut

        self.assertEqual(self.cutoff_power_law_flux.Ecut, Ecut)

    def test_math_function_str(self):
        test_string = "dN/dE = {:.2e} * (E / {:.2e} {})^-{:.2f} * exp(-E / {:.2e} {})".format(self.Phi0, self.E0, self.energy_unit, self.gamma, self.Ecut, self.energy_unit)

        self.assertEqual(self.cutoff_power_law_flux.math_function_str, test_string)

    def test_call(self):
        E = 5
        flux = self.Phi0 * np.power(E / self.E0, -self.gamma) * np.exp(-E / self.Ecut)

        self.assertEqual(self.cutoff_power_law_flux(E), flux)


class TestLogParabolaPowerLawFlux(unittest.TestCase):
    def setUp(self):
        self.Phi0 = 1.5
        self.E0 = 2.5
        self.alpha = 1
        self.beta = 2
        self.energy_unit = "GeV"
        self.log_parabola_power_law_flux = LogParabolaPowerLawFlux(self.Phi0, self.E0, self.alpha, self.beta)

    def test_alpha(self):
        self.assertEqual(self.log_parabola_power_law_flux.alpha, self.alpha)

        alpha = 1.5
        self.log_parabola_power_law_flux.alpha = alpha

        self.assertEqual(self.log_parabola_power_law_flux.alpha, alpha)

    def test_beta(self):
        self.assertEqual(self.log_parabola_power_law_flux.beta, self.beta)

        beta = 2.5
        self.log_parabola_power_law_flux.beta = beta

        self.assertEqual(self.log_parabola_power_law_flux.beta, beta)

    def test_math_funciton_str(self):
        test_string = 'dN/dE = {:.2e} * (E / {:.2e} {})^(-({:.2e} + {:.2e} * log(E / {:.2e} {})))'.format(self.Phi0, self.E0, self.energy_unit, self.alpha, self.beta, self.E0, self.energy_unit)

        self.assertEqual(self.log_parabola_power_law_flux.math_function_str, test_string)

    def test_call(self):
        E = 5
        flux = self.Phi0 * np.power(E / self.E0, -self.alpha - self.beta * np.log(E / self.E0))

        self.assertEqual(self.log_parabola_power_law_flux(E), flux)


if(__name__ == '__main__'):
    unittest.main()
