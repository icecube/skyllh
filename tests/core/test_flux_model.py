import unittest

import numpy as np
from astropy import (
    units,
)
from scipy.integrate import (
    quad,
)

from skyllh.core.config import (
    Config,
)
from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    CutoffPowerLawEnergyFluxProfile,
    EnergyFluxProfile,
    LogParabolaPowerLawEnergyFluxProfile,
    PointSpatialFluxProfile,
    PowerLawEnergyFluxProfile,
    UnityEnergyFluxProfile,
    UnitySpatialFluxProfile,
)


class UnitySpatialFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.fluxprofile = UnitySpatialFluxProfile(cfg=self.cfg)

    def test_math_function_str(self):
        self.assertEqual(self.fluxprofile.math_function_str, '1')

    def test_call(self):
        alpha = 1.5
        delta = 2.5

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta), np.array([1]))

        alpha = np.array([1.5, 2])
        delta = np.array([2.5, 3])

        np.testing.assert_array_equal(self.fluxprofile(alpha, delta), np.array([1, 1]))


class PointSpatialFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.ra = 1.5
        self.dec = 2.5
        self.angle_unit = units.radian
        self.fluxprofile = PointSpatialFluxProfile(ra=self.ra, dec=self.dec, angle_unit=units.radian, cfg=self.cfg)

    def test_init(self):
        self.assertEqual(self.fluxprofile.ra, self.ra)
        self.assertEqual(self.fluxprofile.dec, self.dec)

    def test_param_names(self):
        param_names = self.fluxprofile.param_names
        self.assertEqual(len(param_names), 2)
        self.assertEqual(param_names[0], 'ra')
        self.assertEqual(param_names[1], 'dec')

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
            (f'delta(ra-{self.ra:g}{self.angle_unit})*delta(dec-{self.dec:g}{self.angle_unit})'),
        )

    def test_call(self):
        ra = 1.5
        dec = 2.5

        np.testing.assert_array_equal(self.fluxprofile(ra, dec), np.array([1]))

        ra = np.array([1.5, 2])
        dec = np.array([2.5, 3])

        np.testing.assert_array_equal(self.fluxprofile(ra, dec), np.array([1, 0]))


class PowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.E0 = 2.5
        self.gamma = 2.7
        self.energy_unit = units.GeV
        self.fluxprofile = PowerLawEnergyFluxProfile(
            E0=self.E0, gamma=self.gamma, energy_unit=self.energy_unit, cfg=self.cfg
        )

    def test_gamma(self):
        self.assertEqual(self.fluxprofile.gamma, self.gamma)

        gamma = 2.6
        self.fluxprofile.gamma = gamma
        self.assertEqual(self.fluxprofile.gamma, gamma)

    def test_math_function_str(self):
        self.assertEqual(self.fluxprofile.math_function_str, f'(E / ({self.E0:g} {self.energy_unit}))^-{self.gamma:g}')

    def test_call(self):
        E = np.array([5])
        values = np.power(E / self.E0, -self.gamma)

        self.assertEqual(self.fluxprofile(E=E, unit=self.energy_unit), values)

    def _assert_inverse_integral(self, fluxprofile, gamma, rtol):
        fluxprofile.gamma = gamma
        u = np.linspace(0, 1, 11)
        E1, E2 = 10.0, 10.0**3.2

        E = fluxprofile.get_inverse_integral(u, E1, E2)

        self.assertEqual(E.shape, u.shape)
        frac = fluxprofile.get_integral(E1, E) / fluxprofile.get_integral(E1, E2)
        np.testing.assert_allclose(frac, u, rtol=rtol, atol=1e-12)
        self.assertAlmostEqual(E[0], E1)
        self.assertAlmostEqual(E[-1], E2, delta=1e-9 * E2)

    def test_get_inverse_integral(self):
        for gamma in (-1.5, 0.0, 1.0, 2.0, 3.5):
            with self.subTest(gamma=gamma):
                self._assert_inverse_integral(self.fluxprofile, gamma, rtol=1e-9)

    def test_get_inverse_integral_gamma_close_to_one(self):
        # get_integral loses precision for gamma close to 1, so compare with
        # the gamma = 1 limit, a log-uniform distribution, instead.
        self.fluxprofile.gamma = 1.0 + 1e-12
        u = np.linspace(0, 1, 11)

        E = self.fluxprofile.get_inverse_integral(u, 10.0, 10.0**3.2)

        np.testing.assert_allclose(E, 10.0 * (10.0**2.2) ** u, rtol=1e-10)

    def test_get_inverse_integral_numerical_default_matches_analytic(self):
        u = np.random.default_rng(1).uniform(size=1000)
        E1 = np.where(u < 0.5, 10.0, 100.0)
        E2 = np.where(u < 0.5, 20.0, 10.0**3.2)
        for gamma in (1.0, 2.0, 3.5):
            with self.subTest(gamma=gamma):
                self.fluxprofile.gamma = gamma
                E_analytic = self.fluxprofile.get_inverse_integral(u, E1, E2)
                E_numeric = EnergyFluxProfile.get_inverse_integral(self.fluxprofile, u, E1, E2)
                np.testing.assert_allclose(E_numeric, E_analytic, rtol=1e-5)

    def test_get_inverse_integral_unit(self):
        u = np.array([0.1, 0.5, 0.9])

        E_TeV = self.fluxprofile.get_inverse_integral(u, 0.01, 1.0, unit=units.TeV)
        E_GeV = self.fluxprofile.get_inverse_integral(u, 10.0, 1000.0)

        np.testing.assert_allclose(E_TeV * 1000, E_GeV, rtol=1e-12)

    def test_get_inverse_integral_invalid_args(self):
        with self.assertRaises(ValueError):
            self.fluxprofile.get_inverse_integral(1.5, 10.0, 100.0)
        with self.assertRaises(ValueError):
            self.fluxprofile.get_inverse_integral(0.5, 100.0, 10.0)
        with self.assertRaises(ValueError):
            self.fluxprofile.get_inverse_integral(0.5, 0.0, 10.0)


class UnityEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def test_get_inverse_integral(self):
        fluxprofile = UnityEnergyFluxProfile(energy_unit=units.GeV, cfg=Config())

        E = fluxprofile.get_inverse_integral(np.array([0, 0.25, 1]), 10.0, 50.0)

        np.testing.assert_allclose(E, [10.0, 20.0, 50.0])


class CutoffPowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.E0 = 2.5
        self.gamma = 2.7
        self.Ecut = 2
        self.energy_unit = units.GeV
        self.fluxprofile = CutoffPowerLawEnergyFluxProfile(
            E0=self.E0, gamma=self.gamma, Ecut=self.Ecut, energy_unit=self.energy_unit, cfg=self.cfg
        )

    def test_Ecut(self):
        self.assertEqual(self.fluxprofile.Ecut, self.Ecut)

        Ecut = 2.5
        self.fluxprofile.Ecut = Ecut
        self.assertEqual(self.fluxprofile.Ecut, Ecut)

    def test_math_function_str(self):
        self.assertEqual(
            self.fluxprofile.math_function_str,
            (f'(E / ({self.E0:g} {self.energy_unit}))^-{self.gamma:g} exp(-E / ({self.Ecut:g} {self.energy_unit}))'),
        )

    def test_call(self):
        E = np.array([5])
        values = np.power(E / self.E0, -self.gamma) * np.exp(-E / self.Ecut)

        self.assertEqual(self.fluxprofile(E=E, unit=self.energy_unit), values)

    def test_get_inverse_integral(self):
        # The cutoff profile has no analytic inverse integral, so this tests
        # the numerical default implementation.
        u = np.linspace(0, 1, 11)
        E1, E2 = 1.0, 10.0

        E = self.fluxprofile.get_inverse_integral(u, E1, E2)

        frac = self.fluxprofile.get_integral(np.full_like(E, E1), E) / self.fluxprofile.get_integral(E1, E2)
        np.testing.assert_allclose(frac, u, rtol=1e-4, atol=1e-12)
        self.assertTrue(np.all(np.diff(E) > 0))


class LogParabolaPowerLawEnergyFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.E0 = 2.5
        self.alpha = 1
        self.beta = 2
        self.energy_unit = units.GeV
        self.fluxprofile = LogParabolaPowerLawEnergyFluxProfile(
            E0=self.E0, alpha=self.alpha, beta=self.beta, energy_unit=self.energy_unit, cfg=self.cfg
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

    @staticmethod
    def _quad_integral(fluxprofile, E1, E2):
        """Integrates the profile with scipy's quad in ln(E), as reference."""
        if E2 / E1 - 1 < 1e-8:
            # quad cannot resolve such a narrow range in ln(E), but the
            # integrand is constant over it to that precision.
            return float(fluxprofile(E1)[0]) * (E2 - E1)

        def integrand(y):
            E = np.exp(y)
            return float(fluxprofile(E)[0]) * E

        return quad(integrand, np.log(E1), np.log(E2), epsabs=0, epsrel=1e-12, limit=500)[0]

    # (alpha, beta, E1, E2) with E0 = 2.5 GeV: the peak inside the range, far
    # tails on both sides of the peak, the beta -> 0 limit, an almost flat
    # spectrum, where the closed form loses precision and the numerical default
    # is used, and beta < 0, which always uses the numerical default.
    CASES = (
        (1.0, 2.0, 1.0, 10.0),
        (2.0, 0.3, 1e-4, 1e-2),
        (2.0, 0.3, 1e3, 1e5),
        (3.0, 1.0, 1e4, 1e6),
        (2.0, 1e-3, 10.0, 1e4),
        (2.0, 1e-12, 10.0, 1e4),
        (1.0, 1e-12, 10.0, 1e4),
        (2.0, -0.05, 10.0, 1e4),
    )

    def test_get_integral(self):
        for alpha, beta, E1, E2 in self.CASES:
            with self.subTest(alpha=alpha, beta=beta, E1=E1, E2=E2):
                self.fluxprofile.alpha = alpha
                self.fluxprofile.beta = beta

                integral = self.fluxprofile.get_integral(E1, E2)

                np.testing.assert_allclose(integral, self._quad_integral(self.fluxprofile, E1, E2), rtol=1e-12)
                np.testing.assert_allclose(self.fluxprofile.get_integral(E2, E1), -integral, rtol=1e-15)

    def test_get_integral_beta_zero_is_power_law(self):
        self.fluxprofile.alpha = 2.0
        powerlaw = PowerLawEnergyFluxProfile(E0=self.E0, gamma=2.0, energy_unit=self.energy_unit, cfg=self.cfg)
        for beta, rtol in ((0.0, 1e-14), (1e-14, 1e-12)):
            with self.subTest(beta=beta):
                self.fluxprofile.beta = beta
                np.testing.assert_allclose(
                    self.fluxprofile.get_integral(10.0, 1e4), powerlaw.get_integral(10.0, 1e4), rtol=rtol
                )
                u = np.array([0.1, 0.5, 0.9])
                np.testing.assert_allclose(
                    self.fluxprofile.get_inverse_integral(u, 10.0, 1e4),
                    powerlaw.get_inverse_integral(u, 10.0, 1e4),
                    rtol=rtol,
                )

    def test_get_inverse_integral(self):
        u = np.concatenate(([0.0, 1e-12, 1e-6], np.linspace(0.01, 0.99, 25), [1 - 1e-9, 1.0]))
        for alpha, beta, E1, E2 in self.CASES:
            with self.subTest(alpha=alpha, beta=beta, E1=E1, E2=E2):
                self.fluxprofile.alpha = alpha
                self.fluxprofile.beta = beta
                # The numerical default is accurate to about 1e-5.
                atol = 1e-12 if (beta > 0 and alpha != 1.0) or beta >= 1e-3 else 1e-4

                E = self.fluxprofile.get_inverse_integral(u, E1, E2)

                self.assertEqual(E[0], E1)
                self.assertEqual(E[-1], E2)
                self.assertTrue(np.all(np.diff(E) >= 0))
                total = self._quad_integral(self.fluxprofile, E1, E2)
                frac = np.array([self._quad_integral(self.fluxprofile, E1, E_i) for E_i in E]) / total
                np.testing.assert_allclose(frac, u, rtol=0, atol=atol)

    def test_get_inverse_integral_unit(self):
        u = np.array([0.1, 0.5, 0.9])

        E_TeV = self.fluxprofile.get_inverse_integral(u, 0.01, 1.0, unit=units.TeV)
        E_GeV = self.fluxprofile.get_inverse_integral(u, 10.0, 1000.0)

        np.testing.assert_allclose(E_TeV * 1000, E_GeV, rtol=1e-12)

    def test_math_funciton_str(self):
        s_E0 = f'{self.E0:g} {self.energy_unit}'
        test_string = f'(E / {s_E0})^(-({self.alpha:g} + {self.beta:g} log(E / {s_E0})))'
        self.assertEqual(self.fluxprofile.math_function_str, test_string)


class BoxTimeFluxProfileTestCase(
    unittest.TestCase,
):
    def setUp(self):
        self.cfg = Config()
        self.t0 = 58430  # MJD time 2018.11.08
        self.tw = 2  # 2 day width of the box profile
        self.time_unit = units.day
        self.profile = BoxTimeFluxProfile(t0=self.t0, tw=self.tw, time_unit=self.time_unit, cfg=self.cfg)

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
        times2 = np.array([self.t0 + self.tw, self.t0 + self.tw / 2, self.t0 + self.tw / 2])

        self.assertEqual(self.profile.get_integral(t1, t1), 0)
        self.assertEqual(self.profile.get_integral(t1, t2), 1.0)
        np.testing.assert_array_equal(self.profile.get_integral(times1, times2), np.array([2, 2, 1]))

        # Test cases when t1 > t2.
        self.assertEqual(self.profile.get_integral(t2, t1), -1)
        np.testing.assert_array_equal(self.profile.get_integral(times2, times1), np.array([0, -2, -1]))

    def test_get_total_integral(self):
        self.assertEqual(self.profile.get_total_integral(), 2)

    def test_call(self):
        self.assertEqual(self.profile(t=self.t0 - self.tw), 0)
        self.assertEqual(self.profile(t=self.t0), 1)
        self.assertEqual(self.profile(t=self.t0 + self.tw), 0)

        times = np.array([self.t0 - self.tw, self.t0, self.t0 + self.tw])
        np.testing.assert_array_equal(self.profile(t=times), np.array([0, 1, 0]))


if __name__ == '__main__':
    unittest.main()
