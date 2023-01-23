# -*- coding: utf-8 -*-

r"""The `flux_model` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a mathematical function for the differential flux:

.. math::

    d^4\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) / (dA d\Omega dE dt)
"""

from __future__ import division

import abc
import numpy as np
import scipy.stats

from astropy import units
from copy import deepcopy

from skyllh.core.config import CFG
from skyllh.core.math import MathFunction
from skyllh.core.model import Model
from skyllh.core.py import (
    classname,
    isproperty,
    issequence,
    issequenceof,
    float_cast
)


class FluxProfile(MathFunction, metaclass=abc.ABCMeta):
    """The abstract base class for a flux profile math function.
    """

    def __init__(self):
        super(FluxProfile, self).__init__()


class SpatialFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for a spatial flux profile function.
    """

    def __init__(
            self, angle_unit=None):
        """Creates a new SpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(SpatialFluxProfile, self).__init__()

        self.angle_unit = angle_unit

    @property
    def angle_unit(self):
        """The set unit of angle used for this spatial flux profile.
        If set to ``Ǹone`` the configured default angle unit for fluxes is used.
        """
        return self._angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['angle']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property angle_unit must be of type '
                'astropy.units.UnitBase!')
        self._angle_unit = unit

    @abc.abstractmethod
    def __call__(self, alpha, delta, unit=None):
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            If ``None``, the set angle unit of this SpatialFluxProfile is
            assumed.

        Returns
        -------
        values : 1D numpy ndarray
            The spatial profile values.
        """
        pass


class UnitySpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1 for any spatial
    coordinates.
    """
    def __init__(self, angle_unit=None):
        """Creates a new UnitySpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(UnitySpatialFluxProfile, self).__init__(
            angle_unit=angle_unit)

    @property
    def math_function_str(self):
        return '1'

    def __call__(self, alpha, delta, unit=None):
        """Returns 1 as numpy ndarray in same shape as alpha and delta.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            By the definition of this class this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray
            1 in same shape as alpha and delta.
        """
        (alpha, delta) = np.atleast_1d(alpha, delta)
        if(alpha.shape != delta.shape):
            raise ValueError('The alpha and delta arguments must be of the '
                'same shape!')

        return np.ones_like(alpha)


class PointSpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for a delta function at the celestrical coordinate
    (alpha_s, delta_s).
    """
    def __init__(self, alpha_s, delta_s, angle_unit=None):
        """Creates a new spatial flux profile for a point.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(PointSpatialFluxProfile, self).__init__(
            angle_unit=angle_unit)

        self.alpha_s = alpha_s
        self.delta_s = delta_s

        # Define the names of the parameters, which can be updated.
        self.param_names = ('alpha_s', 'delta_s')

    @property
    def alpha_s(self):
        """The right-ascention of the point-like source.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._alpha_s
    @alpha_s.setter
    def alpha_s(self, v):
        v = float_cast(v,
            'The alpha_s property must be castable to type float!')
        self._alpha_s = v

    @property
    def delta_s(self):
        """The declination of the point-like source.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._delta_s
    @delta_s.setter
    def delta_s(self, v):
        v = float_cast(v,
            'The delta_s property must be castable to type float!')
        self._delta_s = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance.
        """
        return 'delta(alpha-%g%s)*delta(delta-%g%s)'%(
            self._alpha_s, self._angle_unit.to_string(), self._delta_s,
            self._angle_unit.to_string())

    def __call__(self, alpha, delta, unit=None):
        """Returns a numpy ndarray in same shape as alpha and delta with 1 if
        `alpha` equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        delta : float | 1d numpy ndarray of float
            The declination coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            If set to ``None``, the set angle unit of this SpatialFluxProfile
            instance is assumed.

        Returns
        -------
        value : 1D numpy ndarray of int8
            A numpy ndarray in same shape as alpha and delta with 1 if `alpha`
            equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.
        """
        (alpha, delta) = np.atleast_1d(alpha, delta)
        if(alpha.shape != delta.shape):
            raise ValueError('The alpha and delta arguments must be of the '
                'same shape!')

        if((unit is not None) and (unit != self._angle_unit)):
            angle_unit_conv_factor = unit.to(self._angle_unit)
            alpha = alpha * angle_unit_conv_factor
            delta = delta * angle_unit_conv_factor

        value = ((alpha == self._alpha_s) &
                 (delta == self._delta_s)).astype(np.int8, copy=False)

        return value


class EnergyFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for an energy flux profile function.
    """

    def __init__(self, energy_unit=None):
        """Creates a new energy flux profile with a given energy unit to be used
        for flux calculation.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(EnergyFluxProfile, self).__init__()

        # Set the energy unit.
        self.energy_unit = energy_unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux profile calculation.
        """
        return self._energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['energy']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit = unit

    @abc.abstractmethod
    def __call__(self, E, unit=None):
        """This method is supposed to return the energy profile value for the
        given energy value.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energy.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The energy profile values for the given energies.
        """
        pass


class UnityEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for the constant function 1.
    """
    def __init__(self, energy_unit=None):
        """Creates a new UnityEnergyFluxProfile instance.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(UnityEnergyFluxProfile, self).__init__(
            energy_unit=energy_unit)

    @property
    def math_function_str(self):
        """The string representation of the mathematical function of this energy
        flux profile.
        """
        return '1'

    def __call__(self, E, unit=None):
        """Returns 1 as numpy ndarray in some shape as E.

        Parameters
        ----------
        E : float | 1D numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray of int8
            1 in same shape as E.
        """
        E = np.atleast_1d(E)

        values = np.ones_like(E, dtype=np.int8)

        return values


class PowerLawEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for a power law profile with a reference energy
    ``E0`` and a spectral index ``gamma``.

    .. math::
        (E / E_0)^{-\gamma}
    """
    def __init__(self, E0, gamma, energy_unit=None):
        """Creates a new power law flux profile with the reference energy ``E0``
        and spectral index ``gamma``.

        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        gamma : castable to float
            The spectral index.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(PowerLawEnergyFluxProfile, self).__init__(
            energy_unit=energy_unit)

        self.E0 = E0
        self.gamma = gamma

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('E0', 'gamma',)

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0
    @E0.setter
    def E0(self, v):
        v = float_cast(v,
            'Property E0 must be castable to type float!')
        self._E0 = v

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma
    @gamma.setter
    def gamma(self, v):
        v = float_cast(v,
            'Property gamma must be castable to type float!')
        self._gamma = v

    @property
    def math_function_str(self):
        """The string representation of this EnergyFluxProfile instance.
        """
        return '(E / (%g %s))^-%g'%(self._E0, self._energy_unit, self._gamma)

    def __call__(self, E, unit=None):
        """Returns the power law values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E : float | 1D numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if((unit is not None) and (unit != self._energy_unit)):
            energy_unit_conv_factor = unit.to(self._energy_unit)
            E = E * energy_unit_conv_factor

        value = np.power(E / self._E0, -self._gamma)

        return value


class TimeFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for a time flux profile function.
    """

    def __init__(self, t_start=-np.inf, t_end=np.inf, time_unit=None):
        """Creates a new time flux profile instance.

        Parameters
        ----------
        t_start : float
            The start time of the time profile.
            If set to -inf, it means, that the profile starts at the beginning
            of the entire time-span of the dataset.
        t_end : float
            The end time of the time profile.
            If set to +inf, it means, that the profile ends at the end of the
            entire time-span of the dataset.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super(TimeFluxProfile, self).__init__()

        self.time_unit = time_unit

        self.t_start = t_start
        self.t_end = t_end

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t_start', 't_end')

    @property
    def t_start(self):
        """The start time of the time profile. Can be -inf which means, that
        the profile starts at the beginning of the entire dataset.
        """
        return self._t_start
    @t_start.setter
    def t_start(self, t):
        t = float_cast(t,
            'The t_start property must be castable to type float!')
        self._t_start = t

    @property
    def t_end(self):
        """The end time of the time profile. Can be +inf which means, that
        the profile ends at the end of the entire dataset.
        """
        return self._t_end
    @t_end.setter
    def t_end(self, t):
        t = float_cast(t,
            'The t_end property must be castable to type float!')
        self._t_end = t

    @property
    def duration(self):
        """(read-only) The duration of the time profile.
        """
        return self._t_end - self._t_start

    @property
    def time_unit(self):
        """The unit of time used for the flux profile calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['time']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property time_unit must be of type '
                'astropy.units.UnitBase!')
        self._time_unit = unit

    def get_total_integral(self):
        """Calculates the total integral of the time profile from t_start to
        t_end.

        Returns
        -------
        integral : float
            The integral value of the entire time profile.
            The value is in the set time unit of this TimeFluxProfile instance.
        """
        integral = self.get_integral(self._t_start, self._t_end)

        return integral

    @abc.abstractmethod
    def __call__(self, t, unit=None):
        """This method is supposed to return the time profile value for the
        given times.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The time profile values.
        """
        pass

    @abc.abstractmethod
    def move(self, dt, unit=None):
        """Abstract method to move the time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        pass

    @abc.abstractmethod
    def get_integral(self, t1, t2, unit=None):
        """This method is supposed to calculate the integral of the time profile
        from time ``t1`` to time ``t2``.

        Parameters
        ----------
        t1 : float | array of float
            The start time of the integration.
        t2 : float | array of float
            The end time of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """
        pass


class UnityTimeFluxProfile(TimeFluxProfile):
    """Time flux profile for the constant profile function ``1``.
    """
    def __init__(self, time_unit=None):
        super(UnityTimeFluxProfile, self).__init__(
            time_unit=time_unit)

    @property
    def math_function_str(self):
        return '1'

    def __call__(self, t, unit=None):
        """Returns 1 as numpy ndarray in same shape as t.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray of int8
            1 in same shape as ``t``.
        """
        t = np.atleast_1d(t)

        values = np.ones_like(t, dtype=np.int8)

        return values

    def move(self, dt, unit=None):
        """Moves the time profile by the given amount of time. By definition
        this method does nothing, because the profile is 1 over the entire
        dataset time range.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        pass

    def get_integral(self, t1, t2, unit=None):
        """Calculates the integral of the time profile from time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The start time of the integration.
        t2 : float | array of float
            The end time of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """
        if((unit is not None) and (unit != self._time_unit)):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = t2 - t1

        return integral


class BoxTimeFluxProfile(TimeFluxProfile):
    """This class describes a box-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the box profile.
        tw : float
            The width of the box profile.

    The box is centered at ``t0`` and extends to +/-``tw``/2 around ``t0``.
    """
    def __init__(self, t0, tw, time_unit=None):
        """Creates a new box-shaped time profile instance.

        Parameters
        ----------
        t0 : float
            The mid time of the box profile.
        tw : float
            The width of the box profile.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        t_start = t0 - tw/2.
        t_end = t0 + tw/2.

        super(BoxTimeFluxProfile, self).__init__(
            t_start=t_start, t_end=t_end, time_unit=time_unit)

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t0', 'tw')

    @property
    def t0(self):
        """The time of the mid point of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return 0.5*(self._t_start + self._t_end)
    @t0.setter
    def t0(self, t):
        old_t0 = self.t0
        dt = t - old_t0
        self.move(dt)

    @property
    def tw(self):
        """The time width of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return self._t_end - self._t_start
    @tw.setter
    def tw(self, w):
        t0 = self.t0
        self._t_start = t0 - 0.5*w
        self._t_end = t0 + 0.5*w

    @property
    def math_function_str(self):
        t0 = self.t0
        tw = self.tw
        return '1 for t in [%g-%g/2; %g+%g/2], 0 otherwise'%(
            t0, tw, t0, tw)

    def __call__(self, t, unit=None):
        """Returns 1 for all t within the interval [t0-tw/2; t0+tw/2], and 0
        otherwise.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of int8
            The value(s) of the time flux profile for the given time(s).
        """
        t = np.atleast_1d(t)

        if((unit is not None) and (unit != self._time_unit)):
            time_unit_conv_factor = unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        values = np.zeros((t.shape[0],), dtype=np.int8)
        m = (t >= self._t_start) & (t <= self._t_end)
        values[m] = 1

        return values

    def move(self, dt, unit=None):
        """Moves the box-shaped time profile by the time difference dt.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit : instance of astropy.units.UnitBase | None
            The unit of ``dt``.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        if((unit is not None) and (unit != self._time_unit)):
            dt = dt * unit.to(self._time_unit)

        self._t_start += dt
        self._t_end += dt

    def get_integral(self, t1, t2, unit=None):
        """Calculates the integral of the box-shaped time flux profile from
        time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The start time(s) of the integration.
        t2 : float | array of float
            The end time(s) of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s). The values are in the set time unit of this
            TimeFluxProfile instance.
        """
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

        if((unit is not None) and (unit != self._time_unit)):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = np.zeros((t1.shape[0],), dtype=np.float64)

        m = (t2 >= self._t_start) & (t1 <= self._t_end)
        N = np.count_nonzero(m)

        t1 = np.max(np.vstack((t1[m], np.repeat(self._t_start, N))).T, axis=1)
        t2 = np.min(np.vstack((t2[m], np.repeat(self._t_end, N))).T, axis=1)

        integral[m] = t2 - t1

        return integral


class GaussianTimeFluxProfile(TimeFluxProfile):
    """This class describes a gaussian-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the gaussian profile.
        sigma_t : float
            The one-sigma width of the gaussian profile.
    """
    def __init__(self, t0, sigma_t, tol=1e-12, time_unit=None):
        """Creates a new gaussian-shaped time profile instance.

        Parameters
        ----------
        t0 : float
            The mid time of the gaussian profile.
        sigma_t : float
            The one-sigma width of the gaussian profile.
        tol : float
            The tolerance of the gaussian value. This defines the start and end
            time of the gaussian profile.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        # Calculate the start and end time of the gaussian profile, such that
        # at those times the gaussian values obey the given tolerance.
        dt = np.sqrt(-2*sigma_t*sigma_t*np.log(np.sqrt(2*np.pi)*sigma_t*tol))
        t_start = t0 - dt
        t_end = t0 + dt

        # A Gaussian profile extends to +/- infinity by definition.
        super(GaussianTimeFluxProfile, self).__init__(
            t_start=t_start, t_end=t_end, time_unit=time_unit)

        self.t0 = t0
        self.sigma_t = sigma_t

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t0', 'sigma_t')

    @property
    def t0(self):
        """The time of the mid point of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return 0.5*(self._t_start + self._t_end)
    @t0.setter
    def t0(self, t):
        t = float_cast(t,
            'The t0 property must be castable to type float!')
        old_t0 = self.t0
        dt = t - old_t0
        self.move(dt)

    @property
    def sigma_t(self):
        """The one-sigma width of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return self._sigma_t
    @sigma_t.setter
    def sigma_t(self, sigma):
        sigma = float_cast(sigma,
            'The sigma property must be castable to type float!')
        self._sigma_t = sigma

    def __call__(self, t, unit=None):
        """Returns the gaussian profile value for the given time ``t``.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The value(s) of the time flux profile for the given time(s).
        """
        t = np.atleast_1d(t)

        if((unit is not None) and (unit != self._time_unit)):
            time_unit_conv_factor = unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        s = self._sigma_t
        twossq = 2*s*s
        t0 = 0.5*(self._t_end + self._t_start)
        dt = t - t0

        values = 1/(np.sqrt(np.pi*twossq)) * np.exp(-dt*dt/twossq)

        return values

    def move(self, dt, unit=None):
        """Moves the gaussian time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.
        """
        if((unit is not None) and (unit != self._time_unit)):
            dt = dt * unit.to(self._time_unit)

        self._t_start += dt
        self._t_end += dt

    def get_integral(self, t1, t2, unit=None):
        """Calculates the integral of the gaussian time profile from time ``t1``
        to time ``t2``.

        Parameters
        ----------
        t1 : float | array of float
            The start time(s) of the integration.
        t2 : float | array of float
            The end time(s) of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s). The values are in the set time unit of
            this TimeFluxProfile instance.
        """
        if((unit is not None) and (unit != self._time_unit)):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        t0 = 0.5*(self._t_end + self._t_start)
        sigma_t = self._sigma_t

        c1 = scipy.stats.norm.cdf(t1, loc=t0, scale=sigma_t)
        c2 = scipy.stats.norm.cdf(t2, loc=t0, scale=sigma_t)

        integral = c2 - c1

        return integral


class FluxModel(MathFunction, Model, metaclass=abc.ABCMeta):
    r"""Abstract base class for all flux models
    :math:`\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s)`.

    This base class defines the units used for the flux calculation. The unit
    of the flux is ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).

    At this point the functional form of the flux model is not yet defined.
    """

    def __init__(
            self, angle_unit=None, energy_unit=None, length_unit=None,
            time_unit=None, **kwargs):
        """Creates a new FluxModel instance and defines the user-defined units.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``None``, the configured default angle unit for fluxes is
            used.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        length_unit : instance of astropy.units.UnitBase | None
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super(FluxModel, self).__init__(**kwargs)

        # Define the units.
        self.angle_unit = angle_unit
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.time_unit = time_unit

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation.
        """
        return self._angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['angle']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property angle_unit must be of type '
                'astropy.units.UnitBase!')
        self._angle_unit = unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation.
        """
        return self._energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['energy']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit = unit

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._length_unit
    @length_unit.setter
    def length_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['length']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property length_unit must be of type '
                'astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['time']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property time_unit must be of type '
                'astropy.units.UnitBase!')
        self._time_unit = unit

    @property
    def unit_str(self):
        """The string representation of the flux unit.
        """
        return '1/(%s %s %s^2 %s)'%(
            self.energy_unit.to_string(), (self.angle_unit**2).to_string(),
            self.length_unit.to_string(), self.time_unit.to_string())

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit.
        """
        return r'%s$^{-1}$ %s$^{-1}$ %s$^{-2}$ %s$^{-1}$'%(
            self.energy_unit.to_string(), (self.angle_unit**2).to_string(),
            self.length_unit.to_string(), self.time_unit.to_string())

    def __str__(self):
        """Pretty string representation of this class.
        """
        return self.math_function_str + ' ' + self.unit_str

    @abc.abstractmethod
    def __call__(
            self, alpha, delta, E, t,
            angle_unit=None, energy_unit=None, time_unit=None):
        """The call operator to retrieve a flux value for a given celestrial
        position, energy, and observation time.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1D numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1D numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1D numpy ndarray of float
            The energy for which to retrieve the flux value.
        t : float | (Ntime,)-shaped 1D numpy ndarray of float
            The observation time for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the flux model is assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the flux model is assumed.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If ``None``, the set time unit of the flux model is assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values are in unit of the set flux model units
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        pass


class FactorizedFluxModel(FluxModel):
    r"""This class describes a flux model where the spatial, energy, and time
    profiles of the source factorize. That means the flux can be written as:

    .. math::

        \Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) =
            \Phi_0
            \Psi_{\mathrm{S}}(\alpha,\delta|\vec{p}_s)
            \epsilon_{\mathrm{S}}(E|\vec{p}_s)
            T_{\mathrm{S}}(t|\vec{p}_s)

    where, :math:`\Phi_0` is the normalization constant of the flux, and
    :math:`\Psi_{\mathrm{S}}`, :math:`\epsilon_{\mathrm{S}}`, and
    :math:`T_{\mathrm{S}}` are the spatial, energy, and time profiles of the
    flux, respectively.
    """
    def __init__(
            self, Phi0, spatial_profile, energy_profile, time_profile,
            length_unit=None, **kwargs):
        """Creates a new factorized flux model.

        Parameters
        ----------
        Phi0 : float
            The flux normalization constant.
        spatial_profile : SpatialFluxProfile instance | None
            The SpatialFluxProfile instance providing the spatial profile
            function of the flux.
            If set to None, an instance of UnitySpatialFluxProfile will be used,
            which represents the constant function 1.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        length_unit : instance of astropy.units.UnitBase | None
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        """
        self.Phi0 = Phi0
        self.spatial_profile = spatial_profile
        self.energy_profile = energy_profile
        self.time_profile = time_profile

        # The base class will set the default (internally used) flux unit, which
        # will be set automatically to the particular profile.
        super(FactorizedFluxModel, self).__init__(
            angle_unit=spatial_profile.angle_unit,
            energy_unit=energy_profile.energy_unit,
            time_unit=time_profile.time_unit,
            length_unit=length_unit,
            **kwargs
        )

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('Phi0',)

    @property
    def Phi0(self):
        """The flux normalization constant.
        The unit of this normalization constant is
        ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).
        """
        return self._Phi0
    @Phi0.setter
    def Phi0(self, v):
        v = float_cast(v,
            'The Phi0 property must be castable to type float!')
        self._Phi0 = v

    @property
    def spatial_profile(self):
        """Instance of SpatialFluxProfile describing the spatial profile of the
        flux.
        """
        return self._spatial_profile
    @spatial_profile.setter
    def spatial_profile(self, profile):
        if(profile is None):
            profile = UnitySpatialFluxProfile()
        if(not isinstance(profile, SpatialFluxProfile)):
            raise TypeError('The spatial_profile property must be None, or an '
                'instance of SpatialFluxProfile!')
        self._spatial_profile = profile

    @property
    def energy_profile(self):
        """Instance of EnergyFluxProfile describing the energy profile of the
        flux.
        """
        return self._energy_profile
    @energy_profile.setter
    def energy_profile(self, profile):
        if(profile is None):
            profile = UnityEnergyFluxProfile()
        if(not isinstance(profile, EnergyFluxProfile)):
            raise TypeError('The energy_profile property must be None, or an '
                'instance of EnergyFluxProfile!')
        self._energy_profile = profile

    @property
    def time_profile(self):
        """Instance of TimeFluxProfile describing the time profile of the flux.
        """
        return self._time_profile
    @time_profile.setter
    def time_profile(self, profile):
        if(profile is None):
            profile = UnityTimeFluxProfile()
        if(not isinstance(profile, TimeFluxProfile)):
            raise TypeError('The time_profile property must be None, or an '
                'instance of TimeFluxProfile!')
        self._time_profile = profile

    @property
    def math_function_str(self):
        """The string showing the mathematical function of the flux.
        """
        return '%.3e * %s * %s * %s * %s'%(
            self._Phi0,
            self.unit_str,
            self._spatial_profile.math_function_str,
            self._energy_profile.math_function_str,
            self._time_profile.math_function_str
        )

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation. The unit is
        taken and set from and to the set spatial flux profile, respectively.
        """
        return self._spatial_profile.angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
        self._spatial_profile.angle_unit = unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation. The unit is
        taken and set from and to the set energy flux profile, respectively.
        """
        return self._energy_profile.energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        self._energy_profile.energy_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation. The unit is
        taken and set from and to the set time flux profile, respectively.
        """
        return self._time_profile.time_unit
    @time_unit.setter
    def time_unit(self, unit):
        self._time_profile.time_unit = unit

    @property
    def parameter_names(self):
        """The tuple holding the names of the math function's parameters. This
        is the total set of parameter names for all flux profiles of this
        FactorizedFluxModel instance.
        """
        pnames = list(self._parameter_names)
        pnames += self._spatial_profile.parameter_names
        pnames += self._energy_profile.parameter_names
        pnames += self._time_profile.parameter_names

        return tuple(pnames)
    @parameter_names.setter
    def parameter_names(self, names):
        super(FactorizedFluxModel, self.__class__).parameter_names.fset(self, names)

    def __call__(
            self, alpha, delta, E, t,
            angle_unit=None, energy_unit=None, time_unit=None):
        """Calculates the flux values for the given celestrial positions,
        energies, and observation times.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        t : float | (Ntime,)-shaped 1d numpy ndarray of float
            The observation time for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the spatial flux profile is
            assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the energy flux profile is
            assumed.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If ``None``, the set time unit of the time flux profile is
            assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values are in unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        spatial_profile_values = self._spatial_profile(
            alpha, delta, unit=angle_unit)
        energy_profile_values = self._energy_profile(
            E, unit=energy_unit)
        time_profile_values = self._time_profile(
            t, unit=time_unit)

        flux = (
            self._Phi0 *
            spatial_profile_values[:,np.newaxis,np.newaxis] *
            energy_profile_values[np.newaxis,:,np.newaxis] *
            time_profile_values[np.newaxis,np.newaxis,:]
        )

        return flux

    def set_parameters(self, pdict):
        """Sets the parameters of the flux model. For this factorized flux model
        it means that it sets the parameters of the spatial, energy, and time
        profiles.

        Parameters
        ----------
        pdict : dict
            The flux parameter dictionary.

        Returns
        -------
        updated : bool
            Flag if parameter values were actually updated.
        """
        updated = False

        updated |= super(FactorizedFluxModel, self).set_parameters(pdict)

        updated |= self._spatial_profile.set_parameters(pdict)
        updated |= self._energy_profile.set_parameters(pdict)
        updated |= self._time_profile.set_parameters(pdict)

        return updated


class IsPointlikeSource(object):
    """This is a classifier class that can be used by other classes to indicate
    that the specific class describes a point-like source.
    """
    def __init__(
            self, ra_func_instance=None, get_ra_func=None, set_ra_func=None,
            dec_func_instance=None, get_dec_func=None, set_dec_func=None,
            **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsPointlikeSource class.


        """
        super(IsPointlikeSource, self).__init__(**kwargs)

        self._ra_func_instance = ra_func_instance
        self._get_ra_func = get_ra_func
        self._set_ra_func = set_ra_func

        self._dec_func_instance = dec_func_instance
        self._get_dec_func = get_dec_func
        self._set_dec_func = set_dec_func

    @property
    def ra(self):
        """The right-ascention coordinate of the point-like source.
        """
        return self._get_ra_func(self._ra_func_instance)
    @ra.setter
    def ra(self, v):
        self._set_ra_func(self._ra_func_instance, v)

    @property
    def dec(self):
        """The declination coordinate of the point-like source.
        """
        return self._get_dec_func(self._dec_func_instance)
    @dec.setter
    def dec(self, v):
        self._set_dec_func(self._dec_func_instance, v)


class PointlikeSourceFFM(FactorizedFluxModel, IsPointlikeSource):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point. This class provides the base class for a flux
    model of a point-like source.
    """
    def __init__(
            self, alpha_s, delta_s, Phi0, energy_profile, time_profile,
            angle_unit=None, length_unit=None):
        """Creates a new factorized flux model for a point-like source.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        Phi0 : float
            The flux normalization constant in unit of flux.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit for angles used for the flux unit.
            If set to ``None``, the configured internal angle unit is used.
        length_unit : instance of astropy.units.UnitBase | None
            The unit for length used for the flux unit.
            If set to ``None``, the configured internal length unit is used.
        """
        spatial_profile=PointSpatialFluxProfile(
            alpha_s, delta_s, angle_unit=angle_unit)

        super(PointlikeSourceFFM, self).__init__(
            Phi0=Phi0,
            spatial_profile=spatial_profile,
            energy_profile=energy_profile,
            time_profile=time_profile,
            length_unit=length_unit,
            ra_func_instance=spatial_profile,
            get_ra_func=spatial_profile.__class__.alpha_s.fget,
            set_ra_func=spatial_profile.__class__.alpha_s.fset,
            dec_func_instance=spatial_profile,
            get_dec_func=spatial_profile.__class__.delta_s.fget,
            set_dec_func=spatial_profile.__class__.delta_s.fset
        )


class SteadyPointlikeSourceFFM(PointlikeSourceFFM):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point and the time profile as constant 1. It is
    derived from the ``PointlikeSourceFFM`` class.
    """
    def __init__(
            self, alpha_s, delta_s, Phi0, energy_profile,
            angle_unit=None, length_unit=None, time_unit=None):
        """Creates a new factorized flux model for a point-like source with no
        time dependance.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        Phi0 : float
            The flux normalization constant.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        """
        super(SteadyPointlikeSourceFFM, self).__init__(
            alpha_s=alpha_s,
            delta_s=delta_s,
            Phi0=Phi0,
            energy_profile=energy_profile,
            time_profile=UnityTimeFluxProfile(time_unit=time_unit),
            angle_unit=angle_unit,
            length_unit=length_unit
        )

    def __call__(
            self, alpha, delta, E,
            angle_unit=None, energy_unit=None):
        """Calculates the flux values for the given celestrial positions, and
        energies.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the spatial flux profile is
            assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the energy flux profile is
            assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy)-shaped ndarray of float
            The flux values are in unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        spatial_profile_values = self._spatial_profile(
            alpha, delta, unit=angle_unit)
        energy_profile_values = self._energy_profile(
            E, unit=energy_unit)

        flux = (
            self._Phi0 *
            spatial_profile_values[:,np.newaxis] *
            energy_profile_values[np.newaxis,:]
        )

        return flux
