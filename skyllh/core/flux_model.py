# -*- coding: utf-8 -*-

r"""The `flux_model` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a mathematical function for the differential flux:

.. math::

    \frac{d^4\Phi(\alpha,\delta,E,t | \vec{p}_{\mathrm{s}})}{\mathrm{d}A
    \mathrm{d}\Omega \mathrm{d}E \mathrm{d}t}

"""

import abc
from astropy import (
    units,
)
import numpy as np
from scipy.integrate import (
    quad,
)
import scipy.special
import scipy.stats

from skyllh.core import (
    tool,
)
from skyllh.core.config import (
    Config,
    HasConfig,
)
from skyllh.core.math import (
    MathFunction,
)
from skyllh.core.model import (
    Model,
)
from skyllh.core.py import (
    classname,
    float_cast,
)
from skyllh.core.source_model import (
    IsPointlike,
)


class FluxProfile(
        MathFunction,
        HasConfig,
        metaclass=abc.ABCMeta):
    """The abstract base class for a flux profile math function.
    """
    def __init__(
            self,
            **kwargs,
    ):
        """Creates a new FluxProfile instance.
        """
        super().__init__(**kwargs)


class SpatialFluxProfile(
        FluxProfile,
        metaclass=abc.ABCMeta):
    """The abstract base class for a spatial flux profile function.
    """
    def __init__(
            self,
            angle_unit=None,
            **kwargs):
        """Creates a new SpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(
            **kwargs)

        self.angle_unit = angle_unit

    @property
    def angle_unit(self):
        """The set unit of angle used for this spatial flux profile.
        If set to ``Ǹone`` the configured default angle unit for fluxes is used.
        """
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['angle']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property angle_unit must be of type '
                'astropy.units.UnitBase!')
        self._angle_unit = unit

    @abc.abstractmethod
    def __call__(
            self,
            ra,
            dec,
            unit=None):
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        ra : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        dec : float | 1d numpy ndarray of float
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


class UnitySpatialFluxProfile(
        SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1 for any spatial
    coordinates.
    """
    def __init__(
            self,
            angle_unit=None,
            **kwargs,
    ):
        """Creates a new UnitySpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(
            angle_unit=angle_unit,
            **kwargs)

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance.
        """
        return '1'

    def __call__(
            self,
            ra,
            dec,
            unit=None):
        """Returns 1 as numpy ndarray in same shape as ra and dec.

        Parameters
        ----------
        ra : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        dec : float | 1d numpy ndarray of float
            The declination coordinate.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            By the definition of this class this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray
            1 in same shape as ra and dec.
        """
        (ra, dec) = np.atleast_1d(ra, dec)
        if ra.shape != dec.shape:
            raise ValueError(
                'The ra and dec arguments must be of the same shape!')

        return np.ones_like(ra)


class PointSpatialFluxProfile(
        SpatialFluxProfile):
    """Spatial flux profile for a delta function at the celestrical coordinate
    (ra, dec).
    """
    def __init__(
            self,
            ra,
            dec,
            angle_unit=None,
            **kwargs,
    ):
        """Creates a new spatial flux profile for a point at equatorial
        coordinate (ra, dec).

        Parameters
        ----------
        ra : float | None
            The right-ascention of the point.
            In case it is None, the evaluation of this spatial flux profile will
            return zero, unless evaluated for ra=None.
        dec : float | None
            The declination of the point.
            In case it is None, the evaluation of this spatial flux profile will
            return zero, unless evaluated for dec=None.
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(
            angle_unit=angle_unit,
            **kwargs)

        self.ra = ra
        self.dec = dec

        # Define the names of the parameters, which can be updated.
        self.param_names = ('ra', 'dec')

    @property
    def ra(self):
        """The right-ascention of the point.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._ra

    @ra.setter
    def ra(self, v):
        v = float_cast(
            v,
            'The ra property must be castable to type float!',
            allow_None=True)
        self._ra = v

    @property
    def dec(self):
        """The declination of the point.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._dec

    @dec.setter
    def dec(self, v):
        v = float_cast(
            v,
            'The dec property must be castable to type float!',
            allow_None=True)
        self._dec = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance. It is None, if the right-ascention
        or declination property is set to None.
        """
        if (self._ra is None) or (self._dec is None):
            return None

        s = (f'delta(ra-{self._ra:g}{self._angle_unit})*'
             f'delta(dec-{self._dec:g}{self._angle_unit})')

        return s

    def __call__(
            self,
            ra,
            dec,
            unit=None):
        """Returns a numpy ndarray in same shape as ra and dec with 1 if
        `ra` equals `self.ra` and `dec` equals `self.dec`, and 0 otherwise.

        Parameters
        ----------
        ra : float | 1d numpy ndarray of float
            The right-ascention coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        dec : float | 1d numpy ndarray of float
            The declination coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            If set to ``None``, the set angle unit of this SpatialFluxProfile
            instance is assumed.

        Returns
        -------
        value : 1D numpy ndarray of int8
            A numpy ndarray in same shape as ra and dec with 1 if `ra`
            equals `self.ra` and `dec` equals `self.dec`, and 0 otherwise.
        """
        (ra, dec) = np.atleast_1d(ra, dec)
        if ra.shape != dec.shape:
            raise ValueError(
                'The ra and dec arguments must be of the same shape!')

        if (unit is not None) and (unit != self._angle_unit):
            angle_unit_conv_factor = unit.to(self._angle_unit)
            ra = ra * angle_unit_conv_factor
            dec = dec * angle_unit_conv_factor

        value = (
            (ra == self._ra) &
            (dec == self._dec)
        ).astype(np.int8, copy=False)

        return value


class EnergyFluxProfile(
        FluxProfile,
        metaclass=abc.ABCMeta):
    """The abstract base class for an energy flux profile function.
    """
    def __init__(
            self,
            energy_unit=None,
            **kwargs):
        """Creates a new energy flux profile with a given energy unit to be used
        for flux calculation.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            **kwargs)

        # Set the energy unit.
        self.energy_unit = energy_unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux profile calculation.
        """
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['energy']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit = unit

    @abc.abstractmethod
    def __call__(
            self,
            E,
            unit=None):
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

    def get_integral(
            self,
            E1,
            E2,
            unit=None,
    ):
        """This is the default implementation for calculating the integral value
        of this energy flux profile in the range ``[E1, E2]``.

        .. note::

            This implementation utilizes the ``scipy.integrate.quad`` function
            to perform a generic numeric integration. Hence, this implementation
            is slow and should be reimplemented by the derived class if an
            analytic integral form is available.

        Parameters
        ----------
        E1 : float | 1d numpy ndarray of float
            The lower energy bound of the integration.
        E2 : float | 1d numpy ndarray of float
            The upper energy bound of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral : instance of ndarray
            The (n,)-shaped numpy ndarray holding the integral values of the
            given integral ranges.
        """
        E1 = np.atleast_1d(E1)
        E2 = np.atleast_1d(E2)

        if (unit is not None) and (unit != self._energy_unit):
            time_unit_conv_factor = unit.to(self._energy_unit)
            E1 = E1 * time_unit_conv_factor
            E2 = E2 * time_unit_conv_factor

        integral = np.empty((len(E1),), dtype=np.float64)

        for (i, (E1_i, E2_i)) in enumerate(zip(E1, E2)):
            integral[i] = quad(self, E1_i, E2_i, full_output=True)[0]

        return integral


class UnityEnergyFluxProfile(
        EnergyFluxProfile):
    """Energy flux profile for the constant function 1.
    """
    def __init__(
            self,
            energy_unit=None,
            **kwargs):
        """Creates a new UnityEnergyFluxProfile instance.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            energy_unit=energy_unit,
            **kwargs)

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this energy flux profile.
        """
        return '1'

    def __call__(
            self,
            E,
            unit=None):
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

    def get_integral(
            self,
            E1,
            E2,
            unit=None):
        """Computes the integral of this energy flux profile in the range
        [``E1``, ``E2``], which by definition is ``E2 - E1``.

        Parameters
        ----------
        E1 : float | 1d numpy ndarray of float
            The lower energy bound of the integration.
        E2 : float | 1d numpy ndarray of float
            The upper energy bound of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral : 1d ndarray of float
            The integral values of the given integral ranges.
        """
        E1 = np.atleast_1d(E1)
        E2 = np.atleast_1d(E2)

        if (unit is not None) and (unit != self._energy_unit):
            time_unit_conv_factor = unit.to(self._energy_unit)
            E1 = E1 * time_unit_conv_factor
            E2 = E2 * time_unit_conv_factor

        integral = E2 - E1

        return integral


class PowerLawEnergyFluxProfile(
        EnergyFluxProfile,
):
    r"""Energy flux profile for a power law profile with a reference energy
    ``E0`` and a spectral index ``gamma``.

    .. math::

        (E / E_0)^{-\gamma}

    """
    def __init__(
            self,
            E0,
            gamma,
            energy_unit=None,
            **kwargs):
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
        super().__init__(
            energy_unit=energy_unit,
            **kwargs)

        self.E0 = E0
        self.gamma = gamma

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('E0', 'gamma',)

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0

    @E0.setter
    def E0(self, v):
        v = float_cast(
            v,
            'Property E0 must be castable to type float!')
        self._E0 = v

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        v = float_cast(
            v,
            'Property gamma must be castable to type float!')
        self._gamma = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s = f'(E / ({self._E0:g} {self._energy_unit}))^-{self._gamma:g}'

        return s

    def __call__(
            self,
            E,
            unit=None):
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

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        value = np.power(E / self._E0, -self._gamma)

        return value

    def get_integral(
            self,
            E1,
            E2,
            unit=None):
        """Computes the integral value of this power-law energy flux profile in
        the range ``[E1, E2]``.

        Parameters
        ----------
        E1 : float | 1d numpy ndarray of float
            The lower energy bound of the integration.
        E2 : float | 1d numpy ndarray of float
            The upper energy bound of the integration.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral : 1d ndarray of float
            The integral values of the given integral ranges.
        """
        E1 = np.atleast_1d(E1)
        E2 = np.atleast_1d(E2)

        if (unit is not None) and (unit != self._energy_unit):
            time_unit_conv_factor = unit.to(self._energy_unit)
            E1 = E1 * time_unit_conv_factor
            E2 = E2 * time_unit_conv_factor

        gamma = self._gamma

        # Handle special case for gamma = 1.
        if gamma == 1:
            integral = self._E0 * np.log(E2/E1)
            return integral

        integral = (
            np.power(self._E0, gamma) / (1-gamma) *
            (np.power(E2, 1-gamma) - np.power(E1, 1-gamma))
        )

        return integral


class CutoffPowerLawEnergyFluxProfile(
        PowerLawEnergyFluxProfile,
):
    r"""Cut-off power law energy flux profile of the form

    .. math::

        (E / E_0)^{-\gamma} \exp(-E/E_{\mathrm{cut}})

    """
    def __init__(
            self,
            E0,
            gamma,
            Ecut,
            energy_unit=None,
            **kwargs,
    ):
        """Creates a new cut-off power law flux profile with the reference
        energy ``E0``, spectral index ``gamma``, and cut-off energy ``Ecut``.

        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        gamma : castable to float
            The spectral index.
        Ecut : castable to float
            The cut-off energy.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            E0=E0,
            gamma=gamma,
            energy_unit=energy_unit,
            **kwargs)

        self.Ecut = Ecut

    @property
    def Ecut(self):
        """The energy cut value.
        """
        return self._Ecut

    @Ecut.setter
    def Ecut(self, v):
        v = float_cast(
            v,
            'The Property Ecut  must be castable to type float!')
        self._Ecut = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s = (f'(E / ({self._E0:g} {self._energy_unit}))^-{self._gamma:g} '
             f'exp(-E / ({self._Ecut:g} {self._energy_unit}))')

        return s

    def __call__(
            self,
            E,
            unit=None,
    ):
        """Returns the cut-off power law values for the given energies as
        numpy ndarray in the same shape as E.

        Parameters
        ----------
        E : float | instance of numpy ndarray
            The energy value(s) for which to retrieve the energy profile value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values : instance of numpy ndarray
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        values = super().__call__(E=E, unit=None)
        values *= np.exp(-E / self._Ecut)

        return values


class LogParabolaPowerLawEnergyFluxProfile(
        PowerLawEnergyFluxProfile,
):
    r"""This class provides an energy flux profile for a power-law with a
    spectral index that varies as a log parabola in energy of the form

    .. math::

        \frac{E}{E_0}^{-\left(\alpha + \beta\log(\frac{E}{E_0})\right)}

    """
    def __init__(
            self,
            E0,
            alpha,
            beta,
            energy_unit=None,
            **kwargs,
    ):
        """
        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        alpha : float
            The alpha parameter of the log-parabola spectral index.
        beta : float
            The beta parameter of the log-parabola spectral index.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            E0=E0,
            gamma=np.nan,
            energy_unit=energy_unit,
            **kwargs)

        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self):
        """The alpha parameter of the log-parabola spectral index.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        v = float_cast(
            v,
            'Property alpha must be castable to type float!')
        self._alpha = v

    @property
    def beta(self):
        """The beta parameter of the log-parabola spectral index.
        """
        return self._beta

    @beta.setter
    def beta(self, v):
        v = float_cast(
            v,
            'Property beta must be castable to type float!')
        self._beta = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s_E0 = f'{self._E0:g} {self._energy_unit}'
        s = (
            f'(E / {s_E0})'
            f'^(-({self._alpha:g} + {self._beta:g} log(E / {s_E0})))'
        )

        return s

    def __call__(
            self,
            E,
            unit=None,
    ):
        """Returns the log-parabola power-law values for the given energies as
        numpy ndarray in the same shape as E.

        Parameters
        ----------
        E : float | instance of numpy ndarray
            The energy value(s) for which to retrieve the energy profile value.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values : instance of numpy ndarray
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        values = np.power(
            E / self._E0,
            -self._alpha - self._beta * np.log(E / self._E0)
        )

        return values


class PhotosplineEnergyFluxProfile(
        EnergyFluxProfile,
):
    """The abstract base class for an energy flux profile based on a
    photospline.
    """
    @tool.requires('photospline')
    def __init__(
            self,
            splinetable,
            crit_log10_energy_lower,
            crit_log10_energy_upper,
            energy_unit=None,
            **kwargs,
    ):
        """Creates a new instance of PhotosplineEnergyFluxProfile.

        Parameters
        ----------
        splinetable : instance of photospline.SplineTable
            The instance of photospline.SplineTable representing the energy flux
            profile as a spline.
        crit_log10_energy_lower : float
            The lower edge of the spline's supported energy range in log10(E).
        crit_log10_energy_upper : float
            The upper edge of the spline's supported energy range in log10(E).
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            energy_unit=energy_unit,
            **kwargs)

        self.photospline = tool.get('photospline')

        self.splinetable = splinetable
        self.crit_log10_energy_lower = crit_log10_energy_lower
        self.crit_log10_energy_upper = crit_log10_energy_upper

    @property
    def splinetable(self):
        """The instance of photospline.SplineTable that describes the neutrino
        energy flux profile as function of neutrino energy via B-spline
        interpolation.
        """
        return self._splinetable

    @splinetable.setter
    def splinetable(self, table):
        if not isinstance(table, self.photospline.SplineTable):
            raise TypeError(
                'The splinetable property must be an instance of '
                'photospline.SplineTable! '
                f'Its current type is {classname(table)}!')
        self._splinetable = table

    @property
    def crit_log10_energy_lower(self):
        """The lower energy bound of the spline's support.
        """
        return self._crit_log10_energy_lower

    @crit_log10_energy_lower.setter
    def crit_log10_energy_lower(self, v):
        v = float_cast(
            v,
            'The property crit_log10_energy_lower must be castable to type '
            'float!')
        self._crit_log10_energy_lower = v

    @property
    def crit_log10_energy_upper(self):
        """The upper energy bound of the spline's support.
        """
        return self._crit_log10_energy_upper

    @crit_log10_energy_upper.setter
    def crit_log10_energy_upper(self, v):
        v = float_cast(
            v,
            'The property crit_log10_energy_upper must be castable to type '
            'float!')
        self._crit_log10_energy_upper = v


class TimeFluxProfile(
        FluxProfile,
        metaclass=abc.ABCMeta,
):
    """The abstract base class for a time flux profile function.
    """
    def __init__(
            self,
            t_start=-np.inf,
            t_stop=np.inf,
            time_unit=None,
            **kwargs):
        """Creates a new time flux profile instance.

        Parameters
        ----------
        t_start : float
            The start time of the time profile.
            If set to -inf, it means, that the profile starts at the beginning
            of the entire time-span of the dataset.
        t_stop : float
            The stop time of the time profile.
            If set to +inf, it means, that the profile ends at the end of the
            entire time-span of the dataset.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super().__init__(
            **kwargs)

        self.time_unit = time_unit

        self.t_start = t_start
        self.t_stop = t_stop

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('t_start', 't_stop')

    @property
    def t_start(self):
        """The start time of the time profile. Can be -inf which means, that
        the profile starts at the beginning of the entire dataset.
        """
        return self._t_start

    @t_start.setter
    def t_start(self, t):
        t = float_cast(
            t,
            'The t_start property must be castable to type float! '
            f'Its current type is {classname(t)}!')
        self._t_start = t

    @property
    def t_stop(self):
        """The stop time of the time profile. Can be +inf which means, that
        the profile ends at the end of the entire dataset.
        """
        return self._t_stop

    @t_stop.setter
    def t_stop(self, t):
        t = float_cast(
            t,
            'The t_stop property must be castable to type float! '
            f'Its current type is {classname(t)}!')
        self._t_stop = t

    @property
    def duration(self):
        """(read-only) The duration of the time profile.
        """
        return self._t_stop - self._t_start

    @property
    def time_unit(self):
        """The unit of time used for the flux profile calculation.
        """
        return self._time_unit

    @time_unit.setter
    def time_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['time']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property time_unit must be of type '
                'astropy.units.UnitBase! '
                f'Its current type is {classname(unit)}!')
        self._time_unit = unit

    def get_total_integral(self):
        """Calculates the total integral of the time profile from t_start to
        t_stop.

        Returns
        -------
        integral : float
            The integral value of the entire time profile.
            The value is in the set time unit of this TimeFluxProfile instance.
        """
        integral = self.get_integral(self._t_start, self._t_stop).squeeze()

        return integral

    @abc.abstractmethod
    def __call__(
            self,
            t,
            unit=None,
    ):
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
    def move(
            self,
            dt,
            unit=None,
    ):
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
    def get_integral(
            self,
            t1,
            t2,
            unit=None,
    ):
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


class UnityTimeFluxProfile(
        TimeFluxProfile,
):
    """Time flux profile for the constant profile function ``1``.
    """
    def __init__(
            self,
            time_unit=None,
            **kwargs,
    ):
        """Creates a new instance of UnityTimeFluxProfile.

        Parameters
        ----------
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super().__init__(
            time_unit=time_unit,
            **kwargs)

    @property
    def math_function_str(self):
        return '1'

    def __call__(
            self,
            t,
            unit=None,
    ):
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

    def move(
            self,
            dt,
            unit=None,
    ):
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

    def get_integral(
            self,
            t1,
            t2,
            unit=None,
    ):
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
        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = t2 - t1

        return integral


class BoxTimeFluxProfile(
        TimeFluxProfile,
):
    """This class describes a box-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the box profile.
        tw : float
            The width of the box profile.

    The box is centered at ``t0`` and extends to +/-``tw``/2 around ``t0``.
    """

    @classmethod
    def from_start_and_stop_time(
            cls,
            start,
            stop,
            time_unit=None,
            **kwargs,
    ):
        """Constructs a BoxTimeFluxProfile instance from the given start and
        stop time.

        Parameters
        ----------
        start : float
            The start time of the box profile.
        stop : float
            The stop time of the box profile.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        **kwargs
            Additional keyword arguments, which are passed to the constructor
            of the :class:`BoxTimeFluxProfile` class.

        Returns
        -------
        profile : instance of BoxTimeFluxProfile
            The newly created instance of BoxTimeFluxProfile.
        """
        t0 = 0.5*(start + stop)
        tw = stop - start

        profile = cls(
            t0=t0,
            tw=tw,
            time_unit=time_unit,
            **kwargs)

        return profile

    def __init__(
            self,
            t0,
            tw,
            time_unit=None,
            **kwargs,
    ):
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
        t_stop = t0 + tw/2.

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            time_unit=time_unit,
            **kwargs)

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('t0', 'tw')

    @property
    def t0(self):
        """The time of the mid point of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return 0.5*(self._t_start + self._t_stop)

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
        return self._t_stop - self._t_start

    @tw.setter
    def tw(self, w):
        t0 = self.t0
        self._t_start = t0 - 0.5*w
        self._t_stop = t0 + 0.5*w

    @property
    def math_function_str(self):
        """The string representation of the mathematical function of this
        TimeFluxProfile instance.
        """
        t0 = self.t0
        tw = self.tw

        s = f'1 for t in [{t0:g}-{tw:g}/2; {t0:g}+{tw:g}/2], 0 otherwise'

        return s

    def __call__(
            self,
            t,
            unit=None,
    ):
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

        if (unit is not None) and (unit != self._time_unit):
            t = t * unit.to(self._time_unit)

        values = np.zeros((t.shape[0],), dtype=np.int8)
        m = (t >= self._t_start) & (t <= self._t_stop)
        values[m] = 1

        return values

    def cdf(
            self,
            t,
            unit=None,
    ):
        """Calculates the cumulative distribution function value for the given
        time values ``t``.

        Parameters
        ----------
        t : float | instance of numpy ndarray
            The (N_times,)-shaped numpy ndarray holding the time values for
            which to calculate the CDF values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values : instance of numpy ndarray
            The (N_times,)-shaped numpy ndarray holding the cumulative
            distribution function values for each time ``t``.
        """
        t = np.atleast_1d(t)

        if (unit is not None) and (unit != self._time_unit):
            t = t * unit.to(self._time_unit)

        t_start = self._t_start
        t_stop = self._t_stop

        values = np.zeros(t.size, dtype=np.float64)

        m = (t_start <= t) & (t <= t_stop)
        values[m] = (t[m] - t_start) / (t_stop - t_start)

        m = (t > t_stop)
        values[m] = 1

        return values

    def move(
            self,
            dt,
            unit=None,
    ):
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
        if (unit is not None) and (unit != self._time_unit):
            dt = dt * unit.to(self._time_unit)

        self._t_start += dt
        self._t_stop += dt

    def get_integral(
            self,
            t1,
            t2,
            unit=None,
    ):
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

        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = np.zeros((t1.shape[0],), dtype=np.float64)

        m = (t2 >= self._t_start) & (t1 <= self._t_stop)
        N = np.count_nonzero(m)

        t1 = np.max(np.vstack((t1[m], np.repeat(self._t_start, N))).T, axis=1)
        t2 = np.min(np.vstack((t2[m], np.repeat(self._t_stop, N))).T, axis=1)

        integral[m] = t2 - t1

        return integral


class GaussianTimeFluxProfile(
        TimeFluxProfile,
):
    """This class describes a gaussian-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the gaussian profile.
        sigma_t : float
            The one-sigma width of the gaussian profile.
    """

    def __init__(
            self,
            t0,
            sigma_t,
            tol=1e-12,
            time_unit=None,
            **kwargs):
        """Creates a new gaussian-shaped time flux profile instance.

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
        dt = np.sqrt(-2 * sigma_t**2 * np.log(tol))
        t_start = t0 - dt
        t_stop = t0 + dt

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            time_unit=time_unit,
            **kwargs)

        self.t0 = t0
        self.sigma_t = sigma_t

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('t0', 'sigma_t')

    @property
    def math_function_str(self):
        return 'exp(-(t-t0)^2/(2 sigma_t^2))'

    @property
    def t0(self):
        """The time of the mid point of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return 0.5*(self._t_start + self._t_stop)

    @t0.setter
    def t0(self, t):
        t = float_cast(
            t,
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
        sigma = float_cast(
            sigma,
            'The sigma_t property must be castable to type float!')
        self._sigma_t = sigma

    def __call__(
            self,
            t,
            unit=None,
    ):
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

        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        m = (t >= self.t_start) & (t < self.t_stop)

        s = self._sigma_t
        twossq = 2*s*s
        t0 = 0.5*(self._t_stop + self._t_start)
        dt = t[m] - t0

        values = np.zeros_like(t)
        values[m] = np.exp(-dt*dt/twossq)

        return values

    def cdf(
            self,
            t,
            unit=None,
    ):
        """Calculates the cumulative distribution function values for the given
        time values ``t``.

        Parameters
        ----------
        t : float | instance of numpy ndarray
            The (N_times,)-shaped numpy ndarray holding the time values for
            which to calculate the CDF values.
        unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values : instance of numpy ndarray
            The (N_times,)-shaped numpy ndarray holding the cumulative
            distribution function values for each time ``t``.
        """
        t = np.atleast_1d(t)

        if (unit is not None) and (unit != self._time_unit):
            t = t * unit.to(self._time_unit)

        t_start = self._t_start
        t_stop = self._t_stop

        values = np.zeros(t.size, dtype=np.float64)

        m = (t_start <= t) & (t <= t_stop)
        values[m] = (
            self.get_integral(t1=t_start, t2=t[m]) / self.get_total_integral()
        )

        m = (t > t_stop)
        values[m] = 1

        return values

    def move(
            self,
            dt,
            unit=None,
    ):
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
        if (unit is not None) and (unit != self._time_unit):
            dt = dt * unit.to(self._time_unit)

        self._t_start += dt
        self._t_stop += dt

    def get_integral(
            self,
            t1,
            t2,
            unit=None,
    ):
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
        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        t0 = 0.5*(self._t_stop + self._t_start)
        sigma_t = self._sigma_t

        c1 = np.sqrt(np.pi/2) * sigma_t
        c2 = np.sqrt(2) * sigma_t
        i1 = c1 * scipy.special.erf((t1 - t0)/c2)
        i2 = c1 * scipy.special.erf((t2 - t0)/c2)

        integral = i2 - i1

        return integral


class FluxModel(
        MathFunction,
        HasConfig,
        Model,
        metaclass=abc.ABCMeta,
):
    r"""Abstract base class for all flux models of the form

    .. math::

        \Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s).

    This base class defines the units used for the flux calculation. The unit
    of the flux is ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).

    At this point the functional form of the flux model is not yet defined.
    """
    @staticmethod
    def get_default_units(cfg):
        """Returns the configured default units for flux models.

        Parameters
        ----------
        cfg : instance of Config
            The instance of Config holding the local configuration.

        Returns
        -------
        units_dict : dict
            The dictionary holding the configured default units used for flux
            models.
        """
        return cfg['units']['defaults']['fluxes']

    def __init__(
            self,
            angle_unit=None,
            energy_unit=None,
            length_unit=None,
            time_unit=None,
            **kwargs):
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
        super().__init__(
            **kwargs)

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
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['angle']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property angle_unit must be of type '
                'astropy.units.UnitBase!')
        self._angle_unit = unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation.
        """
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['energy']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit = unit

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._length_unit

    @length_unit.setter
    def length_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['length']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property length_unit must be of type '
                'astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation.
        """
        return self._time_unit

    @time_unit.setter
    def time_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['time']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                'The property time_unit must be of type '
                'astropy.units.UnitBase!')
        self._time_unit = unit

    @property
    def unit_str(self):
        """The string representation of the flux unit.
        """
        if self.angle_unit == units.radian:
            angle_unit_sq = units.steradian
        else:
            angle_unit_sq = self.angle_unit**2

        s = (f'({self.energy_unit.to_string()}'
             f' {angle_unit_sq.to_string()}'
             f' {self.length_unit.to_string()}^2'
             f' {self.time_unit.to_string()})^-1')

        return s

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit.
        """
        if self.angle_unit == units.radian:
            angle_unit_sq = units.steradian
        else:
            angle_unit_sq = self.angle_unit**2

        s = (f'{self.energy_unit.to_string()}''$^{-1}$ '
             f'{angle_unit_sq.to_string()}''$^{-1}$ '
             f'{self.length_unit.to_string()}''$^{-2}$ '
             f'{self.time_unit.to_string()}''$^{-1}$')

        return s

    def __str__(self):
        """Pretty string representation of this class.
        """
        return f'{self.math_function_str} {self.unit_str}'

    @abc.abstractmethod
    def __call__(
            self,
            ra=None,
            dec=None,
            E=None,
            t=None,
            angle_unit=None,
            energy_unit=None,
            time_unit=None):
        """The call operator to retrieve a flux value for a given celestrial
        position, energy, and observation time.

        Parameters
        ----------
        ra : float | (Ncoord,)-shaped 1D numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        dec : float | (Ncoord,)-shaped 1D numpy ndarray of float
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

    def to_internal_flux_unit(self):
        """Calculates the conversion factor to convert the flux unit of this
        flux model instance to the SkyLLH internally used flux unit.

        Returns
        -------
        factor : float
            The conversion factor.
        """
        self_flux_unit = 1 / (
            self.angle_unit**2 *
            self.energy_unit *
            self.length_unit**2 *
            self.time_unit)

        internal_units = self._cfg['units']['internal']
        internal_flux_unit = 1 / (
            internal_units['angle']**2 *
            internal_units['energy'] *
            internal_units['length']**2 *
            internal_units['time'])

        factor = (self_flux_unit).to(internal_flux_unit).value

        return factor


class NullFluxModel(
        FluxModel,
):
    """This class provides a dummy flux model class, which can be used for
    testing purposes, in cases where an actual flux model is not required but
    the framework interface requires one.
    """
    def __init__(
            self,
            *args,
            cfg=None,
            **kwargs,
    ):
        """Creates a new instance of NullFluxModel.

        Parameters
        ----------
        cfg : instance of Config | None
            The instance of Config holding the local configuration. Since this
            flux model does nothing, this argument is optional. If not provided
            the default configuration is used.
        """
        if cfg is None:
            cfg = Config()

        super().__init__(
            *args,
            cfg=cfg,
            **kwargs)

    def math_function_str(self):
        """Since this is a dummy flux model, calling this method will raise a
        NotImplementedError.
        """
        raise NotImplementedError(
            f'The {classname(self)} flux model is a dummy flux model which has '
            'no math function prepresentation!')

    def __call__(self, *args, **kwargs):
        """Since this is a dummy flux model, calling this method will raise a
        NotImplementedError.
        """
        raise NotImplementedError(
            f'The {classname(self)} flux model is a dummy flux model and '
            'cannot be called!')


class FactorizedFluxModel(
        FluxModel,
):
    r"""This class describes a flux model where the spatial, energy, and time
    profiles of the source factorize. That means the flux can be written as:

    .. math::

        \Phi(\alpha,\delta,E,t | \vec{p}_\mathrm{s}) =
            \Phi_0
            \Psi(\alpha,\delta|\vec{p}_\mathrm{s})
            \epsilon(E|\vec{p}_\mathrm{s})
            T(t|\vec{p}_\mathrm{s})

    where, :math:`\Phi_0` is the normalization constant of the flux, and
    :math:`\Psi`, :math:`\epsilon`, and :math:`T` are the spatial, energy, and
    time profiles of the flux given the source parameters
    :math:`\vec{p}_\mathrm{s}`, respectively.
    """
    def __init__(
            self,
            Phi0,
            spatial_profile,
            energy_profile,
            time_profile,
            length_unit=None,
            **kwargs,
    ):
        """Creates a new factorized flux model.

        Parameters
        ----------
        Phi0 : float
            The flux normalization constant.
        spatial_profile : instance of SpatialFluxProfile | None
            The SpatialFluxProfile instance providing the spatial profile
            function of the flux.
            If set to None, an instance of UnitySpatialFluxProfile will be used,
            which represents the constant function 1.
        energy_profile : instance of EnergyFluxProfile | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : instance of TimeFluxProfile | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        length_unit : instance of astropy.units.UnitBase | None
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        """
        cfg = kwargs.get('cfg')

        self.Phi0 = Phi0

        if spatial_profile is None:
            spatial_profile = UnitySpatialFluxProfile(
                cfg=cfg)
        self.spatial_profile = spatial_profile

        if energy_profile is None:
            energy_profile = UnityEnergyFluxProfile(
                cfg=cfg)
        self.energy_profile = energy_profile

        if time_profile is None:
            time_profile = UnityTimeFluxProfile(
                cfg=cfg)
        self.time_profile = time_profile

        # The base class will set the default (internally used) flux unit, which
        # will be set automatically to the particular profile.
        super().__init__(
            angle_unit=self._spatial_profile.angle_unit,
            energy_unit=self._energy_profile.energy_unit,
            time_unit=self._time_profile.time_unit,
            length_unit=length_unit,
            **kwargs,
        )

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('Phi0',)

    @property
    def Phi0(self):
        """The flux normalization constant.
        The unit of this normalization constant is
        ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).
        """
        return self._Phi0

    @Phi0.setter
    def Phi0(self, v):
        v = float_cast(
            v,
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
        if not isinstance(profile, SpatialFluxProfile):
            raise TypeError(
                'The spatial_profile property must be None, or an '
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
        if not isinstance(profile, EnergyFluxProfile):
            raise TypeError(
                'The energy_profile property must be None, or an '
                'instance of EnergyFluxProfile!')
        self._energy_profile = profile

    @property
    def time_profile(self):
        """Instance of TimeFluxProfile describing the time profile of the flux.
        """
        return self._time_profile

    @time_profile.setter
    def time_profile(self, profile):
        if not isinstance(profile, TimeFluxProfile):
            raise TypeError(
                'The time_profile property must be None, or an '
                'instance of TimeFluxProfile!')
        self._time_profile = profile

    @property
    def math_function_str(self):
        """The string showing the mathematical function of the flux.
        """
        s = f'{self._Phi0:.3e}'

        spatial_str = self._spatial_profile.math_function_str
        if spatial_str is not None:
            s += f' * {spatial_str}'
        energy_str = self._energy_profile.math_function_str
        if energy_str is not None:
            s += f' * {energy_str}'

        time_str = self._time_profile.math_function_str
        if time_str is not None:
            s += f' * {time_str}'

        return s

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
    def param_names(self):
        """The tuple holding the names of the math function's parameters. This
        is the total set of parameter names for all flux profiles of this
        FactorizedFluxModel instance.
        """
        pnames = list(super(FactorizedFluxModel, type(self)).param_names)
        pnames += self._spatial_profile.param_names
        pnames += self._energy_profile.param_names
        pnames += self._time_profile.param_names

        return tuple(pnames)

    @param_names.setter
    def param_names(self, names):
        super(FactorizedFluxModel, type(self)).param_names.fset(self, names)

    def __call__(
            self,
            ra=None,
            dec=None,
            E=None,
            t=None,
            angle_unit=None,
            energy_unit=None,
            time_unit=None,
    ):
        """Calculates the flux values for the given celestrial positions,
        energies, and observation times.

        Parameters
        ----------
        ra: float | (Ncoord,)-shaped 1d numpy ndarray of float | None
            The right-ascention coordinate for which to retrieve the flux value.
        dec : float | (Ncoord,)-shaped 1d numpy ndarray of float | None
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float | None
            The energy for which to retrieve the flux value.
        t : float | (Ntime,)-shaped 1d numpy ndarray of float | None
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
        if (ra is not None) and (dec is not None):
            spatial_profile_values = self._spatial_profile(
                ra, dec, unit=angle_unit)
        else:
            spatial_profile_values = np.array([1])

        if E is not None:
            energy_profile_values = self._energy_profile(
                E, unit=energy_unit)
        else:
            energy_profile_values = np.array([1])

        if t is not None:
            time_profile_values = self._time_profile(
                t, unit=time_unit)
        else:
            time_profile_values = np.array([1])

        flux = (
            self._Phi0 *
            spatial_profile_values[:, np.newaxis, np.newaxis] *
            energy_profile_values[np.newaxis, :, np.newaxis] *
            time_profile_values[np.newaxis, np.newaxis, :]
        )

        return flux

    def get_param(self, name):
        """Retrieves the value of the given parameter. It returns ``np.nan`` if
        the parameter does not exist.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        value : float | np.nan
            The value of the parameter.
        """
        for obj in (
                super(),
                self._spatial_profile,
                self._energy_profile,
                self._time_profile):
            value = obj.get_param(name=name)
            if not np.isnan(value):
                return value

        return np.nan

    def set_params(self, pdict):
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

        updated |= super().set_params(pdict)

        updated |= self._spatial_profile.set_params(pdict)
        updated |= self._energy_profile.set_params(pdict)
        updated |= self._time_profile.set_params(pdict)

        return updated


class PointlikeFFM(
        FactorizedFluxModel,
        IsPointlike,
):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point. This class provides the base class for a flux
    model of a point-like source.
    """
    def __init__(
            self,
            Phi0,
            energy_profile,
            time_profile,
            ra=None,
            dec=None,
            angle_unit=None,
            length_unit=None,
            **kwargs,
    ):
        """Creates a new factorized flux model for a point-like source.

        Parameters
        ----------
        Phi0 : float
            The flux normalization constant in unit of flux.
        energy_profile : instance of EnergyFluxProfile | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : instance of TimeFluxProfile | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        ra : float | None
            The right-ascention of the point.
        dec : float | None
            The declination of the point.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit for angles used for the flux unit.
            If set to ``None``, the configured internal angle unit is used.
        length_unit : instance of astropy.units.UnitBase | None
            The unit for length used for the flux unit.
            If set to ``None``, the configured internal length unit is used.
        """
        spatial_profile = PointSpatialFluxProfile(
            cfg=kwargs.get('cfg'),
            ra=ra,
            dec=dec,
            angle_unit=angle_unit)

        super().__init__(
            Phi0=Phi0,
            spatial_profile=spatial_profile,
            energy_profile=energy_profile,
            time_profile=time_profile,
            length_unit=length_unit,
            ra_func_instance=spatial_profile,
            get_ra_func=type(spatial_profile).ra.fget,
            set_ra_func=type(spatial_profile).ra.fset,
            dec_func_instance=spatial_profile,
            get_dec_func=type(spatial_profile).dec.fget,
            set_dec_func=type(spatial_profile).dec.fset,
            **kwargs)

    @property
    def unit_str(self):
        """The string representation of the flux unit.
        """
        # Note:
        #    For a point-like differential flux, there is no solid-angle
        #    element.
        s = (f'({self.energy_unit.to_string()}'
             f' {self.length_unit.to_string()}^2'
             f' {self.time_unit.to_string()})^-1')

        return s

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit.
        """
        # Note:
        #    For a point-like differential flux, there is no solid-angle
        #    element.
        s = (f'{self.energy_unit.to_string()}''$^{-1}$ '
             f'{self.length_unit.to_string()}''$^{-2}$ '
             f'{self.time_unit.to_string()}''$^{-1}$')

        return s


class SteadyPointlikeFFM(
        PointlikeFFM,
):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point and the time profile as constant 1. It is
    derived from the ``PointlikeFFM`` class.
    """
    def __init__(
            self,
            Phi0,
            energy_profile,
            ra=None,
            dec=None,
            angle_unit=None,
            length_unit=None,
            time_unit=None,
            **kwargs,
    ):
        """Creates a new factorized flux model for a point-like source with no
        time dependance.

        Parameters
        ----------
        Phi0 : float
            The flux normalization constant.
        energy_profile : instance of EnergyFluxProfile | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        ra : float | None
            The right-ascention of the point.
        dec : float | None
            The declination of the point.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit for angles used for the flux unit.
            If set to ``None``, the configured default angle unit for fluxes
            is used.
        length_unit : instance of astropy.units.UnitBase | None
            The unit for length used for the flux unit.
            If set to ``None``, the configured default length unit for fluxes
            is used.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes
            is used.
        """
        time_profile = UnityTimeFluxProfile(
            cfg=kwargs.get('cfg'),
            time_unit=time_unit)

        super().__init__(
            Phi0=Phi0,
            ra=ra,
            dec=dec,
            energy_profile=energy_profile,
            time_profile=time_profile,
            angle_unit=angle_unit,
            length_unit=length_unit,
            **kwargs)
