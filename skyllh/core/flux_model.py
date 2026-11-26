r"""The `flux_model` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a mathematical function for the differential flux:

.. math::

    \frac{d^4\Phi(\alpha,\delta,E,t | \vec{p}_{\mathrm{s}})}{\mathrm{d}A
    \mathrm{d}\Omega \mathrm{d}E \mathrm{d}t}

"""

import abc
from collections.abc import Callable

import numpy as np
import scipy.special
import scipy.stats
from astropy import (
    units,
)
from scipy.integrate import (
    quad,
)

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

# numpy deprecated the np.trapz function in favor of np.trapezoid, but to maintain compatibility with older numpy
# versions, we define _trapezoid as np.trapz if np.trapezoid is not available.
_trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz  # pyright: ignore[reportAttributeAccessIssue]


class FluxProfile(MathFunction, HasConfig, metaclass=abc.ABCMeta):
    """The abstract base class for a flux profile math function."""

    def __init__(
        self,
        **kwargs,
    ):
        """Creates a new FluxProfile instance."""
        super().__init__(**kwargs)


class SpatialFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for a spatial flux profile function."""

    def __init__(self, angle_unit: units.UnitBase | None = None, **kwargs):
        """Creates a new SpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(**kwargs)

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
            raise TypeError('The property angle_unit must be of type astropy.units.UnitBase!')
        self._angle_unit = unit

    @abc.abstractmethod
    def __call__(
        self, ra: float | np.ndarray, dec: float | np.ndarray, unit: units.UnitBase | None = None
    ) -> np.ndarray:
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        ra
            The right-ascention coordinate.
        dec
            The declination coordinate.
        unit
            The unit of the given celestrial angles.
            If ``None``, the set angle unit of this SpatialFluxProfile is
            assumed.

        Returns
        -------
        values
            The spatial profile values.
        """


class UnitySpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1 for any spatial
    coordinates.
    """

    def __init__(
        self,
        angle_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new UnitySpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(angle_unit=angle_unit, **kwargs)

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance.
        """
        return '1'

    def __call__(
        self, ra: float | np.ndarray, dec: float | np.ndarray, unit: units.UnitBase | None = None
    ) -> np.ndarray:
        """Returns 1 as numpy ndarray in same shape as ra and dec.

        Parameters
        ----------
        ra
            The right-ascention coordinate.
        dec
            The declination coordinate.
        unit
            The unit of the given celestrial angles.
            By the definition of this class this argument is ignored.

        Returns
        -------
        values
            1 in same shape as ra and dec.
        """
        (ra, dec) = np.atleast_1d(ra, dec)
        if ra.shape != dec.shape:
            raise ValueError('The ra and dec arguments must be of the same shape!')

        return np.ones_like(ra)


class PointSpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for a delta function at the celestrical coordinate
    (ra, dec).
    """

    def __init__(
        self,
        ra: float | None,
        dec: float | None,
        angle_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new spatial flux profile for a point at equatorial
        coordinate (ra, dec).

        Parameters
        ----------
        ra
            The right-ascention of the point.
            In case it is None, the evaluation of this spatial flux profile will
            return zero, unless evaluated for ra=None.
        dec
            The declination of the point.
            In case it is None, the evaluation of this spatial flux profile will
            return zero, unless evaluated for dec=None.
        angle_unit
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super().__init__(angle_unit=angle_unit, **kwargs)

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
        v = float_cast(v, 'The ra property must be castable to type float!', allow_None=True)
        self._ra = v

    @property
    def dec(self):
        """The declination of the point.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._dec

    @dec.setter
    def dec(self, v):
        v = float_cast(v, 'The dec property must be castable to type float!', allow_None=True)
        self._dec = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance. It is None, if the right-ascention
        or declination property is set to None.
        """
        if (self._ra is None) or (self._dec is None):
            return None

        s = f'delta(ra-{self._ra:g}{self._angle_unit})*delta(dec-{self._dec:g}{self._angle_unit})'

        return s

    def __call__(
        self, ra: float | np.ndarray, dec: float | np.ndarray, unit: units.UnitBase | None = None
    ) -> np.ndarray:
        """Returns a numpy ndarray in same shape as ra and dec with 1 if
        `ra` equals `self.ra` and `dec` equals `self.dec`, and 0 otherwise.

        Parameters
        ----------
        ra
            The right-ascention coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        dec
            The declination coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        unit
            The unit of the given celestrial angles.
            If set to ``None``, the set angle unit of this SpatialFluxProfile
            instance is assumed.

        Returns
        -------
        value
            A numpy ndarray in same shape as ra and dec with 1 if `ra`
            equals `self.ra` and `dec` equals `self.dec`, and 0 otherwise.
        """
        (ra, dec) = np.atleast_1d(ra, dec)
        if ra.shape != dec.shape:
            raise ValueError('The ra and dec arguments must be of the same shape!')

        if (unit is not None) and (unit != self._angle_unit):
            angle_unit_conv_factor = unit.to(self._angle_unit)
            ra = ra * angle_unit_conv_factor
            dec = dec * angle_unit_conv_factor

        value = ((ra == self._ra) & (dec == self._dec)).astype(np.int8, copy=False)

        return value


class EnergyFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for an energy flux profile function."""

    def __init__(self, energy_unit: units.UnitBase | None = None, **kwargs):
        """Creates a new energy flux profile with a given energy unit to be used
        for flux calculation.

        Parameters
        ----------
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(**kwargs)

        # Set the energy unit.
        self.energy_unit = energy_unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux profile calculation."""
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['energy']
        if not isinstance(unit, units.UnitBase):
            raise TypeError('The property energy_unit must be of type astropy.units.UnitBase!')
        self._energy_unit = unit

    @abc.abstractmethod
    def __call__(self, E: float | np.ndarray, unit: units.UnitBase | None = None) -> np.ndarray:
        """This method is supposed to return the energy profile value for the
        given energy value.

        Parameters
        ----------
        E
            The energy value for which to retrieve the energy profile value.
        unit
            The unit of the given energy.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """

    def get_integral(
        self,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """This is the default implementation for calculating the integral value
        of this energy flux profile in the range ``[E1, E2]``.

        .. note::

            This implementation utilizes the ``scipy.integrate.quad`` function
            to perform a generic numeric integration. Hence, this implementation
            is slow and should be reimplemented by the derived class if an
            analytic integral form is available.

        Parameters
        ----------
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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

        for i, (E1_i, E2_i) in enumerate(zip(E1, E2, strict=True)):
            integral[i] = quad(self, E1_i, E2_i, full_output=True)[0]

        return integral

    #: The number of grid points per energy range used by the numerical default
    #: implementation of :meth:`get_inverse_integral`.
    inverse_integral_n_grid = 1000

    def _prepare_inverse_integral_args(
        self,
        u: float | np.ndarray,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Validates and broadcasts the arguments of :meth:`get_inverse_integral`
        and converts the energies into the energy unit of this profile.

        Returns
        -------
        u, E1, E2
            The broadcast (n,)-shaped arrays, with E1 and E2 in the energy unit
            of this profile.
        unit_conv_factor
            The factor converting from the given unit into the energy unit of
            this profile.
        """
        u, E1, E2 = np.broadcast_arrays(
            np.atleast_1d(np.asarray(u, dtype=np.float64)),
            np.atleast_1d(np.asarray(E1, dtype=np.float64)),
            np.atleast_1d(np.asarray(E2, dtype=np.float64)),
        )
        if u.ndim != 1:
            raise ValueError(f'The arguments u, E1, and E2 must broadcast to a 1-dimensional shape, not {u.shape}!')
        if np.any(~np.isfinite(u)) or np.any(u < 0) or np.any(u > 1):
            raise ValueError('All values of u must lie within [0, 1]!')
        if np.any(~np.isfinite(E1)) or np.any(~np.isfinite(E2)):
            raise ValueError('All values of E1 and E2 must be finite!')
        if np.any(E1 > E2):
            raise ValueError('All values of E1 must be smaller than or equal to the values of E2!')

        unit_conv_factor = 1.0
        if (unit is not None) and (unit != self._energy_unit):
            unit_conv_factor = float(unit.to(self._energy_unit))  # pyright: ignore[reportArgumentType]

        return (u, E1 * unit_conv_factor, E2 * unit_conv_factor, unit_conv_factor)

    def get_inverse_integral(
        self,
        u: float | np.ndarray,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        r"""Computes the energy :math:`E \in [E_1, E_2]` at which the integral of
        this energy flux profile from :math:`E_1` reaches the fraction ``u`` of
        its integral over :math:`[E_1, E_2]`:

        .. math::

            \int_{E_1}^{E} \Phi(E') \mathrm{d}E' = u \int_{E_1}^{E_2} \Phi(E') \mathrm{d}E'

        Hence, for ``u`` uniformly distributed in [0, 1], the returned energies
        follow this energy flux profile within [E1, E2].

        .. note::

            This default implementation tabulates the cumulative integral of the
            profile on a grid of :attr:`inverse_integral_n_grid` points per
            energy range, which is logarithmic for E1 > 0, and interpolates it
            linearly. Derived classes should reimplement this method if an
            analytic form is available.

        Parameters
        ----------
        u
            The integral fractions, each within [0, 1].
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies and the returned energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        E
            The (n,)-shaped numpy ndarray holding the energies, where n is the
            broadcast length of ``u``, ``E1``, and ``E2``.

        Raises
        ------
        ValueError
            If the profile integrates to zero over an energy range with E1 < E2.
        """
        (u, E1, E2, unit_conv_factor) = self._prepare_inverse_integral_args(u, E1, E2, unit)

        E = np.empty_like(u)

        # Tabulate the cumulative integral once per distinct energy range.
        ranges, range_idxs = np.unique(np.stack((E1, E2), axis=1), axis=0, return_inverse=True)
        range_idxs = range_idxs.ravel()
        for idx, (E1_i, E2_i) in enumerate(ranges):
            m = range_idxs == idx
            if E1_i == E2_i:
                E[m] = E1_i
                continue

            if E1_i > 0:
                grid = np.geomspace(E1_i, E2_i, self.inverse_integral_n_grid)
            else:
                grid = np.linspace(E1_i, E2_i, self.inverse_integral_n_grid)
            values = np.asarray(self(grid), dtype=np.float64)
            cum = np.concatenate(([0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))))
            if not cum[-1] > 0:
                raise ValueError(
                    f'The energy flux profile {classname(self)} does not have a positive integral over the energy '
                    f'range [{E1_i:g}, {E2_i:g}] {self._energy_unit}!'
                )
            E[m] = np.interp(u[m], cum / cum[-1], grid)

        return E / unit_conv_factor


class UnityEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for the constant function 1."""

    def __init__(self, energy_unit: units.UnitBase | None = None, **kwargs):
        """Creates a new UnityEnergyFluxProfile instance.

        Parameters
        ----------
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(energy_unit=energy_unit, **kwargs)

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this energy flux profile.
        """
        return '1'

    def __call__(self, E: float | np.ndarray, unit: units.UnitBase | None = None) -> np.ndarray:
        """Returns 1 as numpy ndarray in some shape as E.

        Parameters
        ----------
        E
            The energy value for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values
            1 in same shape as E.
        """
        E = np.atleast_1d(E)

        values = np.ones_like(E, dtype=np.int8)

        return values

    def get_integral(
        self, E1: float | np.ndarray, E2: float | np.ndarray, unit: units.UnitBase | None = None
    ) -> np.ndarray:
        """Computes the integral of this energy flux profile in the range
        [``E1``, ``E2``], which by definition is ``E2 - E1``.

        Parameters
        ----------
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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

    def get_inverse_integral(
        self,
        u: float | np.ndarray,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Computes the energy at which the integral of this energy flux profile
        from ``E1`` reaches the fraction ``u`` of its integral over
        [``E1``, ``E2``], which by definition is ``E1 + u * (E2 - E1)``.

        Parameters
        ----------
        u
            The integral fractions, each within [0, 1].
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies and the returned energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        E
            The (n,)-shaped numpy ndarray holding the energies.
        """
        (u, E1, E2, unit_conv_factor) = self._prepare_inverse_integral_args(u, E1, E2, unit)

        E = E1 + u * (E2 - E1)

        return E / unit_conv_factor


class PowerLawEnergyFluxProfile(
    EnergyFluxProfile,
):
    r"""Energy flux profile for a power law profile with a reference energy
    ``E0`` and a spectral index ``gamma``.

    .. math::

        (E / E_0)^{-\gamma}

    """

    def __init__(self, E0: float, gamma: float, energy_unit: units.UnitBase | None = None, **kwargs):
        """Creates a new power law flux profile with the reference energy ``E0``
        and spectral index ``gamma``.

        Parameters
        ----------
        E0
            The reference energy.
        gamma
            The spectral index.
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(energy_unit=energy_unit, **kwargs)

        self.E0 = E0
        self.gamma = gamma

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = (
            'E0',
            'gamma',
        )

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0

    @E0.setter
    def E0(self, v):
        v = float_cast(v, 'Property E0 must be castable to type float!')
        self._E0 = v

    @property
    def gamma(self):
        """The spectral index."""
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        v = float_cast(v, 'Property gamma must be castable to type float!')
        self._gamma = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s = f'(E / ({self._E0:g} {self._energy_unit}))^-{self._gamma:g}'

        return s

    def __call__(self, E: float | np.ndarray, unit: units.UnitBase | None = None) -> np.ndarray:
        """Returns the power law values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E
            The energy value for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        value = np.power(E / self._E0, -self._gamma)

        return value

    def get_integral(
        self, E1: float | np.ndarray, E2: float | np.ndarray, unit: units.UnitBase | None = None
    ) -> np.ndarray:
        """Computes the integral value of this power-law energy flux profile in
        the range ``[E1, E2]``.

        Parameters
        ----------
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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
            integral = self._E0 * np.log(E2 / E1)
            return integral

        integral = np.power(self._E0, gamma) / (1 - gamma) * (np.power(E2, 1 - gamma) - np.power(E1, 1 - gamma))

        return integral

    def get_inverse_integral(
        self,
        u: float | np.ndarray,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        r"""Computes the energy at which the integral of this power-law energy
        flux profile from ``E1`` reaches the fraction ``u`` of its integral over
        [``E1``, ``E2``]:

        .. math::

            E = E_1 \left[ 1 + u \left( (E_2/E_1)^{1-\gamma} - 1 \right) \right]^{1/(1-\gamma)}

        which becomes :math:`E_1 (E_2/E_1)^u` for :math:`\gamma = 1`.

        Parameters
        ----------
        u
            The integral fractions, each within [0, 1].
        E1
            The lower energy bound of the integration. It must be positive.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies and the returned energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        E
            The (n,)-shaped numpy ndarray holding the energies.
        """
        # Derived profiles that change the function, e.g. by a cutoff, are no
        # pure power law, so they use the numerical default implementation.
        if type(self).__call__ is not PowerLawEnergyFluxProfile.__call__:
            return super().get_inverse_integral(u, E1, E2, unit=unit)

        (u, E1, E2, unit_conv_factor) = self._prepare_inverse_integral_args(u, E1, E2, unit)
        if np.any(E1 <= 0):
            raise ValueError('All values of E1 must be positive for a power-law energy flux profile!')

        a = 1 - self._gamma
        log_ratio = np.log(E2 / E1)

        # expm1 and log1p keep the result accurate for a close to 0 and for
        # narrow energy ranges.
        E = E1 * np.exp(u * log_ratio) if a == 0 else E1 * np.exp(np.log1p(u * np.expm1(a * log_ratio)) / a)

        # Rounding must not push the energies outside [E1, E2].
        E = np.clip(E, E1, E2)

        return E / unit_conv_factor


class CutoffPowerLawEnergyFluxProfile(
    PowerLawEnergyFluxProfile,
):
    r"""Cut-off power law energy flux profile of the form

    .. math::

        (E / E_0)^{-\gamma} \exp(-E/E_{\mathrm{cut}})

    """

    def __init__(
        self,
        E0: float,
        gamma: float,
        Ecut: float,
        energy_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new cut-off power law flux profile with the reference
        energy ``E0``, spectral index ``gamma``, and cut-off energy ``Ecut``.

        Parameters
        ----------
        E0
            The reference energy.
        gamma
            The spectral index.
        Ecut
            The cut-off energy.
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(E0=E0, gamma=gamma, energy_unit=energy_unit, **kwargs)

        self.Ecut = Ecut

    @property
    def Ecut(self):
        """The energy cut value."""
        return self._Ecut

    @Ecut.setter
    def Ecut(self, v):
        v = float_cast(v, 'The Property Ecut  must be castable to type float!')
        self._Ecut = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s = f'(E / ({self._E0:g} {self._energy_unit}))^-{self._gamma:g} exp(-E / ({self._Ecut:g} {self._energy_unit}))'

        return s

    def __call__(
        self,
        E: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Returns the cut-off power law values for the given energies as
        numpy ndarray in the same shape as E.

        Parameters
        ----------
        E
            The energy value(s) for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        values = super().__call__(E=E, unit=None)
        values *= np.exp(-E / self._Ecut)

        return values

    def get_integral(
        self,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """This falls back to the default implementation for the CutoffPowerLawEnergyFluxProfile
        to avoid using the inhereted function
        .. note::

            This implementation utilizes the ``scipy.integrate.quad`` function
            to perform a generic numeric integration. Hence, this implementation
            is slow and should be reimplemented by the derived class if an
            analytic integral form is available.

        Parameters
        ----------
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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

        for i, (E1_i, E2_i) in enumerate(zip(E1, E2, strict=True)):
            # integration for log(energy) for hopefully better numerical stability
            tmp_e = np.linspace(np.log10(E1_i), np.log10(E2_i), 5000)
            tmp_int = _trapezoid(np.log(10) * self(10**tmp_e) * 10**tmp_e, tmp_e)

            # make sure it is always positive (probably not an issue any more with np.trapz.
            # used to be an issue using the spline integrate self.function.integrate)
            integral[i] = max(0.0, tmp_int)

        return integral


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
        E0: float,
        alpha: float,
        beta: float,
        energy_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        E0
            The reference energy.
        alpha
            The alpha parameter of the log-parabola spectral index.
        beta
            The beta parameter of the log-parabola spectral index.
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(E0=E0, gamma=np.nan, energy_unit=energy_unit, **kwargs)

        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self):
        """The alpha parameter of the log-parabola spectral index."""
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        v = float_cast(v, 'Property alpha must be castable to type float!')
        self._alpha = v

    @property
    def beta(self):
        """The beta parameter of the log-parabola spectral index."""
        return self._beta

    @beta.setter
    def beta(self, v):
        v = float_cast(v, 'Property beta must be castable to type float!')
        self._beta = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of this energy flux profile
        instance.
        """
        s_E0 = f'{self._E0:g} {self._energy_unit}'
        s = f'(E / {s_E0})^(-({self._alpha:g} + {self._beta:g} log(E / {s_E0})))'

        return s

    def __call__(
        self,
        E: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Returns the log-parabola power-law values for the given energies as
        numpy ndarray in the same shape as E.

        Parameters
        ----------
        E
            The energy value(s) for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        values = np.power(E / self._E0, -self._alpha - self._beta * np.log(E / self._E0))

        return values

    #: The largest ratio between the summed magnitudes of the terms of the
    #: closed-form integral and the integral itself. Above it, rounding errors
    #: could become significant, and the numerical default is used instead.
    _closed_form_max_cancellation = 1e4

    def _g(self, y: np.ndarray) -> np.ndarray:
        r"""The exponent :math:`g(y) = (1 - \alpha) y - \beta y^2` of the
        integrand :math:`\Phi(E) \mathrm{d}E = E_0 e^{g(y)} \mathrm{d}y` with
        :math:`y = \ln(E / E_0)`.
        """
        return y * ((1 - self._alpha) - self._beta * y)

    def _get_scaled_y_integral(
        self, y_a: np.ndarray, y_b: np.ndarray, g_ref: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Computes the integral of :math:`e^{g(y)}` over [``y_a``, ``y_b``],
        with ``y_a <= y_b``, for :math:`\beta > 0`.

        Completing the square gives a Gaussian in y with mean
        :math:`\mu = (1 - \alpha) / (2\beta)`, so the integral is a difference
        of normal tail probabilities. Each tail is evaluated as
        :math:`\frac{1}{2} \mathrm{erfcx}(x) e^{g(y)}` with
        :math:`x = \pm\sqrt{\beta}(y - \mu)`, which neither overflows nor
        underflows for small :math:`\beta`.

        Returns
        -------
        integral
            The integral divided by :math:`\sqrt{\pi / \beta} e^{g_\mathrm{ref}}`.
        cancellation
            The summed magnitudes of the terms divided by the integral.
        g_ref
            The scale exponent, the maximum of g over the range unless given.
        """
        beta = self._beta
        mu = (1 - self._alpha) / (2 * beta)
        sqrt_beta = np.sqrt(beta)

        x_a = sqrt_beta * (y_a - mu)
        x_b = sqrt_beta * (y_b - mu)
        g_a = self._g(y_a)
        g_b = self._g(y_b)

        straddle = (x_a < 0) & (x_b > 0)
        # g(mu) = beta * mu^2.
        g_mu = np.where(straddle, beta * mu**2, -np.inf)
        if g_ref is None:
            g_ref = np.maximum(np.maximum(g_a, g_b), g_mu)

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            # The tails below (L) and above (U) y, both times exp(beta mu^2).
            L_a = 0.5 * scipy.special.erfcx(-x_a) * np.exp(g_a - g_ref)
            L_b = 0.5 * scipy.special.erfcx(-x_b) * np.exp(g_b - g_ref)
            U_a = 0.5 * scipy.special.erfcx(x_a) * np.exp(g_a - g_ref)
            U_b = 0.5 * scipy.special.erfcx(x_b) * np.exp(g_b - g_ref)
            M = np.exp(g_mu - g_ref)

            lower = x_b <= 0
            upper = x_a >= 0
            integral = np.where(lower, L_b - L_a, np.where(upper, U_a - U_b, M - L_a - U_b))
            terms = np.where(lower, L_a + L_b, np.where(upper, U_a + U_b, M + L_a + U_b))
            cancellation = np.where(y_a == y_b, 1.0, terms / integral)

        return (integral, cancellation, g_ref)

    def _get_gaussian_inverse_start(self, u: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
        """Returns a start value for the inverse integral in y, the inverse of
        the Gaussian CDF in y, evaluated on the side of the mean where the
        normal probabilities keep their precision. Where these probabilities
        underflow, which happens for small beta, it returns the linear guess.
        """
        mu = (1 - self._alpha) / (2 * self._beta)
        sigma = 1 / np.sqrt(2 * self._beta)
        z_a = (y_a - mu) / sigma
        z_b = (y_b - mu) / sigma

        # Mirror ranges above the mean, so the probabilities are lower tails.
        upper = z_a > 0
        p_a = np.where(upper, scipy.special.ndtr(-z_b), scipy.special.ndtr(z_a))
        p_b = np.where(upper, scipy.special.ndtr(-z_a), scipy.special.ndtr(z_b))
        with np.errstate(divide='ignore', invalid='ignore'):
            z = scipy.special.ndtri(np.where(upper, p_b - u * (p_b - p_a), p_a + u * (p_b - p_a)))
        y = mu + sigma * np.where(upper, -z, z)

        y_linear = y_a + u * (y_b - y_a)
        return np.where(np.isfinite(y) & (y >= y_a) & (y <= y_b) & (p_b > p_a), y, y_linear)

    def _get_numerical_y_integral(self, y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
        r"""Integrates :math:`e^{g(y)}` over [``y_a``, ``y_b``] numerically."""
        return np.array(
            [
                quad(lambda y: float(np.exp(self._g(y))), y_a_i, y_b_i, epsabs=0, epsrel=1e-12, limit=200)[0]
                for (y_a_i, y_b_i) in zip(y_a, y_b, strict=True)
            ],
            dtype=np.float64,
        )

    def get_integral(
        self,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        r"""Computes the integral of this log-parabola energy flux profile in the
        range ``[E1, E2]``.

        With :math:`y = \ln(E / E_0)` the integrand is a Gaussian in y, so the
        integral has a closed form in terms of the error function. For
        :math:`\beta = 0` it is the power-law integral with
        :math:`\gamma = \alpha`. For :math:`\beta < 0`, and where the closed
        form would lose precision to cancellation, which happens for an almost
        flat spectrum, the numerical default implementation is used.

        Parameters
        ----------
        E1
            The lower energy bound of the integration. It must be positive.
        E2
            The upper energy bound of the integration. It must be positive.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
            The integral values of the given integral ranges.
        """
        E1, E2 = np.broadcast_arrays(
            np.atleast_1d(np.asarray(E1, dtype=np.float64)), np.atleast_1d(np.asarray(E2, dtype=np.float64))
        )
        if (unit is not None) and (unit != self._energy_unit):
            unit_conv_factor = float(unit.to(self._energy_unit))  # pyright: ignore[reportArgumentType]
            E1 = E1 * unit_conv_factor
            E2 = E2 * unit_conv_factor
        if np.any(E1 <= 0) or np.any(E2 <= 0):
            raise ValueError('All values of E1 and E2 must be positive for a log-parabola energy flux profile!')

        # Integrate over the ordered range and restore the sign afterwards.
        sign = np.where(E1 <= E2, 1.0, -1.0)
        y_a = np.log(np.minimum(E1, E2) / self._E0)
        y_b = np.log(np.maximum(E1, E2) / self._E0)

        if self._beta < 0:
            return sign * self._E0 * self._get_numerical_y_integral(y_a, y_b)

        if self._beta == 0:
            a = 1 - self._alpha
            integral = y_b - y_a if a == 0 else np.exp(a * y_a) * np.expm1(a * (y_b - y_a)) / a
            return sign * self._E0 * integral

        (integral, cancellation, g_ref) = self._get_scaled_y_integral(y_a, y_b)
        integral = self._E0 * np.sqrt(np.pi / self._beta) * np.exp(g_ref) * integral

        m = ~(cancellation <= self._closed_form_max_cancellation)
        if np.any(m):
            integral[m] = self._E0 * self._get_numerical_y_integral(y_a[m], y_b[m])

        return sign * integral

    def get_inverse_integral(
        self,
        u: float | np.ndarray,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        r"""Computes the energy at which the integral of this log-parabola energy
        flux profile from ``E1`` reaches the fraction ``u`` of its integral over
        [``E1``, ``E2``].

        The closed-form integral (see :meth:`get_integral`) is inverted with
        Newton's method, falling back to bisection, to machine precision. For
        :math:`\beta = 0` the power-law inverse is used. For
        :math:`\beta < 0`, and where the closed form would lose precision to
        cancellation, the numerical default implementation is used.

        Parameters
        ----------
        u
            The integral fractions, each within [0, 1].
        E1
            The lower energy bound of the integration. It must be positive.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies and the returned energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        E
            The (n,)-shaped numpy ndarray holding the energies.
        """
        (u, E1, E2, unit_conv_factor) = self._prepare_inverse_integral_args(u, E1, E2, unit)
        if np.any(E1 <= 0):
            raise ValueError('All values of E1 must be positive for a log-parabola energy flux profile!')

        if self._beta < 0:
            return EnergyFluxProfile.get_inverse_integral(self, u, E1, E2) / unit_conv_factor

        y_a = np.log(E1 / self._E0)
        y_b = np.log(E2 / self._E0)

        if self._beta == 0:
            a = 1 - self._alpha
            y = y_a + u * (y_b - y_a) if a == 0 else y_a + np.log1p(u * np.expm1(a * (y_b - y_a))) / a
            return np.clip(self._E0 * np.exp(y), E1, E2) / unit_conv_factor

        (total, cancellation, g_ref) = self._get_scaled_y_integral(y_a, y_b)
        target = u * total
        deriv_factor = np.sqrt(self._beta / np.pi)

        # Solve log(integral from y_a to y) = log(target) for y with Newton's
        # method, falling back to bisection. The start value is the inverse of
        # the Gaussian CDF where it is representable, so Newton's method only
        # needs to polish it to machine precision.
        lo = y_a.copy()
        hi = y_b.copy()
        y = self._get_gaussian_inverse_start(u, y_a, y_b)
        # Ranges where the closed form loses precision use the numerical
        # default below instead.
        use_numerical = ~(cancellation <= self._closed_form_max_cancellation) & (E1 < E2)
        active = (u > 0) & (u < 1) & (y_a < y_b) & ~use_numerical
        with np.errstate(divide='ignore'):
            log_target = np.log(target)
        for _ in range(100):
            if not np.any(active):
                break
            ya, yy, lt = y_a[active], y[active], log_target[active]
            integral = self._get_scaled_y_integral(ya, yy, g_ref=g_ref[active])[0]
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                f = np.log(integral) - lt
                y_new = yy - f * integral / (deriv_factor * np.exp(self._g(yy) - g_ref[active]))
            lo_a = np.where(f < 0, yy, lo[active])
            hi_a = np.where(f > 0, yy, hi[active])
            # A y whose residual is at the rounding level is the solution.
            eps = np.finfo(np.float64).eps
            solved = np.abs(f) <= 16 * eps
            outside = ~np.isfinite(y_new) | (y_new < lo_a) | (y_new > hi_a)
            y_new = np.where(solved, yy, np.where(outside, 0.5 * (lo_a + hi_a), y_new))
            converged = solved | (np.abs(y_new - yy) <= 4 * eps * np.maximum(1, np.abs(yy)))
            lo[active], hi[active], y[active] = lo_a, hi_a, y_new
            active[np.flatnonzero(active)[converged]] = False

        E = np.clip(self._E0 * np.exp(y), E1, E2)
        E = np.where(u == 0, E1, np.where(u == 1, E2, E))

        if np.any(use_numerical):
            E[use_numerical] = EnergyFluxProfile.get_inverse_integral(
                self, u[use_numerical], E1[use_numerical], E2[use_numerical]
            )

        return E / unit_conv_factor


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
        crit_log10_energy_lower: float,
        crit_log10_energy_upper: float,
        energy_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new instance of PhotosplineEnergyFluxProfile.

        Parameters
        ----------
        splinetable
            The instance of photospline.SplineTable representing the energy flux
            profile as a spline.
        crit_log10_energy_lower
            The lower edge of the spline's supported energy range in log10(E).
        crit_log10_energy_upper
            The upper edge of the spline's supported energy range in log10(E).
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(energy_unit=energy_unit, **kwargs)

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
                f'Its current type is {classname(table)}!'
            )
        self._splinetable = table

    @property
    def crit_log10_energy_lower(self):
        """The lower energy bound of the spline's support."""
        return self._crit_log10_energy_lower

    @crit_log10_energy_lower.setter
    def crit_log10_energy_lower(self, v):
        v = float_cast(v, 'The property crit_log10_energy_lower must be castable to type float!')
        self._crit_log10_energy_lower = v

    @property
    def crit_log10_energy_upper(self):
        """The upper energy bound of the spline's support."""
        return self._crit_log10_energy_upper

    @crit_log10_energy_upper.setter
    def crit_log10_energy_upper(self, v):
        v = float_cast(v, 'The property crit_log10_energy_upper must be castable to type float!')
        self._crit_log10_energy_upper = v


class FunctionEnergyFluxProfile(EnergyFluxProfile):
    r"""Energy flux profile for a callable function with energy as argument."""

    def __init__(self, function: Callable, energy_unit: units.UnitBase | None = None, **kwargs):
        """Creates a new flux profile following a given function.

        Parameters
        ----------
        function
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(energy_unit=energy_unit, **kwargs)

        self.function = function

    def __call__(self, E: float | np.ndarray, unit: units.UnitBase | None = None) -> np.ndarray:
        """Returns the function values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E
            The energy value for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        value = self.function(E)

        return value

    @property
    def math_function_str(self) -> str:
        """(read-only) The string representation of the mathematical function of
        this energy flux profile.
        """
        return 'An arbitrary energy flux profile function'


class EpeakFunctionEnergyProfile(FunctionEnergyFluxProfile):
    r"""Energy flux profile for a callable function with energy as argument where
    Epeak (assuming a peaked spectrum, this is the peak's energy) and the scaling
    will be optimized.
    """

    def __init__(
        self, function: Callable, e_peak_orig, e_peak_offset, energy_unit: units.UnitBase | None = None, **kwargs
    ):
        """Creates a new  flux profile with the peak energy ``E0``.

        Parameters
        ----------
        function
        e_peak_orig
        e_peak_offset
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(function, energy_unit=energy_unit, **kwargs)

        self.e_peak_orig = e_peak_orig
        self.e_peak = e_peak_offset

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = 'e_peak'

    @property
    def e_peak(self):
        """The e_peak parameter of the function."""
        return self._e_peak

    @e_peak.setter
    def e_peak(self, e):
        e = float_cast(e, 'Property e must be castable to type float!')
        self._e_peak = e

    @property
    def e_peak_orig(self):
        """The original peak energy of the original distribution. The shift will
        be relative to this energy
        """
        return self._e_peak_orig

    @e_peak_orig.setter
    def e_peak_orig(self, e):
        e = float_cast(e, 'Property e must be castable to type float!')
        self._e_peak_orig = e

    def __call__(self, E: float | np.ndarray, unit: units.UnitBase | None = None) -> np.ndarray:
        """Returns the function values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E
            The energy value for which to retrieve the energy profile value.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        energy_offset = 10**self.e_peak_orig / 10**self.e_peak

        # we want to conserve energy, so the peak value should stay the same for E^2 dN/dE.

        value = self.function(E * energy_offset) * (energy_offset**2)

        max_flux = self.function(10**self.e_peak * energy_offset) * (energy_offset**2)

        # avoid really small values
        return np.where(value >= max_flux * 1e-100, value, max_flux * 1e-100)

    def get_integral(
        self,
        E1: float | np.ndarray,
        E2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """This is the default implementation for calculating the integral value
        of this energy flux profile in the range ``[E1, E2]``.

        .. note::

            This implementation utilizes the ``scipy.integrate.trapz`` function
            to perform a generic numeric integration. Hence, this implementation
            is slow and should be reimplemented by the derived class if an
            analytic integral form is available.

        Parameters
        ----------
        E1
            The lower energy bound of the integration.
        E2
            The upper energy bound of the integration.
        unit
            The unit of the given energies.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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

        for i, (E1_i, E2_i) in enumerate(zip(E1, E2, strict=True)):
            # integration for log(energy) for hopefully better numerical stability

            tmp_e = np.linspace(np.log10(E1_i), np.log10(E2_i))
            tmp_int = _trapezoid(np.log(10) * self(10**tmp_e) * 10**tmp_e, tmp_e)

            # make sure it is always positive (probably not an issue any more with np.trapz.
            # used to be an issue using the spline integrate self.function.integrate)
            integral[i] = max(tmp_int, 0.0)

        return integral

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance.
        """
        return 'spline of histogram'


class TimeFluxProfile(
    FluxProfile,
    metaclass=abc.ABCMeta,
):
    """The abstract base class for a time flux profile function."""

    def __init__(
        self, t_start: float = -np.inf, t_stop: float = np.inf, time_unit: units.UnitBase | None = None, **kwargs
    ):
        """Creates a new time flux profile instance.

        Parameters
        ----------
        t_start
            The start time of the time profile.
            If set to -inf, it means, that the profile starts at the beginning
            of the entire time-span of the dataset.
        t_stop
            The stop time of the time profile.
            If set to +inf, it means, that the profile ends at the end of the
            entire time-span of the dataset.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super().__init__(**kwargs)

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
        t = float_cast(t, f'The t_start property must be castable to type float! Its current type is {classname(t)}!')
        self._t_start = t

    @property
    def t_stop(self):
        """The stop time of the time profile. Can be +inf which means, that
        the profile ends at the end of the entire dataset.
        """
        return self._t_stop

    @t_stop.setter
    def t_stop(self, t):
        t = float_cast(t, f'The t_stop property must be castable to type float! Its current type is {classname(t)}!')
        self._t_stop = t

    @property
    def duration(self):
        """(read-only) The duration of the time profile."""
        return self._t_stop - self._t_start

    @property
    def time_unit(self):
        """The unit of time used for the flux profile calculation."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['time']
        if not isinstance(unit, units.UnitBase):
            raise TypeError(
                f'The property time_unit must be of type astropy.units.UnitBase! Its current type is {classname(unit)}!'
            )
        self._time_unit = unit

    def get_total_integral(self) -> 'float | np.ndarray':
        """Calculates the total integral of the time profile from t_start to
        t_stop.

        Returns
        -------
        integral
            The integral value of the entire time profile.
            The value is in the set time unit of this TimeFluxProfile instance.
        """
        integral = np.asarray(self.get_integral(self._t_start, self._t_stop)).squeeze()

        return integral

    @abc.abstractmethod
    def __call__(
        self,
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """This method is supposed to return the time profile value for the
        given times.

        Parameters
        ----------
        t
            The time(s) for which to get the time flux profile values.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        values
            The time profile values.
        """

    @abc.abstractmethod
    def move(
        self,
        dt: float,
        unit: units.UnitBase | None = None,
    ):
        """Abstract method to move the time profile by the given amount of time.

        Parameters
        ----------
        dt
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit
            The unit of the given time difference.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """

    @abc.abstractmethod
    def get_integral(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> 'float | np.ndarray':
        """This method is supposed to calculate the integral of the time profile
        from time ``t1`` to time ``t2``.

        Parameters
        ----------
        t1
            The start time of the integration.
        t2
            The end time of the integration.
        unit
            The unit of the given times.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """


class UnityTimeFluxProfile(
    TimeFluxProfile,
):
    """Time flux profile for the constant profile function ``1``."""

    def __init__(
        self,
        time_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new instance of UnityTimeFluxProfile.

        Parameters
        ----------
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super().__init__(time_unit=time_unit, **kwargs)

    @property
    def math_function_str(self):
        """The string representation of the mathematical function of this time
        flux profile.
        """
        return '1'

    def __call__(
        self,
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Returns 1 as numpy ndarray in same shape as t.

        Parameters
        ----------
        t
            The time(s) for which to get the time flux profile values.
        unit
            The unit of the given times.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values
            1 in same shape as ``t``.
        """
        t = np.atleast_1d(t)

        values = np.ones_like(t, dtype=np.int8)

        return values

    def move(
        self,
        dt: float,
        unit: units.UnitBase | None = None,
    ):
        """Moves the time profile by the given amount of time. By definition
        this method does nothing, because the profile is 1 over the entire
        dataset time range.

        Parameters
        ----------
        dt
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit
            The unit of the given time difference.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """

    def get_integral(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> 'float | np.ndarray':
        """Calculates the integral of the time profile from time t1 to time t2.

        Parameters
        ----------
        t1
            The start time of the integration.
        t2
            The end time of the integration.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """
        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = float(unit.to(self._time_unit))  # pyright: ignore[reportArgumentType]
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
        start: float,
        stop: float,
        time_unit: units.UnitBase | None = None,
        **kwargs,
    ) -> 'BoxTimeFluxProfile':
        """Constructs a BoxTimeFluxProfile instance from the given start and
        stop time.

        Parameters
        ----------
        start
            The start time of the box profile.
        stop
            The stop time of the box profile.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        **kwargs
            Additional keyword arguments, which are passed to the constructor
            of the :class:`BoxTimeFluxProfile` class.

        Returns
        -------
        profile
            The newly created instance of BoxTimeFluxProfile.
        """
        t0 = 0.5 * (start + stop)
        tw = stop - start

        profile = cls(t0=t0, tw=tw, time_unit=time_unit, **kwargs)

        return profile

    def __init__(
        self,
        t0: float,
        tw: float,
        time_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new box-shaped time profile instance.

        Parameters
        ----------
        t0
            The mid time of the box profile.
        tw
            The width of the box profile.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        t_start = t0 - tw / 2.0
        t_stop = t0 + tw / 2.0

        super().__init__(t_start=t_start, t_stop=t_stop, time_unit=time_unit, **kwargs)

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('t0', 'tw')

    @property
    def t0(self):
        """The time of the mid point of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return 0.5 * (self._t_start + self._t_stop)

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
        self._t_start = t0 - 0.5 * w
        self._t_stop = t0 + 0.5 * w

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
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Returns 1 for all t within the interval [t0-tw/2; t0+tw/2], and 0
        otherwise.

        Parameters
        ----------
        t
            The time(s) for which to get the time flux profile values.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        values
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
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Calculates the cumulative distribution function value for the given
        time values ``t``.

        Parameters
        ----------
        t
            The (N_times,)-shaped numpy ndarray holding the time values for
            which to calculate the CDF values.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values
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

        m = t > t_stop
        values[m] = 1

        return values

    def move(
        self,
        dt: float,
        unit: units.UnitBase | None = None,
    ):
        """Moves the box-shaped time profile by the time difference dt.

        Parameters
        ----------
        dt
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit
            The unit of ``dt``.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        if (unit is not None) and (unit != self._time_unit):
            dt = dt * unit.to(self._time_unit)  # pyright: ignore[reportAssignmentType]

        self._t_start += dt
        self._t_stop += dt

    def get_integral(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Calculates the integral of the box-shaped time flux profile from
        time t1 to time t2.

        Parameters
        ----------
        t1
            The start time(s) of the integration.
        t2
            The end time(s) of the integration.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral
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
        self, t0: float, sigma_t: float, tol: float = 1e-12, time_unit: units.UnitBase | None = None, **kwargs
    ):
        """Creates a new gaussian-shaped time flux profile instance.

        Parameters
        ----------
        t0
            The mid time of the gaussian profile.
        sigma_t
            The one-sigma width of the gaussian profile.
        tol
            The tolerance of the gaussian value. This defines the start and end
            time of the gaussian profile.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        # Calculate the start and end time of the gaussian profile, such that
        # at those times the gaussian values obey the given tolerance.
        dt = np.sqrt(-2 * sigma_t**2 * np.log(tol))
        t_start = t0 - dt
        t_stop = t0 + dt

        super().__init__(t_start=t_start, t_stop=t_stop, time_unit=time_unit, **kwargs)

        self.t0 = t0
        self.sigma_t = sigma_t

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('t0', 'sigma_t')

    @property
    def math_function_str(self):
        """The string representation of the mathematical function of this time
        flux profile.
        """
        return 'exp(-(t-t0)^2/(2 sigma_t^2))'

    @property
    def t0(self):
        """The time of the mid point of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return 0.5 * (self.t_start + self.t_stop)

    @t0.setter
    def t0(self, t):
        t = float_cast(t, 'The t0 property must be castable to type float!')
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
        sigma = float_cast(sigma, 'The sigma_t property must be castable to type float!')
        self._sigma_t = sigma

    def __call__(
        self,
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Returns the gaussian profile value for the given time ``t``.

        Parameters
        ----------
        t
            The time(s) for which to get the time flux profile values.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values
            The value(s) of the time flux profile for the given time(s).
        """
        t = np.atleast_1d(t)

        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        m = (t >= self.t_start) & (t < self.t_stop)

        s = self._sigma_t
        twossq = 2 * s * s
        t0 = 0.5 * (self.t_stop + self.t_start)
        dt = t[m] - t0

        values = np.zeros_like(t)
        values[m] = np.exp(-dt * dt / twossq)

        return values

    def cdf(
        self,
        t: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Calculates the cumulative distribution function values for the given
        time values ``t``.

        Parameters
        ----------
        t
            The (N_times,)-shaped numpy ndarray holding the time values for
            which to calculate the CDF values.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.

        Returns
        -------
        values
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
        values[m] = self.get_integral(t1=t_start, t2=t[m]) / self.get_total_integral()

        m = t > t_stop
        values[m] = 1

        return values

    def move(
        self,
        dt: float,
        unit: units.UnitBase | None = None,
    ):
        """Moves the gaussian time profile by the given amount of time.

        Parameters
        ----------
        dt
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        unit
            The unit of the given time difference.
            If set to ``None``, the set time unit of this TimeFluxProfile is
            assumed.
        """
        if (unit is not None) and (unit != self._time_unit):
            dt = dt * unit.to(self._time_unit)  # pyright: ignore[reportAssignmentType]

        self._t_start += dt
        self._t_stop += dt

    def get_integral(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
        unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Calculates the integral of the gaussian time profile from time ``t1``
        to time ``t2``.

        Parameters
        ----------
        t1
            The start time(s) of the integration.
        t2
            The end time(s) of the integration.
        unit
            The unit of the given times.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral
            The integral value(s). The values are in the set time unit of
            this TimeFluxProfile instance.
        """
        if (unit is not None) and (unit != self._time_unit):
            time_unit_conv_factor = float(unit.to(self._time_unit))  # pyright: ignore[reportArgumentType]
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        t0 = 0.5 * (self.t_stop + self.t_start)
        sigma_t = self._sigma_t

        c1 = np.sqrt(np.pi / 2) * sigma_t
        c2 = np.sqrt(2) * sigma_t
        i1 = c1 * scipy.special.erf((t1 - t0) / c2)
        i2 = c1 * scipy.special.erf((t2 - t0) / c2)

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
    def get_default_units(cfg: Config) -> dict:
        """Returns the configured default units for flux models.

        Parameters
        ----------
        cfg
            The instance of Config holding the local configuration.

        Returns
        -------
        units_dict
            The dictionary holding the configured default units used for flux
            models.
        """
        return cfg['units']['defaults']['fluxes']

    def __init__(
        self,
        angle_unit: units.UnitBase | None = None,
        energy_unit: units.UnitBase | None = None,
        length_unit: units.UnitBase | None = None,
        time_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new FluxModel instance and defines the user-defined units.

        Parameters
        ----------
        angle_unit
            The used unit for angles.
            If set to ``None``, the configured default angle unit for fluxes is
            used.
        energy_unit
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        length_unit
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super().__init__(**kwargs)

        # Define the units.
        self.angle_unit = angle_unit
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.time_unit = time_unit

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['angle']
        if not isinstance(unit, units.UnitBase):
            raise TypeError('The property angle_unit must be of type astropy.units.UnitBase!')
        self._angle_unit = unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation."""
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['energy']
        if not isinstance(unit, units.UnitBase):
            raise TypeError('The property energy_unit must be of type astropy.units.UnitBase!')
        self._energy_unit = unit

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation."""
        return self._length_unit

    @length_unit.setter
    def length_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['length']
        if not isinstance(unit, units.UnitBase):
            raise TypeError('The property length_unit must be of type astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, unit):
        if unit is None:
            unit = self._cfg['units']['defaults']['fluxes']['time']
        if not isinstance(unit, units.UnitBase):
            raise TypeError('The property time_unit must be of type astropy.units.UnitBase!')
        self._time_unit = unit

    @property
    def unit_str(self):
        """The string representation of the flux unit."""
        if self.angle_unit == units.radian:  # noqa: SIM108
            angle_unit_sq = units.steradian
        else:
            angle_unit_sq = self.angle_unit**2

        s = (
            f'({self.energy_unit.to_string()}'
            f' {angle_unit_sq.to_string()}'
            f' {self.length_unit.to_string()}^2'
            f' {self.time_unit.to_string()})^-1'
        )

        return s

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit."""
        if self.angle_unit == units.radian:  # noqa: SIM108
            angle_unit_sq = units.steradian
        else:
            angle_unit_sq = self.angle_unit**2

        s = (
            f'{self.energy_unit.to_string()}'
            '$^{-1}$ '
            f'{angle_unit_sq.to_string()}'
            '$^{-1}$ '
            f'{self.length_unit.to_string()}'
            '$^{-2}$ '
            f'{self.time_unit.to_string()}'
            '$^{-1}$'
        )

        return s

    def __str__(self):
        """Pretty string representation of this class."""
        return f'{self.math_function_str} {self.unit_str}'

    @abc.abstractmethod
    def __call__(
        self,
        ra: float | np.ndarray | None = None,
        dec: float | np.ndarray | None = None,
        E: float | np.ndarray | None = None,
        t: float | np.ndarray | None = None,
        angle_unit: units.UnitBase | None = None,
        energy_unit: units.UnitBase | None = None,
        time_unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """The call operator to retrieve a flux value for a given celestrial
        position, energy, and observation time.

        Parameters
        ----------
        ra
            The right-ascention coordinate for which to retrieve the flux value.
        dec
            The declination coordinate for which to retrieve the flux value.
        E
            The energy for which to retrieve the flux value.
        t
            The observation time for which to retrieve the flux value.
        angle_unit
            The unit of the given angles.
            If ``None``, the set angle unit of the flux model is assumed.
        energy_unit
            The unit of the given energies.
            If ``None``, the set energy unit of the flux model is assumed.
        time_unit
            The unit of the given times.
            If ``None``, the set time unit of the flux model is assumed.

        Returns
        -------
        flux
            The flux values are in unit of the set flux model units
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """

    def to_internal_flux_unit(self) -> float:
        """Calculates the conversion factor to convert the flux unit of this
        flux model instance to the SkyLLH internally used flux unit.

        Returns
        -------
        factor
            The conversion factor.
        """
        self_flux_unit = 1 / (self.angle_unit**2 * self.energy_unit * self.length_unit**2 * self.time_unit)

        internal_units = self._cfg['units']['internal']
        internal_flux_unit = 1 / (
            internal_units['angle'] ** 2
            * internal_units['energy']
            * internal_units['length'] ** 2
            * internal_units['time']
        )

        factor = (self_flux_unit).to(internal_flux_unit).value  # pyright: ignore[reportAttributeAccessIssue]

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
        cfg: Config | None = None,
        **kwargs,
    ):
        """Creates a new instance of NullFluxModel.

        Parameters
        ----------
        cfg
            The instance of Config holding the local configuration. Since this
            flux model does nothing, this argument is optional. If not provided
            the default configuration is used.
        """
        if cfg is None:
            cfg = Config()

        super().__init__(*args, cfg=cfg, **kwargs)

    @property
    def math_function_str(self):
        """Since this is a dummy flux model, calling this method will raise a
        NotImplementedError.
        """
        raise NotImplementedError(
            f'The {classname(self)} flux model is a dummy flux model which has no math function prepresentation!'
        )

    def __call__(self, *args, **kwargs):
        """Since this is a dummy flux model, calling this method will raise a
        NotImplementedError.
        """
        raise NotImplementedError(f'The {classname(self)} flux model is a dummy flux model and cannot be called!')


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
        Phi0: float,
        spatial_profile: 'SpatialFluxProfile | None',
        energy_profile: 'EnergyFluxProfile | None',
        time_profile: 'TimeFluxProfile | None',
        length_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new factorized flux model.

        Parameters
        ----------
        Phi0
            The flux normalization constant.
        spatial_profile
            The SpatialFluxProfile instance providing the spatial profile
            function of the flux.
            If set to None, an instance of UnitySpatialFluxProfile will be used,
            which represents the constant function 1.
        energy_profile
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        length_unit
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        """
        cfg = kwargs.get('cfg')

        self.Phi0 = Phi0

        if spatial_profile is None:
            spatial_profile = UnitySpatialFluxProfile(cfg=cfg)
        self.spatial_profile = spatial_profile

        if energy_profile is None:
            energy_profile = UnityEnergyFluxProfile(cfg=cfg)
        self.energy_profile = energy_profile

        if time_profile is None:
            time_profile = UnityTimeFluxProfile(cfg=cfg)
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
        v = float_cast(v, 'The Phi0 property must be castable to type float!')
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
            raise TypeError('The spatial_profile property must be None, or an instance of SpatialFluxProfile!')
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
            raise TypeError('The energy_profile property must be None, or an instance of EnergyFluxProfile!')
        self._energy_profile = profile

    @property
    def time_profile(self):
        """Instance of TimeFluxProfile describing the time profile of the flux."""
        return self._time_profile

    @time_profile.setter
    def time_profile(self, profile):
        if not isinstance(profile, TimeFluxProfile):
            raise TypeError('The time_profile property must be None, or an instance of TimeFluxProfile!')
        self._time_profile = profile

    @property
    def math_function_str(self):
        """The string showing the mathematical function of the flux."""
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
        super(FactorizedFluxModel, type(self)).param_names.fset(self, names)  # pyright: ignore[reportAttributeAccessIssue]

    def __call__(
        self,
        ra: float | np.ndarray | None = None,
        dec: float | np.ndarray | None = None,
        E: float | np.ndarray | None = None,
        t: float | np.ndarray | None = None,
        angle_unit: units.UnitBase | None = None,
        energy_unit: units.UnitBase | None = None,
        time_unit: units.UnitBase | None = None,
    ) -> np.ndarray:
        """Calculates the flux values for the given celestrial positions,
        energies, and observation times.

        Parameters
        ----------
        ra
            The right-ascention coordinate for which to retrieve the flux value.
        dec
            The declination coordinate for which to retrieve the flux value.
        E
            The energy for which to retrieve the flux value.
        t
            The observation time for which to retrieve the flux value.
        angle_unit
            The unit of the given angles.
            If ``None``, the set angle unit of the spatial flux profile is
            assumed.
        energy_unit
            The unit of the given energies.
            If ``None``, the set energy unit of the energy flux profile is
            assumed.
        time_unit
            The unit of the given times.
            If ``None``, the set time unit of the time flux profile is
            assumed.

        Returns
        -------
        flux
            The flux values are in unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        if (ra is not None) and (dec is not None):
            spatial_profile_values = self._spatial_profile(ra, dec, unit=angle_unit)
        else:
            spatial_profile_values = np.array([1])

        if E is not None:  # noqa: SIM108
            energy_profile_values = self._energy_profile(E, unit=energy_unit)
        else:
            energy_profile_values = np.array([1])

        if t is not None:  # noqa: SIM108
            time_profile_values = self._time_profile(t, unit=time_unit)
        else:
            time_profile_values = np.array([1])

        flux = (
            self._Phi0
            * spatial_profile_values[:, np.newaxis, np.newaxis]
            * energy_profile_values[np.newaxis, :, np.newaxis]
            * time_profile_values[np.newaxis, np.newaxis, :]
        )

        return flux

    def get_param(self, name: str) -> float:
        """Retrieves the value of the given parameter. It returns ``np.nan`` if
        the parameter does not exist.

        Parameters
        ----------
        name
            The name of the parameter.

        Returns
        -------
        value
            The value of the parameter.
        """
        for obj in (super(), self._spatial_profile, self._energy_profile, self._time_profile):
            value = obj.get_param(name=name)
            if not np.isnan(value):
                return value

        return np.nan

    def set_params(self, pdict: dict) -> bool:
        """Sets the parameters of the flux model. For this factorized flux model
        it means that it sets the parameters of the spatial, energy, and time
        profiles.

        Parameters
        ----------
        pdict
            The flux parameter dictionary.

        Returns
        -------
        updated
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
        Phi0: float,
        energy_profile: 'EnergyFluxProfile | None',
        time_profile: 'TimeFluxProfile | None',
        ra: float | None = None,
        dec: float | None = None,
        angle_unit: units.UnitBase | None = None,
        length_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new factorized flux model for a point-like source.

        Parameters
        ----------
        Phi0
            The flux normalization constant in unit of flux.
        energy_profile
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        ra
            The right-ascention of the point.
        dec
            The declination of the point.
        angle_unit
            The unit for angles used for the flux unit.
            If set to ``None``, the configured internal angle unit is used.
        length_unit
            The unit for length used for the flux unit.
            If set to ``None``, the configured internal length unit is used.
        """
        spatial_profile = PointSpatialFluxProfile(cfg=kwargs.get('cfg'), ra=ra, dec=dec, angle_unit=angle_unit)

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
            **kwargs,
        )

    @property
    def unit_str(self):
        """The string representation of the flux unit."""
        # Note:
        #    For a point-like differential flux, there is no solid-angle
        #    element.
        s = f'({self.energy_unit.to_string()} {self.length_unit.to_string()}^2 {self.time_unit.to_string()})^-1'

        return s

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit."""
        # Note:
        #    For a point-like differential flux, there is no solid-angle
        #    element.
        s = (
            f'{self.energy_unit.to_string()}'
            '$^{-1}$ '
            f'{self.length_unit.to_string()}'
            '$^{-2}$ '
            f'{self.time_unit.to_string()}'
            '$^{-1}$'
        )

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
        Phi0: float,
        energy_profile: 'EnergyFluxProfile | None',
        ra: float | None = None,
        dec: float | None = None,
        angle_unit: units.UnitBase | None = None,
        length_unit: units.UnitBase | None = None,
        time_unit: units.UnitBase | None = None,
        **kwargs,
    ):
        """Creates a new factorized flux model for a point-like source with no
        time dependance.

        Parameters
        ----------
        Phi0
            The flux normalization constant.
        energy_profile
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        ra
            The right-ascention of the point.
        dec
            The declination of the point.
        angle_unit
            The unit for angles used for the flux unit.
            If set to ``None``, the configured default angle unit for fluxes
            is used.
        length_unit
            The unit for length used for the flux unit.
            If set to ``None``, the configured default length unit for fluxes
            is used.
        time_unit
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes
            is used.
        """
        time_profile = UnityTimeFluxProfile(cfg=kwargs.get('cfg'), time_unit=time_unit)

        super().__init__(
            Phi0=Phi0,
            ra=ra,
            dec=dec,
            energy_profile=energy_profile,
            time_profile=time_profile,
            angle_unit=angle_unit,
            length_unit=length_unit,
            **kwargs,
        )
