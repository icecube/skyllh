# -*- coding: utf-8 -*-

r"""The `flux_models` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a function for the differential flux:

.. math::

    d\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) / (dA d\Omega dE dt)
"""

import abc
import numpy as np

from copy import deepcopy

from astropy import units

from skyllh.core.py import (
    classname,
    isproperty,
    issequence,
    issequenceof,
    float_cast
)


class FluxModel(object):
    r"""Abstract base class for all flux models
    :math:`\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s)`.

    This base class defines the units used for the flux calculation. At this
    point the functional form of the flux model is not yet defined.

    Attributes
    ----------
    angle_unit : str
        The used unit of angles.
    energy_unit : str
        The used unit of energy.
    length_unit : str
        The used unit of length.
    time_unit : str
        The used unit of time.
    math_function_str : str
        The string showing the mathematical function of the flux calculation.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(FluxModel, self).__init__()

        # Define the default units.
        self.angle_unit = units.radian
        self.energy_unit = units.GeV
        self.length_unit = units.cm
        self.time_unit = units.s

        self.parameter_names = ()

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation.
        """
        return self._angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
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
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property length_unit must be of type '
                'astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property time_unit must be of type '
                'astropy.units.UnitBase!')
        self._time_unit = unit

    @property
    def unit_str(self):
        return ' '.join((self.energy_unit.to_string()+'^-1',
                         self.angle_unit.to_string()+'^-2',
                         self.length_unit.to_string()+'^-2',
                         self.time_unit.to_string()+'^-1'))

    @property
    def parameter_names(self):
        """The tuple holding the names of the parameters this flux depends on.
        """
        return self._parameter_names
    @parameter_names.setter
    def parameter_names(self, names):
        if(not issequence(names)):
            names = (names,)
        if(not issequenceof(names, str)):
            raise TypeError('The parameter_names property must be a sequence '
                'of str!')
        names = tuple(names)
        # Check if all the given names are actual properties of this flux class.
        for name in names:
            if(not hasattr(self, name)):
                raise KeyError('The flux "%s" does not have a property '
                    'named "%s"!'%(classname(self), name))
            if(not isproperty(self, name)):
                raise TypeError('The attribute "%s" of flux "%s" is not a '
                    'property!'%(classname(self), name))
        self._parameter_names = names

    @property
    def internal_flux_unit_conversion_factor(self):
        """The unit conversion factor for converting the used flux unit of this
        flux model into the SkyLLH internally used flux unit 1/(GeV sr cm2 s).
        """
        unit_conversion_factor = (
            1./self.energy_unit *
            1./self.angle_unit**2 *
            1./self.length_unit**2 *
            1./self.time_unit
        ).to(
            1./units.GeV *
            1./units.sr *
            1./units.cm**2 *
            1./units.s
        ).value
        return unit_conversion_factor

    @property
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of the flux.
        """
        pass

    @abc.abstractmethod
    def __call__(self, alpha, delta, E, time, pdict):
        """The call operator to retrieve a flux value for a given celestrial
        position, energy, and observation time.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        time : float | (Ntime,)-shaped 1d numpy ndarray of float
            The MJD observation time for which to retrieve the flux value.
        pdict : dictionary
            The flux parameter dictionary.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values in unit [energy]^-1 [angle]^-2 [length]^-2 [time]^-1.
            By default that is GeV^-1 sr^-1 cm^-2 s^-1.
        """
        pass

    def __str__(self):
        """Pretty string representation of this class.
        """
        return self.math_function_str + ' ' + self.unit_str

    def copy(self, newprop=None):
        """Copies this flux model object by calling the copy.deepcopy function,
        and sets new properties if requested.

        Parameters
        ----------
        newprop : dict | None
            The dictionary with the new property values to set, where the
            dictionary key is the property name and the dictionary value is the
            new value of the property.
        """
        fluxmodel = deepcopy(self)

        # Set the new property values.
        if(newprop is not None):
            fluxmodel.set_properties(newprop)

        return fluxmodel

    def set_parameters(self, pdict):
        """Sets the parameters of the flux model to the given parameter values.

        Parameters
        ----------
        pdict : dict (name: value)
            The dictionary holding the names of the parameters and their new
            values.
        """
        if(not isinstance(pdict, dict)):
            raise TypeError('The pdict argument must be of type dict!')

        for pname in self._parameter_names:
            pvalue = pdict.get(pname, getattr(self, pname))
            setattr(self, pname, pvalue)


class FluxProfile(object):
    """The abstract base class for a flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(SpatialFluxProfile, self).__init__()

        self.parameter_names = ()

    @property
    def parameter_names(self):
        """The tuple holding the names of the parameters this flux profile
        depends on.
        """
        return self._parameter_names
    @parameter_names.setter
    def parameter_names(self, names):
        if(not issequence(names)):
            names = (names,)
        if(not issequenceof(names, str)):
            raise TypeError('The parameter_names property must be a sequence '
                'of str!')
        names = tuple(names)
        # Check if all the given names are actual properties of this flux
        # profile class.
        for name in names:
            if(not hasattr(self, name)):
                raise KeyError('The flux profile "%s" does not have a property '
                    'named "%s"!'%(classname(self), name))
            if(not isproperty(self, name)):
                raise TypeError('The attribute "%s" of flux profile "%s" is '
                    'not a property!'%(classname(self), name))
        self._parameter_names = names

    @property
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of the spatial flux
        profile.
        """
        pass

    def set_parameters(self, pdict):
        """Sets the parameters of the flux profile to the given parameter
        values.

        Parameters
        ----------
        pdict : dict (name: value)
            The dictionary holding the names of the parameters and their new
            values. Only parameters will be considered, which are part of the
            tuple property `parameter_names`.
        """
        if(not isinstance(pdict, dict)):
            raise TypeError('The pdict argument must be of type dict!')

        for pname in self._parameter_names:
            pvalue = pdict.get(pname, getattr(self, pname))
            setattr(self, pname, pvalue)


class SpatialFluxProfile(FluxProfile):
    """The abstract base class for a spatial flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(SpatialFluxProfile, self).__init__()

    @abc.abstractmethod
    def __call__(self, alpha, delta):
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.

        Returns
        -------
        value : 1d numpy ndarray
            The spatial profile value.
        """
        pass


class UnitySpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1.
    """
    def __init__(self):
        super(UnitySpatialFluxProfile, self).__init__()

    def math_function_str(self):
        return '1'

    def __call__(self, alpha, delta):
        """Returns 1 as numpy ndarray in same shape as alpha and delta.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.

        Returns
        -------
        value : 1d numpy ndarray
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
    def __init__(self, alpha_s, delta_s):
        """Creates a new spatial flux profile for a point.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        """
        super(PointSpatialFluxProfile, self).__init__()

        # Define the names of the parameters, which can be updated.
        self.parameter_names = ('alpha_s', 'delta_s')

        self.alpha_s = alpha_s
        self.delta_s = delta_s

    @property
    def alpha_s(self):
        """The right-ascention of the point-like source.
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
        """
        return self._delta_s
    @delta_s.setter
    def delta_s(self, v):
        v = float_cast(v,
                'The delta_s property must be castable to type float!')
        self._delta_s = v

    def math_function_str(self):
        return 'delta(alpha-%.2e)*delta(delta-%.2e)'%(
            self._alpha_s, self._delta_s)

    def __call__(self, alpha, delta):
        """Returns a numpy ndarray in same shape as alpha and delta with 1 if
        `alpha` equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate at which to evaluate the spatial flux
            profile.
        delta : float | 1d numpy ndarray of float
            The declination coordinate at which to evaluate the spatial flux
            profile.

        Returns
        -------
        value : 1d numpy ndarray
            A numpy ndarray in same shape as alpha and delta with 1 if `alpha`
            equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.
        """
        (alpha, delta) = np.atleast_1d(alpha, delta)
        if(alpha.shape != delta.shape):
            raise ValueError('The alpha and delta arguments must be of the '
                'same shape!')

        value = ((alpha == self._alpha_s) & (delta == self._delta_s)).astype(np.float, copy=False)
        return value


class EnergyFluxProfile(FluxProfile):
    """The abstract base class for an energy flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Creates a new energy flux profile with a given energy unit to be used
        for flux calculation.
        """
        super(EnergyFluxProfile, self).__init__()

        # Set the default energy unit.
        self.energy_unit = units.GeV

    @property
    def energy_unit(self):
        """The unit of energy used for the flux profile calculation.
        """
        return self._energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit_conversion_factor = (
            units.GeV
        ).to(unit).value
        self._energy_unit = unit

    @property
    def energy_unit_conversion_factor(self):
        """(read-only) The unit conversion factor for converting the SkyLLH
        internally used energy unit GeV into the used energy unit of this energy
        flux profile.
        """
        return self._energy_unit_conversion_factor

    @abc.abstractmethod
    def __call__(self, E):
        """This method is supposed to return the energy profile value for the
        given energy value.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.

        Returns
        -------
        value : 1d numpy ndarray
            The energy profile value.
        """
        pass


class UnityEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for the constant function 1.
    """
    def __init__(self):
        super(UnityEnergyFluxProfile, self).__init__()

    def math_function_str(self):
        return '1'

    def __call__(self, E):
        """Returns 1 as numpy ndarray in some shape as E.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.

        Returns
        -------
        value : 1d numpy ndarray
            1 in same shape as E.
        """
        E = np.atleast_1d(E)
        return np.ones_like(E)


class PowerLawEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for a power law profile with a reference energy E0
    and a spectral index gamma.
    """
    def __init__(self, E0, gamma):
        """Creates a new power law flux profile with the reference energy E0
        and spectral index gamma.
        """
        super(PowerLawEnergyFluxProfile, self).__init__()

        self.parameter_names = ('gamma',)

        self.E0 = E0
        self.gamma = gamma

    @property
    def E0(self):
        """The reference energy.
        """
        return self._E0
    @E0.setter
    def E0(self, v):
        v = float_cast(v, 'Property E0 must be castable to type float!')
        self._E0 = v

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma
    @gamma.setter
    def gamma(self, v):
        v = float_cast(v, 'Property gamma must be castable to type float!')
        self._gamma = v

    def math_function_str(self):
        return '(E / %.2e %s)^-%.2f'%(self.E0, self.energy_unit, self.gamma)

    def __call__(self, E):
        """Returns the power law values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.

        Returns
        -------
        value : 1d numpy ndarray
            1 in same shape as E.
        """
        E = E * self._energy_unit_conversion_factor

        value = np.power(E / self._E0, -self._gamma)

        return value


class TimeFluxProfile(FluxProfile):
    """The abstract base class for a time flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TimeFluxProfile, self).__init__()

    @abc.abstractmethod
    def __call__(self, time, pdict):
        """This method is supposed to return the time profile value of the flux
        for the given time.

        Parameters
        ----------
        time : float | 1d numpy ndarray of float
            The MJD time.
        pdict : dict
            The dictionary with time flux parameters.

        Returns
        -------
        value : 1d numpy ndarray
            The time profile value.
        """
        pass


class UnityTimeFluxProfile(TimeFluxProfile):
    """Time flux profile for the constant profile function 1.
    """
    def __init__(self):
        super(UnityTimeFluxProfile, self).__init__()

    def math_function_str(self):
        return '1'

    def __call__(self, time, pdict):
        """Returns 1 as numpy ndarray in same shape as time.

        Parameters
        ----------
        time : float | 1d numpy ndarray of float
            The MJD time.
        pdict : dict
            The dictionary with time flux parameters.
            This argument is ignored.

        Returns
        -------
        value : 1d numpy ndarray
            1 in same shape as time.
        """
        time = np.atleast_1d(time)
        return np.ones_like(time)


class FactorizedFluxModel(FluxModel):
    r"""This class describes a flux model where the spatial, energy, and time
    profiles of the source factorize. That means the flux can be written as:

    .. math::

        \Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) =
            \Phi_0
            \Psi_S(\alpha,\delta|\vec{p}_s)
            \epsilon_S(E|\vec{p}_s)
            T_S(t|\vec{p}_s)

    where, :math:`\Phi_0` is the normalization constant of the flux, and
    :math:`\Psi_S`, :math:`\epsilon_S`, and :math:`T_S` are the spatial,
    energy, and time profiles of the source, respectively.
    """
    def __init__(self, Phi0, spatial_profile, energy_profile, time_profile):
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
        """
        self.Phi0 = Phi0
        self.spatial_profile = spatial_profile
        self.energy_profile = energy_profile
        self.time_profile = time_profile

        # The base class will set the default (internally used) flux unit, which
        # will be set automatically to the particular profile.
        super(FactorizedFluxModel, self).__init__()

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
        return '%.2e * %s * %s * %s'%(
            self.Phi0,
            self.spatial_profile.math_function_str,
            self.energy_profile.math_function_str,
            self.time_profile.math_function_str
        )

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation.
        """
        return self.energy_profile.energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self.energy_profile.energy_unit = unit

    def set_parameters(self, pdict):
        """Sets the parameters of the flux model. For this factorized flux model
        it means that it sets the parameters of the spatial, energy, and time
        profiles.

        Parameters
        ----------
        pdict : dictionary
            The flux parameter dictionary.
        """
        self._spatial_profile.set_parameters(pdict)
        self._energy_profile.set_parameters(pdict)
        self._time_profile.set_parameters(pdict)

    def __call__(self, alpha, delta, E, time):
        """Calculates the flux values for the given celestrial
        positions, energies, and observation times.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        time : float | (Ntime,)-shaped 1d numpy ndarray of float
            The MJD observation time for which to retrieve the flux value.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values in unit [energy]^-1 [angle]^-2 [length]^-2 [time]^-1.
            By default that is GeV^-1 sr^-1 cm^-2 s^-1.
        """
        spatial_profile_values = self._spatial_profile(alpha, delta)
        energy_profile_values = self._energy_profile(E)
        time_profile_values = self._time_profile(time)

        flux = (
            self.Phi0 *
            spatial_profile_values[:,np.newaxis,np.newaxis] *
            energy_profile_values[np.newaxis,:,np.newaxis] *
            time_profile_values[np.newaxis,np.newaxis,:]
        )

        return flux


class PointlikeSourceFFM(FactorizedFluxModel):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point. This class provides the base class for a flux
    model of a point-like source.
    """
    def __init__(self, alpha_s, delta_s, Phi0, energy_profile, time_profile):
        """Creates a new factorized flux model for a point-like source.

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
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        """
        super(PointlikeSourceFFM, self).__init__(
            Phi0=Phi0,
            spatial_profile=PointSpatialFluxProfile(alpha_s, delta_s),
            energy_profile=energy_profile,
            time_profile=time_profile
        )


class SteadyPointlikeSourceFFM(PointlikeSourceFFM):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point and the time profile as constant 1.
    """
    def __init__(self, alpha_s, delta_s, Phi0, energy_profile):
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
        super(PointlikeSourceFFM, self).__init__(
            alpha_s=alpha_s,
            delta_s=delta_s,
            Phi0=Phi0,
            energy_profile=energy_profile,
            time_profile=UnityTimeFluxProfile()
        )
