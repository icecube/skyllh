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

from skyllh.core.py import classname, isproperty


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
    def internal_flux_unit_conversion_factor(self):
        """The unit conversion factor for converting the used flux unit of this
        flux model into the SkyLLH internally used flux unit 1/(GeV sr cm2 s).
        """
        unit_conversion_factor = (
            1./fluxmodel.energy_unit *
            1./fluxmodel.angle_unit**2 *
            1./fluxmodel.length_unit**2 *
            1./fluxmodel.time_unit
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

    def set_properties(self, propdict):
        """Sets the properties of the flux model to the given property values.

        Parameters
        ----------
        propdict : dict (name: value)
            The dictionary holding the names of the properties and their new
            values.
        """
        if(not isinstance(propdict, dict)):
            raise TypeError('The propdict argument must be of type dict!')
        for (prop, val) in propdict.iteritems():
            if(not hasattr(self, prop)):
                raise KeyError('The flux model "%s" does not have a property '
                    'named "%s"!'%(classname(self), prop))
            if(not isproperty(self, prop)):
                raise TypeError('The attribute "%s" of flux model "%s" is no '
                    'property!'%(classname(self), prop))
            setattr(self, prop, val)


class FluxProfile(object):
    """The abstract base class for a flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(SpatialFluxProfile, self).__init__()

    @property
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of the spatial flux
        profile.
        """
        pass


class SpatialFluxProfile(FluxProfile):
    """The abstract base class for a spatial flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(SpatialFluxProfile, self).__init__()

    @abc.abstractmethod
    def __call__(self, alpha, delta, pdict):
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        pdict : dict
            The dictionary with spatial flux parameters.

        Returns
        -------
        value : 1d numpy ndarray
            The spatial profile value.
        """
        pass


class NullSpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1.
    """
    def __init__(self):
        super(NullSpatialFluxProfile, self).__init__()

    def math_function_str(self):
        return '1'

    def __call__(self, alpha, delta, pdict):
        """Returns 1 as numpy ndarray in same shape as alpha and delta.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        pdict : dict
            The dictionary with spatial flux parameters.
            This argument is ignored.

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


class EnergyFluxProfile(FluxProfile):
    """The abstract base class for an energy flux profile function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(EnergyFluxProfile, self).__init__()

    @abc.abstractmethod
    def __call__(self, E, pdict):
        """This method is supposed to return the energy profile value for the
        given energy value.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        pdict : dict
            The dictionary with energy flux parameters.

        Returns
        -------
        value : 1d numpy ndarray
            The energy profile value.
        """
        pass


class NullEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for the constant function 1.
    """
    def __init__(self):
        super(EnergyFluxProfile, self).__init__()

    def math_function_str(self):
        return '1'

    def __call__(self, E, pdict):
        """Returns 1 as numpy ndarray in some shape as E.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        pdict : dict
            The dictionary with energy flux parameters.
            This argument is ignored.

        Returns
        -------
        value : 1d numpy ndarray
            1 in same shape as E.
        """
        E = np.atleast_1d(E)
        return np.ones_like(E)


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


class NullTimeFluxProfile(TimeFluxProfile):
    """Time flux profile for the constant profile function 1.
    """
    def __init__(self):
        super(NullTimeFluxProfile, self).__init__()

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
            If set to None, an instance of NullSpatialFluxProfile will be used,
            which represents the constant function 1.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of NullEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of NullTimeFluxProfile will be used,
            which represents the constant function 1.
        """
        super(FactorizedFluxModel, self).__init__()

        self.Phi0 = Phi0
        self.spatial_profile = spatial_profile
        self.energy_profile = energy_profile
        self.time_profile = time_profile

    @property
    def spatial_profile(self):
        """Instance of SpatialFluxProfile describing the spatial profile of the
        flux.
        """
        return self._spatial_profile
    @spatial_profile.setter
    def spatial_profile(self, profile):
        if(profile is None):
            profile = NullSpatialFluxProfile()
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
            profile = NullEnergyFluxProfile()
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
            profile = NullTimeFluxProfile()
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

    def __call__(self, alpha, delta, E, time, pdict):
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
        pdict : dictionary
            The flux parameter dictionary.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values in unit [energy]^-1 [angle]^-2 [length]^-2 [time]^-1.
            By default that is GeV^-1 sr^-1 cm^-2 s^-1.
        """
        spatial_profile_values = self.spatial_profile(alpha, delta, pdict)
        energy_profile_values = self.energy_profile(E, pdict)
        time_profile_values = self.time_profile(time, pdict)

        flux = (
            self.Phi0 *
            spatial_profile_values[:,np.newaxis,np.newaxis] *
            energy_profile_values[np.newaxis,:,np.newaxis] *
            time_profile_values[np.newaxis,np.newaxis,:]
        )

        return flux
