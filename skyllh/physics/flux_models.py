# -*- coding: utf-8 -*-

"""The `flux_models` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a function for the differential flux::

    d\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) / (dA d\Omega dE dt)

"""

import abc

from copy import deepcopy

from astropy import units

from skyllh.core.py import classname, isproperty


def get_conversion_factor_to_internal_flux_unit(fluxmodel):
    """Calculates the unit conversion factor for converting the used flux
    unit of the given flux model into the skyllh internally used flux unit
    1/(GeV sr cm2 s).

    Parameters
    ----------
    fluxmodel : FluxModel
        The flux model instance for which to calculate the unit conversion
        factor.

    Returns
    -------
    unit_conversion_factor : float
        The unit conversion factor.
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


class FluxModel(object):
    """Abstract base class for all flux models
    ::math:`\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s)`.

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
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of the flux.
        """
        pass

    @abc.abstractmethod
    def __call__(self, alpha, delta, E, t):
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
        t : float | (Ntime,)-shaped 1d numpy ndarray of float
            The MJD observation time for which to retrieve the flux value.

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
