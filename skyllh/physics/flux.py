# -*- coding: utf-8 -*-

"""The ``flux`` module contains all standard flux models for a source.
The abstract class ``FluxModel`` serves as a base class for all flux model
classes.
The unit of the resulting flux value must be [energy]^-1 [length]^-2 [time]^-1.
The units are defined using the astropy.units module and can be set through
the properties ``energy_unit``, ``length_unit``, and ``time_unit``.
The default units are [energy] = GeV, [length] = cm, [time] = s.
"""
from __future__ import division

import abc
import numpy as np

from copy import deepcopy

from astropy import units

from skyllh.core.py import classname, isproperty, float_cast
from skyllh.core.config import CFG


def get_conversion_factor_to_internal_flux_unit(fluxmodel):
    """Calculates the unit conversion factor for converting the used flux
    unit of the given flux model into the skyllh internally used flux unit
    1/(GeV cm2 s).

    Parameters
    ----------
    fluxmodel : ~skyllh.physics.flux.FluxModel
        The flux model instance for which to calculate the unit conversion
        factor.

    Returns
    -------
    unit_conversion_factor : float
        The unit conversion factor.
    """
    fluxmodel_flux_unit = 1/(
        fluxmodel.energy_unit * fluxmodel.length_unit**2 * fluxmodel.time_unit)

    internal_units = CFG['internal_units']
    internal_flux_unit = 1/(
        internal_units['energy'] * internal_units['length']**2 *
        internal_units['time'])

    unit_conversion_factor = (fluxmodel_flux_unit).to(internal_flux_unit).value
    return unit_conversion_factor


class FluxModel(object):
    """Abstract base class for all flux models.
    This base class defines the units used for the flux calculation. At this
    point the function form of the flux model is not yet defined.

    Attributes
    ----------
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
        # Define the default units.
        self.energy_unit = units.GeV
        self.length_unit = units.cm
        self.time_unit = units.s

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation.
        """
        return self._energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type astropy.units.UnitBase!')
        self._energy_unit = unit

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._length_unit
    @length_unit.setter
    def length_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property length_unit must be of type astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property time_unit must be of type astropy.units.UnitBase!')
        self._time_unit = unit

    @property
    def unit_str(self):
        """The string representation of the flux unit.
        """
        return '1/(%s %s^2 %s)'%(
            self.energy_unit.to_string(), self.length_unit.to_string(),
            self.time_unit.to_string())

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit.
        """
        return r'%s$^{-1}$ %s$^{-2}$ %s$^{-1}$'%(
            self.energy_unit.to_string(), self.length_unit.to_string(),
            self.time_unit.to_string())

    @property
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of the flux calculation.
        """
        pass

    @abc.abstractmethod
    def __call__(self, E):
        """The call operator to retrieve a flux value for a given energy.

        Parameters
        ----------
        E : float | 1d numpy.ndarray of float
            The energy for which to retrieve the flux value.

        Returns
        -------
        flux : ndarray of float
            Flux at energy E in unit [energy]^-1 [length]^-2 [time]^-1.
            By default that is GeV^-1 cm^-2 s^-1.
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
        for (prop, val) in propdict.items():
            if(not hasattr(self, prop)):
                raise KeyError('The flux model "%s" does not have a property named "%s"!'%(classname(self), prop))
            if(not isproperty(self, prop)):
                raise TypeError('The attribute "%s" of flux model "%s" is no property!'%(classname(self), prop))
            setattr(self, prop, val)

class NormedFluxModel(FluxModel):
    """Abstract base class for all normalized flux models of the form

        dN/(dEdAdt) = Phi0 * f(E/E0),

    where Phi0 is the flux normalization at E=E0 in the flux unit
    [energy]^-1 [length]^-2 [time]^-1, and f(E/E0) is the unit-less energy
    dependence of the flux.

    The unit of dN/(dEdAdt) is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.

    Attributes
    ----------
    Phi0 : float
        Flux value (dN/(dEdAdt)) at E0 in unit
        [energy]^-1 [length]^-2 [time]^-1.
    E0 : float
        Normalization energy in unit of energy.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, Phi0, E0):
        super(NormedFluxModel, self).__init__()
        self.Phi0 = Phi0
        self.E0 = E0

    @property
    def Phi0(self):
        """The flux value (dN/(dEdAdt)) at energy E0 in unit
        [energy]^-1 [length]^-2 [time]^-1.
        """
        return self._Phi0
    @Phi0.setter
    def Phi0(self, v):
        v = float_cast(v, 'Property Phi0 must be castable to type float!')
        self._Phi0 = v

    @property
    def E0(self):
        """The normalization energy.
        """
        return self._E0
    @E0.setter
    def E0(self, v):
        v = float_cast(v, 'Property E0 must be castable to type float!')
        self._E0 = v

class PowerLawFlux(NormedFluxModel):
    """Power law flux of the form

        dN/(dEdAdt) = Phi0 * (E / E0)^(-gamma)

    The unit of dN/(dEdAdt) is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, Phi0, E0, gamma):
        """Creates a new power law flux object.

        Parameters
        ----------
        Phi0 : float
            Flux value (dN/(dEdAdt)) at E0 in unit
            [energy]^-1 [length]^-2 [time]^-1.
            By default that is GeV^-1 cm^-2 s^-1.
        E0 : float
            Normalization energy.
        gamma : float
            Spectral index
        """
        super(PowerLawFlux, self).__init__(Phi0, E0)
        self.gamma = gamma

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, v):
        v = float_cast(v, 'Property gamma must be castable to type float!')
        self._gamma = v

    @property
    def math_function_str(self):
        return "dN/dE = %.2e * (E / %.2e %s)^-%.2f" \
            % (self.Phi0, self.E0, self.energy_unit, self.gamma)

    def __call__(self, E):
        """The flux value dN/dE at energy E.

        Parameters
        ----------
        E : float | 1d numpy.ndarray of float
            Evaluation energy [GeV]

        Returns
        -------
        flux : float | 1d ndarray of float
            Flux at energy E in unit [energy]^-1 [length]^-2 [time]^-1.
            By default in GeV^-1 cm^-2 s^-1.
        """
        flux = self.Phi0 * np.power(E / self.E0, -self.gamma)
        return flux

class CutoffPowerLawFlux(PowerLawFlux):
    """Cut-off power law flux of the form

        dN/(dEdAdt) = Phi0 * (E / E0)^(-gamma) * exp(-E/Ecut)

    The unit of dN/(dEdAdt) is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, Phi0, E0, gamma, Ecut):
        """Creates a new cut-off power law flux object.

        Parameters
        ----------
        Phi0 : float
            Flux value (dN/(dEdAdt)) at E0 in unit
            [energy]^-1 [length]^-2 [time]^-1. By default the unit is
            GeV^-1 cm^-2 s^-1.
        E0 : float
            Normalization energy [GeV]
        gamma : float
            Spectral index
        Ecut : float
            Cut-off energy [GeV]
        """
        super(CutoffPowerLawFlux, self).__init__(Phi0, E0, gamma)
        self.Ecut = Ecut

    @property
    def Ecut(self):
        return self._Ecut
    @Ecut.setter
    def Ecut(self, val):
        val = float_cast(val, 'Property val must be castable to type float!')
        self._Ecut = val

    @property
    def math_function_str(self):
        return super(CutoffPowerLawFlux, self).math_function_str + ' * exp(-E / %.2e %s)'%(self.Ecut, self.energy_unit)

    def __call__(self, E):
        """The flux value dN/(dEdAdt) at energy E.

        Parameters
        ----------
        E : float | 1d numpy.ndarray of float
            Evaluation energy.

        Returns
        -------
        flux : float | 1d ndarray of float
            Flux at energy E in unit [energy]^-1 [length]^-2 [time]^-1.
            By default that is GeV^-1 cm^-2 s^-1.
        """
        flux = super(CutoffPowerLawFlux, self).__call__(E) * np.exp(-E / self.Ecut)
        return flux

class LogParabolaPowerLawFlux(NormedFluxModel):
    """Power law flux with an index which varies as a log parabola in energy of
    the form

        dN/(dEdAdt) = Phi0 * (E / E0)^(-(alpha + beta*log(E / E0)))

    The unit of dN/(dEdAdt) is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, Phi0, E0, alpha, beta):
        super(LogParabolaPowerLawFlux, self).__init__(Phi0, E0)
        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, v):
        v = float_cast(v, 'Property alpha must be castable to type float!')
        self._alpha = v

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, v):
        v = float_cast(v, 'Property beta must be castable to type float!')
        self._beta = v

    @property
    def math_function_str(self):
        return 'dN/dE = %.2e * (E / %.2e %s)^(-(%.2e + %.2e * log(E / %.2e %s)))'%(self.Phi0, self.E0, self.energy_unit, self.alpha, self.beta, self.E0, self.energy_unit)

    def __call__(self, E):
        """The flux value dN/(dEdAdt) at energy E.

        Parameters
        ----------
        E : float | 1d numpy.ndarray of float
            The evaluation energy.

        Returns
        -------
        flux : float | 1d ndarray of float
            Flux at energy E in unit [energy]^-1 [length]^-2 [time]^-1.
            By default that is GeV^-1 cm^-2 s^-1.
        """
        flux = self.Phi0 * np.power(E / self.E0, -self.alpha - self.beta * np.log(E / self.E0))
        return flux
