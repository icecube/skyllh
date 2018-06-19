# -*- coding: utf-8 -*-

"""The ``flux`` module contains all standard flux models for a source.
The abstract class ``FluxModel`` serves as a base class for all flux model
classes.
The unit of the resulting flux value must be [energy]^-1 [length]^-2 [time]^-1.
The units are defined using the astropy.units module and can be set through
the properties ``energy_unit``, ``length_unit``, and ``time_unit``.
The default units are [energy] = GeV, [length] = cm, [time] = s.
"""

import abc
import numpy as np

from astropy import units

class BaseFluxModel(object):
    """Abstract base class for all flux models.
    This base class defines the units used for the flux calculation. At this
    point the function form of the flux model is not yet defined.
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
        if(not isinstance(unit, units.Unit)):
            raise TypeError('The property energy_unit must be of type astropy.units.Unit!')
        self._energy_unit = unit

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._length_unit
    @length_unit.setter
    def length_unit(self, unit):
        if(not isinstance(unit, units.Unit)):
            raise TypeError('The property length_unit must be of type astropy.units.Unit!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(not isinstance(unit, units.Unit)):
            raise TypeError('The property time_unit must be of type astropy.units.Unit!')
        self._time_unit = unit

    @property
    def unit_str(self):
        return ' '.join((self.energy_unit.to_string()+'^-1', self.length_unit.to_string()+'^-2', self.time_unit.to_string()+'^-1'))

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

class FluxModel(BaseFluxModel):
    """Abstract base class for all flux models of the form

        dN/dE = A * f(E/E0),

    where A is the flux normalization at E=E0 in the flux unit
    [energy]^-1 [length]^-2 [time]^-1, and f(E/E0) is the unit-less energy
    dependence of the flux.

    The unit of dN/dE is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.

    Attributes
    ----------
    A : float
        Flux value (dN/dE) at E0 in unit [energy]^-1 [length]^-2 [time]^-1.
    E0 : float
        Normalization energy in unit of energy.
    energy_unit : str
        The used unit of energy.
    length_unit : str
        The used unit of length.
    time_unit : str
        The used unit of time.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, A, E0):
        super(FluxModel, self).__init__()
        self.A = A
        self.E0 = E0

    @property
    def A(self):
        """The flux value of dN/dE at energy E0 in unit
        [energy]^-1 [length]^-2 [time]^-1.
        """
        return self._A
    @A.setter
    def A(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property A must be of type float!')
        self._A = val

    @property
    def E0(self):
        """The normalization energy.
        """
        return self._E0
    @E0.setter
    def E0(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property E0 must be of type float!')
        self._E0 = val

class PowerLawFlux(FluxModel):
    """Power law flux of the form

        dN/dE = A * (E / E0)^(-gamma)

    The unit of dN/dE is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, A, E0, gamma):
        """Creates a new power law flux object.

        Parameters
        ----------
        A : float
            Flux value (dN/dE) at E0 in unit [energy]^-1 [length]^-2 [time]^-1.
            By default that is GeV^-1 cm^-2 s^-1.
        E0 : float
            Normalization energy.
        gamma : float
            Spectral index
        """
        super(PowerLawFlux, self).__init__(A, E0)
        self.gamma = gamma

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property gamma must be of type float!')
        self._gamma = val

    @property
    def math_function_str(self):
        return "dN/dE = %.2e * (E / %.2e %s)^-%.2f" \
            % (self.A, self.E0, self.energy_unit, self.gamma)

    def __call__(self, E):
        """The flux value dN/dE at energy E.

        Parameters
        ----------
        E : float | 1d numpy.ndarray of float
            Evaluation energy [GeV]

        Returns
        -------
        flux : float | 1d ndarray of float
            Flux at energy E in unit GeV^-1 cm^-2 s^-1.
        """
        flux = self.A * np.power(E / self.E0, -self.gamma)
        return flux

class CutoffPowerLawFlux(PowerLawFlux):
    """Cut-off power law flux of the form

        dN/dE = A * (E / E0)^(-gamma) * exp(-E/Ecut)

    The unit of dN/dE is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, A, E0, gamma, Ecut):
        """Creates a new cut-off power law flux object.

        Parameters
        ----------
        A : float
            Flux value (dN/dE) at E0 [GeV^-1 cm^-2 s^-1]
        E0 : float
            Normalization energy [GeV]
        gamma : float
            Spectral index
        Ecut : float
            Cut-off energy [GeV]
        """
        super(CutoffPowerLawFlux, self).__init__(A, E0, gamma)
        self.Ecut = Ecut

    @property
    def Ecut(self):
        return self._Ecut
    @Ecut.setter
    def Ecut(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property Ecut must be of type float!')
        self._Ecut = val

    @property
    def math_function_str(self):
        return super(CutoffPowerLawFlux, self).math_function_str + ' * exp(-E / %.2e %s)'%(self.Ecut, self.energy_unit)

    def __call__(self, E):
        """The flux value dN/dE at energy E.

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

class LogParabolaPowerLawFlux(FluxModel):
    """Power law flux with an index which varies as a log parabola in energy of
    the form

        dN/dE = A * (E / E0)^(-alpha - beta*log(E / E0))

    The unit of dN/dE is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.
    """
    def __init__(self, A, E0, alpha, beta):
        super(LogParabolaPowerLawFlux, self).__init__(A, E0)
        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property alpha must be of type float!')
        self._alpha = val

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, val):
        if(not isinstance(val, float)):
            raise TypeError('Property beta must be of type float!')
        self._beta = val

    @property
    def math_function_str(self):
        return 'dN/dE = %.2e * (E / %.2e %s)^(-%.2e - %.2e * log(E / %.2e %s))'%(self.A, self.E0, self.energy_unit, self.alpha, self.beta, self.E0, self.energy_unit)

    def __call__(self, E):
        """The flux value dN/dE at energy E.

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
        flux = self.A * np.power(E / self.E0, -self.alpha - self.beta * np.log(E / self.E0))
        return flux
