# -*- coding: utf-8 -*-

"""Note: This module is deprecated and new flux models should be implemented in
    `flux_model.py`. However, the framework currently doesn't support flux
    models derived from `flux_model.FluxModel`.

The ``flux`` module contains all standard flux models for a source.
The abstract class ``FluxModel`` serves as a base class for all flux model
classes.
The unit of the resulting flux value must be [energy]^-1 [length]^-2 [time]^-1.
The units are defined using the astropy.units module and can be set through
the properties ``energy_unit``, ``length_unit``, and ``time_unit``.
The default units are [energy] = GeV, [length] = cm, [time] = s.
"""
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
    fluxmodel : FluxModel
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


class FluxModel(object, metaclass=abc.ABCMeta):
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

    def __init__(self):
        super(FluxModel, self).__init__()

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


class NormedFluxModel(FluxModel, metaclass=abc.ABCMeta):
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


class SplineFluxModel(FluxModel, metaclass=abc.ABCMeta):
    """Abstract base class for all flux models that are represented
    numerically through photospline splinetables in .fits format

        dN/(dEdAdt) = Phi0 * f(E),

    where Phi0 is the relative flux normalization (dimensionless)
    and f(E) represents the photospline in units of
    [energy]^-1 [length]^-2 [time]^-1 i.e. the energy dependence of the flux.

    The unit of dN/(dEdAdt) is [energy]^-1 [length]^-2 [time]^-1.
    By default the unit is GeV^-1 cm^-2 s^-1.

    Outside of support [crit_log_nu_energy_lower, crit_log_nu_energy_upper] the
    flux will be set to 0.

    Attributes
    ----------
    Phi0 : float
        Flux normalization relative to model prediction.
    psp_table : object
        The photospline.SplineTable object.
    crit_log_nu_energy_lower : float
        Lower end of energy range (support) of spline flux.
    crit_log_nu_energy_upper : float
        Upper end of energy range (support) of spline flux.
    """
    def __init__(self, Phi0, psp_table, crit_log_nu_energy_lower, crit_log_nu_energy_upper):
        super(SplineFluxModel, self).__init__()
        self._psp_table = psp_table
        self._Phi0 = Phi0
        self._crit_log_nu_energy_lower = crit_log_nu_energy_lower
        self._crit_log_nu_energy_upper = crit_log_nu_energy_upper

    @property
    def psp_table(self):
        """The photospline.SplineTable object that describes the neutrino flux
        as function of neutrino energy via B-spline interpolation.
        """
        return self._psp_table
    @psp_table.setter
    def psp_table(self, t):
        self._psp_table = t

    @property
    def Phi0(self):
        """The relative flux normalization. Phi0=1 corresponds to the nominal
        model flux.
        """
        return self._Phi0
    @Phi0.setter
    def Phi0(self, v):
        v = float_cast(v, 'Property Phi0 must be castable to type float!')
        self._Phi0 = v

    @property
    def crit_log_nu_energy_lower(self):
        """The lower bound of the support of the spline interpolator.
        """
        return self._crit_log_nu_energy_lower
    @crit_log_nu_energy_lower.setter
    def crit_log_nu_energy_lower(self, v):
        v = float_cast(
            v, 'Property crit_log_nu_energy_lower must be castable to type float!')
        self._crit_log_nu_energy_lower = v

    @property
    def crit_log_nu_energy_upper(self):
        """The upper bound of the support of the spline interpolator.
        """
        return self._crit_log_nu_energy_upper
    @crit_log_nu_energy_upper.setter
    def crit_log_nu_energy_upper(self, v):
        v = float_cast(
            v, 'Property crit_log_nu_energy_upper must be castable to type float!')
        self._crit_log_nu_energy_upper = v


class SeyfertCoreCoronaFlux(SplineFluxModel):
    """Implements the Core-Corona Seyfert Galaxy neutrino flux model of
    A. Kheirandish et al., Astrophys.J. 922 (2021) 45 by means of B-spline
    interpolation.

    Attributes
    ----------
    Phi0 : float
        Flux normalization relative to model prediction.
    log_xray_lumin : float
        log10 of intrinsic x-ray luminosity of source in 2-10 keV band.
    psp_table : object
        photospline.SplineTable object
    crit_log_nu_energy_lower : float
        Lower end of energy range (support) of spline flux.
    crit_log_nu_energy_upper : float
        Upper end of energy range (support) of spline flux.
    src_dist : float
        Distance to source in units of Mpc.
    lumin_scale : float
        A relative flux scaling factor. Can correct cases when the model
        calculation has a different normalization from what is desired.
    crit_log_energy_flux : float
        The spline is parameterized in log10(flux). This value determines
        when the flux should be considered 0.
    """
    def __init__(
            self, psp_table, log_xray_lumin, src_dist, Phi0,
            lumin_scale=1.0,
            crit_log_energy_flux=-50,
            crit_log_nu_energy_lower=2.0,
            crit_log_nu_energy_upper=7.0):

        super(SeyfertCoreCoronaFlux, self).__init__(
            Phi0, psp_table, crit_log_nu_energy_lower, crit_log_nu_energy_upper)

        self._lumin_scale = lumin_scale
        self._crit_log_energy_flux = crit_log_energy_flux
        self._src_dist = src_dist
        self._log_xray_lumin = log_xray_lumin

    @property
    def log_xray_lumin(self):
        """The log10 of the intrinsic source luminosity in 2-10keV x-ray band.
        """
        return self._log_xray_lumin
    @log_xray_lumin.setter
    def log_xray_lumin(self, v):
        v = float_cast(
            v, 'Property log_xray_lumin must be castable to type float!')
        self._log_xray_lumin = v

    @property
    def lumin_scale(self):
        """Relative factor for model flux normalization correction.
        """
        return self._lumin_scale
    @lumin_scale.setter
    def lumin_scale(self, v):
        v = float_cast(
            v, 'Property lumin_scale must be castable to type float!')
        self._lumin_scale = v

    @property
    def src_dist(self):
        """The distance to the source in units of Mpc.
        """
        return self._src_dist
    @src_dist.setter
    def src_dist(self, v):
        v = float_cast(
            v, 'Property src_dist must be castable to type float!')
        self._src_dist = v

    @property
    def crit_log_energy_flux(self):
        """Defines critical log energy when the flux is considered to be 0.
        """
        return self._crit_log_energy_flux
    @crit_log_energy_flux.setter
    def crit_log_energy_flux(self, v):
        v = float_cast(
            v, 'Property crit_log_energy_flux must be castable to type float!')
        self._crit_log_energy_flux = v

    @property
    def math_function_str(self):
        return (
            f'dN/dE = {self.Phi0:.2f} * {self.lumin_scale:.2f} '
            f'* 10^(log10(f(E)) - 2*log10(E) - 2*log10({self.src_dist:.2f}), '
            f'with log_xray_lumin={self.log_xray_lumin:.2f}'
        )

    def __call__(self, E):
        """The flux value dN/dE at energy E.

        Parameters
        ----------
        E : float | 1D ndarray of float
            Evaluation energy [GeV]

        Returns
        -------
        flux : float | 1D ndarray of float
            Flux at energy E in units of GeV^-1 cm^-2 s^-1.
        """

        log_enu = np.log10(E)
        log_energy_flux = self.psp_table.evaluate_simple([log_enu])

        # Convert energy flux to particle flux accounting for source distance.
        flux = 10**(log_energy_flux - 2.0*log_enu - 2.0*np.log10(self.src_dist))

        # Have to take care of very small fluxes (set to 0 beyond critical
        # energy or below the critical flux).
        out_of_bounds1 = log_energy_flux < self.crit_log_energy_flux
        out_of_bounds2 = np.logical_or(log_enu < self.crit_log_nu_energy_lower,
                                       log_enu > self.crit_log_nu_energy_upper)
        flux[np.logical_or(out_of_bounds1, out_of_bounds2)] = 0

        return self.Phi0 * self.lumin_scale * flux

    def __deepcopy__(self, memo):
        """The photospline.SplineTable objects are strictly immutable.
           Hence no copy should be required, ever!
        """
        return SeyfertCoreCoronaFlux(
            self.psp_table, self.log_xray_lumin, self.src_dist, self.Phi0,
            self.lumin_scale, self.crit_log_energy_flux,
            self.crit_log_nu_energy_lower, self.crit_log_nu_energy_upper
        )

    def __hash__(self):
        """We use hash in
        `skyllh.core.source_hypothesis.get_fluxmodel_to_source_mapping()` for
        mapping fluxes to KDE PDFs. Seyfert model KDEs only depend on the
        `log_xray_lumin` parameter.
        """
        hash_arg = (self.log_xray_lumin,)
        return hash(hash_arg)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.log_xray_lumin == other.log_xray_lumin
        )


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
