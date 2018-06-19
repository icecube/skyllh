# -*- coding: utf-8 -*-

import abc
import numpy as np

from astropy import units

import scipy.interpolate

from skylab.core.analysis import BinningDefinition
from skylab.core.detsigeff import DetSigEffImplMethod
from skylab.physics.flux import BaseFluxModel, PowerLawFlux

class I3DetSigEffImplMethod(DetSigEffImplMethod):
    """Abstract base class for all IceCube specific detector signal efficiency
    implementation methods. All IceCube detector signal efficiency
    implementation methods require a sinDec binning definition for the effective
    area.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, sinDec_binning):
        """Initializes a new detector signal efficiency implementation method
        object. It takes a sin(dec) binning definition.

        Parameters
        ----------
        sinDec_binning : BinningDefinition
            The BinningDefinition instance defining the sin(dec) binning that
            should be used to compute the sin(dec) dependency of the detector
            effective area.
        """
        self.sinDec_binning = sinDec_binning

    @property
    def sinDec_binning(self):
        """The BinningDefinition instance for the sinDec binning that should be
        used for computing the sin(dec) dependency of the detector signal
        efficiency.
        """
        return self._sinDec_binning
    @sinDec_bins.setter
    def sinDec_binning(self, binning):
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The sinDec_binning property must be an instance of BinningDefinition!')
        self._sinDec_binning = binning

class I3FixedFluxDetSigEff(DetSigEffImplMethod):
    """This detector signal efficiency implementation method constructs a
    detector signal efficiency for a fixed flux model. This means that the
    detector signal efficiency does not vary with any source flux parameters,
    hence it is only dependent on the detector effective area.

    This detector signal efficiency implementation method works with all flux
    models.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(self, sinDec_binning, spline_order=2):
        """Creates a new IceCube detector signal efficiency implementation
        method object for a fixed flux model. It requires a sinDec binning
        definition to compute the sin(dec) dependency of the detector effective
        area. The construct class method of this implementation method will
        create a spline function of a given order in logarithmic space of the
        effective area.

        Parameters
        ----------
        sinDec_binning : BinningDefinition
            The BinningDefinition instance which defines the sin(dec) binning.
        spline_order : int
            The order of the spline function for the logarithmic values of the
            sin(dec)-dependent detector signal efficiency.
        """
        super(I3FixedFluxDetSigEff, self).__init__(sinDec_binning)

        self.supported_fluxmodels = (BaseFluxModel,)

        self.spline_order = spline_order

    @property
    def spline_order(self):
        """The order (int) of the logarithmic spline function that splines the
        sin(dec) dependent detector signal efficiency histogram.
        """
        return self._spline_order
    @sinDec_bins.setter
    def spline_order(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order property must be of type int!')
        self._spline_order = order

    def construct(self, data_mc, fluxmodel, livetime):
        """Constructs a detector signal efficiency log spline function for the
        given fixed flux model.

        Parameters
        ----------
        data_mc : ndarray
            The numpy record ndarray holding the monte-carlo event data.
        fluxmodel : BaseFluxModel
            The flux model instance. Must be an instance of BaseFluxModel.
        livetime : float
            The live-time in days to use for the detector signal efficiency.
        """
        super(I3FixedFluxDetSigEff, self).construct(data_mc, fluxmodel, livetime)

        # Convert the livetime into the time unit of the flux model.
        T = (livetime * units.day).to(fluxmodel.time_unit).value

        # Calculate conversion factor from the flux model unit into the internal
        # unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = (1./fluxmodel.energy_unit * 1./fluxmodel.length_unit**2 * 1./fluxmodel.time_unit).to(1./units.GeV * 1./units.cm**2 * 1./units.s)

        # Calculate the detector signal efficiency contribution of each event.
        # The unit of mcweight is assumed to be GeV cm^2 sr.
        w = data_mc["mcweight"] * fluxmodel(data_mc["true_energy"])*toGeVcm2s * livetime * 86400.

        # Create a histogram along sin(true_dec).
        (h, bins) = np.histogram(np.sin(mc["true_dec"]),
                                 weights = w,
                                 bins = self.sinDec_binning.binedges,
                                 density = False)

        # Normalize by solid angle of each bin which is
        # 2*\pi*(\Delta sin(\delta)).
        h /= (2.*np.pi * np.diff())

        # Create spline in ln(h) at the histogram's bin centers.
        bincenters = (bins[1:] + bins[:-1]) / 2.
        self._log_spl_sinDec = scipy.interpolate.InterpolatedUnivariateSpline(
            bincenters, np.log(h), k=self.spline_order)

    def get(self, src_pos, src_flux_params):
        """Retrieves the detector signal efficiency for the list of given
        sources.

        Parameters
        ----------
        src_pos : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_flux_params : dict
            The dictionary containing the flux parameters of the sources.
        """
        src_dec = np.atleast_1d(src_pos['dec'])

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float64)

        # Create mask for all source declinations which are inside the
        # declination range.
        mask = (np.sin(src_dec) >= self.sinDec_binning.lower_edge)\
              |(np.sin(src_dec) <= self.sinDec_binning.upper_edge)

        values[mask] = np.exp(self._log_spl_sinDec(np.sin(src_dec[mask])))

        return values

#class I3PowerLawFluxDetSigEff(DetSigEffImplMethod):
    #"""
    #"""
    #def __init__(self):
        #super(I3PowerLawFluxDetSigEff, self).__init__()

        #self.supported_fluxmodels = (PowerLawFlux,)


    #def construct(self, data_mc, fluxmodel, livetime):
        #"""
        #"""



