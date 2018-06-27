# -*- coding: utf-8 -*-

import abc
import numpy as np

import scipy.interpolate

from skylab.core import multiproc
from skylab.core.analysis import BinningDefinition
from skylab.core.detsigeff import get_conversion_factor_to_internal_flux_unit
from skylab.core.detsigeff import DetSigEffImplMethod
from skylab.physics.flux import FluxModel, PowerLawFlux

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
        """The BinningDefinition instance for the sin(dec) binning that should
        be used for computing the sin(dec) dependency of the detector signal
        efficiency.
        """
        return self._sinDec_binning
    @sinDec_binning.setter
    def sinDec_binning(self, binning):
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The sinDec_binning property must be an instance of BinningDefinition!')
        self._sinDec_binning = binning

class I3FixedFluxDetSigEff(I3DetSigEffImplMethod):
    """This detector signal efficiency implementation method constructs a
    detector signal efficiency for a fixed flux model. This means that the
    detector signal efficiency does not vary with any source flux parameters,
    hence it is only dependent on the detector effective area.
    It constructs a one-dimensional spline function in sin(dec), using a
    scipy.interpolate.InterpolatedUnivariateSpline.

    This detector signal efficiency implementation method works with all flux
    models.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(self, sinDec_binning, spline_order_sinDec=2):
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
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal efficiency along the sin(dec) axis.
            The default is 2.
        """
        super(I3FixedFluxDetSigEff, self).__init__(sinDec_binning)

        self.supported_fluxmodels = (FluxModel,)

        self.spline_order_sinDec = spline_order_sinDec

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal efficiency, along the sin(dec) axis.
        """
        return self._spline_order_sinDec
    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_sinDec property must be of type int!')
        self._spline_order_sinDec = order

    def construct(self, data_mc, fluxmodel, livetime):
        """Constructs a detector signal efficiency log spline function for the
        given fixed flux model.

        Parameters
        ----------
        data_mc : ndarray
            The numpy record ndarray holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float
            The live-time in days to use for the detector signal efficiency.
        """
        super(I3FixedFluxDetSigEff, self).construct(data_mc, fluxmodel, livetime)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)

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
        h /= (2.*np.pi * np.diff(self.sinDec_binning.binedges))

        # Create spline in ln(h) at the histogram's bin centers.
        self._log_spl_sinDec = scipy.interpolate.InterpolatedUnivariateSpline(
            self.sinDec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

    def get(self, src_pos, src_params):
        """Retrieves the detector signal efficiency for the list of given
        sources.

        Parameters
        ----------
        src_pos : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_params : dict
            The dictionary containing the parameters of the sources. For this
            implementation method it is empty.

        Returns
        -------
        values : numpy 1d ndarray
            The array with the detector signal efficiency for each source.
        """
        src_dec = np.atleast_1d(src_pos['dec'])

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float64)

        # Create mask for all source declinations which are inside the
        # declination range.
        mask = (np.sin(src_dec) >= self.sinDec_binning.lower_edge)\
              &(np.sin(src_dec) <= self.sinDec_binning.upper_edge)

        values[mask] = np.exp(self._log_spl_sinDec(np.sin(src_dec[mask])))

        return values

class I3PowerLawFluxDetSigEff(I3DetSigEffImplMethod, multiproc.Parallelizable):
    """This detector signal efficiency implementation method constructs a
    detector signal efficiency for a variable power law flux model.
    It constructs a two-dimensional spline function in sin(dec) and gamma, using a
    scipy.interpolate.RectBivariateSpline. Hence, the detector signal efficiency
    can vary with the declination and the spectral index, gamma, of the source.

    This detector signal efficiency implementation method works with a
    PowerLawFlux flux model.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(self, sinDec_binning, gamma_binning,
                 spline_order_sinDec=2, spline_order_gamma=2, ncpu=None):
        """Creates a new IceCube detector signal efficiency implementation
        method object for a power law flux model. It requires a sinDec binning
        definition to compute the sin(dec) dependency of the detector effective
        area, and a gamma value binning definition to compute the gamma
        dependency of the detector signal efficiency.

        Parameters
        ----------
        sinDec_binning : BinningDefinition
            The BinningDefinition instance which defines the sin(dec) binning.
        gamma_binning : BinningDefinition
            The BinningDefinition instance which defines the gamma binning.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal efficiency along the sin(dec) axis.
            The default is 2.
        spline_order_gamma : int
            The order of the spline function for the logarithmic values of the
            detector signal efficiency along the gamma axis.
            The default is 2.
        ncpu : int | None
            The number of CPUs to utilize. Global setting will take place if
            not specified, i.e. set to None.
        """
        super(I3PowerLawFluxDetSigEff, self).__init__(sinDec_binning)

        self.supported_fluxmodels = (PowerLawFlux,)

        self.gamma_binning = gamma_binning
        self.spline_order_sinDec = spline_order_sinDec
        self.spline_order_gamma = spline_order_gamma
        self.ncpu = ncpu

    @property
    def gamma_binning(self):
        """The BinningDefinition instance for the gamma binning that should be
        used for computing the gamma dependency of the detector signal
        efficiency.
        """
        return self._gamma_binning
    @gamma_binning.setter
    def gamma_binning(self, binning):
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The gamma_binning property must be an instance of BinningDefinition!')
        self._gamma_binning = binning

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal efficiency, along the sin(dec) axis.
        """
        return self._spline_order_sinDec
    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_sinDec property must be of type int!')
        self._spline_order_sinDec = order

    @property
    def spline_order_gamma(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal efficiency, along the gamma axis.
        """
        return self._spline_order_gamma
    @spline_order_gamma.setter
    def spline_order_gamma(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_gamma property must be of type int!')
        self._spline_order_gamma = order

    def construct(self, data_mc, fluxmodel, livetime):
        """Constructs a detector signal efficiency 2-dimensional log spline
        function for the given power law flux model with varying gamma values.

        Parameters
        ----------
        data_mc : ndarray
            The numpy record ndarray holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float
            The live-time in days to use for the detector signal efficiency.
        """
        super(I3PowerLawFluxDetSigEff, self).construct(data_mc, fluxmodel, livetime)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)

        # Define a function that creates a detector signal efficiency histogram
        # along sin(dec) for a given flux model, i.e. for given spectral index,
        # gamma.
        def hist(data_sin_true_dec, data_true_energy, sinDec_binning, weights, fluxmodel):
            """Creates a histogram of the detector signal efficiency with the
            given sin(dec) binning.

            Parameters
            ----------
            data_sin_true_dec : 1d ndarray
                The sin(true_dec) values of the monte-carlo events.
            data_true_energy : 1d ndarray
                The true energy of the monte-carlo events.
            sinDec_binning : BinningDefinition
                The sin(dec) binning definition to use for the histogram.
            weights : 1d ndarray
                The weight factors of each monte-carlo event where only the
                flux value needs to be multiplied with in order to get the
                detector signal efficiency.
            fluxmodel : FluxModel
                The flux model to get the flux values from.

            Returns
            -------
            h : 1d ndarray
                The numpy array containing the histogram values.
            """
            (h, edges) = np.histogram(data_sin_true_dec,
                                      bins = sinDec_binning.binedges,
                                      weights = weights * fluxmodel(data_true_energy),
                                      density = False)
            return h

        data_sin_true_dec = np.sin(data_mc["true_dec"])
        weights = data_mc["mcweight"] * toGeVcm2s * livetime * 86400.

        # Construct the arguments for the hist function to be used in the
        # multiproc.parallelize function.
        args_list = [ ((data_sin_true_dec, data_mc['true_energy'], self.sinDec_binning, weights, fluxmodel.copy({'gamma':gamma})),{})
                     for gamma in self.gamma_binning.binedges ]
        h = np.vstack(multiproc.parallelize(hist, args_list, self.ncpu)).T

        # Normalize by solid angle of each bin along the sin(dec) axis.
        # The solid angle is given by 2*\pi*(\Delta sin(\delta))
        h /= (2.*np.pi * np.diff(self.sinDec_binning.binedges)).reshape((self.sinDec_binning.nbins,1))

        self._log_spl_sinDec_gamma = scipy.interpolate.RectBivariateSpline(
            self.sinDec_binning.bincenters, self.gamma_binning.binedges, np.log(h),
            kx = self.spline_order_sinDec, ky = self.spline_order_gamma, s = 0)

    def get(self, src_pos, src_params):
        """Retrieves the detector signal efficiency for the given list of
        sources and their flux parameters.

        Parameters
        ----------
        src_pos : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.

        src_params : dict
            The dictionary containing the source parameter ``gamma``.

        Returns
        -------
        values : numpy 1d ndarray
            The array with the detector signal efficiency for each source.
        """
        src_dec = np.atleast_1d(src_pos['dec'])
        gamma = src_params['gamma']

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float64)

        mask = (np.sin(src_dec) >= self.sinDec_binning.lower_edge)\
              &(np.sin(src_dec) <= self.sinDec_binning.upper_edge)

        values[mask] = np.exp(self._log_spl_sinDec_gamma(np.sin(src_dec[mask]), gamma, grid=False))

        return values


