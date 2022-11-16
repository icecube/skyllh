# -*- coding: utf-8 -*-

"""This module contains classes for IceCube specific detector signal yields,
for a variation of source model and flux model combinations.
"""

import abc
import numpy as np

import scipy.interpolate

from skyllh.core import multiproc
from skyllh.core.py import issequenceof
from skyllh.core.binning import BinningDefinition
from skyllh.core.parameters import ParameterGrid
from skyllh.core.detsigyield import (
    DetSigYield,
    DetSigYieldImplMethod,
    get_integrated_livetime_in_days
)
from skyllh.core.livetime import Livetime
from skyllh.physics.source import PointLikeSource

from skyllh.physics.flux import (
    FluxModel,
    PowerLawFlux,
)


class I3DetSigYield(DetSigYield, metaclass=abc.ABCMeta):
    """Abstract base class for all IceCube specific detector signal yield
    classes. It assumes that sin(dec) binning is required for calculating the
    detector effective area and hence the detector signal yield.
    """

    def __init__(self, implmethod, dataset, fluxmodel, livetime, sin_dec_binning):
        """Constructor of the IceCube specific detector signal yield base
        class.
        """
        super(I3DetSigYield, self).__init__(implmethod, dataset, fluxmodel, livetime)

        self.sin_dec_binning = sin_dec_binning

    @property
    def sin_dec_binning(self):
        """The BinningDefinition instance defining the sin(dec) binning
        definition.
        """
        return self._sin_dec_binning
    @sin_dec_binning.setter
    def sin_dec_binning(self, bd):
        if(not isinstance(bd, BinningDefinition)):
            raise TypeError('The sin_dec_binning property must be an instance '
                'of BinningDefinition!')
        self._sin_dec_binning = bd


class I3DetSigYieldImplMethod(DetSigYieldImplMethod, metaclass=abc.ABCMeta):
    """Abstract base class for an IceCube specific detector signal yield
    implementation method class.
    """

    def __init__(self, sin_dec_binning=None, **kwargs):
        """Constructor of the IceCube specific detector signal yield
        implementation base class.

        Parameters
        ----------
        sin_dec_binning : BinningDefinition instance
            The instance of BinningDefinition defining the binning of sin(dec).
        """
        super(I3DetSigYieldImplMethod, self).__init__(**kwargs)

        self.sin_dec_binning = sin_dec_binning

    @property
    def sin_dec_binning(self):
        """The BinningDefinition instance for the sin(dec) binning that should
        be used for computing the sin(dec) dependency of the detector signal
        yield. If None, the binning is supposed to be taken from the Dataset's
        binning definitions.
        """
        return self._sin_dec_binning
    @sin_dec_binning.setter
    def sin_dec_binning(self, binning):
        if((binning is not None) and
           (not isinstance(binning, BinningDefinition))):
            raise TypeError('The sin_dec_binning property must be None, or '
                'an instance of BinningDefinition!')
        self._sin_dec_binning = binning

    def get_sin_dec_binning(self, dataset):
        """Gets the sin(dec) binning definition either as setting from this
        detector signal yield implementation method itself, or from the
        given dataset.
        """
        sin_dec_binning = self.sin_dec_binning
        if(sin_dec_binning is None):
            if(not dataset.has_binning_definition('sin_dec')):
                raise KeyError('No binning definition named "sin_dec" is '
                    'defined in the dataset and no user defined binning '
                    'definition was provided to this detector signal yield '
                    'implementation method!')
            sin_dec_binning = dataset.get_binning_definition('sin_dec')
        return sin_dec_binning


class PointLikeSourceI3DetSigYieldImplMethod(
        I3DetSigYieldImplMethod, metaclass=abc.ABCMeta):
    """Abstract base class for all IceCube specific detector signal yield
    implementation methods for a point-like source. All IceCube detector signal
    yield implementation methods require a sinDec binning definition for
    the effective area. By default it is taken from the binning definitios
    stored in the dataset, but a user-defined sinDec binning can be specified
    if needed.
    """

    def __init__(self, sin_dec_binning=None, **kwargs):
        """Initializes a new detector signal yield implementation method
        object.

        Parameters
        ----------
        sin_dec_binning : BinningDefinition | None
            The BinningDefinition instance defining the sin(dec) binning that
            should be used to compute the sin(dec) dependency of the detector
            effective area. If set to None, the binning will be taken from the
            Dataset binning definitions.
        """
        super(PointLikeSourceI3DetSigYieldImplMethod, self).__init__(
            sin_dec_binning, **kwargs)

        # Define the supported source models.
        self.supported_sourcemodels = (PointLikeSource,)

    def source_to_array(self, sources):
        """Converts the sequence of PointLikeSource sources into a numpy record
        array holding the spatial information of the sources needed for the
        detector signal yield calculation.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source model containing the spatial information of the source.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the spatial information
            for each source.
        """
        if(isinstance(sources, PointLikeSource)):
            sources = [ sources ]
        if(not issequenceof(sources, PointLikeSource)):
            raise TypeError('The source argument must be an instance of PointLikeSource!')

        arr = np.empty((len(sources),), dtype=[('dec', np.float)])
        for (i, src) in enumerate(sources):
            arr['dec'][i] = src.dec

        return arr


class FixedFluxPointLikeSourceI3DetSigYield(I3DetSigYield):
    """The detector signal yield class for the
    FixedFluxPointLikeSourceI3DetSigYieldImplMethod detector signal yield
    implementation method.
    """
    def __init__(self, implmethod, dataset, fluxmodel, livetime, sin_dec_binning, log_spl_sinDec):
        """Constructs an IceCube detector signal yield instance for a
        point-like source with a fixed flux.

        Parameters
        ----------
        implmethod : FixedFluxPointLikeSourceI3DetSigYieldImplMethod instance
            The instance of the detector signal yield implementation
            method.
        dataset : Dataset instance
            The instance of Dataset holding the monte-carlo data this detector
            signal yield is made for.
        fluxmodel : FluxModel instance
            The instance of FluxModel with fixed parameters this detector signal
            yield is made for.
        livetime : float | Livetime instance
            The livetime in days or an instance of Livetime.
        sin_dec_binning : BinningDefinition instance
            The binning definition for sin(dec).
        log_spl_sinDec : scipy.interpolate.InterpolatedUnivariateSpline
            The spline instance representing the log value of the detector
            signal yield as a function of sin(dec).
        """
        if(not isinstance(implmethod, FixedFluxPointLikeSourceI3DetSigYieldImplMethod)):
            raise TypeError('The implmethod argument must be an instance of '
                'FixedFluxPointLikeSourceI3DetSigYieldImplMethod!')

        super(FixedFluxPointLikeSourceI3DetSigYield, self).__init__(
            implmethod, dataset, fluxmodel, livetime, sin_dec_binning)

        self.log_spl_sinDec = log_spl_sinDec

    @property
    def log_spl_sinDec(self):
        """The :class:`scipy.interpolate.InterpolatedUnivariateSpline` instance
        representing the spline for the log value of the detector signal
        yield as a function of sin(dec).
        """
        return self._log_spl_sinDec
    @log_spl_sinDec.setter
    def log_spl_sinDec(self, spl):
        if(not isinstance(spl, scipy.interpolate.InterpolatedUnivariateSpline)):
            raise TypeError('The log_spl_sinDec property must be an instance '
                'of scipy.interpolate.InterpolatedUnivariateSpline!')
        self._log_spl_sinDec = spl

    def __call__(self, src, src_flux_params=None):
        """Retrieves the detector signal yield for the list of given sources.

        Parameters
        ----------
        src : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_flux_params : None
            Unused interface argument, because this implementation does not
            depend on any source flux fit parameters.

        Returns
        -------
        values : numpy 1d ndarray
            The array with the detector signal yield for each source.
        grads : None
            Because with this implementation the detector signal yield
            does not depend on any fit parameters. So there are no gradients
            and None is returned.
        """
        src_dec = np.atleast_1d(src['dec'])

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float64)

        # Create mask for all source declinations which are inside the
        # declination range.
        mask = (np.sin(src_dec) >= self._sin_dec_binning.lower_edge)\
              &(np.sin(src_dec) <= self._sin_dec_binning.upper_edge)

        values[mask] = np.exp(self._log_spl_sinDec(np.sin(src_dec[mask])))

        return (values, None)


class FixedFluxPointLikeSourceI3DetSigYieldImplMethod(
    PointLikeSourceI3DetSigYieldImplMethod):
    """This detector signal yield implementation method constructs a
    detector signal yield for a fixed flux model, assuming a point-like
    source. This means that the detector signal yield does not depend on
    any source flux parameters, hence it is only dependent on the detector
    effective area.
    It constructs a one-dimensional spline function in sin(dec), using a
    :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

    This detector signal yield implementation method works with all flux
    models.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(self, sin_dec_binning=None, spline_order_sinDec=2, **kwargs):
        """Creates a new IceCube detector signal yield implementation
        method object for a fixed flux model. It requires a sinDec binning
        definition to compute the sin(dec) dependency of the detector effective
        area. The construct class method of this implementation method will
        create a spline function of a given order in logarithmic space of the
        effective area.

        Parameters
        ----------
        sin_dec_binning : BinningDefinition | None
            The BinningDefinition instance which defines the sin(dec) binning.
            If set to None, the binning will be taken from the Dataset binning
            definitions.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(dec) axis.
            The default is 2.
        """
        super(FixedFluxPointLikeSourceI3DetSigYieldImplMethod, self).__init__(
            sin_dec_binning, **kwargs)

        self.supported_fluxmodels = (FluxModel,)

        self.spline_order_sinDec = spline_order_sinDec

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the sin(dec) axis.
        """
        return self._spline_order_sinDec
    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_sinDec property must be of '
                'type int!')
        self._spline_order_sinDec = order

    def construct_detsigyield(self, dataset, data, fluxmodel, livetime, ppbar=None):
        """Constructs a detector signal yield log spline function for the
        given fixed flux model.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance holding meta information about the data.
        data : DatasetData instance
            The DatasetData instance holding the monte-carlo event data.
            The numpy record ndarray holding the monte-carlo event data must
            contain the following data fields:

            - 'true_dec' : float
                The true declination of the data event.
            - 'true_energy' : float
                The true energy value of the data event.
            - 'mcweight' : float
                The monte-carlo weight of the data event in the unit
                GeV cm^2 sr.

        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime
            The live-time in days to use for the detector signal yield.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : FixedFluxPointLikeSourceI3DetSigYield instance
            The DetSigYield instance for point-like source with a fixed flux.
        """
        # Check data types of the input arguments.
        super(FixedFluxPointLikeSourceI3DetSigYieldImplMethod, self).construct_detsigyield(
            dataset, data, fluxmodel, livetime)

        # Get integrated live-time in days.
        livetime_days = get_integrated_livetime_in_days(livetime)

        # Get the sin(dec) binning definition either as setting from this
        # implementation method, or from the dataset.
        sin_dec_binning = self.get_sin_dec_binning(dataset)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = fluxmodel.get_conversion_factor_to_internal_flux_unit()

        # Calculate the detector signal yield contribution of each event.
        # The unit of mcweight is assumed to be GeV cm^2 sr.
        w = data.mc["mcweight"] * fluxmodel(data.mc["true_energy"])*toGeVcm2s * livetime_days * 86400.

        # Create a histogram along sin(true_dec).
        (h, bins) = np.histogram(np.sin(data.mc["true_dec"]),
                                 weights = w,
                                 bins = sin_dec_binning.binedges,
                                 density = False)

        # Normalize by solid angle of each bin which is
        # 2*\pi*(\Delta sin(\delta)).
        h /= (2.*np.pi * np.diff(sin_dec_binning.binedges))

        # Create spline in ln(h) at the histogram's bin centers.
        log_spl_sinDec = scipy.interpolate.InterpolatedUnivariateSpline(
            sin_dec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

        detsigyield = FixedFluxPointLikeSourceI3DetSigYield(
            self, dataset, fluxmodel, livetime, sin_dec_binning, log_spl_sinDec)

        return detsigyield


class SingleFloatingParamFluxPointLikeSourceI3DetSigYield(I3DetSigYield):
    """The detector signal yield class for the
    SingleFloatingParamFluxPointLikeSourceI3DetSigYieldImplMethod detector
    signal yield implementation method.
    """
    def __init__(self, implmethod, dataset, fluxmodel, livetime,
                 sin_dec_binning, log_spl_sinDec_param):
        """Constructs the detector signal yield instance.

        Parameters
        ----------
        implmethod : instance of I3DetSigYieldImplMethod
            The instance of
            SingleFloatingParamFluxPointLikeSourceI3DetSigYieldImplMethod.
        """
        if not isinstance(
                implmethod,
                SingleFloatingParamFluxPointLikeSourceI3DetSigYieldImplMethod):
            raise TypeError('The implmethod argument must be an instance of '
                'PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod!')

        super().__init__(
            implmethod, dataset, fluxmodel, livetime, sin_dec_binning)

        self.log_spl_sinDec_param = log_spl_sinDec_param

    @property
    def log_spl_sinDec_param(self):
        """The :class:`scipy.interpolate.RectBivariateSpline` instance
        representing the spline for the log value of the detector signal
        yield as a function of sin(dec) and the floating parameter.
        """
        return self._log_spl_sinDec_param
    @log_spl_sinDec_param.setter
    def log_spl_sinDec_param(self, spl):
        if(not isinstance(spl, scipy.interpolate.RectBivariateSpline)):
            raise TypeError('The log_spl_sinDec_param property must be an '
                'instance of scipy.interpolate.RectBivariateSpline!')
        self._log_spl_sinDec_param = spl

    def __call__(self, src, src_flux_params):
        """Retrieves the detector signal yield for the given list of
        sources and their flux parameters.

        Parameters
        ----------
        src : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_flux_params : (N_sources,)-shaped numpy record ndarray
            The numpy record ndarray containing the floating flux parameter for
            the sources. Hence, the floating parameter value can be different
            for the different sources.

        Returns
        -------
        values : numpy (N_sources,)-shaped 1D ndarray
            The array with the detector signal yield for each source.
        grads : numpy (N_sources,N_fitparams)-shaped 2D ndarray
            The array containing the gradient values for each source and
            floating parameter.
            Since, this implementation depends on only one floating
            parameter, the array is (N_sources,1)-shaped.
        """
        src_dec = np.atleast_1d(src['dec'])
        src_param = src_flux_params[self.implmethod.param_grid.name]

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float)
        grads = np.zeros_like(src_dec, dtype=np.float)

        # Calculate the detector signal yield only for the sources for
        # which we actually have detector acceptance. For the other sources,
        # the detector signal yield is zero.
        mask = (np.sin(src_dec) >= self._sin_dec_binning.lower_edge) &\
               (np.sin(src_dec) <= self._sin_dec_binning.upper_edge)

        if len(src_param) == len(src_dec):
            src_param = src_param[mask]
        else:
            src_param = src_param[0]

        values[mask] = np.exp(self._log_spl_sinDec_param(
            np.sin(src_dec[mask]), src_param, grid=False))
        grads[mask] = values[mask] * self._log_spl_sinDec_param(
            np.sin(src_dec[mask]), src_param, grid=False, dy=1)

        return (values, np.atleast_2d(grads))


class SingleFloatingParamFluxPointLikeSourceI3DetSigYieldImplMethod(
    PointLikeSourceI3DetSigYieldImplMethod, multiproc.IsParallelizable):
    """This detector signal yield implementation method constructs a
    detector signal yield for a variable flux model with a single floating
    parameter, assuming a point-like source.
    It constructs a two-dimensional spline function in sin(dec) and the fit
    parameter, using a :class:`scipy.interpolate.RectBivariateSpline`.
    Hence, the detector signal yield can vary with the declination and the fit
    parameter of the source.

    This detector signal yield implementation method works with a
    PowerLawFlux flux model.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(
            self, param_grid, sin_dec_binning=None, spline_order_sinDec=2,
            spline_order_param=2, ncpu=None):
        """Creates a new IceCube detector signal yield implementation
        method object for a flux model with a single floating parameter.
        It requires a sinDec binning definition to compute the sin(dec)
        dependency of the detector effective area, and a parameter grid to
        compute the floating parameter dependency of the detector signal yield.

        Parameters
        ----------
        param_grid : ParameterGrid instance
            The ParameterGrid instance which defines the grid of floating
            parameter values. The name of the parameter is defined via the name
            property of the ParameterGrid instance.
        sin_dec_binning : BinningDefinition | None
            The BinningDefinition instance which defines the sin(dec) binning.
            If set to None, the sin(dec) binning will be taken from the
            dataset's binning definitions.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(dec) axis.
            The default is 2.
        spline_order_gamma : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the gamma axis.
            The default is 2.
        ncpu : int | None
            The number of CPUs to utilize. Global setting will take place if
            not specified, i.e. set to None.
        """
        super().__init__(
            sin_dec_binning,
            ncpu=ncpu)

        self.param_grid = param_grid
        self.spline_order_sinDec = spline_order_sinDec
        self.spline_order_param = spline_order_param

    @property
    def param_grid(self):
        """The ParameterGrid instance for the floating parameter grid that
        should be used for computing the floating parameter dependency of the
        detector signal yield.
        """
        return self._param_grid
    @param_grid.setter
    def param_grid(self, grid):
        if(not isinstance(grid, ParameterGrid)):
            raise TypeError('The param_grid property must be an instance of '
                'ParameterGrid!')
        self._param_grid = grid

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the sin(dec) axis.
        """
        return self._spline_order_sinDec
    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_sinDec property must be of '
                'type int!')
        self._spline_order_sinDec = order

    @property
    def spline_order_param(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the floating parameter axis.
        """
        return self._spline_order_param
    @spline_order_param.setter
    def spline_order_param(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_param property must be of '
                'type int!')
        self._spline_order_param = order

    def _get_signal_fitparam_names(self):
        """The list of signal fit parameter names the detector signal yield
        depends on.
        """
        return [self.param_grid.name]

    def construct_detsigyield(
            self, dataset, data, fluxmodel, livetime, ppbar=None):
        """Constructs a detector signal yield 2-dimensional log spline
        function for the given flux model with varying floating parameter
        values.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance holding the sin(dec) binning definition.
        data : DatasetData instance
            The DatasetData instance holding the monte-carlo event data.
            The numpy record array for the monte-carlo data of the dataset must
            contain the following data fields:

            - 'true_dec' : float
                The true declination of the data event.
            - 'mcweight' : float
                The monte-carlo weight of the data event in the unit
                GeV cm^2 sr.
            - 'true_energy' : float
                The true energy value of the data event.

        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime instance
            The live-time in days or an instance of Livetime to use for the
            detector signal yield.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : SingleFloatingParamFluxPointLikeSourceI3DetSigYield instance
            The I3DetSigYield instance for a point-like source with a flux model
            of a single floating parameter.
        """
        # Check for the correct data types of the input arguments.
        super().construct_detsigyield(
            dataset, data, fluxmodel, livetime)

        # Get integrated live-time in days.
        livetime_days = get_integrated_livetime_in_days(livetime)

        # Get the sin(dec) binning definition either as setting from this
        # implementation method, or from the dataset.
        sin_dec_binning = self.get_sin_dec_binning(dataset)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = fluxmodel.get_conversion_factor_to_internal_flux_unit()

        # Define a function that creates a detector signal yield histogram
        # along sin(dec) for a given flux model, i.e. for given spectral index,
        # gamma.
        def _create_hist(
                data_sin_true_dec,
                data_true_energy,
                sin_dec_binning,
                weights,
                fluxmodel):
            """Creates a histogram of the detector signal yield with the
            given sin(dec) binning.

            Parameters
            ----------
            data_sin_true_dec : 1d ndarray
                The sin(true_dec) values of the monte-carlo events.
            data_true_energy : 1d ndarray
                The true energy of the monte-carlo events.
            sin_dec_binning : BinningDefinition
                The sin(dec) binning definition to use for the histogram.
            weights : 1d ndarray
                The weight factors of each monte-carlo event where only the
                flux value needs to be multiplied with in order to get the
                detector signal yield.
            fluxmodel : FluxModel
                The flux model to get the flux values from.

            Returns
            -------
            h : 1d ndarray
                The numpy array containing the histogram values.
            """
            weights = weights * fluxmodel(E=data_true_energy)
            (h, edges) = np.histogram(
                data_sin_true_dec,
                bins=sin_dec_binning.binedges,
                weights=weights,
                density=False)
            return h

        data_sin_true_dec = np.sin(data.mc["true_dec"])
        weights = data.mc["mcweight"] * toGeVcm2s * livetime_days * 86400.

        # Make a copy of the parameter grid and extend the grid by one bin on
        # each side.
        param_grid = self._param_grid.copy()
        param_grid.add_extra_lower_and_upper_bin()

        # Construct the arguments for the hist function to be used in the
        # multiproc.parallelize function.
        args_list = [
            ((data_sin_true_dec,
            data.mc['true_energy'],
            sin_dec_binning,
            weights,
            fluxmodel.copy({param_grid.name:param_val})), {})
                for param_val in param_grid.grid
        ]
        h = np.vstack(
            multiproc.parallelize(
                _create_hist, args_list, self.ncpu, ppbar=ppbar)).T

        # Normalize by solid angle of each bin along the sin(dec) axis.
        # The solid angle is given by 2*\pi*(\Delta sin(\delta)).
        h /= (2.*np.pi * np.diff(sin_dec_binning.binedges)).reshape(
            (sin_dec_binning.nbins,1))

        # Create the 2D spline.
        log_spl_sinDec_param = scipy.interpolate.RectBivariateSpline(
            sin_dec_binning.bincenters,
            param_grid.grid,
            np.log(h),
            kx=self.spline_order_sinDec,
            ky=self.spline_order_param,
            s=0)

        detsigyield = SingleFloatingParamFluxPointLikeSourceI3DetSigYield(
            implmethod=self,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            log_spl_sinDec_param=log_spl_sinDec_param)

        return detsigyield
# Define alias for backward compatibility.
PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod =\
    SingleFloatingParamFluxPointLikeSourceI3DetSigYieldImplMethod
