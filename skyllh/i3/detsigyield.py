# -*- coding: utf-8 -*-

"""This module contains classes for IceCube specific detector signal yields,
for a variation of source model and flux model combinations.
"""

import abc
import numpy as np

import scipy.interpolate

from skyllh.core import multiproc
from skyllh.core.py import (
    classname,
    issequenceof,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.parameters import (
    ParameterGrid,
)
from skyllh.core.detsigyield import (
    DetSigYield,
    DetSigYieldBuilder,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.source_model import (
    PointLikeSource,
)


class I3DetSigYield(
        DetSigYield,
        metaclass=abc.ABCMeta):
    """Abstract base class for all IceCube specific detector signal yield
    classes. It assumes that sin(dec) binning is required for calculating the
    detector effective area and hence the detector signal yield.
    """

    def __init__(
            self,
            param_names,
            dataset,
            fluxmodel,
            livetime,
            sin_dec_binning,
            **kwargs):
        """Constructor of the IceCube specific detector signal yield base
        class.

        Parameters
        ----------
        param_names : sequence of str
            The sequence of parameter names this detector signal yield depends
            on. These are either fixed or floating parameters.
        dataset : Dataset instance
            The Dataset instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime instance
            The live-time.
        sin_dec_binning : BinningDefinition instance
            The BinningDefinition instance defining the sin(dec) binning.
        """
        super().__init__(
            param_names=param_names,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            **kwargs)

        self.sin_dec_binning = sin_dec_binning

    @property
    def sin_dec_binning(self):
        """The BinningDefinition instance defining the sin(dec) binning.
        """
        return self._sin_dec_binning

    @sin_dec_binning.setter
    def sin_dec_binning(self, bd):
        if not isinstance(bd, BinningDefinition):
            raise TypeError(
                'The sin_dec_binning property must be an instance of '
                f'BinningDefinition! Its current type is {classname(bd)}.')
        self._sin_dec_binning = bd


class I3DetSigYieldBuilder(
        DetSigYieldBuilder,
        metaclass=abc.ABCMeta):
    """Abstract base class for an IceCube specific detector signal yield
    builder class.
    """

    def __init__(self, sin_dec_binning=None, **kwargs):
        """Constructor of the IceCube specific detector signal yield
        builder class.

        Parameters
        ----------
        sin_dec_binning : BinningDefinition instance
            The instance of BinningDefinition defining the sin(dec) binning.
        """
        super().__init__(**kwargs)

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
        if (binning is not None) and\
           (not isinstance(binning, BinningDefinition)):
            raise TypeError(
                'The sin_dec_binning property must be None, or an instance of '
                'BinningDefinition! '
                f'Its current type is "{classname(binning)}".')
        self._sin_dec_binning = binning

    def get_sin_dec_binning(self, dataset):
        """Gets the sin(dec) binning definition either as setting from this
        detector signal yield implementation method itself, or from the
        given dataset.
        """
        sin_dec_binning = self.sin_dec_binning
        if sin_dec_binning is None:
            if not dataset.has_binning_definition('sin_dec'):
                raise KeyError(
                    'No binning definition named "sin_dec" is defined in the '
                    f'dataset "{dataset.name}" and no user defined binning '
                    'definition was provided to this detector signal yield '
                    f'builder "{classname(self)}"!')
            sin_dec_binning = dataset.get_binning_definition('sin_dec')
        return sin_dec_binning


class PointLikeSourceI3DetSigYield(
        I3DetSigYield):
    """Abstract base class for all IceCube specific detector signal yield
    classes for point-like sources.
    """

    def __init__(
            self,
            param_names,
            dataset,
            fluxmodel,
            livetime,
            sin_dec_binning,
            **kwargs):
        """Constructor of the IceCube specific detector signal yield base
        class for point-like sources.

        Parameters
        ----------
        param_names : sequence of str
            The sequence of parameter names this detector signal yield depends
            on. These are either fixed or floating parameters.
        implmethod : instance of DetSigYieldImplMethod
            The implementation method to use for constructing and receiving
            the detector signal yield. The appropriate method depends on
            the used flux model.
        dataset : Dataset instance
            The Dataset instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        sin_dec_binning : BinningDefinition instance
            The BinningDefinition instance defining the sin(dec) binning.
        """
        super().__init__(
            param_names=param_names,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            **kwargs)

    def sources_to_recarray(self, sources):
        """Converts the sequence of PointLikeSource sources into a numpy record
        array holding the information of the sources needed for the
        detector signal yield calculation.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source model(s) containing the information of the source(s).

        Returns
        -------
        recarr : numpy record ndarray
            The generated (N_sources,)-shaped 1D numpy record ndarray holding
            the information for each source.
        """
        if isinstance(sources, PointLikeSource):
            sources = [sources]
        if not issequenceof(sources, PointLikeSource):
            raise TypeError(
                'The sources argument must be an instance or a sequence of '
                'instances of PointLikeSource!')

        recarr = np.empty((len(sources),), dtype=[('dec', np.float64)])
        for (i, src) in enumerate(sources):
            recarr['dec'][i] = src.dec

        return recarr


class PointLikeSourceI3DetSigYieldBuilder(
        I3DetSigYieldBuilder,
        metaclass=abc.ABCMeta):
    """Abstract base class for all IceCube specific detector signal yield
    builders for point-like sources. All IceCube detector signal
    yield builders require a sin(dec) binning definition for
    the effective area. By default it is taken from the binning definitions
    stored in the dataset, but a user-defined sin(dec) binning can be specified
    if needed.
    """

    def __init__(self, sin_dec_binning=None, **kwargs):
        """Initializes a new detector signal yield builder object.

        Parameters
        ----------
        sin_dec_binning : BinningDefinition | None
            The BinningDefinition instance defining the sin(dec) binning that
            should be used to compute the sin(dec) dependency of the detector
            effective area. If set to None, the binning will be taken from the
            Dataset binning definitions.
        """
        super().__init__(
            sin_dec_binning=sin_dec_binning,
            **kwargs)


class FixedFluxPointLikeSourceI3DetSigYield(
        PointLikeSourceI3DetSigYield):
    """The detector signal yield class for a point-source with a fixed flux.
    """
    def __init__(
            self,
            param_names,
            dataset,
            fluxmodel,
            livetime,
            sin_dec_binning,
            log_spl_sinDec,
            **kwargs):
        """Constructs an IceCube detector signal yield instance for a
        point-like source with a fixed flux.

        Parameters
        ----------
        param_names : sequence of str
            The sequence of parameter names this detector signal yield depends
            on. These are either fixed or floating parameters.
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
        super().__init__(
            param_names=param_names,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            **kwargs)

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
        if not isinstance(spl, scipy.interpolate.InterpolatedUnivariateSpline):
            raise TypeError(
                'The log_spl_sinDec property must be an instance '
                'of scipy.interpolate.InterpolatedUnivariateSpline!')
        self._log_spl_sinDec = spl

    def __call__(self, src_recarray, src_params_recarray=None):
        """Retrieves the detector signal yield for the list of given sources.

        Parameters
        ----------
        src_recarray : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_params_recarray : None
            Unused interface argument, because this detector signal yield does
            not depend on any source parameters.

        Returns
        -------
        values : numpy 1d ndarray
            The array with the detector signal yield for each source.
        grads : dict
            This detector signal yield does not depend on any parameters.
            So there are no gradients and the dictionary is empty.
        """
        src_dec = np.atleast_1d(src_recarray['dec'])

        # Create results array.
        values = np.zeros_like(src_dec, dtype=np.float64)

        # Create mask for all source declinations which are inside the
        # declination range.
        mask = (
            (np.sin(src_dec) >= self._sin_dec_binning.lower_edge) &
            (np.sin(src_dec) <= self._sin_dec_binning.upper_edge)
        )

        values[mask] = np.exp(self._log_spl_sinDec(np.sin(src_dec[mask])))

        return (values, {})


class FixedFluxPointLikeSourceI3DetSigYieldBuilder(
        PointLikeSourceI3DetSigYieldBuilder):
    """This detector signal yield builder constructs a
    detector signal yield for a fixed flux model, assuming a point-like
    source. This means that the detector signal yield does not depend on
    any source parameters, hence it is only dependent on the detector
    effective area.
    It constructs a one-dimensional spline function in sin(dec), using a
    :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

    This detector signal yield builder works with all flux models.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """

    def __init__(self, sin_dec_binning=None, spline_order_sinDec=2, **kwargs):
        """Creates a new IceCube detector signal yield builder object for a
        fixed flux model. It requires a sinDec binning definition to compute
        the sin(dec) dependency of the detector effective area.
        The construct_detsigyield class method of this builder will create a
        spline function of a given order in logarithmic space of the
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
        super().__init__(
            sin_dec_binning=sin_dec_binning,
            **kwargs)

        self.spline_order_sinDec = spline_order_sinDec

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the sin(dec) axis.
        """
        return self._spline_order_sinDec

    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if not isinstance(order, int):
            raise TypeError(
                'The spline_order_sinDec property must be of type int! '
                f'Its current type is {classname(order)}.')
        self._spline_order_sinDec = order

    def construct_detsigyield(
            self, dataset, data, fluxmodel, livetime, ppbar=None):
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
        self.assert_types_of_construct_detsigyield_arguments(
            dataset=dataset,
            data=data,
            fluxmodel=fluxmodel,
            livetime=livetime,
            ppbar=ppbar)

        # Get integrated live-time in days.
        livetime_days = Livetime.get_integrated_livetime(livetime)

        # Get the sin(dec) binning definition either as setting from this
        # implementation method, or from the dataset.
        sin_dec_binning = self.get_sin_dec_binning(dataset)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit (usually GeV^-1 cm^-2 s^-1).
        to_internal_flux_unit =\
            fluxmodel.get_conversion_factor_to_internal_flux_unit()

        # Calculate the detector signal yield contribution of each event.
        # The unit of mcweight is assumed to be GeV cm^2 sr.
        weights = (
            data.mc["mcweight"] *
            fluxmodel(data.mc["true_energy"])*to_internal_flux_unit *
            livetime_days*86400.
        )

        # Create a histogram along sin(true_dec).
        (h, bins) = np.histogram(
            np.sin(data.mc["true_dec"]),
            weights=weights,
            bins=sin_dec_binning.binedges,
            density=False)

        # Normalize by solid angle of each bin which is
        # 2*\pi*(\Delta sin(\delta)).
        h /= (2.*np.pi * np.diff(sin_dec_binning.binedges))

        # Create spline in ln(h) at the histogram's bin centers.
        log_spl_sinDec = scipy.interpolate.InterpolatedUnivariateSpline(
            sin_dec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

        detsigyield = FixedFluxPointLikeSourceI3DetSigYield(
            param_names=[],
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            log_spl_sinDec=log_spl_sinDec)

        return detsigyield


class SingleParamFluxPointLikeSourceI3DetSigYield(
        PointLikeSourceI3DetSigYield):
    """The detector signal yield class for a flux that depends on a single
    source parameter.
    """
    def __init__(
            self,
            param_name,
            dataset,
            fluxmodel,
            livetime,
            sin_dec_binning,
            log_spl_sinDec_param,
            **kwargs):
        """Constructs the detector signal yield instance.

        Parameters
        ----------
        param_name : str
            The parameter name this detector signal yield depends
            on. These are either fixed or floating parameter.
        dataset : Dataset instance
            The Dataset instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime instance
            The live-time.
        sin_dec_binning : BinningDefinition instance
            The BinningDefinition instance defining the sin(dec) binning.
        log_spl_sinDec_param : scipy.interpolate.RectBivariateSpline instance
            The 2D spline in sin(dec) and the parameter this detector signal
            yield depends on.
        """
        super().__init__(
            param_names=[param_name],
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            **kwargs)

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
        if not isinstance(spl, scipy.interpolate.RectBivariateSpline):
            raise TypeError(
                'The log_spl_sinDec_param property must be an instance of '
                'scipy.interpolate.RectBivariateSpline! '
                f'Its current type is {classname(spl)}.')
        self._log_spl_sinDec_param = spl

    def __call__(self, src_recarray, src_params_recarray):
        """Retrieves the detector signal yield for the given list of
        sources and their flux parameters.

        Parameters
        ----------
        src_recarray : numpy record ndarray
            The numpy record ndarray with the field ``dec`` holding the
            declination of the source.
        src_params_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record ndarray containing the parameter values of the
            sources. The parameter values can be different for the different
            sources.
            The record array needs to contain two fields for each source
            parameter, one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        values : numpy (N_sources,)-shaped 1D ndarray
            The array with the detector signal yield for each source.
        grads : dict
            The dictionary holding the gradient values for each global floating
            parameter. The key is the global floating parameter index and the
            value is the (N_sources,)-shaped numpy ndarray holding the gradient
            value dY_k/dp_s.
        """
        local_param_name = self.param_names[0]

        src_dec = np.atleast_1d(src_recarray['dec'])
        src_param = src_params_recarray[local_param_name]
        src_param_gp_idxs = src_params_recarray[f'{local_param_name}:gpidx']

        n_sources = len(src_dec)

        # Check for correct input format.
        if not (len(src_param) == n_sources and
                len(src_param_gp_idxs) == n_sources):
            raise RuntimeError(
                f'The length ({len(src_param)}) of the array for the '
                f'source parameter "{local_param_name}" does not match the '
                f'number of sources ({n_sources})!')

        # Calculate the detector signal yield only for the sources for
        # which we actually have detector acceptance. For the other sources,
        # the detector signal yield is zero.
        src_mask = (np.sin(src_dec) >= self._sin_dec_binning.lower_edge) &\
                   (np.sin(src_dec) <= self._sin_dec_binning.upper_edge)

        values = np.zeros((n_sources,), dtype=np.float64)
        values[src_mask] = np.exp(self._log_spl_sinDec_param(
            np.sin(src_dec[src_mask]), src_param[src_mask], grid=False))

        # Determine the number of global parameters the local parameter is
        # made of.
        gfp_idxs = np.unique(src_param_gp_idxs)
        gfp_idxs = gfp_idxs[gfp_idxs > 0] - 1

        # Calculate the gradients for each global fit parameter.
        grads = dict()
        for gfp_idx in gfp_idxs:
            # Create the gradient array of shape (n_sources,). This could be
            # a masked array to save memory, when there are many sources and
            # global fit parameters.
            grads[gfp_idx] = np.zeros((n_sources,), dtype=np.float64)

            # Create a mask to select the sources that depend on the global
            # fit parameter with index gfp_idx.
            gfp_src_mask = (src_param_gp_idxs == gfp_idx+1)

            # m is a (n_sources,)-shaped ndarray, which selects only sources
            # that have detector exceptance and depend on the global fit
            # parameter gfp_idx.
            m = src_mask & gfp_src_mask

            grads[gfp_idx][m] = values[m] * self._log_spl_sinDec_param(
                np.sin(src_dec[m]), src_param[m], grid=False, dy=1)

        return (values, grads)


class SingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
        PointLikeSourceI3DetSigYieldBuilder,
        multiproc.IsParallelizable):
    """This detector signal yield builder constructs a
    detector signal yield for a variable flux model with a single parameter,
    assuming a point-like source.
    It constructs a two-dimensional spline function in sin(dec) and the
    parameter, using a :class:`scipy.interpolate.RectBivariateSpline`.
    Hence, the detector signal yield can vary with the declination and the
    parameter of the flux model.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.
    """
    def __init__(
            self,
            param_grid,
            sin_dec_binning=None,
            spline_order_sinDec=2,
            spline_order_param=2,
            ncpu=None,
            **kwargs):
        """Creates a new IceCube detector signal yield builder instance for a
        flux model with a single parameter.
        It requires a sinDec binning definition to compute the sin(dec)
        dependency of the detector effective area, and a parameter grid to
        compute the parameter dependency of the detector signal yield.

        Parameters
        ----------
        param_grid : instance of ParameterGrid
            The instance of ParameterGrid which defines the grid of the
            parameter values. The name of the parameter is defined via the name
            property of the ParameterGrid instance.
        sin_dec_binning : instance of BinningDefinition | None
            The instance of BinningDefinition which defines the sin(dec)
            binning. If set to None, the sin(dec) binning will be taken from the
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
            The number of CPUs to utilize. If set to ``None``, global setting
            will take place.
        """
        super().__init__(
            sin_dec_binning,
            ncpu=ncpu,
            **kwargs)

        self.param_grid = param_grid
        self.spline_order_sinDec = spline_order_sinDec
        self.spline_order_param = spline_order_param

    @property
    def param_grid(self):
        """The ParameterGrid instance for the parameter grid that should be used
        for computing the parameter dependency of the detector signal yield.
        """
        return self._param_grid

    @param_grid.setter
    def param_grid(self, grid):
        if not isinstance(grid, ParameterGrid):
            raise TypeError(
                'The param_grid property must be an instance of '
                'ParameterGrid! '
                f'Its current type is {classname(grid)}.')
        self._param_grid = grid

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the sin(dec) axis.
        """
        return self._spline_order_sinDec

    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if not isinstance(order, int):
            raise TypeError(
                'The spline_order_sinDec property must be of type int! '
                f'Its current type is {classname(order)}.')
        self._spline_order_sinDec = order

    @property
    def spline_order_param(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the parameter axis.
        """
        return self._spline_order_param

    @spline_order_param.setter
    def spline_order_param(self, order):
        if not isinstance(order, int):
            raise TypeError(
                'The spline_order_param property must be of type int! '
                f'Its current type is {classname(order)}.')
        self._spline_order_param = order

    def construct_detsigyield(
            self,
            dataset,
            data,
            fluxmodel,
            livetime,
            ppbar=None):
        """Constructs a detector signal yield 2-dimensional log spline
        function for the given flux model with varying parameter values.

        Parameters
        ----------
        dataset : instance of Dataset
            The instance of Dataset holding the sin(dec) binning definition.
        data : instance of DatasetData
            The instance of DatasetData holding the monte-carlo event data.
            The numpy record array for the monte-carlo data of the dataset must
            contain the following data fields:

            ``'true_dec'`` : float
                The true declination of the data event.
            ``'mcweight'`` : float
                The monte-carlo weight of the data event in the unit
                GeV cm^2 sr.
            ``'true_energy'`` : float
                The true energy value of the data event.

        fluxmodel : instance of FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | instance of Livetime
            The live-time in days or an instance of Livetime to use for the
            detector signal yield.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : instance of SingleParamFluxPointLikeSourceI3DetSigYield
            The I3DetSigYield instance for a point-like source with a flux model
            of a single parameter.
        """
        self.assert_types_of_construct_detsigyield_arguments(
            dataset=dataset,
            data=data,
            fluxmodel=fluxmodel,
            livetime=livetime,
            ppbar=ppbar)

        # Get integrated live-time in days.
        livetime_days = Livetime.get_integrated_livetime(livetime)

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
            weights = weights * fluxmodel(E=data_true_energy).squeeze()
            (h, edges) = np.histogram(
                data_sin_true_dec,
                bins=sin_dec_binning.binedges,
                weights=weights,
                density=False)
            return h

        data_sin_true_dec = np.sin(data.mc['true_dec'])

        # Generate a list of indices that would sort the data according to the
        # sin(true_dec) values. We will sort the MC data according to it,
        # because the histogram creation is much faster (2x) when the
        # to-be-histogrammed values are already sorted.
        sorted_idxs = np.argsort(data_sin_true_dec)

        data_sin_true_dec = np.take(data_sin_true_dec, sorted_idxs)
        data_true_energy = np.take(data.mc['true_energy'], sorted_idxs)

        weights = (
            np.take(data.mc['mcweight'], sorted_idxs) *
            toGeVcm2s * livetime_days * 86400.
        )

        # Make a copy of the parameter grid and extend the grid by one bin on
        # each side.
        param_grid = self._param_grid.copy()
        param_grid.add_extra_lower_and_upper_bin()

        # Construct the arguments for the hist function to be used in the
        # multiproc.parallelize function.
        args_list = [
            (
                (
                    data_sin_true_dec,
                    data_true_energy,
                    sin_dec_binning,
                    weights,
                    fluxmodel.copy({param_grid.name: param_val})
                ),
                {}
            )
            for param_val in param_grid.grid
        ]
        h = np.vstack(
            multiproc.parallelize(
                _create_hist, args_list, self.ncpu, ppbar=ppbar)).T

        # Normalize by solid angle of each bin along the sin(dec) axis.
        # The solid angle is given by 2*\pi*(\Delta sin(\delta)).
        h /= (2.*np.pi * np.diff(sin_dec_binning.binedges)).reshape(
            (sin_dec_binning.nbins, 1))

        # Create the 2D spline.
        log_spl_sinDec_param = scipy.interpolate.RectBivariateSpline(
            sin_dec_binning.bincenters,
            param_grid.grid,
            np.log(h),
            kx=self.spline_order_sinDec,
            ky=self.spline_order_param,
            s=0)

        detsigyield = SingleParamFluxPointLikeSourceI3DetSigYield(
            param_name=self._param_grid.name,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            log_spl_sinDec_param=log_spl_sinDec_param)

        return detsigyield
