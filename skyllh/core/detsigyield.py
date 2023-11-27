# -*- coding: utf-8 -*-

import abc

import numpy as np

import scipy.interpolate

from astropy import (
    units,
)

from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.config import (
    Config,
    HasConfig,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.flux_model import (
    FluxModel,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize,
)
from skyllh.core.parameters import (
    ParameterGrid,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    issequence,
    issequenceof,
)
from skyllh.core.sidereal_time import (
    SiderealTimeService,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.types import (
    SourceHypoGroup_t,
)


class DetSigYield(
        object,
        metaclass=abc.ABCMeta,
):
    """This is the abstract base class for a detector signal yield.

    The detector signal yield, Y_s(p_s), is defined as the expected mean
    number of signal events detected by the detector from a given source with
    source parameters p_s.

    To construct a detector signal yield object, four ingredients are
    needed: the dataset holding the monte-carlo data events, a signal flux
    model, the livetime, and a builder instance that knows how to construct
    the actual detector yield in an efficient way.
    In general, the implementation method depends on the detector, the source,
    the flux model with its flux model's signal parameters, and the dataset.
    Hence, for a given detector, source, flux model, and dataset, an appropriate
    implementation method needs to be chosen.
    """
    def __init__(
            self,
            param_names,
            detector_model,
            dataset,
            fluxmodel,
            livetime,
            **kwargs,
    ):
        """Constructs a new detector signal yield object. It takes
        the monte-carlo data events, a flux model of the signal, and the live
        time to compute the detector signal yield.

        Parameters
        ----------
        param_names : sequence of str
            The sequence of parameter names this detector signal yield depends
            on. These are either fixed or floating parameters.
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the detector for this
            detector signal yield.
        dataset : Dataset instance
            The Dataset instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime
            The live-time in days to use for the detector signal yield.
        """
        super().__init__(**kwargs)

        self.param_names = param_names
        self.detector_model = detector_model
        self.dataset = dataset
        self.fluxmodel = fluxmodel
        self.livetime = livetime

    @property
    def param_names(self):
        """The tuple of parameter names this detector signal yield instance
        is a function of.
        """
        return self._param_names

    @param_names.setter
    def param_names(self, names):
        if not issequence(names):
            names = [names]
        if not issequenceof(names, str):
            raise TypeError(
                'The param_names property must be a sequence of str '
                'instances! '
                f'Its current type is {classname(names)}.')
        self._param_names = tuple(names)

    @property
    def detector_model(self):
        """The instance of DetectorModel, for which this detector signal yield
        is made for.
        """
        return self._detector_model

    @detector_model.setter
    def detector_model(self, model):
        if not isinstance(model, DetectorModel):
            raise TypeError(
                'The detector_model property must be an instance of '
                'DetectorModel! '
                f'Its current type is {classname(model)}!')
        self._detector_model = model

    @property
    def dataset(self):
        """The Dataset instance, for which this detector signal yield is made
        for.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, ds):
        if not isinstance(ds, Dataset):
            raise TypeError(
                'The dataset property must be an instance of Dataset! '
                f'Its current type is {classname(ds)}.')
        self._dataset = ds

    @property
    def fluxmodel(self):
        """The flux model, which should be used to calculate the detector
        signal yield.
        """
        return self._fluxmodel

    @fluxmodel.setter
    def fluxmodel(self, model):
        if not isinstance(model, FluxModel):
            raise TypeError(
                'The fluxmodel property must be an instance of FluxModel! '
                f'Its current type is {classname(model)}.')
        self._fluxmodel = model

    @property
    def livetime(self):
        """The live-time in days.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if not (isinstance(lt, float) or isinstance(lt, Livetime)):
            raise TypeError(
                'The livetime property must be of type float or an instance '
                'of Livetime! '
                f'Its current type is {classname(lt)}.')
        self._livetime = lt

    @abc.abstractmethod
    def sources_to_recarray(
            self,
            sources,
    ):
        """This method is supposed to convert a (list of) source model(s) into
        a numpy record array that is understood by the detector signal yield
        class.
        This is for efficiency reasons only. This way the user code can
        pre-convert the list of sources into a numpy record array and cache the
        array.
        The fields of the array are detector signal yield implementation
        dependent, i.e. what kind of sources: point-like source, or extended
        source for instance. Because the sources usually don't change their
        position in the sky, this has to be done only once.

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
        pass

    @abc.abstractmethod
    def __call__(
            self,
            src_recarray,
            src_params_recarray,
    ):
        """Abstract method to retrieve the detector signal yield for the given
        sources and source parameter values.

        Parameters
        ----------
        src_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record array containing the information of the sources.
            The required fields of this record array are implementation
            dependent. In the most generic case for a point-like source, it
            must contain the following three fields: ra, dec.
        src_params_recarray : (N_sources,)-shaped numpy record ndarray
            The numpy record ndarray containing the parameter values of the
            sources. The parameter values can be different for the different
            sources.
            The record array must contain two fields for each source parameter,
            one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        detsigyield : (N_sources,)-shaped 1D ndarray of float
            The array with the mean number of signal in the detector for each
            given source.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter. The key is the global fit parameter index and the value
            is the (N_sources,)-shaped numpy ndarray holding the gradient value
            dY_k/dp_s.
        """
        pass


class DetSigYieldBuilder(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """Abstract base class for a builder of a detector signal yield. Via the
    ``construct_detsigyield`` method it creates a DetSigYield instance holding
    the internal objects to calculate the detector signal yield.
    """

    def __init__(
            self,
            **kwargs,
    ):
        """Constructor.
        """
        super().__init__(
            **kwargs)

    def assert_types_of_construct_detsigyield_arguments(
            self,
            detector_model,
            dataset,
            data,
            shgs,
            ppbar,
    ):
        """Checks the types of the arguments for the ``construct_detsigyield``
        method. It raises errors if the arguments have the wrong type.
        """
        if not isinstance(detector_model, DetectorModel):
            raise TypeError(
                'The detector_model argument must be an instance of '
                'DetectorModel! '
                f'Its current type is {classname(detector_model)}.')

        if not isinstance(dataset, Dataset):
            raise TypeError(
                'The dataset argument must be an instance of Dataset! '
                f'Its current type is {classname(dataset)}.')

        if not isinstance(data, DatasetData):
            raise TypeError(
                'The data argument must be an instance of DatasetData! '
                f'Its current type is {classname(data)}.')

        if (not isinstance(shgs, SourceHypoGroup_t)) and\
           (not issequenceof(shgs, SourceHypoGroup_t)):
            raise TypeError(
                'The shgs argument must be an instance of SourceHypoGroup '
                'or a sequence of SourceHypoGroup instances!'
                f'Its current type is {classname(shgs)}.')

        if ppbar is not None:
            if not isinstance(ppbar, ProgressBar):
                raise TypeError(
                    'The ppbar argument must be an instance of ProgressBar! '
                    f'Its current type is {classname(ppbar)}.')

    def get_detsigyield_construction_factory(self):
        """This method is supposed to return a callable with the call-signature

        .. code::

            __call__(
                dataset,
                data,
                shgs,
                ppbar,
            )


        to construct several DetSigYield instances, one for each provided
        source hypo group (i.e. sources and fluxmodel).
        The return value of this callable must be a sequence of DetSigYield
        instances of the same length as the sequence of ``shgs``.

        Returns
        -------
        factory : callable | None
            This default implementation returns ``None``, indicating that a
            factory is not supported by this builder.
        """
        return None

    @abc.abstractmethod
    def construct_detsigyield(
            self,
            detector_model,
            dataset,
            data,
            shg,
            ppbar=None,
    ):
        """Abstract method to construct the DetSigYield instance.
        This method must be called by the derived class method implementation
        to ensure the compatibility check of the given flux model with the
        supported flux models.

        Parameters
        ----------
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the detector for the
            detector signal yield.
        dataset : instance of Dataset
            The instance of Dataset holding possible dataset specific settings.
        data : instance of DatasetData
            The instance of DatasetData holding the monte-carlo event data.
        shg : instance of SourceHypoGroup
            The instance of SourceHypoGroup (i.e. sources and flux model) for
            which the detector signal yield should be constructed.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : instance of DetSigYield
            An instance derived from DetSigYield.
        """
        pass


class NullDetSigYieldBuilder(
        DetSigYieldBuilder):
    """This class provides a dummy detector signal yield builder, which can
    be used for testing purposes, when an actual builder is not required.
    """
    def __init__(
            self,
            cfg=None,
            **kwargs,
    ):
        """Creates a new instance of NullDetSigYieldBuilder.

        Parameters
        ----------
        cfg : instance of Config | None
            The instance of Config holding the local configuration. Since this
            detector signal yield builder does nothing, this argument is
            optional. If not provided the default configuration is used.
        """
        if cfg is None:
            cfg = Config()

        super().__init__(
            cfg=cfg,
            **kwargs)

    def construct_detsigyield(
            self,
            *args,
            **kwargs,
    ):
        """Since this is a dummy detector signal yield builder, calling this
        method will raise a NotImplementedError!
        """
        raise NotImplementedError(
            f'The {classname(self)} detector signal yield builder cannot '
            'actually build a DetSigYield instance!')


class PointLikeSourceDetSigYield(
        DetSigYield,
):
    """Abstract base class for all detector signal yield classes for point-like
    sources.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sources_to_recarray(self, sources):
        """Converts the sequence of PointLikeSource sources into a numpy
        structured array holding the information of the sources needed for the
        detector signal yield calculation.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source model(s) containing the information of the source(s).

        Returns
        -------
        arr : instance of numpy.ndarray
            The generated (N_sources,)-shaped structured numpy ndarray holding
            the information for each source. This array contains the following
            fields:

                ``'dec'`` : float
                    The declination of the source.
                ``'ra'`` : float
                    The right-ascention of the source.

        """
        if isinstance(sources, PointLikeSource):
            sources = [sources]
        if not issequenceof(sources, PointLikeSource):
            raise TypeError(
                'The sources argument must be an instance or a sequence of '
                'instances of PointLikeSource!')

        arr_dtype = [
            ('dec', np.float64),
            ('ra', np.float64),
        ]

        arr = np.empty((len(sources),), dtype=arr_dtype)
        for (i, src) in enumerate(sources):
            arr['dec'][i] = src.dec
            arr['ra'][i] = src.ra

        return arr


class SingleParamFluxPointLikeSourceDetSigYield(
        PointLikeSourceDetSigYield,
):
    """The detector signal yield class for a point-like source with a flux model
    that depends on a single parameter.
    """

    def __init__(
            self,
            param_name,
            detector_model,
            dataset,
            fluxmodel,
            livetime,
            st_bin_width_deg,
            source_list,
            sin_true_alt_binning,
            log_spl_sintruealt_param,
            **kwargs,
    ):
        """Constructs the detector signal yield instance.

        Parameters
        ----------
        param_name : str
            The parameter name this detector signal yield depends
            on. This is either a fixed or floating parameter.
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the detector for this
            detector signal yield.
        dataset : instance of Dataset
            The instance of Dataset holding the monte-carlo event data.
        fluxmodel : instance of FluxModel
            The instance of FluxModel for which the detector signal yield is
            constructed.
        livetime : instance of Livetime
            The instance of Livetime defining the live-time of the dataset.
        st_bin_width_deg : float
            The sidereal time bin width in degree. This value must be in the
            range [0, 360].
        source_list : list of instance of PointLikeSource
            The list of instance of PointLikeSource defining the sources
            for this detector signal yield.
        sin_true_alt_binning : instance of BinningDefinition
            The instance of BinningDefinition defining the sin(true_altitude)
            binning.
        log_spl_sintruealt_param : instance of scipy.interpolate.RectBivariateSpline
            The 2D spline in sin(true_altitute) and the flux model's parameter
            this detector signal yield depends on.
        """
        super().__init__(
            param_names=[param_name],
            detector_model=detector_model,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            **kwargs,
        )

        if not isinstance(livetime, Livetime):
            raise TypeError(
                'The livetime argument must be an instance of Livetime! '
                f'Its current type is {classname(livetime)}!')

        self.src_recarray = self.sources_to_recarray(source_list)
        self.sin_true_alt_binning = sin_true_alt_binning
        self.log_spl_sintruealt_param = log_spl_sintruealt_param

        # Construct a sidereal time histogram which will preserve the angular
        # resolution of the detector.
        self.st_service = SiderealTimeService(
            detector_model=self._detector_model,
            livetime=self._livetime,
            st_bin_width_deg=st_bin_width_deg)

        self.src_st_alt_arr = self.st_service.create_src_st_alt_array(
            src_array=self.src_recarray,
        )

    @property
    def sin_true_alt_binning(self):
        """The instance of BinningDefinition defining the sin(true_altitude)
        binning.
        """
        return self._sin_true_alt_binning

    @sin_true_alt_binning.setter
    def sin_true_alt_binning(self, bd):
        if not isinstance(bd, BinningDefinition):
            raise TypeError(
                'The sin_true_alt_binning property must be an instance of '
                'BinningDefinition! '
                f'Its current type is {classname(bd)}.')
        self._sin_true_alt_binning = bd

    @property
    def log_spl_sintruealt_param(self):
        """The :class:`scipy.interpolate.RectBivariateSpline` instance
        representing the spline for the log value of the detector signal
        yield as a function of sin(true_altitude) and the flux model's
        parameter.
        """
        return self._log_spl_sintruealt_param

    @log_spl_sintruealt_param.setter
    def log_spl_sintruealt_param(self, spl):
        if not isinstance(spl, scipy.interpolate.RectBivariateSpline):
            raise TypeError(
                'The log_spl_sintruealt_param property must be an instance of '
                'scipy.interpolate.RectBivariateSpline! '
                f'Its current type is {classname(spl)}.')
        self._log_spl_sintruealt_param = spl

    def __call__(
            self,
            src_recarray,
            src_params_recarray,
    ):
        """Retrieves the detector signal yield for the given list of
        sources and their flux model parameters.

        Parameters
        ----------
        src_recarray : instance of numpy.ndarray
            The (N_sources,)-shaped structured numpy ndarray with the fields
            ``dec`` and ``ra`` holding the declination and right-ascention of
            the sources.
        src_params_recarray : instance of numpy.ndarray
            The (N_sources,)-shaped numpy structured ndarray containing the
            parameter values of the sources. The parameter values can be
            different for the different sources.
            The structured array needs to contain two fields for each source
            parameter, one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        values : instance of numpy.ndarray
            The (N_sources,)-shaped numpy ndarray with the detector signal yield
            for each source.
        grads : dict
            The dictionary holding the gradient values for each global floating
            parameter. The key is the global floating parameter index and the
            value is the (N_sources,)-shaped numpy ndarray holding the gradient
            value dY_k/dp_s.
        """
        if np.any(src_recarray != self.src_recarray):
            self.src_st_zen_arr = self.st_service.create_src_st_zen_array(
                src_array=self.src_recarray,
            )

        local_param_name = self.param_names[0]
        src_param = src_params_recarray[local_param_name]
        src_param_gp_idxs = src_params_recarray[f'{local_param_name}:gpidx']

        n_sources = len(src_recarray)

        # Check for correct input format.
        if not (len(src_param) == n_sources and
                len(src_param_gp_idxs) == n_sources):
            raise RuntimeError(
                f'The length ({len(src_param)}) of the array for the '
                f'source parameter "{local_param_name}" does not match the '
                f'number of sources ({n_sources})!')

        # Do the time integration by summing the sidereal time intervals. The
        # total time period for a single sidereal time interval is simply the
        # number of on-times falling into the sidereal time interval multiplied
        # by the time span of that interval.

        sec2fluxtimeunit = units.second.to(self._fluxmodel.time_unit)

        st_livetime_sec_arr = self.st_service.st_livetime_sec_arr

        values_st_arr = (
            np.exp(self._log_spl_sintruealt_param(
                np.sin(self.src_st_alt_arr), src_param[np.newaxis, :],
                grid=False,
            )) *
            st_livetime_sec_arr[:, np.newaxis] * sec2fluxtimeunit
        )
        values = np.sum(values_st_arr, axis=0)

        # Determine the number of global fit parameters the local parameter
        # is made of.
        gfp_idxs = np.unique(src_param_gp_idxs)
        gfp_idxs = gfp_idxs[gfp_idxs > 0] - 1

        grads = dict()

        if len(gfp_idxs) == 0:
            return (values, grads)

        sin_src_st_alt_arr = np.sin(self.src_st_alt_arr)

        for gfp_idx in gfp_idxs:
            # Create the gradient array of shape (n_sources,). This could be
            # a masked array to save memory, when there are many sources and
            # global fit parameters.
            grads[gfp_idx] = np.zeros((n_sources,), dtype=np.float64)

            # Create a mask to select the sources that depend on the global
            # fit parameter with index gfp_idx.
            m = (src_param_gp_idxs == gfp_idx+1)

            grad_st_src = (
                values_st_arr[:, m] *
                self._log_spl_sintruealt_param(
                    sin_src_st_alt_arr[:, m], src_param[np.newaxis, m],
                    grid=False,
                    dy=1,
                )
            )
            grads[gfp_idx][m] = np.sum(grad_st_src, axis=0)

        return (values, grads)


class SingleParamFluxPointLikeSourceDetSigYieldBuilder(
        DetSigYieldBuilder,
        IsParallelizable,
):
    """This detector signal yield builder constructs a detector signal yield
    for a variable flux model of a single parameter, assuming a point-like
    source.

    It constructs a two-dimensional spline function in sin(true_altitude) and
    the parameter, using a :class:`scipy.interpolate.RectBivariateSpline`.
    Hence, the detector signal yield can vary with the zenith angle and the
    parameter of the flux model.

    This builder can be used for any detector, because it is detector location
    independent.
    """

    def __init__(
            self,
            param_grid,
            st_bin_width_deg=0.1,
            sin_true_alt_binning=None,
            spline_order_sin_true_alt=2,
            spline_order_param=2,
            ncpu=None,
            **kwargs,
    ):
        """Creates a new detector signal yield builder instance for a point-like
        source with a flux model of a single parameter.
        It requires a sin_true_alt binning definition to compute the
        sin(true_altitude) dependency of the detector effective area, and a
        parameter grid to compute the parameter dependency of the detector
        signal yield.

        Parameters
        ----------
        param_grid : instance of ParameterGrid
            The instance of ParameterGrid which defines the grid of the
            parameter values. The name of the parameter is defined via the name
            property of the ParameterGrid instance.
        st_bin_width_deg : float
            The sidereal time bin width in degree. This value must be in the
            range [0, 360].
        sin_true_alt_binning : instance of BinningDefinition | None
            The instance of BinningDefinition which defines the
            sin(true_altitude) binning. If set to None, the sin(true_altitude)
            binning will be taken from the dataset's binning definitions.
        spline_order_sin_true_alt : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(true_altitude) axis.
            The default is 2.
        spline_order_param : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the flux model's parameter axis.
            The default is 2.
        ncpu : int | None
            The number of CPUs to utilize. If set to ``None``, configuration
            setting will take place.
        """
        super().__init__(
            ncpu=ncpu,
            **kwargs,
        )

        self.param_grid = param_grid
        self.st_bin_width_deg = st_bin_width_deg
        self.sin_true_alt_binning = sin_true_alt_binning
        self.spline_order_sin_true_alt = spline_order_sin_true_alt
        self.spline_order_param = spline_order_param

    @property
    def param_grid(self):
        """The instance of ParameterGrid for the parameter grid that should be
        used for computing the parameter dependency of the detector signal
        yield.
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
    def st_bin_width_deg(self):
        """The sidereal time bin width in degree.
        """
        return self._st_bin_width_deg

    @st_bin_width_deg.setter
    def st_bin_width_deg(self, bw):
        bw = float_cast(
            bw,
            'The st_bin_width_deg property must be castable to type float! '
            f'Its current type is {classname(bw)}!')
        self._st_bin_width_deg = bw

    @property
    def sin_true_alt_binning(self):
        """The instance of BinningDefinition for the sin(true_altitude) binning
        that should be used for computing the sin(true_altitude) dependency of
        the detector signal yield. If None, the binning is supposed to be taken
        from the Dataset's binning definitions.
        """
        return self._sin_true_alt_binning

    @sin_true_alt_binning.setter
    def sin_true_alt_binning(self, binning):
        if (binning is not None) and\
           (not isinstance(binning, BinningDefinition)):
            raise TypeError(
                'The sin_true_alt_binning property must be None, or an '
                'instance of BinningDefinition! '
                f'Its current type is "{classname(binning)}".')
        self._sin_true_alt_binning = binning

    @property
    def spline_order_sin_true_alt(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the sin(true_altitude) axis.
        """
        return self._spline_order_sin_true_alt

    @spline_order_sin_true_alt.setter
    def spline_order_sin_true_alt(self, order):
        order = int_cast(
            order,
            'The spline_order_sin_true_alt property must be castable to type '
            'int! '
            f'Its current type is {classname(order)}.')
        self._spline_order_sin_true_alt = order

    @property
    def spline_order_param(self):
        """The order (int) of the logarithmic spline function, that splines the
        detector signal yield, along the flux model's parameter axis.
        """
        return self._spline_order_param

    @spline_order_param.setter
    def spline_order_param(self, order):
        order = int_cast(
            order,
            'The spline_order_param property must be castable to type int! '
            f'Its current type is {classname(order)}.')
        self._spline_order_param = order

    def get_sin_true_alt_binning(self, dataset):
        """Gets the sin(true_altitude) binning definition either as setting from
        this detector signal yield builder itself, or from the given dataset.

        Parameters
        ----------
        dataset : instance of Dataset
            The instance of Dataset that defines the binning for
            sin(true_altitude).

        Returns
        -------
        sin_true_alt_binning : instance of BinningDefinition
            The binning definition for the sin(true_altitude) binning.
        """
        sin_true_alt_binning = self.sin_true_alt_binning
        if sin_true_alt_binning is None:
            if not dataset.has_binning_definition('sin_true_alt'):
                raise KeyError(
                    'No binning definition named "sin_true_alt" is defined in '
                    f'the dataset "{dataset.name}" and no user defined binning '
                    'definition was provided to the detector signal yield '
                    f'builder "{classname(self)}"!')
            sin_true_alt_binning = dataset.get_binning_definition(
                'sin_true_alt')
        return sin_true_alt_binning

    def construct_detsigyield(
        self,
        detector_model,
        dataset,
        data,
        shg,
        ppbar=None,
    ):
        """Constructs a 2-dimensional log spline function for a detector signal
        yield for the given flux model with varying parameter values.

        Parameters
        ----------
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the detector for this
            detector signal yield.
        dataset : instance of Dataset
            The instance of Dataset holding the sin(true_altitude) binning
            definition.
        data : instance of DatasetData
            The instance of DatasetData holding the monte-carlo event data.
            The DataFieldRecordArray for the monte-carlo data of the dataset
            must contain the following data fields:

            ``'true_alt'`` : float
                The true altitude of the MC event.
            ``'mcweight'`` : float
                The monte-carlo weight of the MC event in the unit
                GeV cm^2 sr.
            ``'true_energy'`` : float
                The true energy value of the MC event.

        shg : instance of SourceHypoGroup
            The instance of SourceHypoGroup for which the detector signal yield
            should get constructed.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : instance of SingleParamFluxPointLikeSourceDetSigYield
            The DetSigYield instance for a point-like source with a flux model
            of a single parameter.
        """
        self.assert_types_of_construct_detsigyield_arguments(
            detector_model=detector_model,
            dataset=dataset,
            data=data,
            shgs=shg,
            ppbar=ppbar)

        # Get the sin(true_altitude) binning definition either as setting from
        # this builder, or from the dataset.
        sin_true_alt_binning = self.get_sin_true_alt_binning(dataset)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit.
        to_internal_flux_unit_factor = shg.fluxmodel.to_internal_flux_unit()

        # Define a function that creates a detector signal yield histogram
        # along sin(true_altitude) for a given flux model, i.e. for a given
        # parameter value.
        def _create_hist(
                data_sin_true_alt,
                data_true_energy,
                sin_true_alt_binning,
                weights,
                fluxmodel,
                to_internal_flux_unit_factor,
        ):
            """Creates a histogram of the detector signal yield with the
            given sin(true_altitude) binning.

            Parameters
            ----------
            data_sin_true_alt : instance of numpy.ndarray
                The 1d numpy.ndarray holding the sin(true_altitude) values of
                the monte-carlo events.
            data_true_energy : instance of numpy.ndarray
                The 1d numpy.ndarray holding the true energy values of the
                monte-carlo events.
            sin_true_alt_binning : instance of BinningDefinition
                The sin(true_altitude) binning definition to use for the
                histogram.
            weights : instance of numpy.ndarray
                The 1d numpy.ndarray holding the weight factors of each
                monte-carlo event where only the flux and time interval values
                need to be multiplied with in order to get the detector signal
                yield.
            fluxmodel : instance of FluxModel
                The flux model to get the flux values from.
            to_internal_flux_unit_factor : float
                The conversion factor to convert the flux unit into the internal
                flux unit.

            Returns
            -------
            h : instance of numpy.ndarray
                The 1d numpy.ndarray containing the histogram values.
            """
            weights = (
                weights *
                fluxmodel(E=data_true_energy).squeeze() *
                to_internal_flux_unit_factor
            )

            (h, _) = np.histogram(
                data_sin_true_alt,
                bins=sin_true_alt_binning.binedges,
                weights=weights,
                density=False)

            return h

        data_sin_true_alt = data.mc['sin_true_alt']

        # Generate a list of indices that would sort the data according to the
        # sin(true_altitude) values. We will sort the MC data according to it,
        # because the histogram creation is much faster (2x) when the
        # to-be-histogrammed values are already sorted.
        sorted_idxs = np.argsort(data_sin_true_alt)

        data_sin_true_alt = np.take(data_sin_true_alt, sorted_idxs)
        data_true_energy = np.take(data.mc['true_energy'], sorted_idxs)
        weights = np.take(data.mc['mcweight'], sorted_idxs)

        # Make a copy of the parameter grid and extend the grid by one bin on
        # each side.
        param_grid = self._param_grid.copy()
        param_grid.add_extra_lower_and_upper_bin()

        # Construct the arguments for the hist function to be used in the
        # parallelize function.
        args_list = [
            (
                (
                    data_sin_true_alt,
                    data_true_energy,
                    sin_true_alt_binning,
                    weights,
                    shg.fluxmodel.copy({param_grid.name: param_val}),
                    to_internal_flux_unit_factor,
                ),
                {}
            )
            for param_val in param_grid.grid
        ]
        h = np.vstack(
            parallelize(
                _create_hist, args_list, self.ncpu, ppbar=ppbar)).T

        # Normalize by solid angle of each bin along the sin(true_altitude)
        # axis. The solid angle is given by 2*\pi*(\Delta sin(\phi)).
        h /= (2.*np.pi * np.diff(sin_true_alt_binning.binedges)).reshape(
            (sin_true_alt_binning.nbins, 1))

        # Create the 2D spline.
        log_spl_sintruealt_param = scipy.interpolate.RectBivariateSpline(
            sin_true_alt_binning.bincenters,
            param_grid.grid,
            np.log(h),
            kx=self.spline_order_sin_true_alt,
            ky=self.spline_order_param,
            s=0,
        )

        detsigyield = SingleParamFluxPointLikeSourceDetSigYield(
            param_name=self._param_grid.name,
            detector_model=detector_model,
            dataset=dataset,
            fluxmodel=shg.fluxmodel,
            livetime=data.livetime,
            st_bin_width_deg=self.st_bin_width_deg,
            source_list=shg.source_list,
            sin_true_alt_binning=sin_true_alt_binning,
            log_spl_sintruealt_param=log_spl_sintruealt_param,
        )

        return detsigyield
