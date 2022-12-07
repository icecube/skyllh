# -*- coding: utf-8 -*-

import abc
import numpy as np

from astropy import units

from skyllh.core.py import (
    classname,
    issequence,
    issequenceof,
    issequenceofsubclass,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.model import (
    SourceModel,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.physics.flux_model import (
    FluxModel,
)


class DetSigYield(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for a detector signal yield.

    The detector signal yield, Y_s(p_s), is defined as the expected mean
    number of signal events detected by the detector from a given source with
    source parameters p_s.

    To construct a detector signal yield object, four ingredients are
    needed: the dataset holding the monte-carlo data events, a signal flux
    model, the live-time, and a builder instance that knows howto contruct
    the actual detector yield in an efficient way.
    In general, the implementation method depends on the detector, the source,
    the flux model with its flux model's signal parameters, and the dataset.
    Hence, for a given detector, source, flux model, and dataset, an appropriate
    implementation method needs to be chosen.
    """
    def __init__(self, param_names, dataset, fluxmodel, livetime, **kwargs):
        """Constructs a new detector signal yield object. It takes
        the monte-carlo data events, a flux model of the signal, and the live
        time to compute the detector signal yield.

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
        livetime : float | Livetime
            The live-time in days to use for the detector signal yield.
        """
        super().__init__(**kwargs)

        self.param_names = param_names
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
                'The param_names property must be a sequence of str instances!')
        self._param_names = tuple(names)

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
                'The dataset property must be an instance of Dataset!')
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
               'The fluxmodel property must be an instance of FluxModel!')
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
                'of Livetime!')
        self._livetime = lt

    @abc.abstractmethod
    def sources_to_recarray(self, sources):
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
    def __call__(self, src_recarray, src_params_recarray):
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
            The record array needs to contain two fields for each source
            parameter, one named <name> with the source's local parameter name
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


class DetSigYieldBuilder(object, metaclass=abc.ABCMeta):
    """Abstract base class for a builder of a detector signal yield. Via the
    ``construct_detsigyield`` method it creates a DetSigYield instance holding
    the internal objects to calculate the detector signal yield.
    """

    def __init__(self, **kwargs):
        """Constructor.
        """
        super().__init__(**kwargs)

        self.supported_sourcemodels = ()
        self.supported_fluxmodels = ()

    @property
    def supported_sourcemodels(self):
        """The tuple with the SourceModel classes, which are supported by this
        detector signal yield implementation method.
        """
        return self._supported_sourcemodels
    @supported_sourcemodels.setter
    def supported_sourcemodels(self, models):
        if not isinstance(models, tuple):
            raise TypeError(
                'The supported_sourcemodels property must be of type tuple!')
        if not issequenceofsubclass(models, SourceModel):
            raise TypeError(
                'The supported_sourcemodels property must be a sequence of '
                'SourceModel classes!')
        self._supported_sourcemodels = models

    @property
    def supported_fluxmodels(self):
        """The tuple with the FluxModel classes, which are supported by this
        detector signal yield implementation method.
        """
        return self._supported_fluxmodels
    @supported_fluxmodels.setter
    def supported_fluxmodels(self, models):
        if not isinstance(models, tuple):
            raise TypeError(
                'The supported_fluxmodels property must be of type tuple!')
        if not issequenceofsubclass(models, FluxModel):
            raise TypeError(
                'The supported_fluxmodels property must be a sequence of '
                'FluxModel instances!')
        self._supported_fluxmodels = models

    def supports_sourcemodel(self, sourcemodel):
        """Checks if the given source model is supported by this detected signal
        yield implementation method.
        """
        for ssm in self._supported_sourcemodels:
            if isinstance(sourcemodel, ssm):
                return True
        return False

    def supports_fluxmodel(self, fluxmodel):
        """Checks if the given flux model is supported by this detector signal
        yield implementation method.
        """
        for sfm in self._supported_fluxmodels:
            if isinstance(fluxmodel, sfm):
                return True
        return False

    @abc.abstractmethod
    def construct_detsigyield(
            self, dataset, data, fluxmodel, livetime, ppbar=None):
        """Abstract method to construct the DetSigYield instance.
        This method must be called by the derived class method implementation
        to ensure the compatibility check of the given flux model with the
        supported flux models.

        Parameters
        ----------
        dataset : Dataset
            The Dataset instance holding possible dataset specific settings.
        data : DatasetData
            The DatasetData instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime
            The live-time in days to use for the detector signal yield.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : DetSigYield instance
            An instance derived from DetSigYield.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'The dataset argument must be an instance of Dataset!')

        if not isinstance(data, DatasetData):
            raise TypeError(
                'The data argument must be an instance of DatasetData!')

        if not self.supports_fluxmodel(fluxmodel):
            raise TypeError(
                f'The DetSigYieldBuilder "{classname(self)}" does not support '
                f'the flux model "{classname(fluxmodel)}"!')

        if (not isinstance(livetime, float)) and\
           (not isinstance(livetime, Livetime)):
            raise TypeError(
                'The livetime argument must be an instance of float or '
                f'Livetime! It is {classname(livetime)}')

        if ppbar is not None:
            if not isinstance(ppbar, ProgressBar):
                raise TypeError(
                    'The ppbar argument must be an instance of ProgressBar!')
