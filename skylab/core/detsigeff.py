# -*- coding: utf-8 -*-

import abc
import numpy as np

from astropy import units

from skylab.core.py import issequenceofsubclass
from skylab.core.dataset import Dataset
from skylab.core.livetime import Livetime
from skylab.physics.source import SourceModel
from skylab.physics.flux import FluxModel


class DetSigEff(object):
    """This is the abstract base class for a detector signal efficiency.

    The detector signal efficiency, Y_s(x_s,p_s), is defined as the mean number
    of signal events per steradian detected from a given source at position x_s
    with flux fit parameters p_s.

    To construct a detector signal efficiency object, four ingredients are
    needed: the dataset holding the monte-carlo data events, a signal flux
    model, the live time, and an implementation method that knows howto contruct
    the actual detector efficiency in an efficient way.
    In general, the implementation method depends on the detector, the source,
    the flux model with its flux model's signal parameters, and the dataset.
    Hence, for a given detector, source, flux model, and dataset, an appropriate
    implementation method needs to be chosen.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, implmethod, dataset, fluxmodel, livetime):
        """Constructs a new detector signal efficiency object. It takes
        the monte-carlo data events, a flux model of the signal, and the live
        time to compute the detector signal efficiency.

        Parameters
        ----------
        implmethod : instance of DetSigEffImplMethod
            The implementation method to use for constructing and receiving
            the detector signal efficiency. The appropriate method depends on
            the used flux model.
        dataset : Dataset instance
            The Dataset instance holding the monte-carlo event data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime
            The live-time in days to use for the detector signal efficiency.
        """
        self.implmethod = implmethod
        self.dataset = dataset
        self.fluxmodel = fluxmodel
        self.livetime = livetime

    @property
    def implmethod(self):
        return self._implmethod
    @implmethod.setter
    def implmethod(self, method):
        if(not isinstance(method, DetSigEffImplMethod)):
            raise TypeError('The implmethod property must be an instance of DetSigEffImplMethod!')
        self._implmethod = method

    @property
    def dataset(self):
        """The Dataset instance, for which this detector signal efficiency is
        made for.
        """
        return self._dataset
    @dataset.setter
    def dataset(self, ds):
        if(not isinstance(ds, Dataset)):
            raise TypeError('The dataset property must be an instance of Dataset!')
        self._dataset = ds

    @property
    def fluxmodel(self):
        """The flux model, which should be used to calculate the detector
        signal efficiency.
        """
        return self._fluxmodel
    @fluxmodel.setter
    def fluxmodel(self, model):
        if(not isinstance(model, FluxModel)):
           raise TypeError('The fluxmodel property must be an instance of FluxModel!')
        self._fluxmodel = model

    @property
    def livetime(self):
        """The live-time in days.
        """
        return self._livetime
    @livetime.setter
    def livetime(self, lt):
        if(not (isinstance(lt, float) or isinstance(lt, Livetime))):
            raise TypeError('The livetime property must be of type float or an instance of Livetime!')
        self._livetime = lt

    def n_fitparams(self):
        """(read-only) The number of fit parameters this detector signal
        efficiency depends on.
        """
        return self._implmethod.n_fitparams

    def fitparam_names(self):
        """(read-only) The list of fit parameter names this detector signal
        efficiency depends on.
        """
        return self._implmethod.fitparam_names

    def source_to_array(self, source):
        """Converts the (sequence of) source(s) into a numpy record array needed
        for the __call__ method. This convertion is intrinsic to the
        implementation method.

        Parameters
        ----------
        source : SourceModel | sequence of SourceModel
            The source model containing the spatial information of the source.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the spatial information
            for each source.
        """
        return self._implmethod.source_to_array(source)

    @abc.abstractmethod
    def __call__(self, src, src_flux_params):
        """Abstract method  Retrieves the detector signal efficiency for the given sources
        and source flux parameters. The unit is mean number of signal
        events per steradian, i.e. sr^-1.

        Parameters
        ----------
        src : numpy record ndarray
            The numpy record array containing the spatial information of the
            signal sources. The required fields of this record array are
            implementation method dependent. In the most generic case for a
            point-like source, it must contain the following three fields:
            ra, dec, time.
        src_flux_params : numpy record ndarray
            The numpy record ndarray containing the flux parameters of the
            sources. The flux parameters can be different for the different
            sources.

        Returns
        -------
        detsigeff : (N_sources,)-shaped 1D ndarray
            The array with the detector signal efficiency value for each given
            source in unit mean number of signal events per steradian,
            i.e. sr^-1.
        grads : None | (N_sources,N_fitparams)-shaped 2D ndarray
            The gradient of the detector signal efficiency w.r.t. each fit
            parameter for each source. If the detector signal efficiency depends
            on no fit parameter, None is returned.
        """
        pass


class DetSigEffImplMethod(object):
    """Abstract base class for an implementation method of a detector signal
    efficiency. Via the ``construct_detsigeff`` method it creates a DetSigEff
    instance holding the internal objects to calculate the detector signal
    efficiency.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(DetSigEffImplMethod, self).__init__(*args, **kwargs)

        self.supported_sourcemodels = ()
        self.supported_fluxmodels = ()

    @property
    def supported_sourcemodels(self):
        """The tuple with the SourceModel classes, which are supported by this
        detector signal efficiency implementation method.
        """
        return self._supported_sourcemodels
    @supported_sourcemodels.setter
    def supported_sourcemodels(self, models):
        if(not isinstance(models, tuple)):
            raise TypeError('The supported_sourcemodels property must be of type tuple!')
        if(not issequenceofsubclass(models, SourceModel)):
            raise TypeError('The supported_sourcemodels property must be a sequence of SourceModel classes!')
        self._supported_sourcemodels = models

    @property
    def supported_fluxmodels(self):
        """The tuple with the FluxModel classes, which are supported by this
        detector signal efficiency implementation method.
        """
        return self._supported_fluxmodels
    @supported_fluxmodels.setter
    def supported_fluxmodels(self, models):
        if(not isinstance(models, tuple)):
            raise TypeError('The supported_fluxmodels property must be of type tuple!')
        if(not issequenceofsubclass(models, FluxModel)):
            raise TypeError('The supported_fluxmodels property must be a sequence of FluxModel instances!')
        self._supported_fluxmodels = models

    @property
    def n_signal_fitparams(self):
        """(read-only) The number of signal fit parameters the detector signal
        efficiency depends on.
        """
        return len(self._get_signal_fitparam_names())

    @property
    def signal_fitparam_names(self):
        """(read-only) The list of fit parameter names the detector signal
        efficiency depends on. An empty list indicates that it does not depend
        on any fit parameter.
        """
        return self._get_signal_fitparam_names()

    def _get_signal_fitparam_names(self):
        """This method must be re-implemented by the derived class and needs to
        return the list of fit parameter names, this detector signal efficiency
        is a function of. If it returns an empty list, the detector signal
        efficiency is independent of any fit parameters.

        Returns
        -------
        list of str
            The list of the fit parameter names, this detector signal efficiency
            is a function of. By default this method returns an empty list
            indicating that the detector signal efficiency depends on no fit
            parameter.
        """
        return []

    def supports_sourcemodel(self, sourcemodel):
        """Checks if the given source model is supported by this detected signal
        efficiency implementation method.
        """
        for ssm in self._supported_sourcemodels:
            if(isinstance(sourcemodel, ssm)):
                return True
        return False

    def supports_fluxmodel(self, fluxmodel):
        """Checks if the given flux model is supported by this detector signal
        efficiency implementation method.
        """
        for sfm in self._supported_fluxmodels:
            if(isinstance(fluxmodel, sfm)):
                return True
        return False

    @abc.abstractmethod
    def construct_detsigeff(self, dataset, fluxmodel, livetime):
        """Abstract method to construct the DetSigEff instance.
        This method must be called by the derived class method implementation
        to ensure the compatibility check of the given flux model with the
        supported flux models.

        Parameters
        ----------
        dataset : Dataset
            The Dataset instance holding the monte-carlo event data, and
            possible dataset specific settings.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of FluxModel.
        livetime : float | Livetime
            The live-time in days to use for the detector signal efficiency.

        Returns
        -------
        detsigeff : DetSigEff instance
            An instance derived from DetSigEff.
        """
        if(not isinstance(dataset, Dataset)):
            raise TypeError('The dataset argument must be an instance of Dataset!')
        if(not self.supports_fluxmodel(fluxmodel)):
            raise TypeError('The DetSigEffImplMethod "%s" does not support the flux model "%s"!'%(self.__class__.__name__, fluxmodel.__class__.__name__))
        if((not isinstance(livetime, float)) and
           (not isinstance(livetime, Livetime))):
            raise TypeError('The livetime argument must be an instance of float or Livetime!')

    @abc.abstractmethod
    def source_to_array(self, source):
        """This method is supposed to convert a (list of) source model(s) into
        a numpy record array that is understood by the implementation method.
        This is for efficiency reasons only. This way the user code can
        pre-convert the list of sources into a numpy record array and cache the
        array.
        The fields of the array are detector signal efficiency implementation
        dependent, i.e. what kind of sources: point-like source, or extended
        source for instance. Because the sources usually don't change their
        position in the sky, this has to be done only once.

        Parameters
        ----------
        source : SourceModel | sequence of SourceModel
            The source model containing the spatial information of the source.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the spatial information
            for each source.
        """
        pass
