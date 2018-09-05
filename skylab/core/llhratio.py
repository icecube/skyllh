# -*- coding: utf-8 -*-

"""The llhratio module provides classes implementing the log-likelihood ratio
functions. In general these should be detector independent, because they
implement the pure math of the log-likelihood ratio function.
"""

from __future__ import division

import abc

from skylab.core.py import (
    issequence,
    issequenceof,
    float_cast
)
from skylab.core.detsigeff import (
    DetectorSignalEfficiency
)
from skylab.core.parameters import (
    SourceFitParameterMapper,
    SingleSourceFitParameterMapper,
    MultiSourceFitParameterMapper
)
from skylab.core.pdfratio import (
    PDFRatio,
    SingleSourcePDFRatioArrayArithmetic
)
from skylab.physics.source import (
    SourceModel
)

class LLHRatio(object):
    """Abstract base class for a log-likelihood (LLH) ratio function.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(LLHRatio, self).__init__()

    @abc.abstractmethod
    def evaluate(self, fitparam_values):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        fitparam_values : numpy 1D ndarray
            The ndarray holding the current values of the (global) fit
            parameters.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each (global) fit
            parameter.
        """
        pass


class TCLLHRatio(LLHRatio):
    """Abstract base class for a two-component (TC) log-likelihood (LLH) ratio
    function with a list of independent PDF ratio components.
    """
    __metaclass__ = abc.ABCMeta

    # The (1 + alpha)-threshold float value for which the log-likelihood ratio
    # function of a single event should get approximated by a Taylor expansion.
    # This is to prevent a divergence of the log-function for each event, where
    # (1 + alpha_i) < (1 + alpha).
    # This setting is implemented as a class type member instead of a class
    # instance member, because it is supposed to be the same for all instances.
    _one_plus_alpha = 1e-3

    def __init__(self, pdfratios, src_fitparam_mapper):
        """Constructor of the two-component log-likelihood ratio function.

        Parameters
        ----------
        pdfratios : sequence of PDFRatio
            The sequence of PDFRatio instances. A PDFRatio instance might depend
            on none, one, or several fit parameters.
        src_fitparam_mapper : SourceFitParameterMapper
            The instance of SourceFitParameterMapper defining the global fit
            parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process of the log-likelihood-ratio
            function. The names of the source fit parameters must coincide with
            the signal fit parameter names of the PDF ratio instances.
        """
        super(TCLLHRatio, self).__init__()

        self.pdfratio_list = pdfratios
        self.src_fitparam_mapper = src_fitparam_mapper

        # These properties will be set via the ``initialize_for_new_trial``
        # method.
        self._events = events = None
        self._n_pure_bkg_events = None

        # Define cache variables for evaluate method to store values needed for
        # a possible calculation of the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_fitparam_values = None
        self._cache_nsgrad_i = None

    @property
    def events(self):
        """The numpy record array holding the data events, which should get
        evaluated.
        """
        return self._events
    @events.setter
    def events(self, arr):
        if(not isinstance(arr, np.ndarray)):
            raise TypeError('The events property must be an instance of ndarray!')
        self._events = arr

    @property
    def n_events(self):
        """(read-only) The number of events which should get evaluated.
        """
        return len(self._events)

    @property
    def n_pure_bkg_events(self):
        """The number of pure background events, which are not part of the
        `events` array property, but must be considered for the TS value.
        """
        return self._n_pure_bkg_events
    @n_pure_bkg_events.setter
    def n_pure_bkg_events(self, n):
        if(not isinstance(n, int)):
            raise TypeError('The n_pure_bkg_events property must be of type int!')
        self._n_pure_bkg_events = n

    @property
    def n_selected_events(self):
        """(read-only) The number of selected events.
        """
        return self.n_events

    @property
    def pdfratio_list(self):
        """The list of PDFRatio instances.
        """
        return self._pdfratio_list
    @pdfratio_list.setter
    def pdfratio_list(self, seq):
        if(not issequenceof(seq, PDFRatio)):
            raise TypeError('The pdfratio_list property must be a sequence of PDFRatio instances!')
        self._pdfratio_list = list(seq)

    @property
    def src_fitparam_mapper(self):
        """The SourceFitParameterMapper instance defining the global fit
        parameters and their mapping to the source fit parameters.
        """
        return self._src_fitparam_mapper
    @src_fitparam_mapper.setter
    def src_fitparam_mapper(self, mapper):
        if(not isinstance(mapper, SourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper property must be an instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    def initialize_for_new_trial(self, events, n_pure_bkg_events):
        """Initializes the log-likelihood ratio function for a new trial.
        It must be re-implemented by the derived class, which calls the
        method of the base class first.

        Parameters
        ----------
        events : numpy record array
            The numpy record array holding the new data events which should get
            evaluated in the new trial.
        n_pure_bkg_events : int
            The number of pure background events, which are not part of
            `events`, but must be considered for the log_lambda value.
        """
        self.events = events
        self.n_pure_bkg_events = n_pure_bkg_events

    def calculate_log_lambda_and_grads(self, fitparam_values, N, ns, Xi, dXi_ps):
        """Calculates the log(Lambda) value and its gradient for each global fit
        parameter. This calculation is source and detector independent.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.
            These numbers are used as cache key to validate the ``nsgrad_i``
            values for the given fit parameter values for a possible later
            calculation of the second derivative w.r.t. ns of the log-likelihood
            ratio function.
        N : int
            The total number of events.
        ns : float
            The current fit parameter value for ns.
        Xi : numpy (n_selected_events,)-shaped 1D ndarray
            The X value of each selected event.
        dXi_ps : numpy (N_fitparams,n_selected_events)-shaped 2D ndarray
            The derivative value for each fit parameter ps of each event's X
            value.

        Returns
        -------
        log_lambda : float
            The value of the log-likelihood ratio function.
        grads : 1D numpy (N_fitparams+1,)-shaped ndarray
            The gradient value of log_lambda for each fit parameter.
            The first element is the gradient for ns.
        """
        # Get the number of selected events.
        Nprime = len(Xi)

        alpha = TCLLHRatio._one_plus_alpha - 1
        alpha_i = ns*Xi

        # Create a mask for events which have a stable non-diverging
        # log-function argument, and an inverted mask thereof.
        stablemask = alpha_i > alpha
        unstablemask = ~stablemask

        # Allocate memory for the log_lambda_i values.
        log_lambda_i = np.empty_like(alpha_i, dtype=np.float)

        # Calculate the log_lambda_i value for the numerical stable events.
        log_lambda_i[stablemask] = np.log1p(alpha_i[stablemask])
        # Calculate the log_lambda_i value for the numerical unstable events.
        tildealpha_i = (alpha_i[unstablemask] - alpha) / _one_plus_alpha
        log_lambda_i[unstablemask] = np.log1p(alpha) + tildealpha_i - 0.5*tildealpha_i**2

        # Calculate the log_lambda value and account for pure background events.
        log_lambda = np.sum(log_lambda_i) + (N - Nprime)*np.log1p(-ns/N)

        # Calculate the gradient for each fit parameter.
        grads = np.empty((dXi_ps.shape[1]+1,), dtype=np.float)

        # Pre-calculate value that is used twice for the gradients of the
        # numerical stable events.
        one_over_one_plus_alpha_i_stablemask = 1 / (1 + alpha_i[stablemask])

        # For ns.
        nsgrad_i = np.empty_like(alpha_i, dtype=np.float)
        nsgrad_i[stablemask] = Xi[stablemask] * one_over_one_plus_alpha_i_stablemask
        nsgrad_i[unstablemask] = (1 - tildealpha_i)*Xi[unstablemask] / TCLLHRatio._one_plus_alpha
        # Cache the nsgrad_i values for a possible later calculation of the
        # second derivative w.r.t. ns of the log-likelihood ratio function.
        # Note: We create a copy of the fitparam_values array here to make sure
        #       that the values don't get changed outside this method before the
        #       calculate_ns_grad2 method is called.
        self._cache_fitparam_values = fitparam_values.copy()
        self._cache_nsgrad_i = nsgrad_i
        # Calculate the first derivative w.r.t. ns.
        grads[0] = np.sum(nsgrad_i) - (N - Nprime)/(N - ns)

        # For each other fit parameter.
        # For all numerical stable events.
        grads[1:] = np.sum(ns * one_over_one_plus_alpha_i_stablemask * dXi_ps[:,stablemask], axis=1)
        # For all numerical unstable events.
        grads[1:] += np.sum(ns*(1 - tildealpha_i)*dXi_ps[:,unstablemask] / TCLLHRatio._one_plus_alpha, axis=1)

        return (log_lambda, grads)

    def calculate_ns_grad2(self, fitparam_values):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.
        This method tries to use cached values for the first derivative
        w.r.t. ns of the log-likelihood ratio function for the individual
        events. If cached values don't exist or do not match the given fit
        parameter values, they will get calculated automatically by calling the
        evaluate method with the given fit parameter values.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        # Check if the cached nsgrad_i values match the given fitparam_values.
        if((self._cache_fitparam_values is None) or
           (not np.all(self._cache_fitparam_values == fitparam_values))):
            # Calculate the cache values by evaluating the log-likelihood ratio
            # function.
            self.evaluate(fitparam_values)

        ns = fitparam_values[0]
        Nprime = self.n_selected_events
        N = Nprime + self.n_pure_bkg_events

        nsgrad2 = -np.sum(self._cache_nsgrad_i**2) - (N - Nprime)/(N - ns)**2

        return nsgrad2


class SingleSourceTCLLHRatio(TCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a list of independent PDFRatio instances
    assuming a single source.
    """
    def __init__(self, pdfratios, src_fitparam_mapper):
        """Constructor for creating a 2-component, i.e. signal and background,
        log-likelihood ratio function assuming a single source.

        Parameters
        ----------
        pdfratios : list of PDFRatio
            The list of PDFRatio instances. A PDFRatio instance might depend on
            none, one, or several fit parameters.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process.
            The names of the source fit parameters must coincide with the signal
            fit parameter names of the PDF ratio objects.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an instance of SingleSourceFitParameterMapper!')

        super(SingleSourceSpatialEnergyTCLLHRatio, self).__init__(
            pdfratios, src_fitparam_mapper)

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any fit parameters.
        self._pdfratioarray = SingleSourcePDFRatioArrayArithmetic(
            self._pdfratio_list,
            self._src_fitparam_mapper.fitparam_list)

    def initialize_for_new_trial(self, events, n_pure_bkg_events):
        """Initializes the log-likelihood ratio function for a new trial.

        Parameters
        ----------
        events : numpy record array
            The numpy record array holding the new data events which should get
            evaluated in the new trial.
        n_pure_bkg_events : int
            The number of pure background events, which are not part of
            `events`, but must be considered for the log_lambda value.
        """
        super(SingleSourceTCLLHRatio, self).initialize_for_new_trial(
            events, n_pure_bkg_events)
        self._pdfratioarray.initialize_for_new_trial(events)

    def evaluate(self, fitparam_values):
        """Evaluates the log-likelihood ratio function for the given set of
        data events.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams+1,)-shaped 1D ndarray
            The ndarray holding the gradient value of log_lambda for each fit
            parameter and ns.
            The first element is the gradient for ns.
        """
        ns = fitparam_values[0]

        N = len(self._events) + self._n_pure_bkg_events

        # Create the fitparams dictionary with the fit parameter names and
        # values.
        fitparams = self._src_fitparam_mapper.get_src_fitparams(fitparam_values[1:])

        # Calculate the PDF ratio values of all PDF ratio objects, which depend
        # on any fit parameter.
        self._pdfratioarray.calculate_pdfratio_values(fitparams)

        # Calculate the product of all the PDF ratio values for each (selected)
        # event.
        Ri = self._pdfratioarray.get_ratio_product()

        # Calculate Xi for each (selected) event.
        Xi = (Ri - 1.) / N

        # Calculate the gradients of Xi for each fit parameter (without ns).
        dXi_ps = np.empty((len(fitparam_values)-1,len(Xi)), dtype=np.float)
        for (idx, fitparam_value) in enumerate(fitparam_values[1:]):
            fitparam_name = self._src_fitparam_mapper.get_src_fitparam_name(idx)
            # Get the PDFRatio instance from which we need the derivative from.
            pdfratio = self._pdfratioarray.get_pdfratio(idx)

            # Calculate the derivative of Ri.
            dRi = pdfratio.get_gradient(self.events, fitparams, fitparam_name) * self._pdfratioarray.get_ratio_product(excluded_fitparam_idx=idx)

            # Calculate the derivative of Xi w.r.t. the fit parameter.
            dXi_ps[idx] = dRi / N

        (log_lambda, grads) = self.calculate_log_lambda_and_grads(
            fitparam_values, N, ns, Xi, dXi_ps)

        return (log_lambda, grads)


class MultiSourceTCLLHRatio(TCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a list of independent PDFRatio instances
    assuming multiple sources (stacking).
    """
    def __init__(self, events, n_pure_bkg_events, pdfratios, src_fitparam_mapper):
        """
        Parameters
        ----------
        events : numpy record array
            The numpy record array holding the data events which should get
            evaluated.
        n_pure_bkg_events : int
            The number of pure background events, which are not part of
            `events`, but must be considered for the log_lambda value.
        pdfratios : sequence of PDFRatio
            The sequence of PDFRatio instances. A PDFRatio instance might depend
            on none, one, or several fit parameters.
        src_fitparam_mapper : MultiSourceFitParameterMapper
            The multi source fit parameter mapper that defines the fit
            parameters and their relation to the source fit parameters of the
            individual sources.
        """
        if(not isinstance(src_fitparam_mapper, MultiSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an instance of MultiSourceFitParameterMapper!')

        super(MultiSourceTCLLHRatio, self).__init__(
            events, n_pure_bkg_events, pdfratios, src_fitparam_mapper)

    # TODO: Implement this class!!


class DatasetSignalWeights(object):
    """Abstract base class for a dataset signal weight calculator class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, src_hypo_group_manager, src_fitparam_mapper, detsigeffs):
        """Base class constructor.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SourceFitParameterMapper
            The SourceFitParameterMapper instance that defines the global fit
            parameters and their mapping to the source fit parameters.

        detsigeffs : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetectorSignalEfficiency instances
            The collection of DetectorSignalEfficiency instances for each
            dataset and source group combination. The detector signal efficiency
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.detsigeff_arr = detsigeffs

        if(self._detsigeff_arr.shape[0] != self._src_hypo_group_manager.n_src_hypo_groups):
            raise ValueError('The detsigeffs array must have the same number of source hypothesis groups as the source hypothesis group manager defines!')

        # Pre-convert the source list of each source hypothesis group into a
        # source array needed for the detector signal efficiency evaluation.
        # Since all the detector signal efficiency instances must be of the same
        # kind for each dataset, we can just use the one of the first dataset of
        # each source hypothesis group.
        self._src_arr_list = []
        for (gidx, src_hypo_group) in enumerate(src_hypo_group_manager.src_hypo_group_list):
            self._src_arr_list.append(
                self._detsigeff_arr[gidx,0].source_to_array(src_hypo_group.source_list)
            )

    @property
    def src_hypo_group_manager(self):
        """The instance of SourceHypoGroupManager, which defines the source
        hypothesis groups.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    @property
    def src_fitparam_mapper(self):
        """The SourceFitParameterMapper instance defining the global fit
        parameters and their mapping to the source fit parameters.
        """
        return self._src_fitparam_mapper
    @src_fitparam_mapper.setter
    def src_fitparam_mapper(self, mapper):
        if(not isinstance(mapper, SourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper property must be an instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def detsigeff_arr(self):
        """The 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
        DetectorSignalEfficiency instances.
        """
        return self._detsigeff_arr
    @detsigeff_arr.setter
    def detsigeff_arr(self, detsigeffs):
        if(not isinstance(detsigeffs, np.ndarray)):
            raise TypeError('The detsigeff_arr property must be an instance of numpy.ndarray!')
        if(detsigeffs.ndim != 2):
            raise ValueError('The detsigeff_arr property must be a numpy.ndarray with 2 dimensions!')
        if(not issequenceof(detsigeffs.flat, DetectorSignalEfficiency)):
            raise TypeError('The detsigeff_arr property must contain DetectorSignalEfficiency instances, one for each source hypothesis group and dataset combination!')
        self._detsigeff_arr = detsigeffs

    @property
    def n_datasets(self):
        """(read-only) The number of datasets this DatasetSignalWeights instance
        is for.
        """
        return self._detsigeff_arr.shape[1]

    @abc.abstractmethod
    def __call__(self, fitparam_values):
        """This method is supposed to calculate the dataset signal weights and
        their gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_datasets,)-shaped 1D ndarray
            The dataset signal weight factor for each dataset.
        f_grads : (N_datasets,N_fitparams)-shaped 2D ndarray
            The gradients of the dataset signal weight factors, one for each
            fit parameter.
        """
        pass


class SingleSourceDatasetSignalWeights(DatasetSignalWeights):
    """This class calculates the dataset signal weight factors for each dataset
    assuming a single source.
    """
    def __init__(self, src_hypo_group_manager, src_fitparam_mapper, detsigeffs):
        """Constructs a new DatasetSignalWeights instance assuming a single
        source.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        detsigeffs : sequence of DetectorSignalEfficiency
            The sequence of DetectorSignalEfficiency instances, one for each
            dataset.
        """

        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an instance of SingleSourceFitParameterMapper!')

        # Convert sequence into a 2D numpy array.
        detsigeffs = np.atleast_2d(detsigeffs)

        super(SingleSourceDatasetSignalWeights, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper, detsigeffs)

    def __call__(self, fitparam_values):
        """Calculates the dataset signal weight and its fit parameter gradients
        for each dataset.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_datasets,)-shaped 1D ndarray
            The dataset signal weight factor for each dataset.
        f_grads : (N_datasets,N_fitparams)-shaped 2D ndarray | None
            The gradients of the dataset signal weight factors, one for each
            fit parameter. None is returned if there are no fit parameters
            beside ns.
        """
        fitparams_arr = self._src_fitparam_mapper.get_fitparams_array(fitparam_values[1:])

        N_datasets = self.n_datasets
        N_fitparams = self._src_fitparam_mapper.n_global_fitparams

        Y = np.empty((N_datasets,), dtype=np.float)
        if(N_fitparams > 0):
            Y_grads = np.empty((N_datasets, N_fitparams), dtype=np.float)
        # Loop over the detector signal efficiency instances for the first and
        # only source hypothesis group.
        for (j, detsigeff) in enumerate(self._detsigeff_arr[0]):
            (Yj, Yj_grads) = detsigeff(self._src_arr_list[0], fitparams_arr)
            # Store the detector signal efficiency and its fit parameter
            # gradients for the first and only source (element 0).
            Y[j] = Yj[0]
            if(N_fitparams > 0):
                Y_grads[j] = Yj_grads[0]

        # sumj_Y is a scalar.
        sumj_Y = np.sum(Y, axis=0)

        # f is a (N_datasets,)-shaped 1D ndarray.
        f = Y/sumj_Y

        # f_grads is a (N_datasets, N_fitparams)-shaped 2D ndarray.
        if(N_fitparams > 0):
            # sumj_Y_grads is a (N_fitparams,)-shaped 1D array.
            sumj_Y_grads = np.sum(Y_grads, axis=0)
            f_grads = (Y_grads*sumj_Y - Y[...,np.newaxis]*sumj_Y_grads) / sumj_Y**2
        else:
            f_grads = None

        return (f, f_grads)

#TODO: Implement MultiSourceDatasetSignalWeights class!


class MultiDatasetTCLLHRatio(LLHRatio):
    """This class describes a two-component log-likelihood ratio function for
    multiple datasets. The final log-likelihood ratio value is the sum of the
    individual log-likelihood ratio values.

    The different datasets contribute according to their dataset signal weight
    factor, f_j(p_s), which depends on possible signal fit parameters. By
    definition the signal fit parameters are assumed to be the same for each
    dataset.

    By mathematical definition this class is suitable for single and multi
    source hypotheses.
    """
    def __init__(self, dataset_signal_weights, llhratios):
        """Creates a new composite two-component log-likelihood ratio function.

        Parameters
        ----------
        dataset_signal_weights : DatasetSignalWeights
            An instance of DatasetSignalWeights, which calculates the relative
            dataset weight factors.
        llhratios : sequence of TCLLHRatio instances
            The sequence of the two-component log-likelihood ratio functions,
            one for each dataset.
        """
        super(MultiDatasetTCLLHRatio, self).__init__()

        self.dataset_signal_weights = datasetweights
        self.llhratio_list = llhratios

        # Check if the number of datasets the DatasetSignalWeights instance is
        # made for equals the number of log-likelihood ratio functions.
        if(self.dataset_signal_weights.n_datasets != len(self._llhratio_list)):
            raise ValueError('The number of datasets the DatasetSignalWeights instance is made for must be equal to the number of log-likelihood ratio functions!')

        # Define cache variable for the dataset signal weight factors, which
        # will be needed when calculating the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_fitparam_values_ns = None
        self._cache_f = None

    @property
    def dataset_signal_weights(self):
        """The DatasetSignalWeights instance that provides the relative dataset
        weight factors.
        """
        return self._dataset_signal_weights
    @dataset_signal_weights.setter
    def dataset_signal_weights(self, obj):
        if(not isinstance(obj, DatasetSignalWeights)):
            raise TypeError('The dataset_signal_weights property must be an instance of DatasetSignalWeights!')
        self._dataset_signal_weights = obj

    @property
    def llhratio_list(self):
        """(read-only) The list of TCLLHRatio instances, which are part of this
        composite log-likelihood-ratio function.
        """
        return self._llhratio_list
    @llhratio_list.setter
    def llhratio_list(self, llhratios):
        if(not issequenceof(llhratios, TCLLHRatio)):
            raise TypeError('The llhratio_list property must be a sequence of TCLLHRatio instances!')
        self._llhratio_list = list(llhratios)

    @property
    def n_selected_events(self):
        """The sum of selected events of each individual log-likelihood ratio
        function.
        """
        n_selected_events = 0
        for llhratio in self._llhratio_list:
            n_selected_events += llhratio.n_selected_events
        return n_selected_events

    def evaluate(self, fitparam_values):
        """Evaluates the composite log-likelihood-ratio function and returns its
        value and global fit parameter gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams)-shaped numpy 1D ndarray
            The ndarray holding the current values of the global fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value of the composite
            log-likelihood-ratio function.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value of the composite
            log-likelihood-ratio function for ns and each global fit parameter.
            By definition the first element is the gradient for ns.
        """
        ns = fitparam_values[0]

        # Get the dataset signal weights and their gradients.
        # f is a (N_datasets,)-shaped 1D ndarray.
        # f_grads is a (N_datasets,N_fitparams)-shaped 2D ndarray.
        (f, f_grads) = self._dataset_signal_weights(fitparam_values)
        # Cache f for possible later calculation of the second derivative w.r.t.
        # ns of the log-likelihood ratio function.
        self._cache_fitparam_values_ns = ns
        self._cache_f = f

        nsf = ns * f

        # Calculate the composite log-likelihood-ratio function value and the
        # gradient of the composite log-likelihood ratio function for each
        # global fit parameter.
        log_lambda = 0

        # Allocate an array for the gradients of the composite log-likelihood
        # function. It is always at least one element long, i.e. the gradient
        # for ns.
        grads = np.zeros((len(fitparam_values),), dtype=np.float)

        # Create an array holding the fit parameter values for a particular
        # llh ratio function. Since we need to adjust ns with nsj it's more
        # efficient to create this array once and use it within the for loop
        # over the llh ratio functions.
        llhratio_fitparam_values = np.empty((len(fitparam_values),), dtype=np.float)
        # Loop over the llh ratio functions.
        for (j, llhratio) in enumerate(self._llhratio_list):
            llhratio_fitparam_values[0] = nsf[j]
            llhratio_fitparam_values[1:] = fitparam_values[1:]
            (log_lambda_j, grads_j) = llhratio.evaluate(llhratio_fitparam_values)
            log_lambda += log_lambda_j
            # Gradient for ns.
            grads[0] += grads_j[0] * f[j]
            # Gradient for each global fit parameter, if there are any.
            if(len(grads) > 1):
                grads[1:] += grads_j[0] * ns * f_grads[j] + grads_j[1:]

        return (log_lambda, grads)

    def calculate_ns_grad2(self, fitparam_values):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.
        This method tries to use cached values for the dataset signal weight
        factors. If cached values don't exist or do not match the given fit
        parameter values, they will get calculated automatically by calling the
        evaluate method with the given fit parameter values.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        ns = fitparam_values[0]

        # Check if the cached fit parameters match the given ones. The ns value
        # is special to the multi-dataset LLH ratio function, but all the other
        # fit parameters are shared by all the LLH ratio functions of the
        # different datasets. So those we just query from the first LLH ratio
        # function.
        if((self._cache_fitparam_values_ns is None) or
           (self._cache_fitparam_values_ns != ns) or
           (not np.all(self._llhratio_list[0]._cache_fitparam_values[1:] == fitparam_values[1:]))):
            self.evaluate(fitparam_values)

        nsf = ns * self._cache_f

        nsgrad2j = np.empty((len(self._llhratio_list),), dtype=np.float)
        # Loop over the llh ratio functions and their second derivative.
        llhratio_fitparam_values = np.empty((len(fitparam_values),), dtype=np.float)
        for (j, llhratio) in enumerate(self._llhratio_list):
            llhratio_fitparam_values[0] = nsf[j]
            llhratio_fitparam_values[1:] = fitparam_values[1:]
            nsgrad2j[j] = llhratio.calculate_ns_grad2(llhratio_fitparam_values)

        nsgrad2 = np.sum(nsgrad2j * self._cache_f**2)

        return nsgrad2
