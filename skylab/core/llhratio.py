# -*- coding: utf-8 -*-

"""The llhratio module provides classes implementing the log-likelihood ratio
functions. In general these should be detector independent, because they
implement the pure math of the log-likelihood ratio function.
"""

from __future__ import division

import abc

from skylab.core.py import\
    issequence,\
    issequenceof

from skylab.core.detsigeff import\
    DetectorSignalEfficiency

from skylab.core.parameters import\
    SourceFitParameterMapper,\
    SingleSourceFitParameterMapper,\
    MultiSourceFitParameterMapper

from skylab.core.pdfratio import\
    PDFRatio,\
    SingleSourcePDFRatioArrayArithmetic

from skylab.physics.source import\
    SourceModel


class TCLLHRatio(object):
    """Abstract base class for a two-component (TC) log-likelihood (LLH) ratio
    function with a list of independent PDF ratio components.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, events, n_pure_bkg_events, pdfratios, src_fitparam_mapper):
        """Constructor of the two-component log-likelihood ratio function.

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
        src_fitparam_mapper : SourceFitParameterMapper
            The instance of SourceFitParameterMapper defining the global fit
            parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process of the log-likelihood-ratio
            function. The names of the source fit parameters must coincide with
            the signal fit parameter names of the PDF ratio instances.
        """
        super(TCLLHRatio, self).__init__()

        self.events = events
        self.n_pure_bkg_events = n_pure_bkg_events
        self.pdfratio_list = pdfratios
        self.src_fitparam_mapper = src_fitparam_mapper

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

    @abc.abstractmethod
    def evaluate(self, fitparam_values):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        fitparam_values : numpy 1D ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each fit parameter.
        """
        pass


class SingleSourceTCLLHRatio(TCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a list of independent PDFRatio instances
    assuming a single source.
    """
    def __init__(self, events, n_pure_bkg_events, pdfratios, src_fitparam_mapper):
        """Constructor for creating a 2-component, i.e. signal and background,
        log-likelihood ratio function assuming a single source.

        Parameters
        ----------
        events : numpy record array
            The numpy record array holding the data events which should get
            evaluated.
        n_pure_bkg_events : int
            The number of pure background events, which are not part of
            `events`, but must be considered for the log_lambda value.
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
            events, n_pure_bkg_events, pdfratios, src_fitparam_mapper)

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any fit parameters.
        self._pdfratioarray = SingleSourcePDFRatioArrayArithmetic(
            self._pdfratio_list,
            self._src_fitparam_mapper.fitparam_list, self._events)

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
        fitparam_values : numpy 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each fit parameter.
            The first element is the gradient for ns.
        """
        ns = fitparam_values[0]

        Nprime = self.n_events
        N = Nprime + self.n_pure_bkg_events

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

        log_lambda = np.sum(np.log1p(ns*Xi))
        if(Nprime != N):
            # Account for the pure background events.
            log_lambda += (N - Nprime)*np.log1p(-ns/N)

        # Calculate the gradient for each fit parameter.
        grads = np.zeros((len(fitparam_values),), dtype=np.float)

        # Precalculate the denumerator which is used in all the derivatives.
        one_plus_ns_times_Xi = 1 + ns*Xi

        # For ns.
        grads[0] = np.sum(Xi/one_plus_ns_times_Xi) - (N - Nprime)/(N - ns)

        # For each other fit parameter.
        for (idx, fitparam_value) in enumerate(fitparam_values[1:]):
            fitparam_name = self._src_fitparam_mapper.get_src_fitparam_name(idx)
            # Get the PDFRatio instance from which we need the derivative from.
            pdfratio = self._pdfratioarray.get_pdfratio(idx)

            # Calculate the derivative of Ri.
            dRi = pdfratio.get_gradient(self.events, fitparams, fitparam_name) * self._pdfratioarray.get_ratio_product(excluded_fitparam_idx=idx)

            # Calculate the derivative of Xi.
            dXi = dRi / N

            grads[idx+1] = np.sum(ns/one_plus_ns_times_Xi * dXi)

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

    def __init__(self, source, src_fitparam_mapper, detsigeffs):
        """Base class constructor.

        Parameters
        ----------
        source : SourceModel | sequence of SourceModel
            The source or sequence of sources for which the dataset signal
            weights should get calculated for.
        src_fitparam_mapper : SourceFitParameterMapper
            The SourceFitParameterMapper instance that defines the global fit
            parameters and their mapping to the source fit parameters.
        detsigeffs : sequence of DetectorSignalEfficiency instances
            The sequence of detector signal efficiency instances, one for each
            dataset, which should be used to calculate the dataset signal
            weight factors. The order of the detector signal efficiency
            instances must match the order to datasets, which are evaluated in
            the multi-dataset log-likelihood ratio function.
        """
        self.source_list = source
        self.src_fitparam_mapper = src_fitparam_mapper
        self.detsigeff_list = detsigeffs

        # Pre-convert the source list into a source array needed for the
        # detector signal efficiency evaluation. Since all the detector signal
        # efficiency instances must be of the same kind, we can just use the
        # first one.
        self._src_arr = self._detsigeff_list[0].source_to_array(self._source_list)

    @property
    def source_list(self):
        """The list of SourceModel instances for which the dataset signal
        weights should get calculated for.
        """
        return self._source_list
    @source_list.setter
    def source_list(self, sources):
        if(not issequence(sources)):
            sources = [ sources ]
        if(not issequenceof(sources, SourceModel)):
            raise TypeError('The source_list property must be a sequence of SourceModel instances!')
        self._source_list = list(sources)

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
    def detsigeff_list(self):
        """The list of DetectorSignalEfficiency instances which should be used
        to calculate the dataset signal weight factors.
        """
        return self._detsigeff_list
    @detsigeff_list.setter
    def detsigeff_list(self, detsigeffs):
        if(not issequence(detsigeffs)):
            detsigeffs = [ detsigeffs ]
        if(not issequenceof(detsigeffs, DetectorSignalEfficiency)):
            raise TypeError('The detsigeff_list property must be a sequence of DetectorSignalEfficiency instances, one for each dataset!')
        self._detsigeff_list = list(detsigeffs)

    @abc.abstractmethod
    def __call__(self, fitparam_values):
        """This method is supposed to calculate the dataset signal weights and
        their gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams,)-shaped 1D numpy ndarray
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
    def __init__(self, source, src_fitparam_mapper, detsigeffs):
        """Constructs a new DatasetSignalWeights instance assuming a single
        source.

        Parameters
        ----------
        source : SourceModel
            The source for which the dataset signal weights should get
            calculated for.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        detsigeffs : sequence of DetectorSignalEfficiency
            The sequence of DetectorSignalEfficiency instances, one for each
            dataset.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an instance of SingleSourceFitParameterMapper!')

        super(SingleSourceDatasetSignalWeights, self).__init__(
            source, src_fitparam_mapper, detsigeffs)

    def __call__(self, fitparam_values):
        """Calculates the dataset signal weight and its fit parameter gradients
        for each dataset.

        Parameters
        ----------
        fitparam_values : (N_fitparams,)-shaped 1D numpy ndarray
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

        N_datasets = len(self._detsigeff_list)
        N_fitparams = len(self._src_fitparam_mapper.N_global_fitparams)

        Y = np.empty((N_datasets,), dtype=np.float)
        if(N_fitparams > 0):
            Y_grads = np.empty((N_datasets, N_fitparams), dtype=np.float)
        for (j, detsigeff) in enumerate(self._detsigeff_list):
            (Yj, Yj_grads) = detsigeff(self._src_arr, fitparams_arr)
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
        else
            f_grads = None

        return (f, f_grads)

# For a general implementation the MultiSourceDatasetSignalWeights class needs
# to be provides with different detector signal efficiency instances for each
# dataset AND source, because each source could have a different spectrum model
# with different fit parameters.

class MultiDatasetTCLLHRatio(object):
    """This class describes a two-component log-likelihood ratio function for
    multiple datasets. The final log-likelihood ratio value is the sum of the
    individual log-likelihood ratio values.

    The different datasets contribute according to their dataset signal weight
    factor, f_j(p_s), which depends on possible signal fit parameters. By
    definition the signal fit parameters are the same for each dataset.
    """
    def __init__(self, datasetweights):
        """Creates a new composite two-component log-likelihood ratio function.

        Parameters
        ----------
        datasetweights : DatasetSignalWeights
            An instance of DatasetSignalWeights, which calculates the relative
            dataset weight factors.
        """
        self.dataset_signal_weights = datasetweights

        self._llhratio_list = []

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

    def add_llhratio(self, llhratio):
        """Adds the given two-component log-likelihood ratio function to this
        multi-dataset two-component log-likelihood ratio function.

        Parameters
        ----------
        llhratio : TCLLHRatio
            The instance of TCLLHRatio that should be added.
        """
        if(not isinstance(llhratio, TCLLHRatio)):
            raise TypeError('The llhratio argument must be an instance of TCLLHRatio!')

        self._llhratio_list.append(llhratio)

    def evaluate(self, fitparam_values):
        """Evaluates the composite log-likelihood-ratio function.

        Parameters
        ----------
        fitparam_values : (N_fitparams)-shaped numpy 1D ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each fit parameter.
        """
        ns = fitparam_values[0]

        # Get the dataset signal weights and their gradients.
        # f is a (N_datasets,)-shaped 1D ndarray.
        # f_grads is a (N_datasets,N_fitparams)-shaped 2D ndarray.
        (f, f_grads) = self._dataset_signal_weights(fitparam_values)

        # TODO: Finish implementation
