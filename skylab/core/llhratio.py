# -*- coding: utf-8 -*-

"""The llhratio module provides classes implementing the log-likelihood ratio
functions. In general these should be detector independent, because they
implement the pure math of the log-likelihood ratio function.
"""

from __future__ import division

import abc

from skylab.core.parameters import SourceFitParameterManager
from skylab.core.pdfratio import PDFRatio, SingleSourcePDFRatioArrayArithmetic

class TCLLHRatio(object):
    """Abstract base class for a two-component (TC) log-likelihood (LLH) ratio
    function with a list of independent PDF ratio components.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, events, n_pure_bkg_events, pdfratios):
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
        """
        super(TCLLHRatio, self).__init__()

        self.events = events
        self.n_pure_bkg_events = n_pure_bkg_events
        self.pdfratio_list = pdfratios

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
    def evaluate(self, fitvalues):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        fitvalues : numpy 1D ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal event parameter, ns.

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
    def __init__(self, events, n_pure_bkg_events, pdfratios, fitparams):
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
        fitparams : sequence of FitParameter
            The sequence of FitParameter instances defining the fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process.
            The names of the fit parameters must coincide with the signal fit
            parameter names of the PDF ratio objects.
        """
        super(SingleSourceSpatialEnergyTCLLHRatio, self).__init__(
            events, n_pure_bkg_events, pdfratios)

        self.fitparam_list = fitparams

        # Pre-create the list of fit parameter names, which is needed later in
        # the evaluate method.
        self._fitparam_names = [ fitparam.name for fitparam in self.fitparam_list ]

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any fit parameters.
        self._pdfratioarray = SingleSourcePDFRatioArrayArithmetic(
            self._pdfratio_list,
            self._fitparam_list, self._events)

    @property
    def fitparam_list(self):
        """The list of FitParameter instances. The order of the fit parameters
        in this list defines the order of the fit parameter values in the
        `fitparam_values` argument of the evaluate method.
        """
        return self._fitparam_list
    @fitparam_list.setter
    def fitparam_list(self, seq):
        if(not issequenceof(seq, FitParameter)):
            raise TypeError('The fitparam_list property must be a sequence of FitParameter!')
        self._fitparam_list = list(seq)

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
        """
        ns = fitvalues[0]

        Nprime = self.n_events
        N = Nprime + self.n_pure_bkg_events

        # Create the fitparams dictionary with the fit parameter names and
        # values.
        fitparams = dict([ (self._fitparam_names[idx], fitparam_value)
                          for (idx, fitparam_value) in enumerate(fitparam_values) ] )

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
        grads = np.zeros((1 + len(self._fitparam_list),), dtype=np.float)

        # Precalculate the denumerator which is used in all the derivatives.
        one_plus_ns_times_Xi = 1 + ns*Xi

        # For ns.
        grads[0] = np.sum(Xi/one_plus_ns_times_Xi) - (N - Nprime)/(N - ns)

        # For each other fit parameter.
        for (idx, fitparam_value) in enumerate(fitparam_values[1:]):
            fitparam_name = self._fitparam_names[idx]
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
    def __init__(self, events, n_pure_bkg_events, pdfratios, src_fit_param_manager):
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
        src_fit_param_manager : SourceFitParameterManager
            The source fit parameter manager that defines the fit parameter and
            their relation to the source fit parameters of the different
            sources.
        """
        super(MultiSourceTCLLHRatio, self).__init__(events, n_pure_bkg_events, pdfratios)

        self.src_fit_param_manager = src_fit_param_manager

    @property
    def src_fit_param_manager(self):
        """The instance of SourceFitParameterManager that defines the fit
        parameters and their relation to the source fit parameters of the
        different sources.
        """
        return self._src_fit_param_manager
    @src_fit_param_manager.setter
    def src_fit_param_manager(self, manager):
        if(not isinstance(manager, SourceFitParameterManager)):
            raise TypeError('The src_fit_param_manager property must be an instance of SourceFitParameterManager!')
        self._src_fit_param_manager = manager

    # TODO: Implement this class!!
