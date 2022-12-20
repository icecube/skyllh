# -*- coding: utf-8 -*-

"""The llhratio module provides classes implementing the log-likelihood ratio
functions. In general these should be detector independent, because they
implement the pure math of the log-likelihood ratio function.
"""

from __future__ import division

import abc
import numpy as np

from skyllh.core.config import CFG
from skyllh.core.debugging import get_logger
from skyllh.core.py import (
    classname,
    int_cast,
    issequence,
    issequenceof,
    float_cast
)
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.trialdata import TrialDataManager
from skyllh.core.detsigyield import DetSigYield
from skyllh.core.minimizer import (
    Minimizer,
    NR1dNsMinimizerImpl,
    NRNsScan2dMinimizerImpl
)
from skyllh.core.parameters import (
    SourceFitParameterMapper,
    SingleSourceFitParameterMapper,
    MultiSourceFitParameterMapper
)
from skyllh.core.pdfratio import (
    PDFRatio,
    SingleSourcePDFRatioArrayArithmetic
)
from skyllh.core.timing import TaskTimer
from skyllh.physics.source import SourceModel


logger = get_logger(__name__)


class LLHRatio(object, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function.
    """

    def __init__(self, minimizer):
        """Creates a new LLH ratio function instance.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        """
        super(LLHRatio, self).__init__()

        self.minimizer = minimizer

    @property
    def minimizer(self):
        """The Minimizer instance used to minimize the negative of the
        log-likelihood ratio function.
        """
        return self._minimizer
    @minimizer.setter
    def minimizer(self, minimizer):
        if(not isinstance(minimizer, Minimizer)):
            raise TypeError('The minimizer property must be an instance '
                'of Minimizer!')
        self._minimizer = minimizer

    def initialize_for_new_trial(self, tl=None):
        """This method will be called by the Analysis class after new trial data
        has been initialized to the trial data manager. Derived classes can make
        use of this call hook to perform LLHRatio specific trial initialization.

        Parameters
        ----------
        tl : TimeLord | None
            The optional TimeLord instance to use for timing measurements.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, fitparam_values, tl=None):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        fitparam_values : numpy 1D ndarray
            The ndarray holding the current values of the (global) fit
            parameters.
        tl : TimeLord | None
            The optional TimeLord instance to use for measuring timeing.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each (global) fit
            parameter.
        """
        pass

    def maximize(self, rss, fitparamset, tl=None):
        """Maximize the log-likelihood ratio function, by using the ``evaluate``
        method.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance to draw random numbers from.
            This is needed to generate random parameter initial values.
        fitparamset : FitParameterSet instance
            The instance of FitParameterSet holding the global fit parameter
            definitions used in the maximization process.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization process.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        fitparam_values : (N_fitparam)-shaped 1D ndarray
            The ndarray holding the global fit parameter values.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        tracing = CFG['debugging']['enable_tracing']

        # Define the negative llhratio function, that will get minimized.
        self_evaluate = self.evaluate
        def negative_llhratio_func(fitparam_values, func_stats, tl=None):
            func_stats['n_calls'] += 1
            with TaskTimer(tl, 'Evaluate llh-ratio function.'):
                (f, grads) = self_evaluate(fitparam_values, tl=tl)
                if(tracing): logger.debug(
                    'LLH-ratio func value f={:g}, grads={}'.format(
                        f, str(grads)))
            return (-f, -grads)

        minimize_kwargs = {'func_provides_grads': True}

        func_stats = {'n_calls': 0}
        with TaskTimer(tl, 'Minimize -llhratio function.'):
            (fitparam_values, fmin, status) = self._minimizer.minimize(
                rss, fitparamset, negative_llhratio_func, args=(func_stats,tl),
                kwargs=minimize_kwargs)
        log_lambda_max = -fmin
        status['n_llhratio_func_calls'] = func_stats['n_calls']

        logger.debug(
            'Maximized LLH ratio function with {:d} calls'.format(
                status['n_llhratio_func_calls']))

        return (log_lambda_max, fitparam_values, status)


class TCLLHRatio(LLHRatio, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function with two
    components, i.e. signal and background.
    """

    def __init__(self, minimizer, mean_n_sig_0):
        """Creates a new two-component LLH ratio function instance.

        Parameters
        ----------
        mean_n_sig_0 : float
            The fixed mean number of signal events for the null-hypothesis.
        """
        super(TCLLHRatio, self).__init__(minimizer)

        self.mean_n_sig_0 = mean_n_sig_0

    @property
    def mean_n_sig_0(self):
        """The parameter value for the mean number of signal events of the
        null-hypothesis.
        """
        return self._mean_n_sig_0
    @mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        v = float_cast(v, 'The mean_n_sig_0 property must be castable to a '
            'float value!')
        self._mean_n_sig_0 = v

    @abc.abstractmethod
    def calculate_ns_grad2(self, fitparam_values):
        """This method is supposed to calculate the second derivative of the
        log-likelihood ratio function w.r.t. the fit parameter ns, the number
        of signal events in the data set.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        pass

    def maximize(self, rss, fitparamset, tl=None):
        """Maximizes this log-likelihood ratio function, by minimizing its
        negative.
        This method has a special implementation when a 1D Newton-Rapson
        minimizer is used. In that case only the first and second derivative
        of the log-likelihood ratio function is calculated.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used to draw random
            numbers from. It is used by the minimizer to generate random
            fit parameter initial values.
        fitparamset : FitParameterSet instance
            The instance of FitParameterSet holding the global fit parameter
            definitions used in the maximization process.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization of the LLH-ratio function.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        fitparam_values : (N_fitparam)-shaped 1D ndarray
            The ndarray holding the global fit parameter values.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        if(isinstance(self._minimizer.minimizer_impl, NR1dNsMinimizerImpl) or
           isinstance(self._minimizer.minimizer_impl, NRNsScan2dMinimizerImpl)):
            # Define the negative llhratio function, that will get minimized
            # when using the Newton-Rapson 1D minimizer for llhratio functions
            # depending solely on ns.
            self__evaluate = self.evaluate
            self__calculate_ns_grad2 = self.calculate_ns_grad2
            def negative_llhratio_func_nr1d_ns(fitparam_values, tl):
                with TaskTimer(tl, 'Evaluate llh-ratio function.'):
                    (f, grads) = self__evaluate(fitparam_values, tl=tl)
                with TaskTimer(tl, 'Calculate 2nd derivative of llh-ratio '
                        'function w.r.t. ns'):
                    grad2_ns = self__calculate_ns_grad2(fitparam_values)

                return (-f, -grads[0], -grad2_ns)

            (fitparam_values, fmin, status) = self._minimizer.minimize(
                rss, fitparamset, negative_llhratio_func_nr1d_ns, args=(tl,))
            log_lambda_max = -fmin

            return (log_lambda_max, fitparam_values, status)

        return super(TCLLHRatio, self).maximize(rss, fitparamset, tl=tl)


class SingleDatasetTCLLHRatio(TCLLHRatio, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function with two
    components, i.e. signal and background, for a single data set.
    """

    def __init__(
            self, minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            mean_n_sig_0):
        """Creates a new two-component LLH ratio function instance for a single
        data set.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance that defines the source
            hypotheses.
        src_fitparam_mapper : SourceFitParameterMapper
            The instance of SourceFitParameterMapper defining the global fit
            parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process of the log-likelihood-ratio
            function. The names of the source fit parameters must coincide with
            the signal fit parameter names of the PDF instances.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        mean_n_sig_0 : float
            The fixed mean number of signal events for the null-hypothesis.
        """
        super(SingleDatasetTCLLHRatio, self).__init__(
            minimizer, mean_n_sig_0)

        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.tdm = tdm

        # Calculate the data fields that solely depend on source parameters.
        self._tdm.calculate_source_data_fields(src_hypo_group_manager)

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance that defines the source
        hypotheses.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                'instance of SourceHypoGroupManager!')
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
            raise TypeError('The src_fitparam_mapper property must be an '
                'instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def tdm(self):
        """The TrialDataManager instance that holds the trial event data and
        additional data fields for this LLH ratio function.
        """
        return self._tdm
    @tdm.setter
    def tdm(self, manager):
        if(not isinstance(manager, TrialDataManager)):
            raise TypeError('The tdm property must be an instance of '
                'TrialDataManager!')
        self._tdm = manager

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the source hypothesis group manager of this two-component LLH
        ratio function.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._tdm.change_source_hypo_group_manager(src_hypo_group_manager)


class ZeroSigH0SingleDatasetTCLLHRatio(SingleDatasetTCLLHRatio):
    """This class implements a two-component (TC) log-likelihood (LLH) ratio
    function for a single data assuming zero signal for the null-hypothesis.
    The log-likelihood-ratio function uses a list of independent PDF ratio
    instances.
    """
    # The (1 + alpha)-threshold float value for which the log-likelihood ratio
    # function of a single event should get approximated by a Taylor expansion.
    # This is to prevent a divergence of the log-function for each event, where
    # (1 + alpha_i) < (1 + alpha).
    # This setting is implemented as a class type member instead of a class
    # instance member, because it is supposed to be the same for all instances.
    _one_plus_alpha = 1e-3

    def __init__(
            self, minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            pdfratios):
        """Constructor of the two-component log-likelihood ratio function.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance that defines the source
            hypotheses.
        src_fitparam_mapper : SourceFitParameterMapper
            The instance of SourceFitParameterMapper defining the global fit
            parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process of the log-likelihood-ratio
            function. The names of the source fit parameters must coincide with
            the signal fit parameter names of the PDF ratio instances.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        pdfratios : sequence of PDFRatio
            The sequence of PDFRatio instances. A PDFRatio instance might depend
            on none, one, or several fit parameters.
        """
        super(ZeroSigH0SingleDatasetTCLLHRatio, self).__init__(
            minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            mean_n_sig_0=0)

        self.pdfratio_list = pdfratios

        # Define cache variables for evaluate method to store values needed for
        # a possible calculation of the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_fitparam_values = None
        self._cache_nsgrad_i = None

    @SingleDatasetTCLLHRatio.mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        SingleDatasetTCLLHRatio.mean_n_sig_0.fset(self, v)
        if(self._mean_n_sig_0 != 0):
            raise ValueError('The %s class is only valid for '
                'mean_n_sig_0 = 0!'%(classname(self)))

    @property
    def pdfratio_list(self):
        """The list of PDFRatio instances.
        """
        return self._pdfratio_list
    @pdfratio_list.setter
    def pdfratio_list(self, seq):
        if(not issequenceof(seq, PDFRatio)):
            raise TypeError('The pdfratio_list property must be a sequence of '
                'PDFRatio instances!')
        self._pdfratio_list = list(seq)

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
        tracing = CFG['debugging']['enable_tracing']

        # Get the number of selected events.
        Nprime = len(Xi)

        if(tracing):
            logger.debug(
                'N={:d}, Nprime={:d}'.format(
                    N, Nprime))

        one_plus_alpha = ZeroSigH0SingleDatasetTCLLHRatio._one_plus_alpha

        alpha = one_plus_alpha - 1
        alpha_i = ns*Xi

        # Create a mask for events which have a stable non-diverging
        # log-function argument, and an inverted mask thereof.
        stablemask = alpha_i > alpha
        unstablemask = ~stablemask
        if(tracing):
            logger.debug(
                '# of events doing Taylor expansion for (unstable events): '
                '{:d}'.format(
                    np.count_nonzero(unstablemask)))

        # Allocate memory for the log_lambda_i values.
        log_lambda_i = np.empty_like(alpha_i, dtype=np.float64)

        # Calculate the log_lambda_i value for the numerical stable events.
        log_lambda_i[stablemask] = np.log1p(alpha_i[stablemask])
        # Calculate the log_lambda_i value for the numerical unstable events.
        tildealpha_i = (alpha_i[unstablemask] - alpha) / one_plus_alpha
        log_lambda_i[unstablemask] = np.log1p(alpha) + tildealpha_i - 0.5*tildealpha_i**2

        # Calculate the log_lambda value and account for pure background events.
        log_lambda = np.sum(log_lambda_i) + (N - Nprime)*np.log1p(-ns/N)

        # Calculate the gradient for each fit parameter.
        grads = np.empty((dXi_ps.shape[0]+1,), dtype=np.float64)

        # Pre-calculate value that is used twice for the gradients of the
        # numerical stable events.
        one_over_one_plus_alpha_i_stablemask = 1 / (1 + alpha_i[stablemask])

        # For ns.
        nsgrad_i = np.empty_like(alpha_i, dtype=np.float64)
        nsgrad_i[stablemask] = Xi[stablemask] * one_over_one_plus_alpha_i_stablemask
        nsgrad_i[unstablemask] = (1 - tildealpha_i)*Xi[unstablemask] / one_plus_alpha
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
        grads[1:] += np.sum(ns*(1 - tildealpha_i)*dXi_ps[:,unstablemask] / one_plus_alpha, axis=1)

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
        Nprime = self._tdm.n_selected_events
        N = Nprime + self._tdm.n_pure_bkg_events

        nsgrad2 = -np.sum(self._cache_nsgrad_i**2) - (N - Nprime)/(N - ns)**2

        return nsgrad2


class SingleSourceZeroSigH0SingleDatasetTCLLHRatio(
        ZeroSigH0SingleDatasetTCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a single data set. The
    log-likelihood-ratio function assumes a zero signal null-hypothesis and uses
    a list of independent PDFRatio instances assuming a single source.
    """
    def __init__(
            self, minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            pdfratios):
        """Constructor for creating a 2-component, i.e. signal and background,
        log-likelihood ratio function assuming a single source.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance that defines the source
            hypotheses.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process.
            The names of the source fit parameters must coincide with the signal
            fit parameter names of the PDF ratio objects.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        pdfratios : list of PDFRatio
            The list of PDFRatio instances. A PDFRatio instance might depend on
            none, one, or several fit parameters.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(SingleSourceZeroSigH0SingleDatasetTCLLHRatio, self).__init__(
            minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            pdfratios)

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any fit parameters.
        self._pdfratioarray = SingleSourcePDFRatioArrayArithmetic(
            self._pdfratio_list,
            self._src_fitparam_mapper.fitparamset.fitparam_list)

    def initialize_for_new_trial(self, tl=None):
        """Initializes the log-likelihood ratio function for a new trial.

        Parameters
        ----------
        tl : TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        self._pdfratioarray.initialize_for_new_trial(self._tdm)

    def evaluate(self, fitparam_values, tl=None):
        """Evaluates the log-likelihood ratio function for the given set of
        data events.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.
        tl : TimeLord instance | None
            The optional TimeLord instance to measure the timing of evaluating
            the LLH ratio function.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams+1,)-shaped 1D ndarray
            The ndarray holding the gradient value of log_lambda for each fit
            parameter and ns.
            The first element is the gradient for ns.
        """
        tracing = CFG['debugging']['enable_tracing']

        # Define local variables to avoid (.)-lookup procedure.
        tdm = self._tdm
        pdfratioarray = self._pdfratioarray

        ns = fitparam_values[0]

        N = tdm.n_events

        # Create the fitparams dictionary with the fit parameter names and
        # values.
        with TaskTimer(tl, 'Create fitparams dictionary.'):
            fitparams = self._src_fitparam_mapper.get_src_fitparams(
                fitparam_values[1:])

        # Calculate the data fields that depend on fit parameter values.
        with TaskTimer(tl, 'Calc fit param dep data fields.'):
            tdm.calculate_fitparam_data_fields(
                self._src_hypo_group_manager, fitparams)

        # Calculate the PDF ratio values of all PDF ratio objects, which depend
        # on any fit parameter.
        with TaskTimer(tl, 'Calc pdfratio values.'):
            pdfratioarray.calculate_pdfratio_values(tdm, fitparams, tl=tl)

        # Calculate the product of all the PDF ratio values for each (selected)
        # event.
        with TaskTimer(tl, 'Calc pdfratio value product Ri'):
            Ri = pdfratioarray.get_ratio_product()

        # Calculate Xi for each (selected) event.
        Xi = (Ri - 1.) / N
        if(tracing):
            logger.debug('dtype(Xi)={:s}'.format(str(Xi.dtype)))

        # Calculate the gradients of Xi for each fit parameter (without ns).
        dXi_ps = np.empty((len(fitparam_values)-1,len(Xi)), dtype=np.float64)
        for (idx, fitparam_value) in enumerate(fitparam_values[1:]):
            fitparam_name = self._src_fitparam_mapper.get_src_fitparam_name(idx)

            dRi = np.zeros((len(Xi),), dtype=np.float64)
            for (num_k) in np.arange(len(pdfratioarray._pdfratio_list)):
                # Get the PDFRatio instance from which we need the derivative from.
                pdfratio = pdfratioarray.get_pdfratio(num_k)
                # Calculate the derivative of Ri.
                dRi += pdfratio.get_gradient(tdm, fitparams, fitparam_name) * pdfratioarray.get_ratio_product(excluded_idx=num_k)

            # Calculate the derivative of Xi w.r.t. the fit parameter.
            dXi_ps[idx] = dRi / N

        if(tracing):
            logger.debug(
                '{:s}.evaluate: N={:d}, Nprime={:d}, ns={:.3f}, '.format(
                    classname(self), N, len(Xi), ns))

        with TaskTimer(tl, 'Calc logLamds and grads'):
            (log_lambda, grads) = self.calculate_log_lambda_and_grads(
                fitparam_values, N, ns, Xi, dXi_ps)

        return (log_lambda, grads)


class MultiSourceZeroSigH0SingleDatasetTCLLHRatio(
        SingleSourceZeroSigH0SingleDatasetTCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a single data set assuming zero signal for
    the null-hypothesis. It uses a list of independent PDFRatio instances
    assuming multiple sources (stacking).
    """
    def __init__(
            self, minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            pdfratios, detsigyields):
        """Constructor for creating a 2-component, i.e. signal and background,
        log-likelihood ratio function assuming a single source.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance that defines the source
            hypotheses.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
            The order of the fit parameters defines the order of the fit values
            during the maximization process.
            The names of the source fit parameters must coincide with the signal
            fit parameter names of the PDF ratio objects.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        pdfratios : list of PDFRatio
            The list of PDFRatio instances. A PDFRatio instance might depend on
            none, one, or several fit parameters.
        detsigyields : (N_source_hypo_groups,)-shaped 1D ndarray of DetSigYield
                instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(MultiSourceZeroSigH0SingleDatasetTCLLHRatio, self).__init__(
            minimizer, src_hypo_group_manager, src_fitparam_mapper, tdm,
            pdfratios)

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any fit parameters.
        self._pdfratioarray = SingleSourcePDFRatioArrayArithmetic(
            self._pdfratio_list,
            self._src_fitparam_mapper.fitparamset.fitparam_list)

        self._calc_source_weights = MultiPointSourcesRelSourceWeights(
            src_hypo_group_manager, src_fitparam_mapper, detsigyields)

    def evaluate(self, fitparam_values, tl=None):
        """Evaluates the log-likelihood ratio function for the given set of
        data events.

        Parameters
        ----------
        fitparam_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.
        tl : TimeLord instance | None
            The optional TimeLord instance to measure the timing of evaluating
            the LLH ratio function.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams+1,)-shaped 1D ndarray
            The ndarray holding the gradient value of log_lambda for each fit
            parameter and ns.
            The first element is the gradient for ns.
        """
        _src_w, _src_w_grads = self._calc_source_weights(
                fitparam_values)
        self._tdm.get_data('src_array')['src_w'] = _src_w
        if _src_w_grads is not None:
            self._tdm.get_data('src_array')['src_w_grad'] = _src_w_grads.flatten()
        else:
            self._tdm.get_data('src_array')['src_w_grad'] = np.zeros_like(_src_w)

        (log_lambda, grads) = super(
            MultiSourceZeroSigH0SingleDatasetTCLLHRatio, self).evaluate(
                fitparam_values, tl)

        return (log_lambda, grads)


class DatasetSignalWeights(object, metaclass=abc.ABCMeta):
    """Abstract base class for a dataset signal weight calculator class.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Base class constructor.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SourceFitParameterMapper
            The SourceFitParameterMapper instance that defines the global fit
            parameters and their mapping to the source fit parameters.
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.detsigyield_arr = detsigyields

        if(self._detsigyield_arr.shape[0] != self._src_hypo_group_manager.n_src_hypo_groups):
            raise ValueError('The detsigyields array must have the same number '
                'of source hypothesis groups as the source hypothesis group '
                'manager defines!')

        # Pre-convert the source list of each source hypothesis group into a
        # source array needed for the detector signal yield evaluation.
        # Since all the detector signal yield instances must be of the same
        # kind for each dataset, we can just use the one of the first dataset of
        # each source hypothesis group.
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

    def _create_src_arr_list(self, src_hypo_group_manager, detsigyield_arr):
        """Pre-convert the source list of each source hypothesis group into a
        source array needed for the detector signal yield evaluation.
        Since all the detector signal yield instances must be of the same
        kind for each dataset, we can just use the one of the first dataset of
        each source hypothesis group.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the sources.

        detsigyield_arr : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                        DetSigYield instances
            The collection of DetSigYield instances for each dataset and source
            group combination.
        Returns
        -------
        src_arr_list : list of numpy record ndarrays
            The list of the source numpy record ndarrays, one for each source
            hypothesis group, which is needed by the detector signal yield
            instance.
        """
        src_arr_list = []
        for (gidx, src_hypo_group) in enumerate(src_hypo_group_manager.src_hypo_group_list):
            src_arr_list.append(
                detsigyield_arr[gidx,0].source_to_array(src_hypo_group.source_list)
            )

        return src_arr_list

    @property
    def src_hypo_group_manager(self):
        """The instance of SourceHypoGroupManager, which defines the source
        hypothesis groups.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                'instance of SourceHypoGroupManager!')
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
            raise TypeError('The src_fitparam_mapper property must be an '
                'instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def detsigyield_arr(self):
        """The 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
        DetSigYield instances.
        """
        return self._detsigyield_arr
    @detsigyield_arr.setter
    def detsigyield_arr(self, detsigyields):
        if(not isinstance(detsigyields, np.ndarray)):
            raise TypeError('The detsigyield_arr property must be an instance '
                'of numpy.ndarray!')
        if(detsigyields.ndim != 2):
            raise ValueError('The detsigyield_arr property must be a '
                'numpy.ndarray with 2 dimensions!')
        if(not issequenceof(detsigyields.flat, DetSigYield)):
            raise TypeError('The detsigyield_arr property must contain '
                'DetSigYield instances, one for each source hypothesis group '
                'and dataset combination!')
        self._detsigyield_arr = detsigyields

    @property
    def n_datasets(self):
        """(read-only) The number of datasets this DatasetSignalWeights instance
        is for.
        """
        return self._detsigyield_arr.shape[1]

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the SourceHypoGroupManager instance of this
        DatasetSignalWeights instance. This will also recreate the internal
        source numpy record arrays needed for the detector signal efficiency
        calculation.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance, that should be used for
            this dataset signal weights instance.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

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
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
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
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """

        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        # Convert sequence into a 2D numpy array.
        detsigyields = np.atleast_2d(detsigyields)

        super(SingleSourceDatasetSignalWeights, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper, detsigyields)

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

        Y = np.empty((N_datasets,), dtype=np.float64)
        if(N_fitparams > 0):
            Y_grads = np.empty((N_datasets, N_fitparams), dtype=np.float64)

        # Loop over the detector signal efficiency instances for the first and
        # only source hypothesis group.
        for (j, detsigyield) in enumerate(self._detsigyield_arr[0]):
            (Yj, Yj_grads) = detsigyield(self._src_arr_list[0], fitparams_arr)
            # Store the detector signal yield and its fit parameter
            # gradients for the first and only source (element 0).
            Y[j] = Yj[0]
            if(N_fitparams > 0):
                if Yj_grads is None:
                    Y_grads[j] = np.zeros_like(Yj[0])
                else:
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


class MultiSourceDatasetSignalWeights(SingleSourceDatasetSignalWeights):
    """This class calculates the dataset signal weight factors for each dataset
    assuming multiple sources.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Constructs a new DatasetSignalWeights instance assuming multiple
        sources.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """

        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(MultiSourceDatasetSignalWeights, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper, detsigyields)

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

        Y = np.empty((N_datasets, len(self._src_arr_list[0])), dtype=np.float64)
        if(N_fitparams > 0):
            Y_grads = np.empty((N_datasets, len(self._src_arr_list[0]), N_fitparams), dtype=np.float64)

        # Loop over the detector signal efficiency instances for the first and
        # only source hypothesis group.
        for (k, detsigyield_k) in enumerate(self._detsigyield_arr):
            for (j, detsigyield) in enumerate(detsigyield_k):
                (Yj, Yj_grads) = detsigyield(self._src_arr_list[k], fitparams_arr)
                # Store the detector signal yield and its fit parameter
                # gradients for the first and only source (element 0).
                Y[j] = Yj
                if(N_fitparams > 0):
                    Y_grads[j] = Yj_grads.T

        sum_Y = np.sum(Y)

        # f is a (N_datasets,)-shaped 1D ndarray.
        f = np.sum(Y, axis=1) / sum_Y

        # f_grads is a (N_datasets, N_fitparams)-shaped 2D ndarray.
        if(N_fitparams > 0):
            # sum_Y_grads is a (N_datasets, N_fitparams,)-shaped 2D array.
            sum_Y_grads = np.sum(Y_grads, axis=1)
            f_grads = (sum_Y_grads*sum_Y - (f*sum_Y)[...,np.newaxis]*np.sum(sum_Y_grads, axis=0)) / sum_Y**2
        else:
            f_grads = None

        return (f, f_grads)


class SourceWeights(object, metaclass=abc.ABCMeta):
    """Abstract base class for a source weight calculator class.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Constructs a new SourceWeights instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SourceFitParameterMapper
            The SourceFitParameterMapper instance that defines the global fit
            parameters and their mapping to the source fit parameters.
        detsigyields : (N_source_hypo_groups,)-shaped 1D ndarray of DetSigYield
                instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.detsigyield_arr = np.atleast_1d(detsigyields)

        if(self._detsigyield_arr.shape[0] != self._src_hypo_group_manager.n_src_hypo_groups):
            raise ValueError('The detsigyields array must have the same number '
                'of source hypothesis groups as the source hypothesis group '
                'manager defines!')

        # Pre-convert the source list of each source hypothesis group into a
        # source array needed for the detector signal yield evaluation.
        # Since all the detector signal yield instances must be of the same
        # kind for each dataset, we can just use the one of the first dataset of
        # each source hypothesis group.
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

    def _create_src_arr_list(self, src_hypo_group_manager, detsigyield_arr):
        """Pre-convert the source list of each source hypothesis group into a
        source array needed for the detector signal yield evaluation.
        Since all the detector signal yield instances must be of the same
        kind for each dataset, we can just use the one of the first dataset of
        each source hypothesis group.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the sources.

        detsigyield_arr : (N_source_hypo_groups,)-shaped 1D ndarray of
                DetSigYield instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        Returns
        -------
        src_arr_list : list of numpy record ndarrays
            The list of the source numpy record ndarrays, one for each source
            hypothesis group, which is needed by the detector signal yield
            instance.
        """
        src_arr_list = []
        for (gidx, src_hypo_group) in enumerate(src_hypo_group_manager.src_hypo_group_list):
            src_arr_list.append(
                detsigyield_arr[gidx].source_to_array(src_hypo_group.source_list)
            )

        return src_arr_list

    @property
    def src_hypo_group_manager(self):
        """The instance of SourceHypoGroupManager, which defines the source
        hypothesis groups.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                'instance of SourceHypoGroupManager!')
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
            raise TypeError('The src_fitparam_mapper property must be an '
                'instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def detsigyield_arr(self):
        """The (N_source_hypo_groups,)-shaped 1D ndarray of DetSigYield
        instances.
        """
        return self._detsigyield_arr
    @detsigyield_arr.setter
    def detsigyield_arr(self, detsigyields):
        if(not isinstance(detsigyields, np.ndarray)):
            raise TypeError('The detsigyield_arr property must be an instance '
                'of numpy.ndarray!')
        if(detsigyields.ndim != 1):
            raise ValueError('The detsigyield_arr property must be a '
                'numpy.ndarray with 1 dimensions!')
        if(not issequenceof(detsigyields.flat, DetSigYield)):
            raise TypeError('The detsigyield_arr property must contain '
                'DetSigYield instances, one for each source hypothesis group!')
        self._detsigyield_arr = detsigyields

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the SourceHypoGroupManager instance of this
        DatasetSignalWeights instance. This will also recreate the internal
        source numpy record arrays needed for the detector signal efficiency
        calculation.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance, that should be used for
            this dataset signal weights instance.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

    @abc.abstractmethod
    def __call__(self, fitparam_values):
        """This method is supposed to calculate source weights and
        their gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_sources,)-shaped 1D ndarray
            The source weight factor for each source.
        f_grads : (N_sources,)-shaped 1D ndarray | None
            The gradients of the source weight factors. None is returned if
            there are no fit parameters beside ns.
        """
        pass


class MultiPointSourcesRelSourceWeights(SourceWeights):
    """This class calculates the relative source weights for a group of point
    sources.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Constructs a new MultiPointSourcesRelSourceWeights instance assuming
        multiple sources.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        detsigyields : (N_source_hypo_groups,)-shaped 1D ndarray of
                DetSigYield instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        """

        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(MultiPointSourcesRelSourceWeights, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper, detsigyields)

    def __call__(self, fitparam_values):
        """Calculates the source weights and its fit parameter gradients
        for each source.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_sources,)-shaped 1D ndarray
            The source weight factor for each source.
        f_grads : (N_sources,)-shaped 1D ndarray | None
            The gradients of the source weight factors. None is returned if
            there are no fit parameters beside ns.
        """
        fitparams_arr = self._src_fitparam_mapper.get_fitparams_array(fitparam_values[1:])

        N_fitparams = self._src_fitparam_mapper.n_global_fitparams

        Y = []
        Y_grads = []

        # Loop over detector signal efficiency instances for each source
        # hypothesis group in source hypothesis group manager.
        for (g, detsigyield) in enumerate(self._detsigyield_arr):
            (Yg, Yg_grads) = detsigyield(self._src_arr_list[g], fitparams_arr)

            # Store the detector signal yield and its fit parameter
            # gradients for all sources.
            Y.append(Yg)
            if(N_fitparams > 0):
                Y_grads.append(Yg_grads.T)

        Y = np.array(Y)
        sum_Y = np.sum(Y)

        # f is a (N_sources,)-shaped 1D ndarray.
        f = Y / sum_Y

        # Flatten the array so that each relative weight corresponds to specific
        # source.
        f = f.flatten()

        if(N_fitparams > 0):
            Y_grads = np.concatenate(Y_grads)

            # Sum over fit parameter gradients axis.
            # f_grads is a (N_sources,)-shaped 1D ndarray.
            f_grads = np.sum(Y_grads, axis=1) / sum_Y
        else:
            f_grads = None

        return (f, f_grads)


class MultiDatasetTCLLHRatio(TCLLHRatio):
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
    def __init__(self, minimizer, dataset_signal_weights, llhratios):
        """Creates a new composite two-component log-likelihood ratio function.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        dataset_signal_weights : DatasetSignalWeights
            An instance of DatasetSignalWeights, which calculates the relative
            dataset weight factors.
        llhratios : sequence of SingleDatasetTCLLHRatio instances
            The sequence of the two-component log-likelihood ratio functions,
            one for each dataset.
        """
        self.dataset_signal_weights = dataset_signal_weights
        self.llhratio_list = llhratios

        super(MultiDatasetTCLLHRatio, self).__init__(
            minimizer, self._llhratio_list[0].mean_n_sig_0)

        # Check if the number of datasets the DatasetSignalWeights instance is
        # made for equals the number of log-likelihood ratio functions.
        if(self.dataset_signal_weights.n_datasets != len(self._llhratio_list)):
            raise ValueError('The number of datasets the DatasetSignalWeights '
                'instance is made for must be equal to the number of '
                'log-likelihood ratio functions!')

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
            raise TypeError('The dataset_signal_weights property must be an '
                'instance of DatasetSignalWeights!')
        self._dataset_signal_weights = obj

    @property
    def llhratio_list(self):
        """(read-only) The list of TCLLHRatio instances, which are part of this
        composite log-likelihood-ratio function.
        """
        return self._llhratio_list
    @llhratio_list.setter
    def llhratio_list(self, llhratios):
        if(not issequenceof(llhratios, SingleDatasetTCLLHRatio)):
            raise TypeError('The llhratio_list property must be a sequence of '
                'SingleDatasetTCLLHRatio instances!')
        self._llhratio_list = list(llhratios)

    @property
    def n_selected_events(self):
        """(read-only) The sum of selected events of each individual
        log-likelihood ratio function.
        """
        n_selected_events = 0
        for llhratio in self._llhratio_list:
            n_selected_events += llhratio.tdm.n_selected_events
        return n_selected_events

    @TCLLHRatio.mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        TCLLHRatio.mean_n_sig_0.fset(self, v)
        for llhratio in self._llhratio_list:
            llhratio.mean_n_sig_0 = self._mean_n_sig_0

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the source hypo group manager of all objects of this LLH
        ratio function, hence, calling the `change_source_hypo_group_manager`
        method of all TCLLHRatio objects of this LLHRatio instance.
        """
        # Change the source hypo group manager of the DatasetSignalWeights
        # instance.
        self._dataset_signal_weights.change_source_hypo_group_manager(
            src_hypo_group_manager)

        for llhratio in self._llhratio_list:
            llhratio.change_source_hypo_group_manager(src_hypo_group_manager)

    def initialize_for_new_trial(self, tl=None):
        """Initializes the log-likelihood-ratio function for a new trial.
        """
        for llhratio in self._llhratio_list:
            llhratio.initialize_for_new_trial(tl=tl)

    def evaluate(self, fitparam_values, tl=None):
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
        tracing = CFG['debugging']['enable_tracing']

        ns = fitparam_values[0]
        if(tracing):
            logger.debug(
                '{:s}.evaluate: ns={:.3f}'.format(
                    classname(self), ns))

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
        grads = np.zeros((len(fitparam_values),), dtype=np.float64)

        # Create an array holding the fit parameter values for a particular
        # llh ratio function. Since we need to adjust ns with nsj it's more
        # efficient to create this array once and use it within the for loop
        # over the llh ratio functions.
        llhratio_fitparam_values = np.empty(
            (len(fitparam_values),), dtype=np.float64)
        # Loop over the llh ratio functions.
        for (j, llhratio) in enumerate(self._llhratio_list):
            if(tracing):
                logger.debug(
                    'nsf[j={:d}] = {:.3f}'.format(
                        j, nsf[j]))
            llhratio_fitparam_values[0] = nsf[j]
            llhratio_fitparam_values[1:] = fitparam_values[1:]
            (log_lambda_j, grads_j) = llhratio.evaluate(
                llhratio_fitparam_values, tl=tl)
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

        nsgrad2j = np.empty((len(self._llhratio_list),), dtype=np.float64)
        # Loop over the llh ratio functions and their second derivative.
        llhratio_fitparam_values = np.empty(
            (len(fitparam_values),), dtype=np.float64)
        for (j, llhratio) in enumerate(self._llhratio_list):
            llhratio_fitparam_values[0] = nsf[j]
            llhratio_fitparam_values[1:] = fitparam_values[1:]
            nsgrad2j[j] = llhratio.calculate_ns_grad2(llhratio_fitparam_values)

        nsgrad2 = np.sum(nsgrad2j * self._cache_f**2)

        return nsgrad2


class NsProfileMultiDatasetTCLLHRatio(TCLLHRatio):
    r"""This class implements a profile log-likelihood ratio function that has
    only ns as fit parameter. It uses a MultiDatasetTCLLHRatio instance as
    log-likelihood function. Hence, mathematically it is

    .. math::

        \Lambda(n_s) = \frac{L(n_s)}{L(n_s=n_{s,0})},

    where :math:`n_{s,0}` is the fixed mean number of signal events for the
    null-hypothesis.
    """
    def __init__(self, minimizer, mean_n_sig_0, llhratio):
        """Creates a new ns-profile log-likelihood-ratio function with a
        null-hypothesis where ns is fixed to `mean_n_sig_0`.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        mean_n_sig_0 : float
            The fixed parameter value for the mean number of signal events of
            the null-hypothesis.
        llhratio : instance of MultiDatasetTCLLHRatio
            The instance of MultiDatasetTCLLHRatio, which should be used as
            log-likelihood function.
        """
        super(NsProfileMultiDatasetTCLLHRatio, self).__init__(
            minimizer, mean_n_sig_0)

        self.llhratio = llhratio

        # Check that the given log-likelihood-ratio function has no fit
        # parameters, i.e. only ns in the end.
        for sub_llhratio in llhratio.llhratio_list:
            n_global_fitparams = sub_llhratio.src_fitparam_mapper.n_global_fitparams
            if(n_global_fitparams != 0):
                raise ValueError('The log-likelihood-ratio functions of the '
                    'MultiDatasetTCLLHRatio instance must have no global fit '
                    'parameters, i.e. only ns in the end! Currently it has %d '
                    'global fit parameters'%(n_global_fitparams))

        # Define a member to hold the constant null-hypothesis log-likelihood
        # function value for ns=mean_n_sig_0.
        self._logL_0 = None

    @property
    def llhratio(self):
        """The instance of MultiDatasetTCLLHRatio, which should be used as
        log-likelihood function.
        """
        return self._llhratio
    @llhratio.setter
    def llhratio(self, obj):
        if(not isinstance(obj, MultiDatasetTCLLHRatio)):
            raise TypeError('The llhratio property must be an instance of '
                'MultiDatasetTCLLHRatio!')
        self._llhratio = obj

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the source hypo group manager of all objects of this LLH
        ratio function, hence, calling the `change_source_hypo_group_manager`
        method of the underlaying MultiDatasetTCLLHRatio instance of this
        LLHRatio instance.
        """
        self._llhratio.change_source_hypo_group_manager(src_hypo_group_manager)

    def initialize_for_new_trial(self, tl=None):
        """Initializes the log-likelihood-ratio function for a new trial.

        Parameters
        ----------
        tl : TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        self._llhratio.initialize_for_new_trial(tl=tl)

        # Compute the constant log-likelihood function value for the
        # null-hypothesis.
        fitparam_values_0 = np.array([self._mean_n_sig_0], dtype=np.float64)
        (self._logL_0, grads_0) = self._llhratio.evaluate(fitparam_values_0)

    def evaluate(self, fitparam_values):
        """Evaluates the log-likelihood-ratio function and returns its value and
        global fit parameter gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams)-shaped numpy 1D ndarray
            The ndarray holding the current values of the global fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value of this log-likelihood-ratio
            function.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value of this log-likelihood-ratio
            for ns.
            By definition the first element is the gradient for ns.
        """
        (logL, grads) = self._llhratio.evaluate(fitparam_values)

        return (logL - self._logL_0, grads)

    def calculate_ns_grad2(self, fitparam_values):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.

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
        return self._llhratio.calculate_ns_grad2(fitparam_values)

#class NestedProfileLLHRatio(LLHRatio, metaclass=abc.ABCMeta):
    #r"""This class provides the abstract base class for a nested profile
    #log-likelihood ratio function, which is, by definition, of the form

    #.. math::

        #\Lambda = \frac{\sup_{\Theta_0} L(\theta|D)}{\sup_{\Theta} L(\theta|D)}

    #where :math:`\theta` are the possible fit parameters, and :math:`\Theta`
    #and :math:`\Theta_0` are the total and nested fit parameter spaces,
    #respectively.
    #"""

    #def __init__(self, ):
        #super(NestedProfileLLHRatio, self).__init__()


#class MultiDatasetNestedProfileLLHRatio(NestedProfileLLHRatio):
    #"""This class provides a nested profile log-likelihood ratio function for
    #multiple data sets.
    #"""
    #def __init__(self):
        #super(MultiDatasetNestedProfileLLHRatio, self).__init__()

