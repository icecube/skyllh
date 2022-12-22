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
from skyllh.core.dataset_signal_weights import (
    DatasetSignalWeights,
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
    ParameterModelMapper,
)
from skyllh.core.pdfratio import (
    PDFRatio,
    SingleSourcePDFRatioArrayArithmetic
)
from skyllh.core.timing import TaskTimer
from skyllh.core.model import SourceModel


logger = get_logger(__name__)


class LLHRatio(object, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function.
    """

    def __init__(
            self,
            pmm,
            minimizer,
            **kwargs):
        """Creates a new LLH ratio function instance.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        """
        super().__init__(**kwargs)

        self.pmm = pmm
        self.minimizer = minimizer

    @property
    def pmm(self):
        """The ParameterModelMapper instance providing the mapping of
        global floating parameters to individual models.
        """
        return self._pmm
    @pmm.setter
    def pmm(self, mapper):
        if not isinstance(mapper, ParameterModelMapper):
            raise TypeError(
                'The pmm property must be an instance of '
                'ParameterModelMapper!')
        self._pmm = mapper

    @property
    def minimizer(self):
        """The Minimizer instance used to minimize the negative of the
        log-likelihood ratio function.
        """
        return self._minimizer
    @minimizer.setter
    def minimizer(self, minimizer):
        if(not isinstance(minimizer, Minimizer)):
            raise TypeError(
                'The minimizer property must be an instance of Minimizer!')
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
    def evaluate(self, gflp_values, tl=None):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        gflp_values : numpy 1D ndarray
            The ndarray holding the current values of the global floating
            parameters.
        tl : TimeLord | None
            The optional TimeLord instance to use for measuring timeing.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the gradient value for each global floating
            parameter.
        """
        pass

    def maximize(self, rss, tl=None):
        """Maximize the log-likelihood ratio function, by using the ``evaluate``
        method.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance to draw random numbers from.
            This is needed to generate random parameter initial values.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization process.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        gflp_values : (N_fitparam)-shaped 1D ndarray
            The ndarray holding the values of the global floating parameters.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        tracing = CFG['debugging']['enable_tracing']

        # Define the negative llhratio function, that will get minimized.
        self_evaluate = self.evaluate
        def negative_llhratio_func(gfp_values, func_stats, tl=None):
            func_stats['n_calls'] += 1
            with TaskTimer(tl, 'Evaluate llh-ratio function.'):
                (f, grads) = self_evaluate(gfp_values, tl=tl)
                if tracing: logger.debug(
                    f'LLH-ratio func value f={f:g}, grads={str(grads)}')
            return (-f, -grads)

        minimize_kwargs = {'func_provides_grads': True}

        func_stats = {'n_calls': 0}
        with TaskTimer(tl, 'Minimize -llhratio function.'):
            (gflp_values, fmin, status) = self._minimizer.minimize(
                rss=rss,
                paramset=self._pmm.global_paramset,
                func=negative_llhratio_func,
                args=(func_stats,tl),
                kwargs=minimize_kwargs)
        log_lambda_max = -fmin
        status['n_llhratio_func_calls'] = func_stats['n_calls']

        logger.debug(
            f'Maximized LLH ratio function with '
            f'{status["n_llhratio_func_calls"]:d} calls')

        return (log_lambda_max, gflp_values, status)


class TCLLHRatio(LLHRatio, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function with two
    components, i.e. signal and background.
    """

    def __init__(
            self,
            pmm,
            minimizer,
            mean_n_sig_0,
            **kwargs):
        """Creates a new two-component LLH ratio function instance.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        mean_n_sig_0 : float
            The fixed mean number of signal events for the null-hypothesis.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            **kwargs)

        self.mean_n_sig_0 = mean_n_sig_0

    @property
    def mean_n_sig_0(self):
        """The parameter value for the mean number of signal events of the
        null-hypothesis.
        """
        return self._mean_n_sig_0
    @mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        v = float_cast(
            v,
            'The mean_n_sig_0 property must be castable to a float value!')
        self._mean_n_sig_0 = v

    @abc.abstractmethod
    def calculate_ns_grad2(self, gflp_values, ns_pidx):
        """This method is supposed to calculate the second derivative of the
        log-likelihood ratio function w.r.t. the fit parameter ns, the number
        of signal events in the data set.

        Parameters
        ----------
        gflp_values : numpy (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the current values of the global floating
            parameters.
        ns_pidx : int
            The index of the global ns floating parameter.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        pass

    def maximize(self, rss, tl=None):
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
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization of the LLH-ratio function.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        gflp_values : (N_fitparam)-shaped 1D ndarray
            The ndarray holding the global floating parameter values.
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

            global_paramset = self._pmm.global_paramset

            # Determine the floating parameter index of the ns parameter.
            ns_pidx = global_paramset.get_floating_pidx(param_name='ns')

            def negative_llhratio_func_nr1d_ns(gflp_values, tl):
                with TaskTimer(tl, 'Evaluate llh-ratio function.'):
                    (f, grads) = self__evaluate(gflp_values, tl=tl)
                with TaskTimer(tl, 'Calculate 2nd derivative of llh-ratio '
                        'function w.r.t. ns'):
                    grad2_ns = self__calculate_ns_grad2(
                        gflp_values=gflp_values,
                        ns_pidx=ns_pidx)

                return (-f, -grads[ns_pidx], -grad2_ns)

            (gflp_values, fmin, status) = self._minimizer.minimize(
                rss=rss,
                paramset=global_paramset,
                func=negative_llhratio_func_nr1d_ns,
                args=(tl,))
            log_lambda_max = -fmin

            return (log_lambda_max, gflp_values, status)

        return super().maximize(rss, tl=tl)


class SingleDatasetTCLLHRatio(TCLLHRatio, metaclass=abc.ABCMeta):
    """Abstract base class for a log-likelihood (LLH) ratio function with two
    components, i.e. signal and background, for a single data set.
    """

    def __init__(
            self,
            pmm,
            minimizer,
            shg_mgr,
            tdm,
            mean_n_sig_0,
            **kwargs):
        """Creates a new two-component LLH ratio function instance for a single
        data set.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        shg_mgr : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance that defines the source
            hypothesis groups.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        mean_n_sig_0 : float
            The fixed mean number of signal events for the null-hypothesis.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            mean_n_sig_0=mean_n_sig_0,
            **kwargs)

        self.shg_mgr = shg_mgr
        self.tdm = tdm

        # Calculate the data fields that solely depend on source parameters.
        self._tdm.calculate_source_data_fields(self.shg_mgr)

    @property
    def shg_mgr(self):
        """The SourceHypoGroupManager instance that defines the source
        hypothesis groups.
        """
        return self._shg_mgr
    @shg_mgr.setter
    def shg_mgr(self, mgr):
        if(not isinstance(mgr, SourceHypoGroupManager)):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager!')
        self._shg_mgr = mgr

    @property
    def tdm(self):
        """The TrialDataManager instance that holds the trial event data and
        additional data fields for this LLH ratio function.
        """
        return self._tdm
    @tdm.setter
    def tdm(self, manager):
        if(not isinstance(manager, TrialDataManager)):
            raise TypeError(
                'The tdm property must be an instance of TrialDataManager!')
        self._tdm = manager

    def change_source_hypo_group_manager(self, shg_mgr):
        """Changes the source hypothesis group manager of this two-component LLH
        ratio function.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance.
        """
        self.shg_mgr = shg_mgr
        self._tdm.change_source_hypo_group_manager(shg_mgr)


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
            self,
            pmm,
            minimizer,
            shg_mgr,
            tdm,
            pdfratios,
            **kwargs):
        """Constructor of the two-component log-likelihood ratio function.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        shg_mgr : instance of SourceHypoGroupManager
            The SourceHypoGroupManager instance that defines the source
            hypothesis groups.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        pdfratios : sequence of PDFRatio
            The sequence of PDFRatio instances. A PDFRatio instance might depend
            on none, one, or several fit parameters.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            shg_mgr=shg_mgr,
            tdm=tdm,
            mean_n_sig_0=0,
            **kwargs)

        self.pdfratio_list = pdfratios

        # Define cache variables for the evaluate method to store values needed
        # for a possible calculation of the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_gflp_values = None
        self._cache_nsgrad_i = None

    @SingleDatasetTCLLHRatio.mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        SingleDatasetTCLLHRatio.mean_n_sig_0.fset(self, v)
        if(self._mean_n_sig_0 != 0):
            raise ValueError(
                f'The {classname(self)} class is only valid for '
                f'mean_n_sig_0 = 0!')

    @property
    def pdfratio_list(self):
        """The list of PDFRatio instances.
        """
        return self._pdfratio_list
    @pdfratio_list.setter
    def pdfratio_list(self, seq):
        if(not issequenceof(seq, PDFRatio)):
            raise TypeError(
                'The pdfratio_list property must be a sequence of PDFRatio '
                'instances!')
        self._pdfratio_list = list(seq)

    def calculate_log_lambda_and_grads(
            self, gflp_values, N, ns_pidx, ns, Xi, dXi_ps):
        """Calculates the log(Lambda) value and its gradient for each global fit
        parameter. This calculation is source and detector independent.

        Parameters
        ----------
        gflp_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the fit parameters.
            By definition, the first element is the fit parameter for the number
            of signal events, ns.
            These numbers are used as cache key to validate the ``nsgrad_i``
            values for the given fit parameter values for a possible later
            calculation of the second derivative w.r.t. ns of the log-likelihood
            ratio function.
        N : int
            The total number of events.
        ns_pidx : int
            The index of the global floating parameter ns.
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
                f'N={N:d}, Nprime={Nprime:d}')

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
                f'{np.count_nonzero(unstablemask):d}')

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
        # Note: We create a copy of the gflp_values array here to make sure
        #       that the values don't get changed outside this method before the
        #       calculate_ns_grad2 method is called.
        self._cache_gflp_values = gflp_values.copy()
        self._cache_nsgrad_i = nsgrad_i
        # Calculate the first derivative w.r.t. ns.
        grads[ns_pidx] = np.sum(nsgrad_i) - (N - Nprime)/(N - ns)

        # For each other fit parameter.
        # For all numerical stable events.
        m = np.ones((len(grads),), dtype=bool)
        m[ns_pidx] = False
        grads[m] = np.sum(ns * one_over_one_plus_alpha_i_stablemask * dXi_ps[:,stablemask], axis=1)
        # For all numerical unstable events.
        grads[m] += np.sum(ns*(1 - tildealpha_i)*dXi_ps[:,unstablemask] / one_plus_alpha, axis=1)

        return (log_lambda, grads)

    def calculate_ns_grad2(self, gflp_values, ns_pidx):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.
        This method tries to use cached values for the first derivative
        w.r.t. ns of the log-likelihood ratio function for the individual
        events. If cached values don't exist or do not match the given fit
        parameter values, they will get calculated automatically by calling the
        evaluate method with the given fit parameter values.

        Parameters
        ----------
        gflp_values : numpy (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the current values of the global floating
            parameters.
        ns_pidx : int
            The parameter index of the global floating parameter ns.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        # Check if the cached nsgrad_i values match the given gflp_values.
        if((self._cache_gflp_values is None) or
           (not np.all(self._cache_gflp_values == gflp_values))):
            # Calculate the cache values by evaluating the log-likelihood ratio
            # function.
            self.evaluate(gflp_values)

        ns = gflp_values[ns_pidx]
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
            self,
            pmm,
            minimizer,
            shg_mgr,
            tdm,
            pdfratios,
            **kwargs):
        """Constructor for creating a 2-component, i.e. signal and background,
        log-likelihood ratio function assuming a single source.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        shg_mgr : instance of SourceHypoGroupManager
            The SourceHypoGroupManager instance that defines the source
            hypothesis groups.
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data and
            additional data fields for this LLH ratio function.
        pdfratios : list of PDFRatio
            The list of PDFRatio instances. A PDFRatio instance might depend on
            none, one, or several fit parameters.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            shg_mgr=shg_mgr,
            tdm=tdm,
            pdfratios=pdfratios,
            **kwargs)

        if pmm.n_sources != 1:
            raise RuntimeError(
                f'The LLH ratio function class {classname(self)} can handle '
                f'only a single source!')

        # Construct a PDFRatio array arithmetic object specialized for a single
        # source. This will pre-calculate the PDF ratio values for all PDF ratio
        # instances, which do not depend on any floating parameters.
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
            By definition, the first element is the floating parameter for the
            number of signal events, ns.
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

        # Determine the index of the global floating parameter ns.
        ns_pidx = self._pmm.global_paramset.get_floating_pidx(
            param_name='ns')
        if ns_pidx != 0:
            raise RuntimeError(
                f'The global floating parameter "ns" must be the first '
                f'floating parameter! But its index is {ns_pidx}!')

        ns = fitparam_values[ns_pidx]

        N = tdm.n_events

        # Create the global_fitparams dictionary with the global fit parameter
        # names and values.
        global_fitparams = self._pmm.get_global_floating_params_dict(
            gflp_values=fitparam_values)

        # Calculate the data fields that depend on global fit parameters.
        with TaskTimer(
                tl,
                'Calculate global fit parameter dependent data fields.'):
            tdm.calculate_global_fitparam_data_fields(
                shg_mgr=self._shg_mgr,
                pmm=self._pmm,
                global_fitparams=global_fitparams)

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
                fitparam_values, N, ns_pidx, ns, Xi, dXi_ps)

        return (log_lambda, grads)


class MultiSourceZeroSigH0SingleDatasetTCLLHRatio(
        SingleSourceZeroSigH0SingleDatasetTCLLHRatio):
    """This class implements a 2-component, i.e. signal and background,
    log-likelihood ratio function for a single data set assuming zero signal for
    the null-hypothesis. It uses a list of independent PDFRatio instances
    assuming multiple sources (stacking).
    """
    def __init__(
            self,
            minimizer,
            src_hypo_group_manager,
            src_fitparam_mapper,
            tdm,
            pdfratios,
            detsigyields):
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

        (log_lambda, grads) = super().evaluate(
                fitparam_values, tl)

        return (log_lambda, grads)


class SourceWeights(object, metaclass=abc.ABCMeta):
    """This class is DEPRECATED!
    Use :py:class:`skyllh.core.weights.SourceDetectorWeights` instead!

    Abstract base class for a source weight calculator class.
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

    def _create_src_arr_list(self, shg_mgr, detsigyield_arr):
        """Pre-convert the source list of each source hypothesis group into a
        source array needed for the detector signal yield evaluation.
        Since all the detector signal yield instances must be of the same
        kind for each dataset, we can just use the one of the first dataset of
        each source hypothesis group.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
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
        for (gidx, shg) in enumerate(shg_mgr.src_hypo_group_list):
            src_arr_list.append(
                detsigyield_arr[gidx].sources_to_recarray(shg.source_list)
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
    """This class is DEPRECATED!
    Use :py:class:`skyllh.core.weights.SourceDetectorWeights` instead!

    This class calculates the relative source weights for a group of point
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
