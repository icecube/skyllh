# -*- coding: utf-8 -*-

"""The llhratio module provides classes implementing the log-likelihood ratio
functions. In general these should be detector independent, because they
implement the pure math of the log-likelihood ratio function.
"""

import abc
import numpy as np

from skyllh.core.config import (
    CFG,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.py import (
    classname,
    issequenceof,
    float_cast,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)
from skyllh.core.minimizer import (
    Minimizer,
    NR1dNsMinimizerImpl,
    NRNsScan2dMinimizerImpl,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.pdfratio import (
    PDFRatio,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.core.weights import (
    DatasetSignalWeightFactors,
)


logger = get_logger(__name__)


class LLHRatio(
        object,
        metaclass=abc.ABCMeta):
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
            global parameters to local parameters of individual models.
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
                'ParameterModelMapper! '
                f'Its current type is {classname(mapper)}.')
        self._pmm = mapper

    @property
    def minimizer(self):
        """The Minimizer instance used to minimize the negative of the
        log-likelihood ratio function.
        """
        return self._minimizer

    @minimizer.setter
    def minimizer(self, minimizer):
        if not isinstance(minimizer, Minimizer):
            raise TypeError(
                'The minimizer property must be an instance of Minimizer! '
                f'Its current type is {classname(minimizer)}.')
        self._minimizer = minimizer

    def initialize_for_new_trial(
            self,
            tl=None):
        """This method will be called by the Analysis class after new trial data
        has been initialized to the trial data manager. Derived classes can make
        use of this call hook to perform LLHRatio specific trial initialization.

        Parameters
        ----------
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for timing measurements.
        """
        pass

    @abc.abstractmethod
    def evaluate(
            self,
            fitparam_values,
            params_recarray=None,
            tl=None):
        """This method evaluates the LLH ratio function for the given set of
        fit parameter values.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the current values
            of the global fit parameters.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D numpy ndarray holding the gradient
            value for each global fit parameter.
        """
        pass

    def maximize(
            self,
            rss,
            tl=None):
        """Maximize the log-likelihood ratio function, by using the ``evaluate``
        method.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to draw random numbers from.
            This is needed to generate random parameter initial values.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time the
            maximization process.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the values of the
            global fit parameters.
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
                (f, grads) = self_evaluate(
                    fitparam_values=fitparam_values,
                    params_recarray=None,
                    tl=tl)
                if tracing:
                    logger.debug(
                        f'LLH-ratio func value f={f:g}, grads={str(grads)}')
            return (-f, -grads)

        minimize_kwargs = {
            'func_provides_grads': True
        }

        func_stats = {
            'n_calls': 0
        }
        with TaskTimer(tl, 'Minimize -llhratio function.'):
            (fitparam_values, fmin, status) = self._minimizer.minimize(
                rss=rss,
                paramset=self._pmm.global_paramset,
                func=negative_llhratio_func,
                args=(func_stats, tl),
                kwargs=minimize_kwargs)
        log_lambda_max = -fmin
        status['n_llhratio_func_calls'] = func_stats['n_calls']

        logger.debug(
            f'Maximized LLH ratio function with '
            f'{status["n_llhratio_func_calls"]:d} calls')

        return (log_lambda_max, fitparam_values, status)


class TCLLHRatio(
        LLHRatio,
        metaclass=abc.ABCMeta):
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
    def calculate_ns_grad2(
            self,
            fitparam_values,
            ns_pidx):
        """This method is supposed to calculate the second derivative of the
        log-likelihood ratio function w.r.t. the fit parameter ns, the number
        of signal events in the data set.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D numpy ndarray holding the current
            values of the global fit parameters.
        ns_pidx : int
            The index of the global ns fit parameter.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        pass

    def maximize_with_1d_newton_rapson_minimizer(
            self,
            rss,
            tl=None):
        """Maximizes this log-likelihood ratio function, by minimizing its
        negative using a 1D Newton-Rapson minimizer.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to draw
            random numbers from. It is used by the minimizer to generate random
            fit parameter initial values.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time the
            maximization of the LLH-ratio function.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the global
            fit parameter values.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        # Define the negative llhratio function, that will get minimized
        # when using the Newton-Rapson 1D minimizer for llhratio functions
        # depending solely on ns.
        self__evaluate = self.evaluate
        self__calculate_ns_grad2 = self.calculate_ns_grad2

        ns_pidx = self._pmm.get_gflp_idx(name='ns')

        def negative_llhratio_func_nr1d_ns(fitparam_values, tl):
            with TaskTimer(
                    tl,
                    'Evaluate llh-ratio function.'):
                (f, grads) = self__evaluate(
                    fitparam_values=fitparam_values,
                    params_recarray=None,
                    tl=tl)
            with TaskTimer(
                    tl,
                    'Calculate 2nd derivative of llh-ratio function w.r.t. ns'):
                grad2_ns = self__calculate_ns_grad2(
                    fitparam_values=fitparam_values,
                    ns_pidx=ns_pidx)

            return (-f, -grads[ns_pidx], -grad2_ns)

        (fitparam_values, fmin, status) = self._minimizer.minimize(
            rss=rss,
            paramset=self._pmm.global_paramset,
            func=negative_llhratio_func_nr1d_ns,
            args=(tl,))
        log_lambda_max = -fmin

        return (log_lambda_max, fitparam_values, status)

    def maximize(
            self,
            rss,
            tl=None):
        """Maximizes this log-likelihood ratio function, by minimizing its
        negative.
        This method has a special implementation when a 1D Newton-Rapson
        minimizer is used. In that case only the first and second derivative
        of the log-likelihood ratio function is calculated.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to draw
            random numbers from. It is used by the minimizer to generate random
            fit parameter initial values.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time the
            maximization of the LLH-ratio function.

        Returns
        -------
        log_lambda_max : float
            The (maximum) value of the log-likelihood ratio (log_lambda)
            function for the best fit parameter values.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the global
            fit parameter values.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        if isinstance(self._minimizer.minimizer_impl, NR1dNsMinimizerImpl) or\
           isinstance(self._minimizer.minimizer_impl, NRNsScan2dMinimizerImpl):
            return self.maximize_with_1d_newton_rapson_minimizer(
                rss=rss,
                tl=tl)

        return super().maximize(
            rss=rss,
            tl=tl)


class SingleDatasetTCLLHRatio(
        TCLLHRatio,
        metaclass=abc.ABCMeta):
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
        self._tdm.calculate_source_data_fields(self._shg_mgr)

    @property
    def shg_mgr(self):
        """The SourceHypoGroupManager instance that defines the source
        hypothesis groups.
        """
        return self._shg_mgr

    @shg_mgr.setter
    def shg_mgr(self, mgr):
        if not isinstance(mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager! '
                f'Its current type is {classname(mgr)}.')
        self._shg_mgr = mgr

    @property
    def tdm(self):
        """The instance of TrialDataManager that holds the trial event data and
        additional data fields for this LLH ratio function.
        """
        return self._tdm

    @tdm.setter
    def tdm(self, mgr):
        if not isinstance(mgr, TrialDataManager):
            raise TypeError(
                'The tdm property must be an instance of TrialDataManager! '
                f'Its current type is {classname(mgr)}.')
        self._tdm = mgr

    def change_shg_mgr(self, shg_mgr):
        """Changes the source hypothesis group manager of this two-component LLH
        ratio function.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        self.shg_mgr = shg_mgr

        self._tdm.change_shg_mgr(
            shg_mgr=shg_mgr,
            pmm=self._pmm)


class ZeroSigH0SingleDatasetTCLLHRatio(
        SingleDatasetTCLLHRatio):
    """This class implements a two-component (TC) log-likelihood (LLH) ratio
    function for a single dataset assuming zero signal for the null-hypothesis.
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
            pdfratio,
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
        pdfratio : instance of PDFRatio
            The instance of PDFRatio. A PDFRatio instance might depend
            on none, one, or several fit parameters.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            shg_mgr=shg_mgr,
            tdm=tdm,
            mean_n_sig_0=0,
            **kwargs)

        self.pdfratio = pdfratio

        # Define cache variables for the evaluate method to store values needed
        # for a possible calculation of the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_fitparam_values = None
        self._cache_nsgrad_i = None

    @SingleDatasetTCLLHRatio.mean_n_sig_0.setter
    def mean_n_sig_0(self, v):
        SingleDatasetTCLLHRatio.mean_n_sig_0.fset(self, v)
        if self._mean_n_sig_0 != 0:
            raise ValueError(
                f'The {classname(self)} class is only valid for '
                f'mean_n_sig_0 = 0!')

    @property
    def pdfratio(self):
        """The instance of PDFRatio.
        """
        return self._pdfratio

    @pdfratio.setter
    def pdfratio(self, r):
        if not isinstance(r, PDFRatio):
            raise TypeError(
                'The pdfratio property must be an instance of PDFRatio! '
                f'Its current type is {classname(r)}.')
        self._pdfratio = r

    def calculate_log_lambda_and_grads(
            self,
            fitparam_values,
            N,
            ns_pidx,
            p_mask,
            Xi,
            dXi_dp):
        """Calculates the log(Lambda) value and its gradient for each global fit
        parameter. This calculation is source and detector independent.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (N_fitparams,)-shaped ndarray holding the current values of the
            global fit parameters.
            These numbers are used as cache key to validate the ``nsgrad_i``
            values for the given fit parameter values for a possible later
            calculation of the second derivative w.r.t. ns of the log-likelihood
            ratio function.
        N : int
            The total number of events.
        ns_pidx : int
            The index of the global floating parameter ns.
        p_mask : instance of numpy ndarray
            The (N_fitparam,)-shaped numpy ndarray of bool selecting all global
            fit parameters, except ns.
        Xi : instance of numpy ndarray
            The (n_selected_events,)-shaped 1D numpy ndarray holding the X value
            of each selected event.
        dXi_dp : instance of numpy ndarray
            The (n_selected_events, N_fitparams-1,)-shaped 2D ndarray holding
            the derivative value for each fit parameter p (i.e. except ns) of
            each event's X value.

        Returns
        -------
        log_lambda : float
            The value of the log-likelihood ratio function.
        grads : instance of numpy ndarray
            The (N_fitparams,)-shaped numpy ndarray holding the gradient value
            of log_lambda for each fit parameter.
        """
        tracing = CFG['debugging']['enable_tracing']

        ns = fitparam_values[ns_pidx]

        # Get the number of selected events.
        Nprime = len(Xi)

        if tracing:
            logger.debug(
                f'N={N:d}, Nprime={Nprime:d}')

        one_plus_alpha = ZeroSigH0SingleDatasetTCLLHRatio._one_plus_alpha

        alpha = one_plus_alpha - 1
        alpha_i = ns*Xi

        # Create a mask for events which have a stable non-diverging
        # log-function argument, and an inverted mask thereof.
        m_stable = alpha_i > alpha
        m_unstable = ~m_stable

        if tracing:
            logger.debug(
                '# of events doing Taylor expansion for (unstable events): '
                f'{np.count_nonzero(m_unstable):d}')

        # Allocate memory for the log_lambda_i values.
        log_lambda_i = np.empty_like(alpha_i, dtype=np.float64)

        # Calculate the log_lambda_i value for the numerical stable events.
        log_lambda_i[m_stable] = np.log1p(alpha_i[m_stable])
        # Calculate the log_lambda_i value for the numerical unstable events.
        tildealpha_i = (alpha_i[m_unstable] - alpha) / one_plus_alpha
        log_lambda_i[m_unstable] =\
            np.log1p(alpha) + tildealpha_i - 0.5 * tildealpha_i**2

        # Calculate the log_lambda value and account for pure background events.
        log_lambda = np.sum(log_lambda_i) + (N - Nprime)*np.log1p(-ns/N)

        # Calculate the gradient for each fit parameter.
        grads = np.empty((dXi_dp.shape[1]+1,), dtype=np.float64)

        # Pre-calculate value that is used twice for the gradients of the
        # numerical stable events.
        one_over_one_plus_alpha_i_stablemask = 1 / (1 + alpha_i[m_stable])

        # For ns.
        nsgrad_i = np.empty_like(alpha_i, dtype=np.float64)
        nsgrad_i[m_stable] =\
            Xi[m_stable] * one_over_one_plus_alpha_i_stablemask
        nsgrad_i[m_unstable] =\
            (1 - tildealpha_i) * Xi[m_unstable] / one_plus_alpha

        # Cache the nsgrad_i values for a possible later calculation of the
        # second derivative w.r.t. ns of the log-likelihood ratio function.
        # Note: We create a copy of the fitparam_values array here to make sure
        #       that the values don't get changed outside this method before the
        #       calculate_ns_grad2 method is called.
        self._cache_fitparam_values = fitparam_values.copy()
        self._cache_nsgrad_i = nsgrad_i

        # Calculate the first derivative w.r.t. ns.
        grads[ns_pidx] = np.sum(nsgrad_i) - (N - Nprime) / (N - ns)

        # Now for each other fit parameter.

        # For all numerical stable events.
        grads[p_mask] = np.sum(
            ns * one_over_one_plus_alpha_i_stablemask * dXi_dp[m_stable],
            axis=1)

        # For all numerical unstable events.
        grads[p_mask] += np.sum(
            ns * (1 - tildealpha_i) * dXi_dp[m_unstable] / one_plus_alpha,
            axis=1)

        return (log_lambda, grads)

    def calculate_ns_grad2(
            self,
            fitparam_values,
            ns_pidx,
            params_recarray=None,
            tl=None):
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
            The ndarray holding the current values of the global fit
            parameters.
        ns_pidx : int
            The parameter index of the global floating parameter ns.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        # Check if the cached nsgrad_i values match the given fitparam_values.
        if (
                (self._cache_fitparam_values is None) or
                (not np.all(np.isclose(
                    self._cache_fitparam_values,
                    fitparam_values)))
           ):
            # Calculate the cache values by evaluating the log-likelihood ratio
            # function.
            self.evaluate(
                fitparam_values=fitparam_values,
                params_recarray=params_recarray,
                tl=tl)

        ns = fitparam_values[ns_pidx]
        Nprime = self._tdm.n_selected_events
        N = Nprime + self._tdm.n_pure_bkg_events

        nsgrad2 = -np.sum(self._cache_nsgrad_i**2) - (N - Nprime)/(N - ns)**2

        return nsgrad2

    def evaluate(
            self,
            fitparam_values,
            params_recarray=None,
            tl=None):
        """Evaluates the log-likelihood ratio function for the given set of
        data events.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D ndarray holding the current values of
            the global fit parameters.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure the timing of
            evaluating the LLH ratio function.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value.
        grads : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D numpy ndarray holding the gradient
            value for each global fit parameter.
        """
        tracing = CFG['debugging']['enable_tracing']

        tdm = self._tdm

        ns_pidx = self._pmm.get_gflp_idx('ns')

        ns = fitparam_values[ns_pidx]

        N = tdm.n_events

        # Calculate the data fields that depend on global fit parameters.
        if tdm.has_global_fitparam_data_fields:
            with TaskTimer(
                    tl,
                    'Calculate global fit parameter dependent data fields.'):
                # Create the global_fitparams dictionary with the global fit
                # parameter names and values.
                global_fitparams = self._pmm.get_global_floating_params_dict(
                    gflp_values=fitparam_values)
                tdm.calculate_global_fitparam_data_fields(
                    shg_mgr=self._shg_mgr,
                    pmm=self._pmm,
                    global_fitparams=global_fitparams)

        # Create the source paramater record array if not already provided.
        if params_recarray is None:
            params_recarray = self._pmm.create_src_params_recarray(
                fitparam_values)

        # Calculate the PDF ratio values for each selected event.
        with TaskTimer(tl, 'Calc pdfratio value Ri'):
            Ri = self._pdfratio.get_ratio(
                tdm=tdm,
                params_recarray=params_recarray,
                tl=tl)

        # Calculate Xi for each selected event.
        Xi = (Ri - 1.) / N

        n_fitparams = len(fitparam_values)

        # Calculate the gradients of Xi for each fit parameter (without ns).
        dXi_dp = np.empty(
            (len(Xi), n_fitparams-1),
            dtype=np.float64)

        # Create a mask that selects all fit parameters except ns.
        p_mask = np.ones((n_fitparams,), dtype=np.bool_)
        p_mask[ns_pidx] = False

        # Loop over the global fit parameters and calculate the derivative of
        # Xi w.r.t. each fit paramater.
        fitparam_ids = np.arange(n_fitparams)
        for (idx, (fitparam_id, fitparam_value)) in enumerate(
                zip(fitparam_ids[p_mask], fitparam_values[p_mask])):

            dRi = self._pdfratio.get_gradient(
                tdm=tdm,
                params_recarray=params_recarray,
                fitparam_id=fitparam_id,
                tl=tl)

            # Calculate the derivative of Xi w.r.t. the global fit parameter
            # with ID fitparam_id.
            dXi_dp[:, idx] = dRi / N

        if tracing:
            logger.debug(
                f'{classname(self)}.evaluate: N={N}, Nprime={len(Xi)}, '
                f'ns={ns:.3f}')

        with TaskTimer(tl, 'Calc logLamds and grads'):
            (log_lambda, grads) = self.calculate_log_lambda_and_grads(
                fitparam_values=fitparam_values,
                N=N,
                ns_pidx=ns_pidx,
                p_mask=p_mask,
                Xi=Xi,
                dXi_dp=dXi_dp)

        return (log_lambda, grads)


class MultiDatasetTCLLHRatio(
        TCLLHRatio):
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
    def __init__(
            self,
            pmm,
            minimizer,
            ds_sig_weight_factors,
            llhratio_list,
            **kwargs):
        """Creates a new composite two-component log-likelihood ratio function.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global floating parameters to individual models.
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of this log-likelihood ratio function.
        ds_sig_weight_factors : instance of DatasetSignalWeightFactors
            An instance of DatasetSignalWeightFactors, which calculates the
            relative dataset signal weight factors.
        llhratio_list : list of instance of SingleDatasetTCLLHRatio
            The list of the two-component log-likelihood ratio functions,
            one for each dataset.
        """
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            mean_n_sig_0=self._llhratio_list[0].mean_n_sig_0,
            **kwargs)

        self.ds_sig_weight_factors = ds_sig_weight_factors
        self.llhratio_list = llhratio_list

        if self.ds_sig_weight_factors.n_datasets != len(self._llhratio_list):
            raise ValueError(
                'The number of datasets the DatasetSignalWeightFactors '
                'instance is made for must be equal to the number of '
                'log-likelihood ratio functions!')

        # Define cache variable for the dataset signal weight factors, which
        # will be needed when calculating the second derivative w.r.t. ns of the
        # log-likelihood ratio function.
        self._cache_fitparam_values_ns = None
        self._cache_f = None

    @property
    def ds_sig_weight_factors(self):
        """The instance of DatasetSignalWeightFactors that provides the
        relative dataset signal weight factors.
        """
        return self._ds_sig_weight_factors

    @ds_sig_weight_factors.setter
    def ds_sig_weight_factors(self, obj):
        if not isinstance(obj, DatasetSignalWeightFactors):
            raise TypeError(
                'The ds_sig_weight_factors property must be an '
                'instance of DatasetSignalWeightFactors! '
                f'Its current type is {classname(obj)}.')
        self._ds_sig_weight_factors = obj

    @property
    def llhratio_list(self):
        """(read-only) The list of TCLLHRatio instances, which are part of this
        composite log-likelihood-ratio function.
        """
        return self._llhratio_list

    @llhratio_list.setter
    def llhratio_list(self, llhratios):
        if not issequenceof(llhratios, SingleDatasetTCLLHRatio):
            raise TypeError(
                'The llhratio_list property must be a sequence of '
                'SingleDatasetTCLLHRatio instances! '
                f'Its current type is {classname(llhratios)}.')
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

    def change_shg_mgr(self, shg_mgr):
        """Changes the source hypo group manager of all objects of this LLH
        ratio function, hence, calling the ``change_shg_mgr``
        method of all TCLLHRatio instances of this LLHRatio instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the groups of
            source hypotheses.
        """
        # Change the source hypo group manager of the DatasetSignalWeights
        # instance.
        self._dataset_signal_weights.change_shg_mgr(shg_mgr=shg_mgr)

        for llhratio in self._llhratio_list:
            llhratio.change_shg_mgr(shg_mgr=shg_mgr)

    def initialize_for_new_trial(self, tl=None):
        """Initializes the log-likelihood-ratio function for a new trial.
        """
        for llhratio in self._llhratio_list:
            llhratio.initialize_for_new_trial(tl=tl)

    def evaluate(
            self,
            fitparam_values,
            params_recarray=None,
            tl=None):
        """Evaluates the composite log-likelihood-ratio function and returns its
        value and global fit parameter gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams,)-shaped numpy 1D ndarray
            The (N_fitparams,)-shaped numpy 1D ndarray holding the current
            values of the global fit parameters.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value of the composite
            log-likelihood-ratio function.
        grads : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D ndarray holding the gradient value of
            the composite log-likelihood-ratio function for each global fit
            parameter.
        """
        tracing = CFG['debugging']['enable_tracing']

        n_fitparams = len(fitparam_values)

        ns_pidx = self._pmm.get_gflp_idx('ns')

        ns = fitparam_values[ns_pidx]
        if tracing:
            logger.debug(
                f'{classname(self)}.evaluate: ns={ns:.3f}')

        # Get the dataset signal weights and their gradients.
        # f is a (N_datasets,)-shaped 1D ndarray.
        # f_grads is a dictionary holding (N_datasets,)-shaped 1D ndarrays for
        # each global fit parameter.
        (f, f_grads_dict) = self._ds_sig_weight_factors(fitparam_values)

        # Convert the f_grads dictionary into a (N_datasets,N_fitparams)
        f_grads = np.zeros((len(f), n_fitparams), dtype=np.float64)
        for pidx in f_grads_dict.keys():
            f_grads[pidx] = f_grads_dict[pidx]

        # Cache f for possible later calculation of the second derivative w.r.t.
        # ns of the log-likelihood ratio function.
        self._cache_fitparam_values = fitparam_values.copy()
        self._cache_f = f

        nsf = ns * f

        # Calculate the composite log-likelihood-ratio function value and the
        # gradient of the composite log-likelihood ratio function for each
        # global fit parameter.
        log_lambda = 0

        # Allocate an array for the gradients of the composite log-likelihood
        # function. It is always at least one element long, i.e. the gradient
        # for ns.
        grads = np.zeros((n_fitparams,), dtype=np.float64)

        # Create an array holding the fit parameter values for a particular
        # llh ratio function. Since we need to adjust ns with nsj it's more
        # efficient to create this array once and use it within the for loop
        # over the llh ratio functions.
        llhratio_fitparam_values = fitparam_values.copy()

        # Loop over the llh ratio functions.
        for (j, llhratio) in enumerate(self._llhratio_list):
            if tracing:
                logger.debug(
                    f'nsf[j={j}] = {nsf[j]:.3f}')

            llhratio_fitparam_values[ns_pidx] = nsf[j]

            (log_lambda_j, grads_j) = llhratio.evaluate(
                fitparam_values=llhratio_fitparam_values,
                params_recarray=params_recarray,
                tl=tl)
            log_lambda += log_lambda_j

            # Gradient for ns.
            grads[ns_pidx] += grads_j[ns_pidx] * f[j]

            # Gradient for each global fit parameter, if there are any.
            if len(grads) > 1:
                ns_summand = grads_j[ns_pidx] * ns * f_grads[j]
                grads[:ns_pidx] += ns_summand + grads_j[:ns_pidx]
                grads[ns_pidx+1:] += ns_summand + grads_j[ns_pidx+1:]

        return (log_lambda, grads)

    def calculate_ns_grad2(
            self,
            fitparam_values,
            ns_pidx,
            params_recarray=None,
            tl=None):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.
        This method tries to use cached values for the dataset signal weight
        factors. If cached values don't exist or do not match the given fit
        parameter values, they will get calculated automatically by calling the
        :meth:`skyllh.core.llhratio.MultiDatasetTCLLHRatio.evaluate` method with
        the given fit parameter values.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (N_fitparams,)-shaped 1D ndarray holding the current values of
            the fit parameters.

        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        ns = fitparam_values[ns_pidx]

        # Check if the cached fit parameters match the given ones. The ns value
        # is special to the multi-dataset LLH ratio function, but all the other
        # fit parameters are shared by all the LLH ratio functions of the
        # different datasets. So those we just query from the first LLH ratio
        # function.
        if (
            (self._cache_fitparam_values is None) or
            (self._cache_f is None) or
            (not np.all(np.isclose(
                self._cache_fitparam_values[:ns_pidx],
                fitparam_values[:ns_pidx]))) or
            (not np.all(np.isclose(
                self._cache_fitparam_values[ns_pidx+1:],
                fitparam_values[ns_pidx+1:])))
        ):
            self._cache_fitparam_values = fitparam_values.copy()
            (self._cache_f, f_grads_dict) = self._ds_sig_weight_factors(
                fitparam_values=fitparam_values)

        nsf = ns * self._cache_f

        nsgrad2j = np.empty((len(self._llhratio_list),), dtype=np.float64)

        # Loop over the llh ratio functions and calculate their second
        # derivative.
        llhratio_fitparam_values = fitparam_values.copy()

        for (j, llhratio) in enumerate(self._llhratio_list):
            llhratio_fitparam_values[ns_pidx] = nsf[j]

            nsgrad2j[j] = llhratio.calculate_ns_grad2(
                fitparam_values=llhratio_fitparam_values,
                ns_pidx=ns_pidx,
                params_recarray=params_recarray,
                tl=tl)

        nsgrad2 = np.sum(nsgrad2j * self._cache_f**2)

        return nsgrad2


class NsProfileMultiDatasetTCLLHRatio(
        TCLLHRatio):
    r"""This class implements a profile log-likelihood ratio function that has
    only ns as fit parameter. It uses a MultiDatasetTCLLHRatio instance as
    log-likelihood function. Hence, mathematically it is

    .. math::

        \Lambda(n_{\mathrm{s}}) = \frac{L(n_{\mathrm{s}})}{L(n_{\mathrm{s}}=n_{\mathrm{s},0})},

    where :math:`n_{\mathrm{s},0}` is the fixed mean number of signal events for
    the null-hypothesis.
    """
    def __init__(
            self,
            pmm,
            minimizer,
            mean_n_sig_0,
            llhratio,
            **kwargs):
        r"""Creates a new ns-profile log-likelihood-ratio function with a
        null-hypothesis where :math:`n_{\mathrm{s}}` is fixed to
        ``mean_n_sig_0``.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global parameters to local parameters of individual models.
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
        super().__init__(
            pmm=pmm,
            minimizer=minimizer,
            mean_n_sig_0=mean_n_sig_0,
            **kwargs)

        self.llhratio = llhratio

        if pmm.n_global_floating_params != 1:
            raise ValueError(
                'The log-likelihood-ratio function implemented by '
                f'{classname(self)} provides functionality only for LLH '
                'function with a single global fit parameter! Currently there '
                f'are {pmm.n_global_floating_params} global fit parameters '
                'defined!')

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
        if not isinstance(obj, MultiDatasetTCLLHRatio):
            raise TypeError(
                'The llhratio property must be an instance of '
                'MultiDatasetTCLLHRatio! '
                f'Its current type is {classname(obj)}.')
        self._llhratio = obj

    def change_shg_mgr(
            self,
            shg_mgr):
        """Changes the source hypo group manager of all objects of this LLH
        ratio function, hence, calling the ``change_shg_mgr``
        method of the underlaying MultiDatasetTCLLHRatio instance of this
        LLHRatio instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        self._llhratio.change_shg_mgr(shg_mgr=shg_mgr)

    def initialize_for_new_trial(
            self,
            tl=None):
        """Initializes the log-likelihood-ratio function for a new trial.

        Parameters
        ----------
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.
        """
        self._llhratio.initialize_for_new_trial(tl=tl)

        # Compute the constant log-likelihood function value for the
        # null-hypothesis.
        fitparam_values_0 = np.array([self._mean_n_sig_0], dtype=np.float64)
        (self._logL_0, grads_0) = self._llhratio.evaluate(
            fitparam_values=fitparam_values_0,
            tl=tl)

    def evaluate(
            self,
            fitparam_values,
            params_recarray=None,
            tl=None):
        """Evaluates the log-likelihood-ratio function and returns its value and
        global fit parameter gradients.

        Parameters
        ----------
        fitparam_values : instance of numpy ndarray
            The (1,)-shaped numpy 1D ndarray holding the current
            values of the global fit parameters.
            By definition of this LLH ratio function, it must contain the single
            fit paramater value for ns.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.

        Returns
        -------
        log_lambda : float
            The calculated log-lambda value of this log-likelihood-ratio
            function.
        grads : (1,)-shaped 1D ndarray
            The ndarray holding the gradient value of this log-likelihood-ratio
            for ns.
        """
        (logL, grads) = self._llhratio.evaluate(
            fitparam_values=fitparam_values,
            params_recarray=params_recarray,
            tl=tl)

        log_lambda = logL - self._logL_0

        return (log_lambda, grads)

    def calculate_ns_grad2(
            self,
            fitparam_values,
            ns_pidx=0,
            params_recarray=None,
            tl=None):
        """Calculates the second derivative w.r.t. ns of the log-likelihood
        ratio function.

        Parameters
        ----------
        fitparam_values : numpy  ndarray
            The (1,)-shaped 1D numpy ndarray holding the current
            value of the global fit parameter ns.
        ns_pidx : int
            The index of the global fit paramater ns. By definition this must
            be ``0``.
        params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values of all sources. If set to ``None`` it will be
            created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.

        Returns
        -------
        nsgrad2 : float
            The second derivative w.r.t. ns of the log-likelihood ratio function
            for the given fit parameter values.
        """
        if ns_pidx != 0:
            raise ValueError(
                'The value of the ns_pidx argument must be 0! '
                f'Its current value is {ns_pidx}.')

        nsgrad2 = self._llhratio.calculate_ns_grad2(
            fitparam_values=fitparam_values,
            ns_pidx=ns_pidx,
            params_recarray=params_recarray,
            tl=tl)

        return nsgrad2
