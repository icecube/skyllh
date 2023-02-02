# -*- coding: utf-8 -*-

"""The test_statistic module provides the classes for different test statistic
functions.
"""

import abc
import numpy as np


class TestStatistic(
        object,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for a test statistic class.
    """

    def __init__(self, **kwargs):
        """Constructs the test-statistic function instance.
        """
        super().__init__(**kwargs)

    @abc.abstractmethod
    def __call__(
            self,
            pmm,
            log_lambda,
            fitparam_values,
            **kwargs):
        """This method is supposed to evaluate the test-statistic function.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance that defines the global
            parameter set.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the
            global fit parameter values of the log-likelihood ratio
            function for the given log_lambda value.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        pass


class WilksTestStatistic(
        TestStatistic):
    r"""This class implements the standard Wilks theorem test-statistic function:

    .. math::

        TS = 2 \text{sign}(\hat{n}_{\text{s}}) \log \left(
        \frac{\mathcal{L}(\hat{\vec{p}})}{\mathcal{L}(n_{\text{s}} = 0)} \right)

    where the :math:`\text{sign}(\hat{n}_{\text{s}})` is negative for
    :math:`\hat{n}_{\text{s}} < 0`, and positive otherwise.
    """
    def __init__(self, ns_param_name='ns', **kwargs):
        """Constructs the test-statistic function instance.

        Parameters
        ----------
        ns_param_name : str
            The name of the global fit parameter for the number of signal
            events in the detector, ns.
        """
        super().__init__(**kwargs)

        self._ns_param_name = ns_param_name

    @property
    def ns_param_name(self):
        """(read-only) The name of the global fit parameter for the number of
        signal events in the detector, ns.
        """
        return self._ns_param_name

    def __call__(
            self,
            pmm,
            log_lambda,
            fitparam_values,
            **kwargs):
        """Evaluates the test-statistic function.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance that defines the global
            parameter set.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the
            global fit parameter values of the log-likelihood ratio
            function for the given log_lambda value.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        ns_pidx = pmm.get_gflp_idx(
            name=self._ns_param_name)

        ns = fitparam_values[ns_pidx]

        # We need to distinguish between ns=0 and ns!=0, because the np.sign(ns)
        # function returns 0 for ns=0, but we want it to be 1 in such cases.
        sgn_ns = np.where(ns == 0, 1., np.sign(ns))

        TS = 2 * sgn_ns * log_lambda

        return TS


class LLHRatioZeroNsTaylorWilksTestStatistic(
        TestStatistic):
    r"""Similar to the TestStatisticWilks class, this class implements the
    standard Wilks theorem test-statistic function. But for zero ns values, the
    log-likelihood ratio function is taylored up to second order and the
    resulting apex is used as log_lambda value. Hence, the TS function is
    defined as:

    .. math::

        TS = 2 \text{sign}(\hat{n}_{\text{s}}) \log \left(
        \frac{\mathcal{L}(\hat{\vec{p}})}{\mathcal{L}(n_{\text{s}} = 0)} \right)

    for :math:`\hat{n}_{\text{s}} \neq 0`, and

    .. math::

        TS = -2 \frac{a^2}{4b}

    for :math:`\hat{n}_{\text{s}} = 0`, with

    .. math::

        a = \frac{\text{d}}{\text{d}n_{\text{s}}} \left(
        \frac{\mathcal{L}(\hat{\vec{p}})}{\mathcal{L}(n_{\text{s}} = 0)} \right)

    being the derivative w.r.t. :math:`n_{\text{s}}` of the log-likelihood ratio
    function, and

    .. math::

        b = \frac{\text{d}a}{\text{d}n_{\text{s}}}

    being its second derivative w.r.t. ns.
    """
    def __init__(self, ns_param_name='ns', **kwargs):
        """Constructs the test-statistic function instance.

        Parameters
        ----------
        ns_param_name : str
            The name of the global fit parameter for the number of signal
            events in the detector, ns.
        """
        super().__init__(**kwargs)

        self._ns_param_name = ns_param_name

    @property
    def ns_param_name(self):
        """(read-only) The name of the global fit parameter for the number of
        signal events in the detector, ns.
        """
        return self._ns_param_name

    def __call__(
            self,
            pmm,
            log_lambda,
            fitparam_values,
            llhratio,
            grads,
            tl=None,
            **kwargs):
        """Evaluates the test-statistic function.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance that defines the global
            parameter set.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the
            global fit parameter values of the log-likelihood ratio
            function for the given log_lambda value.
        llhratio : instance of LLHRatio
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        grads : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D numpy ndarray holding the
            values of the first derivative of the log-likelihood ratio function
            w.r.t. each global fit parameter.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure timing information.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        ns_pidx = pmm.get_gflp_idx(
            name=self._ns_param_name)

        ns = fitparam_values[ns_pidx]

        if ns == 0:
            nsgrad = grads[ns_pidx]
            nsgrad2 = llhratio.calculate_ns_grad2(
                fitparam_values=fitparam_values,
                ns_pidx=ns_pidx,
                tl=tl)

            TS = -2 * nsgrad**2 / (4*nsgrad2)

            return TS

        TS = 2 * np.sign(ns) * log_lambda

        return TS
