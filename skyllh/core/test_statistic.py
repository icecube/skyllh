# -*- coding: utf-8 -*-

"""The test_statistic module provides the classes for different test statistic
functions.
"""

import abc
import numpy as np


class TestStatistic(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for a test statistic class.
    """

    def __init__(self):
        """Constructs the test-statistic function instance.
        """
        super(TestStatistic, self).__init__()

    @abc.abstractmethod
    def evaluate(self, llhratio, log_lambda, fitparam_values, *args, **kwargs):
        """This method is supposed to evaluate the test-statistic function.

        Parameters
        ----------
        llhratio : LLHRatio instance
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : (N_fitparams+1)-shaped 1D numpy ndarray
            The numpy ndarray holding the fit parameter values of the
            log-likelihood ratio function for the given log_lambda value.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        pass


class TestStatisticWilks(TestStatistic):
    """This class implements the standard Wilks theorem test-statistic function:

        TS = 2 * sign(ns_best) * log( L(fitparam_best) / L(ns = 0) )

    where the sign(ns_best) is negative for ns_best < 0, and positive otherwise.
    """
    def __init__(self):
        """Constructs the test-statistic function instance.
        """
        super(TestStatisticWilks, self).__init__()

    def evaluate(self, llhratio, log_lambda, fitparam_values):
        """Evaluates this test-statistic function.

        Parameters
        ----------
        llhratio : LLHRatio instance
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : (N_fitparams+1)-shaped 1D numpy ndarray
            The numpy ndarray holding the fit parameter values of the
            log-likelihood ratio function for the given log_lambda value.
            By definition, the first element is the value of 'ns'.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        ns = fitparam_values[0]

        # We need to distinguish between ns=0 and ns!=0, because the np.sign(ns)
        # function returns 0 for ns=0, but we want it to be 1 in such cases.
        sgn_ns = np.where(ns == 0, 1., np.sign(ns))

        TS = 2 * sgn_ns * log_lambda

        return TS


class TestStatisticWilksZeroNsTaylor(TestStatistic):
    """Similar to the TestStatisticWilks class, this class implements the
    standard Wilks theorem test-statistic function. But for zero ns values, the
    log-likelihood ratio function is taylored up to second order and the
    resulting apex is used as log_lambda value. Hence, the TS function is
    defined as:

        TS = 2 * sign(ns_best) * log( L(fitparam_best) / L(ns = 0) )

    for ns_best != 0, and

        TS = 2 * a^2 / (4*b)

    for ns_best == 0, with

        a = d/dns ( L(fitparam_best) / L(ns = 0) )

    being the derivative w.r.t. ns of the log-likelihood ratio function, and

        b = d/dns ( a )

    being its second derivative w.r.t. ns.
    """
    def __init__(self):
        """Constructs the test-statistic function instance.
        """
        super(TestStatisticWilksZeroNsTaylor, self).__init__()

    def evaluate(self, llhratio, log_lambda, fitparam_values, grads):
        """Evaluates this test-statistic function.

        Parameters
        ----------
        llhratio : LLHRatio instance
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : (N_fitparams+1)-shaped 1D numpy ndarray
            The numpy ndarray holding the fit parameter values of the
            log-likelihood ratio function for the given log_lambda value.
            By definition, the first element is the value of 'ns'.
        grads : (N_fitparams+1)-shaped 1D ndarray
            The ndarray holding the values of the first derivative of the
            log-likelihood ratio function w.r.t. each global fit parameter.
            By definition the first element is the first derivative
            w.r.t. the fit parameter ns.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        ns = fitparam_values[0]

        if(ns == 0):
            nsgrad = grads[0]
            nsgrad2 = llhratio.calculate_ns_grad2(fitparam_values)

            TS = -2 * nsgrad**2 / (4*nsgrad2)
        else:
            TS = 2 * np.sign(ns) * log_lambda

        return TS
