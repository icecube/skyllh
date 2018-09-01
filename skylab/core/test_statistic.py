# -*- coding: utf-8 -*-

"""The test_statistic module provides the classes for different test statistic
functions.
"""

import abc


class TestStatistic(object):
    """This is the abstract base class for a test statistic class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Constructs the test-statistic function instance.
        """
        super(TestStatistic, self).__init__()

    @abc.abstractmethod
    def evaluate(self, llhratio, log_lambda, fitparam_dict, *args, **kwargs):
        """This method is supposed to evaluate the test-statistic function.

        Parameters
        ----------
        llhratio : LLHRatio instance
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_dict : dict
            The dictionary holding the fit parameter names and their values of
            the log-likelihood ratio function for the given log_lambda value.

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

    def evaluate(self, llhratio, log_lambda, fitparam_dict):
        """Evaluates this test-statistic function.

        Parameters
        ----------
        llhratio : LLHRatio instance
            The log-likelihood ratio function, which should be used for the
            test-statistic function.
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_dict : dict
            The dictionary holding the fit parameter names and their values of
            the log-likelihood ratio function for the given log_lambda value.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        ns = fitparam_dict['ns']

        # We need to distinguish between ns=0 and ns!=0, because the np.sign(ns)
        # function returns 0 for ns=0, but we want it to be 1 in such cases.
        sgn_ns = np.where(ns == 0, 1., np.sign(ns))

        TS = 2 * sgn_ns * log_lambda

        return TS
