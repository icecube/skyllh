"""The minimizer module provides functionality for the minimization process of
a function.
"""
import abc
import numpy as np

import scipy.optimize

from skyllh.core.parameters import FitParameterSet

class MinimizerImpl(object):
    """Abstract base class for a minimizer implementation. It defines the
    interface between the implementation and the Minimizer class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(MinimizerImpl, self).__init__()

    @abc.abstractmethod
    def minimize(self, initials, bounds, func, func_args=None, **kwargs):
        """This method is supposed to minimize the given function with the given
        initials.

        Parameters
        ----------
        initials : 1D (N_fitparams)-shaped numpy ndarray
            The ndarray holding the initial values of all the fit parameters.
        bounds : 2D (N_fitparams,2)-shaped numpy ndarray
            The ndarray holding the boundary values (vmin, vmax) of the fit
            parameters.
        func : callable
            The function that should get minimized.
            The call signature must be

                ``__call__(x, *args)``

            The return value of ``func`` is minimizer implementation dependent.
        func_args : sequence | None
            Optional sequence of arguments for ``func``.

        Additional Keyword Arguments
        ----------------------------
        Additional keyword arguments include options for this minimizer
        implementation. These are implementation dependent.

        Returns
        -------
        xmin : 1D ndarray
            The array containing the function parameter values at the function's
            minimum.
        fmin : float
            The function value at its minimum.
        status : dict
            The status dictionary with information about the minimization
            process.
        """
        pass

    @abc.abstractmethod
    def has_converged(self, status):
        """This method is supposed to analyze the status information dictionary
        if the last minimization process has converged.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        converged : bool
            The flag if the minimization has converged (True), or not (False).
        """
        pass

    @abc.abstractmethod
    def is_repeatable(self, status):
        """This method is supposed to analyze the status information dictionary
        if the last minimization process can be repeated to obtain a better
        minimum.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        repeatable : bool
            The flag if the minimization process can be repeated to obtain a
            better minimum.
        """
        pass


class LBFGSMinimizerImpl(MinimizerImpl):
    """The LBFGSMinimizerImpl class provides the minimizer implementation for
    L-BFG-S minimizer used from the :mod:`scipy.optimize` module.
    """
    def __init__(self):
        """Creates a new L-BGF-S minimizer instance to minimize the given
        likelihood function with its given partial derivatives.
        """
        super(LBFGSMinimizerImpl, self).__init__()

        self._fmin_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b

    def minimize(self, initials, bounds, func, func_args=None, **kwargs):
        """Minimizes the given function ``func`` with the given initial function
        argument values ``initials``.

        Parameters
        ----------
        initials : 1D numpy ndarray
            The ndarray holding the initial values of all the fit parameters.
        bounds : 2D (N_fitparams,2)-shaped numpy ndarray
            The ndarray holding the boundary values (vmin, vmax) of the fit
            parameters.
        func : callable
            The function that should get minimized.
            The call signature must be

                ``__call__(x, *args)``

            The return value of ``func`` must be (f, grads), the function value
            at the function arguments ``x`` and the ndarray with the values of
            the function gradient for each fit parameter, if the
            ``func_provides_grads`` keyword argument option is set to True.
            If set to False, ``func`` must return only the function value.
        func_args : sequence | None
            Optional sequence of arguments for ``func``.

        Additional Keyword Arguments
        ----------------------------
        Additional keyword arguments include options for this minimizer
        implementation. Possible options are:

            func_provides_grads : bool
                Flag if the function ``func`` also returns its gradients.
                Default is ``True``.

        Any additional keyword arguments are passed on to the underlaying
        :func:`scipy.optimize.fmin_l_bfgs_b` minimization function.

        Returns
        -------
        xmin : 1D ndarray
            The array containing the function arguments at the function's
            minimum.
        fmin : float
            The function value at its minimum.
        status : dict
            The status dictionary with information about the minimization
            process.
        """
        if(func_args is None):
            func_args = tuple()

        func_provides_grads = kwargs.pop('func_provides_grads', True)

        (xmin, fmin, status) = self._fmin_l_bfgs_b(
            func, initials,
            bounds = bounds,
            args = func_args,
            approx_grad = not func_provides_grads,
            **kwargs
        )

        return (xmin, fmin, status)

    def has_converged(self, status):
        """Analyzes the status information dictionary if the minimization
        process has converged. By definition the minimization process has
        converged if ``status['warnflag']`` equals 0.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the minimization
            process.

        Returns
        -------
        converged : bool
            The flag if the minimization has converged (True), or not (False).
        """
        if(status['warnflag'] == 0):
            return True
        return False

    def is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result. It's repeatable if

            `status['warnflag'] == 2 and 'FACTR' in str(status['task'])`

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        repeatable : bool
            The flag if the minimization process can be repeated to obtain a
            better minimum.
        """
        if(status['warnflag'] == 2 and 'FACTR' in str(status['task'])):
            return True
        return False


class NR1dNsMinimizerImpl(MinimizerImpl):
    """The NR1dNsMinimizerImpl class provides a minimizer implementation for the
    Newton-Raphson method for finding the minimum of a one-dimensional R1->R1
    function, i.e. a function that depends solely on one parameter, the number of
    signal events ns.
    """
    def __init__(self, ns_tol=1e-4):
        """Creates a new NRNs minimizer instance to minimize the given
        likelihood function with its given partial derivatives.

        Parameters
        ----------
        ns_tol : float
            The tolerance / precision for the ns parameter value.
        """
        super(NR1dNsMinimizerImpl, self).__init__()

        self.ns_tol = ns_tol

    def minimize(self, initials, bounds, func, func_args=None, **kwargs):
        """Minimizes the given function ``func`` with the given initial function
        argument values ``initials``.

        Parameters
        ----------
        initials : 1D numpy ndarray
            The ndarray holding the initial values of all the fit parameters.
            By definition of this 1D minimizer, this array must be of length 1.
        bounds : 2D (N_fitparams,2)-shaped numpy ndarray
            The ndarray holding the boundary values (vmin, vmax) of the fit
            parameters.
        func : callable
            The function that should get minimized.
            The call signature must be

                ``__call__(x, *args)``

            The return value of ``func`` must be (f, grad, grad2), i.e. the
            function value at the function arguments ``x``, the value of the
            function first derivative for the one fit parameter, and the value
            of the second derivative for the one fit parameter.
        func_args : sequence | None
            Optional sequence of arguments for ``func``.

        Additional Keyword Arguments
        ----------------------------
        There are no additional options defined for this minimization
        implementation.

        Returns
        -------
        xmin : 1D ndarray
            The array containing the function parameter values at the function's
            minimum.
        fmin : float
            The function value at its minimum.
        status : dict
            The status dictionary with information about the minimization
            process. The following information are provided:

            niter : int
                The number of iterations needed to find the minimum.
            last_nr_step : float
                The Newton-Raphson step size of the last iteration.
            warnflag : int
                The warning flag indicating if the minimization did converge.
                The possible values are:

                    0: The minimization converged with a iteration step size
                       smaller than the specified precision.
                    1: The function minimum is below the minimum bound of the
                       parameter value. The last iteration's step size did not
                       achieve the specified precision.
                    2: The function minimum is above the maximum bound of the
                       parameter value. The last iteration's step size did not
                       achieve the specified precision.
            warnreason: str
                The description for the set warn flag.

        """
        if(func_args is None):
            func_args = tuple()

        (ns_min, ns_max) = bounds[0]
        if(ns_min > initials[0]):
            raise ValueError('The initial value for ns (%g) must be equal or '
                'greater than the minimum bound value for ns (%g)'%(
                    initials[0], ns_min))

        ns_tol = self.ns_tol

        niter = 0
        step = ns_tol + 1
        ns = initials[0]
        status = {'warnflag': 0, 'warnreason': ''}
        f = None
        fprime = 0
        x = np.empty((1,), dtype=np.float)
        # We do the minimization process while the precision of ns is not
        # reached yet or the function is still rising or falling fast, i.e. the
        # minimum is in a deep well.
        while (ns_tol < np.fabs(step)) or (np.fabs(fprime) > 1):
            niter += 1
            x[0] = ns
            (f, fprime, fprimeprime) = func(x, *func_args)
            if(ns == ns_min and fprime >= 0):
                # We found the function minimum to be below the minimum bound of
                # the parameter value, but the function is rising. This can be
                # considered as converged.
                break

            step = -fprime / fprimeprime
            ns += step

            if(ns < ns_min):
                # The function minimum is below the minimum bound of the
                # parameter value.
                ns = ns_min
                if((ns_tol < np.fabs(step)) or (np.fabs(fprime) > 1)):
                    status['warnflag'] = 1
                    status['warnreason'] = 'Function minimum is below the '\
                                           'minimum bound of the parameter '\
                                           'value.'
                break
            if(ns > ns_max):
                # The function minimum is above the maximum bound of the
                # parameter value.
                ns = ns_max
                if((ns_tol < np.fabs(step)) or (np.fabs(fprime) > 1)):
                    status['warnflag'] = 2
                    status['warnreason'] = 'Function minimum is above the '\
                                           'maximum bound of the parameter '\
                                           'value.'
                break

        status['niter'] = niter
        status['last_nr_step'] = step

        return (x, f, status)

    def has_converged(self, status):
        """Analyzes the status information dictionary if the minimization
        process has converged. By definition the minimization process has
        converged if ``status['warnflag']`` equals 0.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the minimization
            process.

        Returns
        -------
        converged : bool
            The flag if the minimization has converged (True), or not (False).
        """
        return not status['warnflag']

    def is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result. By definition of this minimization method, this method will
        always return ``False``.
        """
        return False


class Minimizer(object):
    """The Minimizer class provides the general interface for minimizing a
    function. The class takes an instance of MinimizerImpl for a specific
    minimizer implementation.
    """
    def __init__(self, minimizer_impl, max_repetitions=100):
        """Creates a new Minimizer instance.

        Parameters
        ----------
        minimizer_impl : instance of MinimizerImpl
            The minimizer implementation for a specific minimizer algorithm.
        max_repetitions : int
            In case the minimization process did not converge at the first time
            this option specifies the maximum number of repetitions with
            different initials.
        """
        self.minimizer_impl = minimizer_impl
        self.max_repetitions = max_repetitions

    @property
    def minimizer_impl(self):
        """The instance of MinimizerImpl, which provides the implementation of
        the minimizer.
        """
        return self._minimizer_impl
    @minimizer_impl.setter
    def minimizer_impl(self, impl):
        if(not isinstance(impl, MinimizerImpl)):
            raise TypeError('The minimizer_impl property must be an instance '
                'of MinimizerImpl!')
        self._minimizer_impl = impl

    @property
    def max_repetitions(self):
        """In case the minimization process did not converge at the first time
        this option specifies the maximum number of repetitions with
        different initials.
        """
        return self._max_repetitions
    @max_repetitions.setter
    def max_repetitions(self, n):
        if(not isinstance(n, int)):
            raise TypeError('The maximal repetitions property must be of type '
                'int!')
        self._max_repetitions = n

    def minimize(self, rss, fitparamset, func, args=None, kwargs=None):
        """Minimizes the the given function ``func`` by calling the ``minimize``
        method of the minimizer implementation.

        After the minimization process it calls the ``has_converged`` and
        ``is_repeatable`` methods of the minimizer implementation to determine
        if an additional minimization attempt has to be performed.
        This is repeated until the minimization process did converge or if the
        maximal number of repetitions has occurred.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance to draw random numbers from.
        fitparamset : instance of FitParameterSet
            The set of FitParameter instances defining fit parameters of the
            function ``func``.
        func : callable ``f(x, *args)``
            The function to be minimized. It must have the call signature

                ``__call__(x, *args)``

            The return value of ``func`` is minimizer implementation dependent.
        args : sequence of arguments for ``func`` | None
            The optional sequence of arguments for ``func``.
        kwargs : dict | None
            The optional dictionary with keyword arguments for the minimizer
            implementation minimize method.

        Returns
        -------
        xmin : 1d numpy ndarray
            The array holding the parameter values for which the function has
            a minimum.
        fmin : float
            The function value at its minimum.
        status : dict
            The status dictionary with information about the minimization
            process.
        """
        if(not isinstance(fitparamset, FitParameterSet)):
            raise TypeError('The fitparamset argument must be an instance of '
                'FitParameterSet!')

        bounds = fitparamset.bounds

        (xmin, fmin, status) = self._minimizer_impl.minimize(
            fitparamset.initials, bounds, func, args, kwargs=kwargs)

        reps = 0
        while((not self._minimizer_impl.has_converged(status)) and
              self._minimizer_impl.is_repeatable(status) and
              reps < self._max_repetitions
        ):
            # The minimizer did not converge at the first time, but it is
            # possible to repeat the minimization process with different
            # initials to obtain a better result.

            # Create a new set of random parameter initials based on the
            # parameter bounds.
            initials = fitparamset.generate_random_initials(rss)

            # Repeat the minimization process.
            (xmin, fmin, status) = self._minimizer_impl.minimize(
                initials, bounds, func, args, kwargs=kwargs)

            reps += 1

        # Check if any fit value is outside its bounds due to rounding errors by
        # the minimizer. If so, set those fit values to their respective bound
        # value and re-evaluate the function with the corrected fit values.
        condmin = xmin < bounds[:,0]
        condmax = xmin > bounds[:,1]
        if(np.any(condmin) or np.any(condmax)):
            xmin = np.where(condmin, bounds[:,0], xmin)
            xmin = np.where(condmax, bounds[:,1], xmin)
            if(args is None):
                args = tuple()
            (fmin, grads) = func(xmin, *args)

        return (xmin, fmin, status)
