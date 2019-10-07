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
    def minimize(self, initials, bounds, func, args=None,
                 func_provides_grads=True, kwargs=None):
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

            The return value of ``func`` must be (f, grads), the function value
            at the function arguments ``x`` and the 1D ndarray with the values
            of the function gradient for each fit parameter.
        args : sequence | None
            Optional sequence of arguments for ``func``.
        kwargs : dict | None
            Optional additional keyword arguments for the underlaying
            minimization process. Those are minimizer implementation dependent.

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
        pass

    @abc.abstractmethod
    def has_converged(self, status):
        """This method is supposed to analyze the status information dictionary
        if the last minimization process has convered.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        convered : bool
            The flag if the minimization has convered (True), or not (False).
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

    def minimize(self, initials, bounds, func, args=None,
                 func_provides_grads=True, kwargs=None):
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
            the function gradient for each fit parameter, if
            ``func_provides_grads`` is set to True. If set to False, it must
            return only the function value.
        args : sequence | None
            Optional sequence of arguments for ``func``.
        func_provides_grads : bool
            Flag if function ``func`` provides its gradients.
            Default is True.
        kwargs : dict | None
            Optional additional keyword arguments for the underlaying
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
        if(args is None):
            args = tuple()
        if(kwargs is None):
            kwargs = dict()

        (xmin, fmin, status) = self._fmin_l_bfgs_b(
            func, initials,
            bounds = bounds,
            args = args,
            approx_grad = not func_provides_grads,
            **kwargs
        )

        return (xmin, fmin, status)

    def has_converged(self, status):
        """Analyzes the status information dictionary if the last minimization
        process has convered. By definition the minimization process has
        convered if ``status['warnflag']`` equals 0.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        convered : bool
            The flag if the minimization has convered (True), or not (False).
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

    def minimize(
            self, rss, fitparamset, func, args=None, func_provides_grads=True,
            kwargs=None):
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

            The return value of ``func`` must be (f, grads), the function value
            at the function arguments ``x`` and the 1D ndarray with the values
            of the function gradient for each fit parameter, if
            ``func_provides_grads`` is set to True. If set to False, it must
            return only the function value.
        args : sequence of arguments for ``func`` | None
            The optional sequence of arguments for ``func``.
        func_provides_grads : bool
            Flag if function ``func`` provides its gradients.
            Default is True.
        kwargs : dict | None
            The optional dictionary with keyword arguments for the minimizer
            implementation minimize method.

        Other Parameters
        ----------------
        Additional keyword arguments will be passed to the actual used
        minimizer method.
        """
        if(not isinstance(fitparamset, FitParameterSet)):
            raise TypeError('The fitparamset argument must be an instance of '
                'FitParameterSet!')

        bounds = fitparamset.bounds

        (xmin, fmin, status) = self._minimizer_impl.minimize(
            fitparamset.initials, bounds, func, args, func_provides_grads,
            kwargs=kwargs)

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
                initials, bounds, func, args, func_provides_grads,
                kwargs=kwargs)

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
