"""
The minimizer module provides functionality for the minimization process of
a function.
"""
import abc
import logging
import scipy.optimize

import numpy as np

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.parameters import (
    ParameterSet,
)
from skyllh.core.py import (
    classname,
)


logger = logging.getLogger(__name__)


class MinimizerImpl(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """Abstract base class for a minimizer implementation. It defines the
    interface between the implementation and the Minimizer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def minimize(
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
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
    def get_niter(self, status):
        """This method is supposed to return the number of iterations that were
        required to find the minimum.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the last
            minimization process.

        Returns
        -------
        niter : int
            The number of iterations needed to find the minimum.
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


class ScipyMinimizerImpl(
        MinimizerImpl
):
    """Wrapper for `scipy.optimize.minimize`"""

    def __init__(
            self,
            method: str,
            **kwargs,
    ) -> None:
        """Creates a new instance of ScipyMinimizerImpl.

        Parameters
        ----------
        method : str
            The minimizer method to use. See the documentation for the method
            argument of the :func:`scipy.optimize.minimize` function for
            possible values.
        """
        super().__init__(
            **kwargs)

        self._method = method

    def minimize(
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
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
        :func:`scipy.optimize.minimize` minimization function.

        Returns
        -------
        xmin : instance of numpy.ndarray
            The 1D array containing the function arguments at the function's
            minimum.
        fmin : float
            The function value at its minimum.
        res : instance of scipy.optimize.OptimizeResult
            The scipy OptimizeResult.
        """

        method_supports_bounds = False

        # constraints: List[Dict[str, Any]]
        constraints = None

        # Check if method allows for bounds
        if self._method in ["L-BFGS-B", "TNC", "SLSQP"]:
            method_supports_bounds = True
        elif self._method == "COBYLA":
            # COBYLA doesn't allow for bounds, but we can convert bounds
            # to a linear constraint

            constraints = []
            for (bound_num, bound) in enumerate(bounds):
                lower, upper = bound
                lc = {"type": "ineq",
                      "fun": lambda x, lb=lower, i=bound_num: x[i] - lb}
                uc = {"type": "ineq",
                      "fun": lambda x, ub=upper, i=bound_num: ub - x[i]}
                constraints.append(lc)
                constraints.append(uc)
            bounds = None

        if (bounds is not None) and (not method_supports_bounds):
            logger.warn(
                f'Selected minimization method "{self._method}" does not '
                'support bounds. Continue at your own risk!')
            bounds = None

        if func_args is None:
            func_args = tuple()
        if kwargs is None:
            kwargs = {}

        func_provides_grads = kwargs.pop('func_provides_grads', True)

        res = scipy.optimize.minimize(
            func,
            initials,
            bounds=bounds,
            constraints=constraints,
            args=func_args,
            jac=func_provides_grads,
            **kwargs)

        return (res.x, res.fun, res)

    def get_niter(self, status):
        """Returns the number of iterations needed to find the minimum.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the minimization
            process.

        Returns
        -------
        niter : int
            The number of iterations needed to find the minimum.
        """
        return status['nit']

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
        return bool(status["success"])

    def is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result.

        TODO: Method specific checks. For now just return False


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
        return False


class LBFGSMinimizerImpl(
        MinimizerImpl
):
    """The LBFGSMinimizerImpl class provides the minimizer implementation for
    L-BFG-S minimizer used from the :mod:`scipy.optimize` module.
    """

    def __init__(
            self,
            ftol=1e-6,
            pgtol=1e-5,
            maxls=100,
            iprint=-1,
            **kwargs,
    ):
        """Creates a new L-BGF-S minimizer instance to minimize the given
        likelihood function with its given partial derivatives.

        Parameters
        ----------
        ftol : float
            The function value tolerance.
        pgtol : float
            The gradient value tolerance.
        maxls : int
            The maximum number of line search steps for an iteration.
        iprint : int
            Controls the frequency of output. See scipy documentation for more
            information.
        """
        super().__init__(**kwargs)

        self._ftol = ftol
        self._pgtol = pgtol
        self._maxls = maxls
        self._iprint = iprint

        self._fmin_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b

    def minimize(
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
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


        Any additional keyword arguments are passed on to the underlying
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
            process. The following information are provided:

            niter : int
                The number of iterations needed to find the minimum.
            warnflag : int
                The warning flag indicating if the minimization did converge.
                The possible values are:

                    0: The minimization converged.
        """
        if func_args is None:
            func_args = tuple()
        if kwargs is None:
            kwargs = {}

        if 'factr' not in kwargs:
            kwargs['factr'] = self._ftol / np.finfo(float).eps
        if 'pgtol' not in kwargs:
            kwargs['pgtol'] = self._pgtol
        if 'maxls' not in kwargs:
            kwargs['maxls'] = self._maxls
        if 'iprint' not in kwargs:
            kwargs['iprint'] = self._iprint

        func_provides_grads = kwargs.pop('func_provides_grads', True)

        (xmin, fmin, status) = self._fmin_l_bfgs_b(
            func, initials,
            bounds=bounds,
            args=func_args,
            approx_grad=not func_provides_grads,
            **kwargs
        )

        return (xmin, fmin, status)

    def get_niter(
            self,
            status,
    ):
        """Returns the number of iterations needed to find the minimum.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the minimization
            process.

        Returns
        -------
        niter : int
            The number of iterations needed to find the minimum.
        """
        return status['nit']

    def has_converged(
            self,
            status,
    ):
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
        if status['warnflag'] == 0:
            return True
        return False

    def is_repeatable(
            self,
            status,
    ):
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
        if status['warnflag'] == 2:
            task = str(status['task'])
            if 'FACTR' in task:
                return True
            if 'ABNORMAL_TERMINATION_IN_LNSRCH' in task:
                # This is causes most probably by starting the minimization at
                # a parameter boundary.
                return True
        return False


class NR1dNsMinimizerImpl(
        MinimizerImpl
):
    """The NR1dNsMinimizerImpl class provides a minimizer implementation for the
    Newton-Raphson method for finding the minimum of a one-dimensional R1->R1
    function, i.e. a function that depends solely on one parameter, the number of
    signal events ns.
    """

    def __init__(
            self,
            ns_tol=1e-3,
            max_steps=100,
            **kwargs,
    ):
        """Creates a new NRNs minimizer instance to minimize the given
        likelihood function with its given partial derivatives.

        Parameters
        ----------
        ns_tol : float
            The tolerance / precision for the ns parameter value.
        max_steps : int
            The maximum number of NR steps. If max_step is reached,
            the fit is considered NOT converged.
        """
        super().__init__(**kwargs)

        self.ns_tol = ns_tol
        self.max_steps = max_steps

    def minimize(  # noqa: C901
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
        """Minimizes the given function ``func`` with the given initial function
        argument values ``initials``. This minimizer implementation will only
        vary the first parameter. All other parameters will be set to their
        initial value.

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

                    -1: The function minimum is above the upper bound of the
                        parameter value. Convergence forced at upper bound.
                    -2: The function minimum is below the lower bound of the
                        parameter value. Convergence forced at lower bound.
                    0: The minimization converged with a iteration step size
                       smaller than the specified precision.
                    1: The minimization did NOT converge within self.max_steps
                       number of steps

            warnreason: str
                The description for the set warn flag.

        """
        if func_args is None:
            func_args = tuple()

        (ns_min, ns_max) = bounds[0]
        if ns_min > initials[0]:
            raise ValueError(
                f'The initial value for ns ({initials[0]:g}) must be equal or '
                f'greater than the minimum bound value for ns ({ns_min:g})')

        ns_tol = self.ns_tol

        niter = 0
        x = np.copy(initials).astype(np.float64)
        ns = x[0]

        # Initialize stepsize to be larger than ns tolerance.
        # Also initialize first derivative to large value.
        # Want to perform at least one NR iteration.
        step = ns_tol + 1
        fprime = 1000
        # NR does not guarantee convergence, thus limit iterations.
        max_steps = self.max_steps
        status = {
            'warnflag': 0,
            'warnreason': '',
        }
        f = None
        at_boundary = False

        # We do the minimization process while the precision of ns is not
        # reached yet or the function is still rising or falling fast, i.e. the
        # minimum is in a deep well.
        # In case the optimum is found outside the bounds on ns the best fit
        # will be set to the boundary value and the fit considered converged.
        while ((ns_tol < np.fabs(step)) or (np.fabs(fprime) > 1.e-1)) and\
              (niter < max_steps):

            x[0] = ns
            (f, fprime, fprimeprime) = func(x, *func_args)
            step = -fprime / fprimeprime

            # Exit optimization if ns is at boundary but next step would be outside.
            if (ns == ns_min and step < 0.0) or (ns == ns_max and step > 0.0):
                at_boundary = True

                if ns == ns_min:
                    status['warnflag'] = -2
                    status['warnreason'] = (
                        'Function minimum is below the minimum bound of the '
                        'parameter value. Convergence forced at boundary.')
                elif ns == ns_max:
                    status['warnflag'] = -1
                    status['warnreason'] = (
                        'Function minimum is above the maximum bound of the '
                        'parameter value. Convergence forced at boundary.')
                break

            # Always perform step in ns as it improves the solution.
            ns += step

            # Do not allow ns outside boundaries.
            if ns < ns_min:
                ns = ns_min
            elif ns > ns_max:
                ns = ns_max

            # Increase counter since a step was taken.
            niter += 1

        x[0] = ns
        # Once converged evaluate function at minimum value unless
        # Convergence was forced at boundary
        # in which case function value is already known.
        if not at_boundary:
            (f, fprime, fprimeprime) = func(x, *func_args)

        if niter == max_steps:
            status['warnflag'] = 1
            status['warnreason'] = (
                f'NR optimization did not converge within {niter} NR steps.')

        status['niter'] = niter
        status['last_nr_step'] = step
        return (x, f, status)

    def get_niter(
            self,
            status,
    ):
        """Returns the number of iterations needed to find the minimum.

        Parameters
        ----------
        status : dict
            The dictionary with the status information about the minimization
            process.

        Returns
        -------
        niter : int
            The number of iterations needed to find the minimum.
        """
        return status['niter']

    def has_converged(
            self,
            status,
    ):
        """Analyzes the status information dictionary if the minimization
        process has converged. By definition the minimization process has
        converged if ``status['warnflag']`` is smaller or equal to 0.

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
        if status['warnflag'] <= 0:
            return True

        return False

    def is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result. By definition of this minimization method, this method will
        always return ``False``.
        """
        return False


class NRNsScan2dMinimizerImpl(
        NR1dNsMinimizerImpl
):
    """The NRNsScan2dMinimizerImpl class provides a minimizer implementation for
    the R2->R1 function where the first dimension is minimized using the
    Newton-Raphson minimization method and the second dimension is scanned.
    """

    def __init__(
            self,
            p2_scan_step,
            ns_tol=1e-3,
            **kwargs,
    ):
        """Creates a new minimizer implementation instance.

        Parameters
        ----------
        p2_scan_step : float
            The step size for the scan of the second parameter of the function
            to minimize.
        ns_tol : float
            The tolerance / precision for the ns parameter value.
        """
        super().__init__(
            ns_tol=ns_tol,
            **kwargs)

        self.p2_scan_step = p2_scan_step

    def minimize(
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
        """Minimizes the given function ``func`` with the given initial function
        argument values ``initials``. This minimizer implementation will only
        vary the first two parameters. The first parameter is the number of
        signal events, ns, and it is minimized using the Newton-Rapson method.
        The second parameter is scanned through its boundaries in step sizes of
        ``p2_scan_step``.

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

            The return value of ``func`` must be (f, grad, grad2), i.e. the
            function value at the function arguments ``x``, the value of the
            function first derivative for the first fit parameter, and the value
            of the second derivative for the first fit parameter.
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
            p2_n_steps : int
                The number of scanning steps performed for the 2nd parameter.
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
        p2_low = bounds[1][0]
        p2_high = bounds[1][1]
        p2_scan_values = np.linspace(
            p2_low, p2_high, int((p2_high-p2_low)/self.p2_scan_step)+1)

        logger.debug(
            'Minimize func by scanning 2nd parameter in '
            f'{len(p2_scan_values):d} steps with a step size of '
            f'{np.mean(np.diff(p2_scan_values)):g}')

        niter_total = 0
        best_xmin = None
        best_fmin = None
        best_status = None
        for p2_value in p2_scan_values:
            initials[1] = p2_value
            (xmin, fmin, status) = super().minimize(
                initials, bounds, func, func_args, **kwargs)
            niter_total += status['niter']
            if (best_fmin is None) or (fmin < best_fmin):
                best_xmin = xmin
                best_fmin = fmin
                best_status = status

        best_status['p2_n_steps'] = len(p2_scan_values)
        best_status['niter'] = niter_total

        return (best_xmin, best_fmin, best_status)


class Minimizer(
        object
):
    """The Minimizer class provides the general interface for minimizing a
    function. The class takes an instance of MinimizerImpl for a specific
    minimizer implementation.
    """

    def __init__(
            self,
            minimizer_impl,
            max_repetitions=100,
            **kwargs,
    ):
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
        super().__init__(**kwargs)

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
        if not isinstance(impl, MinimizerImpl):
            raise TypeError(
                'The minimizer_impl property must be an instance of '
                'MinimizerImpl!')
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
        if not isinstance(n, int):
            raise TypeError(
                'The maximal repetitions property must be of type int!')
        self._max_repetitions = n

    def minimize(
            self,
            rss,
            paramset,
            func,
            args=None,
            kwargs=None,
    ):
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
        paramset : instance of ParameterSet
            The ParameterSet instances holding the floating parameters of the
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
        if not isinstance(paramset, ParameterSet):
            raise TypeError(
                'The paramset argument must be an instance of ParameterSet!')

        if kwargs is None:
            kwargs = dict()

        bounds = paramset.floating_param_bounds
        initials = paramset.floating_param_initials
        logger.debug(f'Doing function minimization: initials: {initials}.')

        (xmin, fmin, status) = self._minimizer_impl.minimize(
            initials, bounds, func, args, **kwargs)

        reps = 0
        while (not self._minimizer_impl.has_converged(status)) and\
              (self._minimizer_impl.is_repeatable(status)) and\
              (reps < self._max_repetitions):
            # The minimizer did not converge at the first time, but it is
            # possible to repeat the minimization process with different
            # initials to obtain a better result.

            # Create a new set of random parameter initials based on the
            # parameter bounds.
            initials = paramset.generate_random_floating_param_initials(
                rss=rss)

            logger.debug(
                'Previous rep ({}) status={}, new initials={}'.format(
                    reps, str(status), str(initials)))

            # Repeat the minimization process.
            (xmin, fmin, status) = self._minimizer_impl.minimize(
                initials, bounds, func, args, **kwargs)

            reps += 1

        # Store the number of repetitions in the status dictionary.
        status['skyllh_minimizer_n_reps'] = reps

        if not self._minimizer_impl.has_converged(status):
            raise ValueError(
                f'The minimizer did not converge after {reps:d} repetitions! '
                'The maximum number of repetitions is '
                f'{self._max_repetitions:d}. The status dictionary is '
                f'"{str(status)}".')

        # Check if any fit value is outside its bounds due to rounding errors by
        # the minimizer. If so, set those fit values to their respective bound
        # value and re-evaluate the function with the corrected fit values.
        condmin = xmin < bounds[:, 0]
        condmax = xmin > bounds[:, 1]
        if np.any(condmin) or np.any(condmax):
            xmin = np.where(condmin, bounds[:, 0], xmin)
            xmin = np.where(condmax, bounds[:, 1], xmin)
            if args is None:
                args = tuple()
            (fmin, grads) = func(xmin, *args)

        logger.debug(
            '%s (%s): Minimized function: %d iterations, %d repetitions, '
            'xmin=%s' % (
                classname(self), classname(self._minimizer_impl),
                self._minimizer_impl.get_niter(status), reps, str(xmin)))

        return (xmin, fmin, status)
