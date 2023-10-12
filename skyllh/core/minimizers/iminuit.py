"""
The minimizers/iminuit module provides a SkyLLH interface to the iminuit
minimizer.
"""

import numpy as np

from skyllh.core import (
    tool,
)
from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.minimizer import (
    MinimizerImpl,
)


logger = get_logger(__name__)


class FuncWithGradsFunctor(
        HasConfig,
):
    """Helper class to evaluate the LLH function that returns the function value
    and its gradients in two seperate calls, one for the LLH function value and
    one for its gradient values.
    """

    def __init__(
            self,
            func,
            func_args=None,
            **kwargs,
    ):
        """Initializes a new functor instance for the given function ``func``.

        Parameters
        ----------
        func : callable
            The function with call signature

                ``__call__(x, *args)``

            returning the a two-element tuple (f, grads).
        func_args : tuple | None
            The optional positional arguments for the function ``func``.
        """
        super().__init__(
            **kwargs)

        if func_args is None:
            func_args = tuple()

        self._func = func
        self._func_args = func_args

        self._tracing = self._cfg.is_tracing_enabled

        self._cache_x = None
        self._cache_f = None
        self._cache_grads = None

    def get_f(self, x):
        tracing = self._tracing

        if self._cache_x is None:
            self._cache_x = np.copy(x)
        else:
            if np.all(x == self._cache_x):
                if tracing:
                    logger.debug(
                        f'call_func(x={x}): Return cached f={self._cache_f}')
                return self._cache_f
            else:
                np.copyto(self._cache_x, x)

        (self._cache_f, self._cache_grads) = self._func(
            x, *self._func_args)

        if tracing:
            logger.debug(
                f'call_func(x={x}): Return calculated f={self._cache_f}')
        return self._cache_f

    def get_grads(self, x):
        tracing = self._tracing

        if self._cache_x is None:
            self._cache_x = np.copy(x)
        else:
            if np.all(x == self._cache_x):
                if tracing:
                    logger.debug(
                        f'call_grads(x={x}): Return cached '
                        f'grads={self._cache_grads}')
                return self._cache_grads
            else:
                np.copyto(self._cache_x, x)

        (self._cache_f, self._cache_grads) = self._func(
            x, *self._func_args)

        if tracing:
            logger.debug(
                f'call_grads(x={x}): Return calculated '
                f'grads={self._cache_grads}')
        return self._cache_grads


class IMinuitMinimizerImpl(
        MinimizerImpl,
):
    """The SkyLLH minimizer implementation that utilizes the iminuit minimizer.
    """

    @tool.requires('iminuit')
    def __init__(
            self,
            ftol=1e-6,
            **kwargs,
    ):
        """Creates a new IMinuit minimizer instance to minimize a given
        function.

        Parameters
        ----------
        ftol : float
            The function value tolerance as absolute value.
        """
        super().__init__(**kwargs)

        self._ftol = ftol

    def minimize(
            self,
            initials,
            bounds,
            func,
            func_args=None,
            **kwargs,
    ):
        """Minimizes the given function ``func`` with the given initial function
        argument values ``initials`` and within the given parameter bounds
        ``bounds``.

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
        :func:`iminuit.minimize` minimization function.

        Returns
        -------
        xmin : 1D ndarray
            The array containing the function arguments at the function's
            minimum.
        fmin : float
            The function value at its minimum.
        res : iminuit.OptimizeResult
            The iminuit OptimizeResult dictionary with additional information.
        """
        if func_args is None:
            func_args = tuple()
        if kwargs is None:
            kwargs = dict()

        iminuit = tool.get('iminuit')

        func_provides_grads = kwargs.pop('func_provides_grads', True)

        if func_provides_grads:
            # The function func returns the function value and its gradients,
            # so we need to use the FuncWithGradsFunctor helper class.
            functor = FuncWithGradsFunctor(
                cfg=self._cfg,
                func=func,
                func_args=func_args)

            res = iminuit.minimize(
                fun=functor.get_f,
                x0=initials,
                bounds=bounds,
                jac=functor.get_grads,
                tol=self._ftol,
                **kwargs
            )
        else:
            # The function func returns only the function value, so we can use
            # the
            res = iminuit.minimize(
                func,
                initials,
                bounds=bounds,
                args=func_args,
                tol=self._ftol,
                **kwargs
            )

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
        return status['nfev']

    def has_converged(self, status):
        """Analyzes the status information dictionary if the minimization
        process has converged. By definition the minimization process has
        converged if ``status['is_valid']`` equals True.

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
        return bool(status['success'])

    def is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result.

        TODO: Implement a proper check. For now just return True.


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
        return True
