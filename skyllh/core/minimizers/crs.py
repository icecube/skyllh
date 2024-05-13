from numpy import *

from skyllh.core import (
    tool,
)
from skyllh.core.minimizer import (
    MinimizerImpl,
)

from skyllh.core.minimizers.iminuit import FuncWithGradsFunctor


class CRSMinimizerImpl(
        MinimizerImpl,
):
    """The SkyLLH minimizer implementation that utilizes the CRS minimizer from nlopt.
    """

    @tool.requires('nlopt')
    def __init__(
            self,
            ftol=1e-6,
            **kwargs,
    ):
        """Creates a new CRS minimizer instance to minimize a given
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
        res : dictionary
            A dictionary with additional information.
        """
        if func_args is None:
            func_args = tuple()
        if kwargs is None:
            kwargs = dict()

        nlopt = tool.get('nlopt')

        opt = nlopt.opt(nlopt.GN_CRS2_LM, 2)

        opt.set_lower_bounds(bounds[:,0])
        opt.set_upper_bounds(bounds[:,1])



        func_provides_grads = kwargs.pop('func_provides_grads', True)

        # objective function needs to take x and grad as arguments
        def helper_function(x, grad):

            if func_provides_grads:
                # The function func returns the function value and its gradients,
                # so we need to use the FuncWithGradsFunctor helper class.
                functor = FuncWithGradsFunctor(
                    cfg=self._cfg,
                    func=func,
                    func_args=func_args)
                
                print(x, functor.get_f(x))
                return functor.get_f(x)

            else:
                # we can use the function directly
                print(x, func(x, *func_args))
                return (lambda x, grad: func(x, *func_args))
            
        opt.set_ftol_abs(self._ftol)
        opt.set_stopval(1e-6)
        opt.set_xtol_abs((1e-5, 1e-4))

        opt.set_min_objective(helper_function)

        opt.set_maxtime(30)
        x = opt.optimize(initials)
        val = opt.last_optimum_value()

        status = opt.last_optimize_result()

        res = {"x": x, "success": True if status > 0 else False, "status": status, "message": self.get_message(status), 
               "nfev": opt.get_numevals(), "fun": val}

        return (x, val, res)
    
    def get_message(self, status):
        """returns the message belonging to the minimizer status

        Parameters
        ----------
        status : int
            The number of the minimizer status
        """

        message = {1: "Generic success return value.",
            2: "Optimization stopped because stopval was reached.",
            3: "Optimization stopped because ftol_rel or ftol_abs was reached.", 
            4: "Optimization stopped because xtol_rel or xtol_abs was reached.", 
            5: "Optimization stopped because maxeval was reached.", 
            6: "Optimization stopped because maxtime was reached.", 
            -1: "Generic failure code.", 
            -2: "Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).",
            -3: "Ran out of memory.",
            -4: "Halted because roundoff errors limited progress. (In this case, the optimization still typically returns a useful result.)",
            -5: "Halted because of a forced termination"}
        
        return message[status]

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