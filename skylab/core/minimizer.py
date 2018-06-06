"""The minimizer module provides functionality for the minimization process of
the likelihood function. In general the user wants to maximize the likelihood
function, but numeric software libraries provide only minimizers. So the user
needs to provide the negative likelihood function that has a minimum at its
maximum. 
"""
import abc
import numpy as np

from py import is_sequence
from lh import LHFunction

class Minimizer(object):
    __metaclass__ = abc.ABCMeta
    
    """The Minimizer class holds the likelihood function and the partial
    derivatives for each parameter thereof. The derivatives are needed for the
    minimizer to work more stable. 
    """
    def __init__(self, lhfunc, partialderivatives, max_repetitions=100):
        """Creates a new Minimizer instance which holds the likelihood function
        and its partial derivative functions.
        
        Parameters
        ----------
        lhfunc : LHFunction
            The likelihood function that is going to be minimized.
        partialderivatives : list of LHFunction
            The list of LHFunction objects, one for each partial derivative of
            the likelihood function.
        max_repetitions : int
            In case the minimization process did not converge at the first time
            this option specifies the maximum number of repetitions with
            different initials.
        """
        self.lh_func = lhfunc
        self.partial_derivatives = partialderivatives
        self.max_repetitions = max_repetitions

    #___________________________________________________________________________
    @property
    def lh_func(self):
        """The likelihood function that the minimizer should minimize.
        It must be of type LHFunction. It must have the parameters already
        defined.
        """
        return self._lh_func
    @lh_func.setter
    def lh_func(self, lhfunc):
        if(not isinstance(lhfunc, LHFunction)):
            raise TypeError('The likelihood function object needs to be an object of type LHFunction!')
        if(lhfunc.params.N == 0):
            raise ValueError('Parameters for the LHFunction object must be defined first!')
        self._lh_func = lhfunc
    
    #___________________________________________________________________________
    @property
    def partial_derivatives(self):
        """The list of partial derivatives of the likelihood function. Each
        derivative must be an object of type LHFunction. The order of the
        derivatives must follow the same order as of the parameter definitions.
        
        The defined parameters for each derivative must be the same as for the
        likelihood function. If they are not defined, they will be defined
        automatically through the likelihood function.
        """
        return self._partial_derivatives
    @partial_derivatives.setter
    def partial_derivatives(self, partialderivatives):
        # Check for correct types.
        if(not is_sequence(partialderivatives)):
            raise TypeError('The partial derivatives must be a sequence!')
        partialderivatives = list(partialderivatives)
        for pd in partialderivatives:
            if(not isinstance(pd, LHFunction)):
                raise TypeError('Not all specified partial derivatives are of type LHFunction!')
        
        # Define the parameters for all the partial derivatives in case they are
        # not defined yet.
        lh_func_params = self.lh_func.params
        for pd in partialderivatives:
            if(pd.params != lh_func_params):
                pd.params.def_param(lh_func_params)
            
        self._partial_derivatives = partialderivatives
    
    #___________________________________________________________________________
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
            raise TypeError('The maximal repetitions property must be of type int!')
        self._max_repetitions = n
    
    ###
    # Minimizer interface methods.
    @abc.abstractmethod
    def _minimize_impl(self, initials, events, fmin_kwargs):
        """This is the implementation method of the actual used minimizer.
        
        Parameters
        ----------
        initials : list
            The list of initial parameter values.
        events : numpy.recarray
            The events, i.e. the data, for the likelihood function.
        fmin_kwargs : dict
            Optional additional keyword arguments for the function minimizer
            method. 
        
        Returns
        -------
        xmin : numpy.ndarray
            The 1-dim numpy.ndarray holding the coordinates of the minimum.
        fmin : number
            The function value at the minimum.
        status : object
            The status object that is understood by the derived minimizer class
            to determine convergence.
        """
        pass
    
    @abc.abstractmethod
    def _has_converged(self, status):
        """Interface method to determine if the minimization process has
        converged.
        
        Parameters
        ----------
        status : object
            The status object returned by the `_minimize_impl` method.
        """
        pass
    
    @abc.abstractmethod
    def _is_repeatable(self, status): 
        """Checks if the minimization process can be repeated with a different
        parameter initial within the parameter bounds.
        
        Parameters
        ----------
        status : object
            The status object returned by the `_minimize_impl` method.
        """
    ###
    # Public methods.
    #___________________________________________________________________________
    def get_minimizer_lh_func(self):
        """Provides the interface to the minimizer, which requires a ndarray `x`
        with the n-dim parameter values and optional arguments.
        """
        lh_func_param_names = self._lh_func.params.names
        initials = np.array(self._lh_func.params.initials)
        isconst = np.array(self._lh_func.params.isconst_list)
        param_storage = np.empty_like(initials)
        
        def func(x, *args):
            # Translate the indexed parameters into a name => value dictionary
            # and mix in the constant parameter initials.
            param_storage[isconst] = initials[isconst]
            param_storage[~isconst] = x
            params = dict(zip(lh_func_param_names, param_storage))
            print(params)
            return self._lh_func.evaluate(params, *args)
        
        return func
    
    def get_minimizer_partial_derivatives_func(self):
        """Returns a minimizer function with the call signature
        `func(x, *args)`, which evaluates all the likelihood derivative
        functions and returns a numpy.ndarray with all the derivative values.
        """
        lh_func_param_names = self._lh_func.params.names
        initials = np.array(self._lh_func.params.initials)
        isconst = np.array(self._lh_func.params.isconst_list)
        param_storage = np.empty_like(initials)
        
        results = np.ndarray((self._lh_func.params.N_var,), np.float64) 
        
        def func(x, *args):
            # Translate the indexed parameters into a name => value dictionary
            # and mix in the constant parameter initials.
            param_storage[isconst] = initials[isconst]
            param_storage[~isconst] = x
            params = dict(zip(lh_func_param_names, param_storage))
            print(params)
            
            # Evalute the likelihood function derivative for each parameter.
            for (i, pd) in enumerate(self._partial_derivatives):
                results[i] = pd.evaluate(params, *args)
            
            return results
        
        return func
    
    def minimize(self, events, **kwargs):
        """Minimizes the likelihood function by calling the _minimize_impl
        method, which has to be implemented by the derived class and is tailored
        to the specific minimizer.
        
        After the minimization process it calls the `_has_converged` and
        `_is_repeatable` methods to determine if an additional minimization
        attempt has to be performed. This is repeated until the minimization
        process did converge or if the maximal number of repetitions has
        occurred. 
        
        Parameters
        ----------
        events : numpy.recarray
            The events, i.e. the data, for the likelihood function.
        
        Other Parameters
        ----------------
        Additional keyword arguments will be passed to the actual used
        minimizer method.
        """
        initials = self.lh_func.params.variable_initials
        (xmin, fmin, status) = self._minimize_impl(initials, events, fmin_kwargs=kwargs)
        
        reps = 0
        while((not self._has_converged(status)) and
              self._is_repeatable(status) and
              reps < self.max_repetitions
        ):
            # The minimizer did not converge at the first time, but it is
            # possible to repeated the minimization process with different
            # initials to obtain a better result.
            
            # Create a new set of random parameter initials based on the
            # parameter bounds.
            initials = self.lh_func.params.generate_random_initials()
            
            # Repeat the minimization process.
            (xmin, fmin, status) = self._minimize_impl(initials, events, fmin_kwargs=kwargs)
            
            reps += 1
        
        return (xmin, fmin, status)
            
class LBFGSMinimizer(Minimizer):
    """The LBFGSMinimizer class provides the L-BFG-S minimizer from the
    scipy.optimize module.
    """
    def __init__(self, lhfunc, partialderivatives, **kwargs):
        """Creates a new L-BGF-S minimizer instance to minimize the given
        likelihood function with its given partial derivatives.
        """
        super(LBFGSMinimizer, self).__init__(lhfunc, partialderivatives, **kwargs)
        
        # Import the external library here to make it an optional dependency. 
        from scipy.optimize import fmin_l_bfgs_b
        self._fmin_l_bfgs_b = fmin_l_bfgs_b
    
    def _minimize_impl(self, initials, events, fmin_kwargs):
        """The implementation of the minimize method. It uses the
        `scipy.optimize.fmin_l_bfgs_b` minimizer.

        See the documentation of the Minimizer class for the documentation of
        the arguments.

        """
        (xmin, fmin, d) = self._fmin_l_bfgs_b(
            self.get_minimizer_lh_func(), initials,
            fprime      = self.get_minimizer_partial_derivatives_func(),
            args        = (events,),
            bounds      = self.lh_func.params.variable_bounds,
            approx_grad = False,
            **fmin_kwargs
        )
        
        return (xmin, fmin, d)
    
    def _has_converged(self, status):
        if(status['warnflag'] == 0):
            return True
        return False
    
    def _is_repeatable(self, status):
        """Checks if the minimization process can be repeated to get a better
        result. It's repeatable if
        
            `status['warnflag'] == 2 and 'FACTR' in str(status['task'])`
        
        """
        if(status['warnflag'] == 2 and 'FACTR' in str(status['task'])):
            return True
        return False