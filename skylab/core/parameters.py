# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np

from skylab.core.py import ObjectCollection

def make_linear_parameter_grid_1d(name, low, high, delta):
    """Utility function to create a ParameterGrid object for a 1-dimensional
    linear parameter grid.

    Parameters
    ----------
    name : str
        The name of the parameter.
    low : float
        The lowest value of the parameter.
    high : float
        The highest value of the parameter.
    delta : float
        The constant distance between the grid values. By definition this
        defines also the precision of the parameter values.

    Returns
    -------
    obj : ParameterGrid
        The ParameterGrid object holding the discrete parameter grid values.
    """
    grid = np.linspace(low, high, np.round((high-low)/delta)+1)
    return ParameterGrid(name, grid, delta)

def make_params_hash(params):
    """Utility function to create a hash value for a given parameter dictionary.

    Parameters
    ----------
    params : dict
        The dictionary holding the parameter (name: value) pairs.

    Returns
    -------
    hash : int
        The hash of the parameter dictionary.
    """
    if(not isinstance(params, dict)):
        raise TypeError('The params argument must be of type dict!')

    # A note on the ordering of Python dictionary items: The items are ordered
    # internally according to the hash value of their keys. Hence, if we don't
    # insert more dictionary items, the order of the items won't change. Thus,
    # we can just take the items list and make a tuple to create a hash of it.
    # The hash will be the same for two dictionaries having the same items.
    return hash(tuple(params.items()))

class ParameterGrid(object):
    """This class provides a data holder for a parameter that has a set of
    discrete values, i.e. has a value grid.
    """
    def __init__(self, name, grid, precision):
        """Creates a new parameter grid.

        Parameters
        ----------
        name : str
            The name of the parameter.
        grid : numpy.ndarray
            The numpy ndarray holding the discrete grid values of the parameter.
        precision : float
            The precision of the parameter values.
        """
        self.name = name
        self.grid = grid
        self.precision = precision

        #self.grid = self.round_to_precision(self.grid)

    @property
    def name(self):
        """The name of the parameter.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

    @property
    def grid(self):
        """The numpy.ndarray with the grid values of the parameter.
        """
        return self._grid
    @grid.setter
    def grid(self, arr):
        if(not isinstance(arr, np.ndarray)):
            raise TypeError('The values property must be of type numpy.ndarray!')
        self._grid = arr

    @property
    def precision(self):
        """The precision (float) of the parameter values.
        """
        return self._precision
    @precision.setter
    def precision(self, value):
        if(isinstance(value, int)):
            value = float(value)
        if(not isinstance(value, float)):
            raise TypeError('The precision property must be of type float!')
        self._precision = value

    @property
    def ndim(self):
        """The dimensionality of the parameter grid.
        """
        return self._grid.ndim

    def round_to_precision(self, value):
        """Rounds the given value to the precision of the parameter.

        Parameters
        ----------
        value : float | ndarray
            The value(s) to round.
        """
        return np.around(value / self._precision) * self._precision

class ParameterGridSet(ObjectCollection):
    """Describes a set of parameter grids.
    """
    def __init__(self):
        super(ParameterGridSet, self).__init__(obj_t=ParameterGrid)

    @property
    def ndim(self):
        """The dimensionality of this parameter grid set. By definition it's the
        number of parameters of the set.
        """
        return len(self)

    @property
    def parameter_permutation_dict_list(self):
        """(read-only) The list of parameter dictionaries constructed from all
        permutations of all the parameter values.
        """
        # Get the list of parameter names.
        param_names = [ paramgrid.name for paramgrid in self.objects ]
        # Get the list of parameter grids, in same order than the parameter
        # names.
        param_grids = [ paramgrid.grid for paramgrid in self.objects ]

        dict_list = [ dict([ (p_i, t_i)
                            for (p_i, t_i) in zip(param_names, tup) ])
                     for tup in itertools.product(*param_grids) ]
        return dict_list

class FitParameterManifoldGridInterpolationMethod(object):
    """This is an abstract base class for implementing a method to interpolate
    a set of fit parameters on a fit parameter manifold grid. In general the
    number of fit parameters can be arbitrary and hence the manifold's
    dimensionality can be arbitrary too. However, in practice the interpolation
    on a multi-dimensional manifold can be rather difficult.
    Nevertheless, we provide this interface to allow for different kinds of
    manifold grids with different dimensionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, f, fitparams_grid_set):
        """Constructor for a FitParameterManifoldGridInterpolationMethod object.
        It must be called by the derived class.

        Parameters
        ----------
        f : callable R^d -> R
            The function that takes d fit parameters as input and returns the
            value of the manifold at this d-dimensional point for each given
            event.
            The call signature of f must be:
                __call__(gridfitparams, eventdata)
            where gridfitparams is the dictionary with the fit parameter values
            on the grid and ``eventdata`` is a 2-dimensional (N,V)-shaped numpy
            ndarray holding the event data, where N is the number of events, and
            V the dimensionality of the event data.
        fitparams_grid_set : ParameterGridSet
            The set of d fit parameter grids. This defines the grid of the
            manifold.
        """
        self.f = f
        self.fitparams_grid_set = fitparams_grid_set

    @property
    def f(self):
        """The R^d -> R manifold function.
        """
        return self._f
    @f.setter
    def f(self, func):
        if(not callable(func)):
            raise TypeError('The f property must be a callable object!')
        self._f = func

    @property
    def fitparams_grid_set(self):
        """The ParameterGridSet object defining the set of d fit parameter
        grids. This defines the grid of the manifold.
        """
        return self._fitparams_grid_set
    @fitparams_grid_set.setter
    def fitparams_grid_set(self, obj):
        if(not isinstance(obj, ParameterGridSet)):
            raise TypeError('The fitparams_grid_set property must be an instance of ParameterGridSet!')
        self._fitparams_grid_set = obj

    @property
    def ndim(self):
        """(read-only) The dimensionality of the manifold.
        """
        return len(self.fitparams_grid_set)

    @abc.abstractmethod
    def get_value_and_gradients(self, eventdata, fitparams):
        """Retrieves the interpolated value of the manifold at the d-dimensional
        point ``fitparams`` for all given events, along with the d gradients,
        i.e. partial derivatives.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
            manifold value should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values, defining the point
            on the manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (N,D) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of fit parameters.
        """
        pass


class ParabolaFitParameterInterpolationMethod(FitParameterManifoldGridInterpolationMethod):
    """This fit parameter manifold grid interpolation method interpolates the
    1-dimensional fit parameter manifold using a parabola.
    """
    def __init__(self, f, fitparams_grid_set):
        super(ParabolaFitParameterInterpolationMethod, self).__init__(f, fitparams_grid_set)

        self.p_grid = self.fitparams_grid_set[0]

        # Create a cache for the parabola parameterization for the last
        # manifold grid point for the different events.
        self._create_cache(None, np.array([]), np.array([]), np.array([]))

    def _create_cache(self, x1, M1, a, b):
        """Create a cache for the parabola parameterization for the last
        manifold grid point p1 for the nevents different events.

        Parameters
        ----------
        x1 : float | None
        M1 : 1d ndarray
        a : 1d ndarray
        b : 1d ndarray
        """
        self._cache = {
            'x1': x1,
            'M1': M1,
            'a': a,
            'b': b
        }

    def get_value_and_gradients(self, eventdata, fitparams):
        """Calculates the interpolted manifold value and its gradient for each
        given event at the point ``fitparams``.
        """
        (xname, x) = fitparams.items()[0]

        # Determine the nearest grid point x1 and the grid precision.
        x1 = self.p_grid.round_to_precision(x)
        dx = self.p_grid.precision

        # Check if the parabola parametrization for x1 is already cached.
        if((self._cache['x1'] == x1) and
           (len(events) == len(self._cache['M1']))
          ):
            M1 = self._cache_parabola['M1']
            a = self._cache_parabola['a']
            b = self._cache_parabola['b']
        else:
            # Calculate the neighboring gridponts to x1: x0 and x2.
            x0 = self.p_grid.round_to_precision(x1 - dp)
            x2 = self.p_grid.round_to_precision(x1 + dp)

            # Parameterize the parabola with parameters a, b, and M1.
            M0 = self.f({xname:x0}, eventdata)
            M1 = self.f({xname:x1}, eventdata)
            M2 = self.f({xname:x2}, eventdata)

            a = 0.5*(M0 - 2.*M1 + M2) / dx**2
            b = 0.5*(M2 - M0) / dx

            # Cache the parabola parametrization.
            self._create_cache(x1, M1, a, b)

        # Calculate the interpolated manifold value.
        value = a * (x - x1)**2 + b * (x - x1) + M1
        # Calculate the gradient of the manifold.
        gradients = 2. * a * (x - x1) + b

        return (value, np.atleast_2d(gradients))
