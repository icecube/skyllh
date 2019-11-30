# -*- coding: utf-8 -*-

"""This module provides functionality for interpolation.
"""

import abc
import numpy as np

from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet
)


class GridManifoldInterpolationMethod(object):
    """This is an abstract base class for implementing a method to interpolate
    a manifold that is defined on a grid of parameter values. In general the
    number of parameters can be arbitrary and hence the manifold's
    dimensionality can be arbitrary, too. However, in practice the interpolation
    on a multi-dimensional manifold can be rather difficult.
    Nevertheless, we provide this interface to allow for manifold grids with
    different dimensionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, f, param_grid_set):
        """Constructor for a GridManifoldInterpolationMethod object.
        It must be called by the derived class.

        Parameters
        ----------
        f : callable R^d -> R
            The function that takes d parameters as input and returns the
            value of the d-dimensional manifold at this point for each given
            event.
            The call signature of f must be:

                ``__call__(gridparams, eventdata)``

            where gridparams is the dictionary with the parameter values
            on the grid, and ``eventdata`` is a 2-dimensional (N,V)-shaped numpy
            ndarray holding the event data, where N is the number of events, and
            V the dimensionality of the event data.
        param_grid_set : ParameterGridSet instance | ParameterGrid instance
            The set of d parameter grids. This defines the grid of the
            manifold.
        """
        super(GridManifoldInterpolationMethod, self).__init__()

        self.f = f
        self.param_grid_set = param_grid_set

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
    def param_grid_set(self):
        """The ParameterGridSet object defining the set of d parameter grids.
        This defines the grid of the manifold.
        """
        return self._param_grid_set
    @param_grid_set.setter
    def param_grid_set(self, obj):
        if(isinstance(obj, ParameterGrid)):
            obj = ParameterGridSet([obj])
        elif(not isinstance(obj, ParameterGridSet)):
            raise TypeError('The param_grid_set property must be an instance '
                'of ParameterGridSet!')
        self._param_grid_set = obj

    @property
    def ndim(self):
        """(read-only) The dimensionality of the manifold.
        """
        return len(self.param_grid_set)

    @abc.abstractmethod
    def get_value_and_gradients(self, eventdata, params):
        """Retrieves the interpolated value of the manifold at the d-dimensional
        point ``params`` for all given events, along with the d gradients,
        i.e. partial derivatives.

        Parameters
        ----------
        eventdata : numpy (N_events,V)-shaped 2D ndarray
            The 2D (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params : dict
            The dictionary with the parameter values, defining the point on the
            manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of parameters. The order of the D parameters is defined
            by the ParameterGridSet that has been provided at construction time
            of this interpolation method object.
        """
        pass


class Linear1DGridManifoldInterpolationMethod(GridManifoldInterpolationMethod):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a line.
    """
    def __init__(self, f, param_grid_set):
        """Creates a new Linear1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        f : callable R -> R
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given event.

                ``__call__(gridparams, eventdata)``

            where ``gridparams`` is the dictionary with the parameter names and
            values on the grid, and ``eventdata`` is a 2-dimensional
            (N,V)-shaped numpy ndarray holding the event data, where N is the
            number of events, and V the dimensionality of the event data.
            The return value of ``f`` must be a (N,)-shaped 1d ndarray of float.
        param_grid_set : instance of ParameterGridSet
            The set of parameter grids. This defines the grid of the
            1-dimensional manifold. By definition, only the first parameter grid
            is considered.
        """
        super(Linear1DGridManifoldInterpolationMethod, self).__init__(
            f, param_grid_set)

        self.p_grid = self.param_grid_set[0]

        # Create a cache for the line parameterization for the last
        # manifold grid point for the different events.
        self._create_cache(None, np.array([]), np.array([]))

    def _create_cache(self, x0, m, b):
        """Creates a cache for the line parameterization for the last manifold
        grid point for the nevents different events.

        Parameters
        ----------
        x0 : float | None
            The parameter grid value for the lower point of the grid manifold
            used to estimate the line.
        m : 1d ndarray
            The slope of the line for each event.
        b : 1d ndarray
            The offset coefficient of the line for each event.
        """
        self._cache = {
            'x0': x0,
            'm': m,
            'b': b
        }

    def get_value_and_gradients(self, eventdata, params):
        """Calculates the interpolated manifold value and its gradient for each
        given event at the point ``params``.

        Parameters
        ----------
        eventdata : numpy (N_events,V)-shaped 2D ndarray
            The 2D (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params : dict
            The dictionary with the parameter values, defining the point on the
            manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of parameters.
        """
        (xname, x) = tuple(params.items())[0]

        # Determine the nearest grid point that is lower than x and use that as
        # x0.
        x0 = self.p_grid.round_to_lower_grid_point(x)

        # Check if the line parametrization for x0 is already cached.
        self__cache = self._cache
        if((self__cache['x0'] == x0) and
           (eventdata.shape[0] == len(self__cache['m']))
          ):
            m = self__cache['m']
            b = self__cache['b']
        else:
            # Calculate the line parametrization for all the given events.

            # Calculate the upper grid point of x.
            x1 = self.p_grid.round_to_upper_grid_point(x)

            self__f = self.f
            M0 = self__f({xname:x0}, eventdata)
            M1 = self__f({xname:x1}, eventdata)

            m = (M1 - M0) / (x1 - x0)
            b = M0 - m*x0

            # Cache the line parametrization.
            self._create_cache(x0, m, b)

        # Calculate the interpolated manifold value. The gradient is m.
        value = m*x + b

        return (value, np.atleast_2d(m))


class Parabola1DGridManifoldInterpolationMethod(GridManifoldInterpolationMethod):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a parabola.
    """
    def __init__(self, f, param_grid_set):
        """Creates a new Parabola1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        f : callable R -> R
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given event.
            The call signature of f must be:

                ``__call__(gridparams, eventdata)``

            where ``gridparams`` is the dictionary with the parameter names and
            values on the grid, and ``eventdata`` is a 2-dimensional
            (N,V)-shaped numpy ndarray holding the event data, where N is the
            number of events, and V the dimensionality of the event data.
        param_grid_set : instance of ParameterGridSet
            The set of parameter grids. This defines the grid of the
            1-dimensional manifold. By definition, only the first parameter grid
            is considered.
        """
        super(Parabola1DGridManifoldInterpolationMethod, self).__init__(
            f, param_grid_set)

        self._p_grid = self.param_grid_set[0]

        # Create a cache for the parabola parameterization for the last
        # manifold grid point for the different events.
        self._create_cache(None, np.array([]), np.array([]), np.array([]))

    def _create_cache(self, x1, M1, a, b):
        """Creates a cache for the parabola parameterization for the last
        manifold grid point for the nevents different events.

        Parameters
        ----------
        x1 : float | None
            The parameter grid value for middle point of the grid manifold used
            to estimate the parabola.
        M1 : 1d ndarray
            The grid manifold value for each event of the middle point (x1,).
        a : 1d ndarray
            The parabola coefficient ``a`` for each event.
        b : 1d ndarray
            The parabola coefficient ``b`` for each event.
        """
        self._cache = {
            'x1': x1,
            'M1': M1,
            'a': a,
            'b': b
        }

    def get_value_and_gradients(self, eventdata, params):
        """Calculates the interpolated manifold value and its gradient for each
        given event at the point ``params``.

        Parameters
        ----------
        eventdata : numpy (N_events,V)-shaped 2D ndarray
            The 2D (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params : dict
            The dictionary with the parameter values, defining the point on the
            manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of parameters.
        """
        (xname, x) = tuple(params.items())[0]

        # Create local variable name alias to avoid Python dot lookups.
        self__p_grid = self._p_grid
        self__p_grid__round_to_nearest_grid_point = \
            self__p_grid.round_to_nearest_grid_point
        self__cache = self._cache

        # Determine the nearest grid point x1.
        x1 = self__p_grid__round_to_nearest_grid_point(x)

        # Check if the parabola parametrization for x1 is already cached.
        if((self__cache['x1'] == x1) and
           (eventdata.shape[0] == len(self__cache['M1']))
          ):
            M1 = self__cache['M1']
            a = self__cache['a']
            b = self__cache['b']
        else:
            dx = self__p_grid.delta

            # Calculate the neighboring grid points to x1: x0 and x2.
            x0 = self__p_grid__round_to_nearest_grid_point(x1 - dx)
            x2 = self__p_grid__round_to_nearest_grid_point(x1 + dx)

            # Parameterize the parabola with parameters a, b, and M1.
            self__f = self.f
            M0 = self__f({xname:x0}, eventdata)
            M1 = self__f({xname:x1}, eventdata)
            M2 = self__f({xname:x2}, eventdata)

            a = 0.5*(M0 - 2.*M1 + M2) / dx**2
            b = 0.5*(M2 - M0) / dx

            # Cache the parabola parametrization.
            self._create_cache(x1, M1, a, b)

        # Calculate the interpolated manifold value.
        value = a * (x - x1)**2 + b * (x - x1) + M1
        # Calculate the gradient of the manifold.
        gradients = 2. * a * (x - x1) + b

        return (value, np.atleast_2d(gradients))

