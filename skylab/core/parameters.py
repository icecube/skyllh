# -*- coding: utf-8 -*-

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
