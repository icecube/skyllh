# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np
from copy import deepcopy

from skyllh.physics.source import (
    SourceCollection,
    SourceModel
)
from skyllh.core.py import (
    ObjectCollection,
    bool_cast,
    float_cast,
    issequence,
    issequenceof,
    range
)
from skyllh.core.random import RandomStateService


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


class Parameter(object):
    """This class describes a parameter of a mathematical function, like a PDF,
    or source flux function. A parameter has a name, a value range, and an
    initial value. Furthermore, it has a flag that determines whether this
    parameter has a fixed value or not.
    """
    def __init__(self, name, initial, isfixed=True, valmin=None, valmax=None):
        """Creates a new Parameter instance.

        Parameters
        ----------
        name : str
            The name of the parameter.
        initial : float
            The initial value of the parameter.
        isfixed : bool
            Flag if the value of this parameter is mutable (False), or not
            (True). If set to `True`, the value of the parameter will always be
            the `initial` value.
        valmin : float | None
            The minimum value of the parameter in case this parameter is
            mutable.
        valmax : float | None
            The maximum value of the parameter in case this parameter is
            mutable.
        """
        self.name = name
        self.initial = initial
        self.isfixed = isfixed
        self.valmin = valmin
        self.valmax = valmax
        self.value = self._initial

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
    def initial(self):
        """The initial value of the parameter.
        """
        return self._initial
    @initial.setter
    def initial(self, v):
        v = float_cast(v, 'The initial property must be castable to type '
            'float!')
        self._initial = v

    @property
    def isfixed(self):
        """The flag if the parameter is mutable (False) or not (True).
        """
        return self._isfixed
    @isfixed.setter
    def isfixed(self, b):
        b = bool_cast(b, 'The isfixed property must be castable to type bool!')
        self._isfixed = b

    @property
    def valmin(self):
        """The minimum bound value of the parameter.
        """
        return self._valmin
    @valmin.setter
    def valmin(self, v):
        v = float_cast(v, 'The valmin property must be castable to type float!',
            allow_None=True)
        self._valmin = v

    @property
    def valmax(self):
        """The maximum bound value of the parameter.
        """
        return self._valmax
    @valmax.setter
    def valmax(self, v):
        v = float_cast(v, 'The valmax property must be castable to type float!',
            allow_None=True)
        self._valmax = v

    @property
    def value(self):
        """The current value of the parameter.
        """
        return self._value
    @value.setter
    def value(self, v):
        v = float_cast(v, 'The value property must be castable to type float!')
        if(self._isfixed):
            if(v != self._initial):
                raise ValueError('The value (%f) of the fixed parameter "%s" '
                    'must to equal to the parameter\'s initial value (%f)!'%(
                    v, self._initial))
        else:
            if((v < self._valmin) or (v > self._valmax)):
                raise ValueError('The value (%f) of parameter "%s" must be '
                    'within the range (%f, %f)!'%(
                    v, self._name, self._valmin, self._valmax))
        self._value = v

    def __str__(self):
        """Creates and returns a pretty string representation of this Parameter
        instance.
        """
        indstr = ' ' * display.INDENTATION_WIDTH

        s = 'Parameter: %s = %.3f '%(self._name, self._value)

        if(self.isfixed):
            s += '[fixed]'
        else:
            s += '[floating] {\n'
            s += indstr + 'initial: %.3f\n'%(self._initial)
            s += indstr + 'range: (%.3f, %.3f)\n'%(self._valmin, self._valmax)
            s += '}'

        return s

    def as_linear_grid(self, delta):
        """Creates a ParameterGrid instance with a linear grid with constant
        grid value distances delta.

        Parameters
        ----------
        delta : float
            The constant distance between the grid values. By definition this
            defines also the precision of the parameter values.

        Returns
        -------
        grid : ParameterGrid instance
            The ParameterGrid instance holding the grid values.
        """
        delta = float_cast(delta, 'The delta argument must be castable to type '
            'float!')
        grid = make_linear_parameter_grid_1d(
            self._name, self._valmin, self._valmax, delta)
        return grid

    def make_fixed(self, initial=None):
        """Fixes this parameter to the given initial value.

        Parameters
        ----------
        initial : float | None
            The new fixed initial value of the Parameter. If set to None, the
            parameter's current value will be used as initial value.

        Returns
        -------
        value : float
            The parameters new value.
        """
        self._isfixed = True

        # Set the new initial value if requested.
        if(initial is None):
            self._initial = self._value
            return self._value

        self.initial = initial
        self._value = self._initial

        return self._value

    def make_floating(self, initial=None, valmin=None, valmax=None):
        """Defines this parameter as floating with the given initial, minimal,
        and maximal value.

        Parameters
        ----------
        initial : float | None
            The initial value of the parameter. If set to `None`, the
            parameter's current value will be used as initial value.
        valmin : float | None
            The minimal value the parameter's value can take.
            If set to `None`, the parameter's current minimal value will be
            used.
        valmax : float | None
            The maximal value the parameter's value can take.
            If set to `None`, the parameter's current maximal value will be
            used.

        Returns
        -------
        value : float
            The parameter's new value.
        """
        self._isfixed = False

        if(initial is None):
            initial = self._value
        if(valmin is None):
            if(self._valmin is None):
                raise ValueError('The current minimal value of parameter "%s" '
                    'is not set. So it must be defined through the valmin '
                    'argument!'%(self._name))
            valmin = self._valmin
        if(valmax is None):
            if(self._valmax is None):
                raise ValueError('The current maximal value of parameter "%s" '
                    'is not set. So it must be defined through the valmax '
                    'argument!'%(self._name))
            valmax = self._valmax

        self.initial = initial
        self.valmin = valmin
        self.valmax = valmax
        self.value = self._initial

        return self._value


class ParameterSet(object):
    """This class holds a set of Parameter instances.
    """
    @staticmethod
    def union(*paramsets):
        """Creates a ParameterSet instance that is the union of the given
        ParameterSet instances.

        Parameters
        ----------
        *paramsets : ParameterSet instances
            The sequence of ParameterSet instances.

        Returns
        -------
        paramset : ParameterSet instance
            The newly created ParameterSet instance that holds the union of the
            parameters provided by all the ParameterSet instances.
        """
        if(not issequenceof(paramsets, ParameterSet)):
            raise TypeError('The arguments of the union static function must '
                'be instances of ParameterSet!')
        if(not len(paramsets) >= 1):
            raise ValueError('At least 1 ParameterSet instance must be '
                'provided to the union static function!')

        paramset = ParameterSet(params=paramsets[0])
        for paramset_i in paramsets[1:]:
            for param in paramset_i._params:
                if(not paramset.has_param(param)):
                    paramset.add_param(param)

        return paramset

    def __init__(self, params=None):
        """Constructs a new ParameterSet instance.

        Parameters
        ----------
        params : instance of Parameter | sequence of Parameter instances | None
            The initial sequence of Parameter instances of this ParameterSet
            instance.
        """
        # Define the list of parameters.
        # Define the (n_params,)-shaped numpy array of Parameter objects.
        self._params = np.empty((0,), dtype=np.object)
        # Define the (n_params,)-shaped numpy mask array that masks the fixed
        # parameters in the list of all parameters.
        self._params_fixed_mask = np.empty((0,), dtype=np.bool)

        # Define two lists for the parameter names. One for the fixed
        # parameters, and one for the floating parameters.
        # This is for optimization purpose only.
        self._fixed_param_name_list = []
        self._floating_param_name_list = []

        # Define dictionaries to map parameter names to storage index of the
        # parameter for fixed and floating parameters.
        self._fixed_param_name_to_idx = dict()
        self._floating_param_name_to_idx = dict()

        # Define a (n_fixed_params,)-shaped ndarray holding the values of the
        # fixed parameters. This is for optimization purpose only.
        self._fixed_param_values = np.empty((0,), dtype=np.float)

        # Add the initial Parameter instances.
        if(params is not None):
            if(isinstance(params, Parameter)):
                params = [params]
            if(not issequenceof(params, Parameter)):
                raise TypeError('The params argument must be None, an instance '
                    'of Parameter, or a sequence of Parameter instances!')
            for param in params:
                self.add_param(param)

    @property
    def params(self):
        """(read-only) The 1D ndarray holding the Parameter instances.
        """
        return self._params

    @property
    def fixed_params(self):
        """(read-only) The 1D ndarray holding the Parameter instances, whose
        values are fixed.
        """
        return self._params[self._params_fixed_mask]

    @property
    def floating_params(self):
        """(read-only) The 1D ndarray holding the Parameter instances,
        whose values are floating.
        """
        return self._params[np.invert(self._params_fixed_mask)]

    @property
    def n_params(self):
        """(read-only) The number of parameters this ParameterSet has.
        """
        return len(self._params)

    @property
    def n_fixed_params(self):
        """(read-only) The number of fixed parameters defined in this parameter
        set.
        """
        return len(self._fixed_param_name_list)

    @property
    def n_floating_params(self):
        """(read-only) The number of floating parameters defined in this
        parameter set.
        """
        return len(self._floating_param_name_list)

    @property
    def fixed_param_name_list(self):
        """(read-only) The list of the fixed parameter names.
        """
        return self._fixed_param_name_list

    @property
    def floating_param_name_list(self):
        """(read-only) The list of the floating parameter names.
        """
        return self._floating_param_name_list

    @property
    def fixed_param_values(self):
        """(read-only) The (n_fixed_params,)-shaped ndarray holding values of
        the fixed parameters.
        """
        return self._fixed_param_values

    @property
    def floating_param_initials(self):
        """(read-only) The 1D (n_floating_params,)-shaped ndarray holding the
        initial values of all the global floating parameters.
        """
        floating_params = self.floating_params
        if(len(floating_params) == 0):
            return np.empty((0,), dtype=np.float)
        return np.array(
            [ param.initial
             for param in floating_params ], dtype=np.float)

    @property
    def floating_param_bounds(self):
        """(read-only) The 2D (n_floating_params,2)-shaped ndarray holding the
        boundaries for all the floating parameters.
        """
        floating_params = self.floating_params
        if(len(floating_params) == 0):
            return np.empty((0,2), dtype=np.float)
        return np.array(
            [ (param.valmin, param.valmax)
             for param in floating_params ], dtype=np.float)

    def __len__(self):
        """The number of parameters this ParameterSet has.
        """
        return len(self._params)

    def __str__(self):
        """Creates and returns a pretty string representation of this
        ParameterSet instance.
        """
        s = '%s: %d parameters (%d floating, %d fixed) {'%(
            classname(self), self.n_params, self.n_floating_params,
            self.n_fixed_params)
        for param in self._params:
            s += '\n'
            s += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, str(param))
        s += '\n}'
        return s

    def get_fixed_pidx(self, param_name):
        """Returns the parameter index of the given fixed parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter.

        Returns
        -------
        pidx : int
            The index of the fixed parameter.

        Raises
        ------
        KeyError
            If the given parameter is not part of the set of fixed parameters.
        """
        return self._fixed_param_name_to_idx[param_name]

    def get_floating_pidx(self, param_name):
        """Returns the parameter index of the given floating parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter.

        Returns
        -------
        pidx : int
            The index of the floating parameter.

        Raises
        ------
        KeyError
            If the given parameter is not part of the set of floating
            parameters.
        """
        return self._floating_param_name_to_idx[param_name]

    def has_fixed_param(self, param_name):
        """Checks if this ParameterSet instance has a fixed parameter named
        ``param_name``.

        Parameters
        ----------
        param_name : str
            The name of the parameter.

        Returns
        -------
        check : bool
            ``True`` if this ParameterSet instance has a fixed parameter
            of the given name, ``False`` otherwise.
        """
        return (param_name in self._fixed_param_name_list)

    def has_floating_param(self, param_name):
        """Checks if this ParameterSet instance has a floating parameter named
        ``param_name``.

        Parameters
        ----------
        param_name : str
            The name of the parameter.

        Returns
        -------
        check : bool
            ``True`` if this ParameterSet instance has a floating parameter
            of the given name, ``False`` otherwise.
        """
        return (param_name in self._floating_param_name_list)

    def fix_params(self, fix_params):
        """Fixes the given parameters to the given values.

        Parameters
        ----------
        fix_params : dict
            The dictionary defining the parameters that should get fixed to the
            given dictionary entry values.
        """
        fix_params_keys = fix_params.keys()
        self._fixed_param_name_list = []
        self._floating_param_name_list = []
        self._fixed_param_name_to_idx = dict()
        self._floating_param_name_to_idx = dict()
        self._fixed_param_values = np.empty((0,), dtype=np.float)
        for (pidx, param) in enumerate(self._params):
            pname = param.name
            if(pname in fix_params_keys):
                # The parameter of name `pname` should get fixed.
                if(param.isfixed is True):
                    raise ValueError('The parameter "%s" is already a fixed '
                        'parameter!'%(pname))
                initial = fix_params[pname]
                param.make_fixed(initial)
                self._params_fixed_mask[pidx] = True
                self._fixed_param_name_list += [ pname ]
                self._fixed_param_values = np.concatenate(
                    (self._fixed_param_values, [param.value]))
                self._fixed_param_name_to_idx[pname] = len(
                    self._fixed_param_name_list) - 1
            else:
                if(param.isfixed):
                    self._fixed_param_name_list += [ pname ]
                    self._fixed_param_values = np.concatenate(
                        (self._fixed_param_values, [param.value]))
                    self._fixed_param_name_to_idx[pname] = len(
                        self._fixed_param_name_list) - 1
                else:
                    self._floating_param_name_list += [ pname ]
                    self._floating_param_name_to_idx[pname] = len(
                        self._floating_param_name_list) - 1

    def float_params(self, float_params):
        """Makes the given parameters floating with the given initial value and
        within the given bounds.

        Parameters
        ----------
        float_params : dict
            The dictionary defining the parameters that should get set to be
            floating. The format of a dictionary's entry can be one of the
            following formats:

                - None
                    The parameter's initial, minimal and maximal value should be
                    taken from the parameter's current settings.
                - initial : float
                    The parameter's initial value should be set to the given
                    value. The minimal and maximal values of the parameter will
                    be taken from the parameter's current settings.
                - (initial, valmin, valmax)
                    The parameter's initial value, minimal and maximal value
                    should be set to the given values. If `initial` is set to
                    `None`, the parameter's current value will be used as
                    initial value.
        """
        def _parse_float_param_dict_entry(e):
            """Parses the given float_param dictionary entry into initial,
            valmin, and valmax values.
            """
            if(e is None):
                return (None, None, None)
            if(issequence(e)):
                return (e[0], e[1], e[2])
            return (e, None, None)

        float_params_keys = float_params.keys()
        self._fixed_param_name_list = []
        self._floating_param_name_list = []
        self._fixed_param_name_to_idx = dict()
        self._floating_param_name_to_idx = dict()
        self._fixed_param_values = np.empty((0,), dtype=np.float)
        for (pidx, param) in enumerate(self._params):
            pname = param.name
            if(pname in float_params_keys):
                # The parameter of name `pname` should get set floating.
                if(param.isfixed is False):
                    raise ValueError('The parameter "%s" is already a floating '
                        'parameter!'%(pname))
                (initial, valmin, valmax) = _parse_float_param_dict_entry(
                    float_params[pname])
                param.make_floating(initial, valmin, valmax)
                self._params_fixed_mask[pidx] = False
                self._floating_param_name_list += [ pname ]
                self._floating_param_name_to_idx[pname] = len(
                    self._floating_param_name_list) - 1
            else:
                if(param.isfixed):
                    self._fixed_param_name_list += [ pname ]
                    self._fixed_param_values = np.concatenate(
                        (self._fixed_param_values, [param.value]))
                    self._fixed_param_name_to_idx[pname] = len(
                        self._fixed_param_name_list) - 1
                else:
                    self._floating_param_name_list += [ pname ]
                    self._floating_param_name_to_idx[pname] = len(
                        self._floating_param_name_list) - 1

    def update_fixed_param_value_cache(self):
        """Updates the internal cache of the fixed parameter values. This method
        has to be called whenever the values of the fixed Parameter instances
        change.
        """
        for (i, param) in enumerate(self.fixed_params):
            self._fixed_param_values[i] = param.value

    def copy(self):
        """Creates a deep copy of this ParameterSet instance.

        Returns
        -------
        copy : ParameterSet instance
            The copied instance of this ParameterSet instance.
        """
        copy = deepcopy(self)
        return copy

    def add_param(self, param, atfront=False):
        """Adds the given Parameter instance to this set of parameters.

        Parameters
        ----------
        param : instance of Parameter
            The parameter, which should get added.
        atfront : bool
            Flag if the parameter should be added at the front of the parameter
            list. If set to False (default), it will be added at the back.

        Returns
        -------
        self : instance of ParameterSet
            This ParameterSet instance so that multiple add_param calls can just
            be concatenated.

        Raises
        ------
        KeyError
            If given parameter was already added to the set.
        """
        if(not isinstance(param, Parameter)):
            raise TypeError('The param argument must be an instance of '
                'Parameter!')

        if(self.has_param(param)):
            raise KeyError('The parameter named "%s" was already added to the '
                'parameter set!'%(param.name))

        param_fixed_mask = True if param.isfixed else False

        if(atfront):
            # Add parameter at front of parameter list.
            self._params = np.concatenate(
                ([param], self._params))
            self._params_fixed_mask = np.concatenate(
                ([param_fixed_mask], self._params_fixed_mask))
            if(param.isfixed):
                self._fixed_param_name_list = (
                    [param.name] + self._fixed_param_name_list)
                self._fixed_param_values = np.concatenate(
                    ([param.value], self._fixed_param_values))
                # Shift the index of all fixed parameters.
                self._fixed_param_name_to_idx = dict([ (k,v+1)
                    for (k,v) in self._fixed_param_name_to_idx.items() ])
                self._fixed_param_name_to_idx[param.name] = 0
            else:
                self._floating_param_name_list = (
                    [param.name] + self._floating_param_name_list)
                # Shift the index of all floating parameters.
                self._floating_param_name_to_idx = dict([ (k,v+1)
                    for (k,v) in self._floating_param_name_to_idx.items() ])
                self._floating_param_name_to_idx[param.name] = 0
        else:
            # Add parameter at back of parameter list.
            self._params = np.concatenate(
                (self._params, [param]))
            self._params_fixed_mask = np.concatenate(
                (self._params_fixed_mask, [param_fixed_mask]))
            if(param.isfixed):
                self._fixed_param_name_list = (
                    self._fixed_param_name_list + [param.name])
                self._fixed_param_values = np.concatenate(
                    (self._fixed_param_values, [param.value]))
                self._fixed_param_name_to_idx[param.name] = len(
                    self._fixed_param_name_list) - 1
            else:
                self._floating_param_name_list = (
                    self._floating_param_name_list + [param.name])
                self._floating_param_name_list[param.name] = len(
                    self._floating_param_name_list) - 1

        return self

    def has_param(self, param):
        """Checks if the given Parameter is already present in this ParameterSet
        instance. The check is performed based on the parameter name.

        Parameters
        ----------
        param : Parameter instance
            The Parameter instance that should be checked.

        Returns
        -------
        check : bool
            ``True`` if the given parameter is present in this parameter set,
            ``False`` otherwise.
        """
        if((param.name in self._floating_param_name_list) or
           (param.name in self._fixed_param_name_list)):
            return True

        return False

    def floating_param_values_to_dict(self, floating_param_values):
        """Converts the given floating parameter values into a dictionary with
        the floating parameter names and values and also adds the fixed
        parameter names and their values to this dictionary.

        Parameters
        ----------
        floating_param_values : 1D ndarray
            The ndarray holding the values of the floating parameters in the
            order that the floating parameters are defined.

        Returns
        -------
        param_dict : dict
            The dictionary with the floating and fixed parameter names and
            values.
        """
        param_dict = dict(
            zip(self._floating_param_name_list, floating_param_values) +
            zip(self._fixed_param_name_list, self._fixed_param_values))

        return param_dict


class ParameterGrid(object):
    """This class provides a data holder for a parameter that has a set of
    discrete values on a grid. Thus, the parameter has a value grid. By default
    the grid is aligned with zero, but this can be changed by setting the offset
    value to a value other than zero.
    """
    def __init__(self, name, grid, delta, offset=0):
        """Creates a new parameter grid.

        Parameters
        ----------
        name : str
            The name of the parameter.
        grid : numpy.ndarray
            The numpy ndarray holding the discrete grid values of the parameter.
        delta : float
            The width between the grid values.
        offset : float
            The offset from zero to align the grid other than with zero.
            By definition the absolute value of the offset must be smaller than
            `delta`.
        """
        self.name = name
        self.delta = delta
        self.offset = offset

        # Setting the grid, will automatically round the grid values to their
        # next nearest grid value. Hence, we need to set the grid property after
        # setting the delta and offser properties.
        self.grid = grid

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
            raise TypeError('The grid property must be of type numpy.ndarray!')
        if(arr.ndim != 1):
            raise ValueError('The grid property must be a 1D numpy.ndarray!')
        self._grid = self.round_to_nearest_grid_point(arr)

    @property
    def delta(self):
        """The width (float) between the grid values.
        """
        return self._delta
    @delta.setter
    def delta(self, v):
        v = float_cast(v, 'The delta property must be castable to type float!')
        self._delta = v

    @property
    def offset(self):
        """The offset from zero to align the grid other than with zero.
        By definition the absolute value of the offset must be smaller than
        `delta`.
        """
        return self._offset
    @offset.setter
    def offset(self, v):
        v = float_cast(v, 'The offset property must be castable to type float!')
        if(np.abs(v) >= self._delta):
            raise ValueError('The absolute value of the offset property (%f) '
                'must be smaller than the value of the delta property (%f)'%(
                v, self._delta))
        self._offset = v

    @property
    def ndim(self):
        """The dimensionality of the parameter grid.
        """
        return self._grid.ndim

    def add_extra_lower_and_upper_bin(self):
        """Adds an extra lower and upper bin to this parameter grid. This is
        usefull when interpolation or gradient methods require an extra bin on
        each side of the grid.
        """
        newgrid = np.empty((self._grid.size+2,))
        newgrid[1:-1] = self._grid
        newgrid[0] = newgrid[1] - self._delta
        newgrid[-1] = newgrid[-2] + self._delta
        self.grid = newgrid

    def copy(self):
        """Copies this ParameterGrid object and returns the copy.
        """
        copy = deepcopy(self)
        return copy

    def round_to_nearest_grid_point(self, value):
        """Rounds the given value to the nearest grid point.

        Parameters
        ----------
        value : float | ndarray of float
            The value(s) to round.

        Returns
        -------
        grid_point : float | ndarray of float
            The calculated grid point(s).
        """
        return np.around(value / self._delta) * self._delta + self._offset

    def round_to_lower_grid_point(self, value):
        """Rounds the given value to the nearest grid point that is lower than
        the given value.

        Parameters
        ----------
        value : float | ndarray of float
            The value(s) to round.

        Returns
        -------
        grid_point : float | ndarray of float
            The calculated grid point(s).
        """
        return np.floor_divide(value, self._delta) * self._delta + self._offset

    def round_to_upper_grid_point(self, value):
        """Rounds the given value to the nearest grid point that is larger than
        the given value.

        Parameters
        ----------
        value : float | ndarray of float
            The value(s) to round.

        Returns
        -------
        grid_point : float | ndarray of float
            The calculated grid point(s).
        """
        return np.ceil(value / self._delta) * self._delta + self._offset


class ParameterGridSet(ObjectCollection):
    """Describes a set of parameter grids.
    """
    def __init__(self, param_grid_list=None):
        """Constructs a new ParameterGridSet object.

        Parameters
        ----------
        param_grid_list : list of ParameterGrid | ParameterGrid | None
            The list of ParameterGrid objects with which this set should get
            initialized with.
        """
        super(ParameterGridSet, self).__init__(
            obj_type=ParameterGrid, obj_list=param_grid_list)

    @property
    def ndim(self):
        """The dimensionality of this parameter grid set. By definition it's the
        number of parameters of the set.
        """
        return len(self)

    @property
    def parameter_names(self):
        """(read-only) The list of the parameter names.
        """
        return [ paramgrid.name for paramgrid in self.objects ]

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

    def add_extra_lower_and_upper_bin(self):
        """Adds an extra lower and upper bin to all the parameter grids. This is
        usefull when interpolation or gradient methods require an extra bin on
        each side of the grid.
        """
        for paramgrid in self.objects:
            paramgrid.add_extra_lower_and_upper_bin()

    def copy(self):
        """Copies this ParameterGridSet object and returns the copy.
        """
        copy = deepcopy(self)
        return copy


class FitParameter(object):
    """This class is DEPRECATED! Use class Parameter instead!

    This class describes a single fit parameter. A fit parameter has a name,
    a value range, an initial value, and a current value. The current value will
    be updated in the fitting process.
    """
    def __init__(self, name, valmin, valmax, initial):
        """Creates a new fit parameter object.

        Parameters
        ----------
        name : str
            The name of the fit parameter.
        valmin : float
            The minimal bound value of the fit parameter.
        valmax : float
            The maximal bound value of the fit parameter.
        initial : float
            The (initial) value (guess) of the parameter, which will be used as
            start point for the fitting procedure.
        """
        self.name = name
        self.valmin = valmin
        self.valmax = valmax
        self.initial = initial

        self.value = self.initial

    @property
    def name(self):
        """The name of the fit parameter.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

    @property
    def valmin(self):
        """The minimal bound value of the fit parameter.
        """
        return self._valmin
    @valmin.setter
    def valmin(self, v):
        v = float_cast(v, 'The valmin property must castable to type float!')
        self._valmin = v

    @property
    def valmax(self):
        """The maximal bound value of the fit parameter.
        """
        return self._valmax
    @valmax.setter
    def valmax(self, v):
        v = float_cast(v, 'The valmax property must be castable to type float!')
        self._valmax = v

    @property
    def initial(self):
        """The initial value of the fit parameter.
        """
        return self._initial
    @initial.setter
    def initial(self, v):
        v = float_cast(v, 'The initial property must be castable to type float!')
        self._initial = v

    def as_linear_grid(self, delta):
        """Creates a ParameterGrid instance with a linear grid with constant
        grid value distances delta.

        Parameters
        ----------
        delta : float
            The constant distance between the grid values. By definition this
            defines also the precision of the parameter values.

        Returns
        -------
        grid : ParameterGrid instance
            The ParameterGrid instance holding the grid values.
        """
        delta = float_cast(delta, 'The delta argument must be castable to type float!')
        grid = make_linear_parameter_grid_1d(
            self._name, self._valmin, self._valmax, delta)
        return grid


class FitParameterSet(object):
    """This class is DEPRECATED, use ParameterSet instead!

    This class describes a set of FitParameter instances.
    """
    def __init__(self):
        """Constructs a fit parameter set instance.
        """
        # Define the list of fit parameters.
        # Define the (N_fitparams,)-shaped numpy array of FitParameter objects.
        self._fitparams = np.empty((0,), dtype=np.object)
        # Define a list for the fit parameter names. This is for optimization
        # purpose only.
        self._fitparam_name_list = []

    @property
    def fitparams(self):
        """The 1D ndarray holding the FitParameter instances.
        """
        return self._fitparams

    @property
    def fitparam_list(self):
        """(read-only) The list of the global FitParameter instances.
        """
        return list(self._fitparams)

    @property
    def fitparam_name_list(self):
        """(read-only) The list of the fit parameter names.
        """
        return self._fitparam_name_list

    @property
    def initials(self):
        """(read-only) The 1D ndarray holding the initial values of all the
        global fit parameters.
        """
        return np.array([ fitparam.initial
                         for fitparam in self._fitparams ], dtype=np.float)

    @property
    def bounds(self):
        """(read-only) The 2D (N_fitparams,2)-shaped ndarray holding the
        boundaries for all the global fit parameters.
        """
        return np.array([ (fitparam.valmin, fitparam.valmax)
                         for fitparam in self._fitparams ], dtype=np.float)

    def copy(self):
        """Creates a deep copy of this FitParameterSet instance.

        Returns
        -------
        copy : FitParameterSet instance
            The copied instance of this FitParameterSet instance.
        """
        copy = deepcopy(self)
        return copy

    def add_fitparam(self, fitparam, atfront=False):
        """Adds the given FitParameter instance to the list of fit parameters.

        Parameters
        ----------
        fitparam : instance of FitParameter
            The fit parameter, which should get added.
        atfront : bool
            Flag if the fit parameter should be added at the front of the
            parameter list. If set to False (default), it will be added at the
            back.
        """
        if(not isinstance(fitparam, FitParameter)):
            raise TypeError('The fitparam argument must be an instance of FitParameter!')

        if(atfront):
            # Add fit parameter at front of list.
            self._fitparams = np.concatenate(([fitparam], self._fitparams))
            self._fitparam_name_list = [fitparam.name] + self._fitparam_name_list
        else:
            # Add fit parameter at back of list.
            self._fitparams = np.concatenate((self._fitparams, [fitparam]))
            self._fitparam_name_list = self._fitparam_name_list + [fitparam.name]

    def fitparam_values_to_dict(self, fitparam_values):
        """Converts the given fit parameter values into a dictionary with the
        fit parameter names and values.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The ndarray holding the fit parameter values in the order that the
            fit parameters are defined.

        Returns
        -------
        fitparam_dict : dict
            The dictionary with the fit parameter names and values.
        """
        fitparam_dict = dict(zip(self._fitparam_name_list, fitparam_values))
        return fitparam_dict

    def fitparam_dict_to_values(self, fitparam_dict):
        """Converts the given fit parameter dictionary into a 1D ndarray holding
        the fit parameter values in the order the fit parameters are defined.

        Parameters
        ----------
        fitparam_dict : dict
            The dictionary with the fit parameter names and values.

        Returns
        -------
        fitparam_values : 1D ndarray
            The ndarray holding the fit parameter values in the order that the
            fit parameters are defined.
        """
        fitparam_values = np.empty_like(self._fitparams, dtype=np.float)
        for (i, fitparam) in enumerate(self._fitparams):
            fitparam_values[i] = fitparam_dict[fitparam.name]
        return fitparam_values

    def generate_random_initials(self, rss):
        """Generates a set of random initials for all global fit parameters.
        A new random initial is defined as

            lower_bound + RAND * (upper_bound - lower_bound),

        where RAND is a uniform random variable between 0 and 1.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used for drawing
            random numbers from.
        """
        vb = self.bounds
        # Do random_initial = lower_bound + RAND * (upper_bound - lower_bound)
        ri = vb[:,0] + rss.random.uniform(size=vb.shape[0])*(vb[:,1] - vb[:,0])

        return ri


class SourceFitParameterMapper(object):
    """This abstract base class defines the interface of the source fit
    parameter mapper. This mapper provides the functionality to map a global fit
    parameter to a source fit parameter.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Constructor of the source fit parameter mapper.
        """
        self._fitparamset = FitParameterSet()

        # Define the list of source parameter names, which map to the fit
        # parameters.
        # Define the (N_fitparams,)-shaped numpy array of str objects.
        self._src_param_names = np.empty((0,), dtype=np.object)

    @property
    def fitparamset(self):
        """(read-only) The FitParameterSet instance holding the list of global
        fit parameters.
        """
        return self._fitparamset

    @property
    def n_global_fitparams(self):
        """(read-only) The number of defined global fit parameters.
        """
        return len(self._fitparamset.fitparams)

    def get_src_fitparam_name(self, fitparam_idx):
        """Returns the name of the source fit parameter for the given global fit
        parameter index.

        Parameters
        ----------
        fitparam_idx : int
            The index of the global fit parameter.

        Returns
        -------
        src_fitparam_name : str
            The name of the source fit parameter.
        """
        return self._src_param_names[fitparam_idx]

    @abc.abstractmethod
    def def_fit_parameter(self, fit_param, src_param_name=None, sources=None):
        """This method is supposed to define a new fit parameter that maps to a
        given source fit parameter for a list of sources. If no list of sources
        is given, it maps to all sources.

        Parameters
        ----------
        fit_param : FitParameter
            The FitParameter instance defining the fit parameter.
        src_param_name : str | None
            The name of the source parameter. It must match the name of a source
            model property. If set to None (default) the name of the fit
            parameter will be used.
        sources : sequence of SourceModel | None
            The sequence of SourceModel instances for which the fit parameter
            applies. If None (the default) is specified, the fit parameter will
            apply to all sources.
        """
        pass

    @abc.abstractmethod
    def get_src_fitparams(self, fitparam_values, src_idx=0):
        """This method is supposed to create a dictionary of source fit
        parameter names and values for the requested source based on the given
        fit parameter values.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.
        src_idx : int
            The index of the source for which the parameters should get
            retrieved.

        Returns
        -------
        src_fitparams : dict
            The dictionary holding the translated source parameters that are
            beeing fitted.
        """
        pass

    @abc.abstractmethod
    def get_fitparams_array(self, fitparam_values):
        """This method is supposed to create a numpy record ndarray holding the
        unique source fit parameter names as key and their value for each
        source. The returned array must be (N_sources,)-shaped.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.

        Returns
        -------
        fitparams_arr : (N_sources,)-shaped numpy record ndarray | None
            The numpy record ndarray holding the fit parameter names as keys
            and their value for each source in each row.
            None must be returned if no global fit parameters were defined.
        """
        pass


class SingleSourceFitParameterMapper(SourceFitParameterMapper):
    """This class provides the functionality to map the global fit parameters to
    the source fit parameters of the single source. This class assumes a single
    source, hence the mapping can be performed faster than in the multi-source
    case.
    """
    def __init__(self):
        """Constructs a new source fit parameter mapper for a single source.
        """
        super(SingleSourceFitParameterMapper, self).__init__()

    def def_fit_parameter(self, fitparam, src_param_name=None):
        """Define a new fit parameter that maps to a given source fit parameter.

        Parameters
        ----------
        fitparam : FitParameter
            The FitParameter instance defining the fit parameter.
        src_param_name : str | None
            The name of the source parameter. It must match the name of a source
            model property. If set to None (default) the name of the fit
            parameter will be used.
        """
        self._fitparamset.add_fitparam(fitparam)

        if(src_param_name is None):
            src_param_name = fitparam.name
        if(not isinstance(src_param_name, str)):
            raise TypeError('The src_param_name argument must be of type str!')

        # Append the source parameter name to the internal array.
        self._src_param_names = np.concatenate((self._src_param_names, [src_param_name]))

    def get_src_fitparams(self, fitparam_values):
        """Create a dictionary of source fit parameter names and values based on
        the given fit parameter values.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.

        Returns
        -------
        src_fitparams : dict
            The dictionary holding the translated source parameters that are
            beeing fitted.
            An empty dictionary is returned if no fit parameters were defined.
        """
        src_fitparams = dict(zip(self._src_param_names, fitparam_values))

        return src_fitparams

    def get_fitparams_array(self, fitparam_values):
        """Creates a numpy record ndarray holding the fit parameters names as
        key and their value for each source. The returned array is (1,)-shaped
        since there is only one source defined for this mapper class.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.

        Returns
        -------
        fitparams_arr : (1,)-shaped numpy record ndarray | None
            The numpy record ndarray holding the fit parameter names as keys
            and their value for the one single source.
            None is returned if no fit parameters were defined.
        """
        if(self.n_global_fitparams == 0):
            return None

        fitparams_arr = np.array([tuple(fitparam_values)],
                                 dtype=[ (name, np.float)
                                        for name in self._src_param_names ])
        return fitparams_arr


class MultiSourceFitParameterMapper(SourceFitParameterMapper):
    """This class provides the functionality to map the global fit parameters to
    the source fit parameters of the sources.
    Sometimes it's necessary to define a global fit parameter, which relates to
    a source model fit parameter for a set of sources, while another global fit
    parameter relates to the same source model fit parameter, but for another
    set of sources.

    At construction time this manager takes the collection of sources. Each
    source gets an index, which is defined as the position of the source within
    the collection.
    """
    def __init__(self, sources):
        """Constructs a new source fit parameter mapper for multiple sources.

        Parameters
        ----------
        sources : sequence of SourceModel
            The sequence of SourceModel instances defining the list of sources.
        """
        super(MultiSourceFitParameterMapper, self).__init__()

        self.sources = sources

        # (N_fitparams, N_sources) shaped boolean ndarray defining what fit
        # parameter applies to which source.
        self._fit_param_2_src_mask = np.zeros((0, len(self.sources)), dtype=np.bool)

        # Define an array, which will hold the unique source parameter names.
        self._unique_src_param_names = np.empty((0,), dtype=np.object)

    @property
    def sources(self):
        """The SourceCollection defining the sources.
        """
        return self._sources
    @sources.setter
    def sources(self, obj):
        obj = SourceCollection.cast(obj, 'The sources property must be castable to an instance of SourceCollection!')
        self._sources = obj

    @property
    def N_sources(self):
        """(read-only) The number of sources.
        """
        return len(self._sources)

    def def_fit_parameter(self, fitparam, src_param_name=None, sources=None):
        """Defines a new fit parameter that maps to a given source parameter
        for a list of sources. If no list of sources is given, it maps to all
        sources.

        Parameters
        ----------
        fitparam : FitParameter
            The FitParameter instance defining the fit parameter.
        src_param_name : str | None
            The name of the source parameter. It must match the name of a source
            model property. If set to None (default) the name of the fit
            parameter will be used.
        sources : SourceCollection | None
            The instance of SourceCollection with the sources for which the fit
            parameter applies. If None (the default) is specified, the fit
            parameter will apply to all sources.
        """
        self._fitparamset.add_fitparam(fitparam)

        if(src_param_name is None):
            src_param_name = fitparam.name
        if(not isinstance(src_param_name, str)):
            raise TypeError('The src_param_name argument must be of type str!')

        if(sources is None):
            sources = self.sources
        sources = SourceCollection.cast(sources,
            'The sources argument must be castable to an instance of SourceCollection!')

        # Append the source parameter name to the internal array and keep track
        # of the unique names.
        self._src_param_names = np.concatenate((self._src_param_names, [src_param_name]))
        self._unique_src_param_names = np.unique(self._src_param_names)

        # Get the list of source indices for which the fit parameter applies.
        mask = np.zeros((len(self.sources),), dtype=np.bool)
        for ((idx,src), applied_src) in itertools.product(enumerate(self.sources), sources):
            if(applied_src.id == src.id):
                mask[idx] = True
        self._fit_param_2_src_mask = np.vstack((self._fit_param_2_src_mask, mask))

    def get_src_fitparams(self, fitparam_values, src_idx):
        """Constructs a dictionary with the source parameters that are beeing
        fitted. As values the given global fit parameter values will be used.
        Hence, this method translates the global fit parameter values into the
        source parameters.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.
        src_idx : int
            The index of the source for which the parameters should get
            retieved.

        Returns
        -------
        src_fitparams : dict
            The dictionary holding the translated source parameters that are
            beeing fitted.
        """
        # Get the mask of global fit parameters that apply to the requested
        # source.
        fp_mask = self._fit_param_2_src_mask[:,src_idx]

        # Get the source parameter names and values.
        src_param_names = self._src_param_names[fp_mask]
        src_param_values = fitparam_values[fp_mask]

        src_fitparams = dict(zip(src_param_names, src_param_values))

        return src_fitparams

    def get_fitparams_array(self, fitparam_values):
        """Creates a numpy record ndarray holding the fit parameters names as
        key and their value for each source. The returned array is
        (N_sources,)-shaped.

        Parameters
        ----------
        fitparam_values : 1D ndarray
            The array holding the current global fit parameter values.

        Returns
        -------
        fitparams_arr : (N_sources,)-shaped numpy record ndarray | None
            The numpy record ndarray holding the unique source fit parameter
            names as keys and their value for each source per row.
            None is returned if no fit parameters were defined.
        """
        if(self.n_global_fitparams == 0):
            return None

        fitparams_arr = np.empty((self.N_sources,),
                                 dtype=[ (name, np.float)
                                         for name in self._unique_src_param_names ])

        for src_idx in range(self.N_sources):
            # Get the mask of global fit parameters that apply to the requested
            # source.
            fp_mask = self._fit_param_2_src_mask[:,src_idx]

            # Get the source parameter names and values.
            src_param_names = self._src_param_names[fp_mask]
            src_param_values = fitparam_values[fp_mask]

            # Fill the fit params array.
            for (name, value) in zip(src_param_names, src_param_values):
                fitparams_arr[name][src_idx] = value

        return fitparams_arr

