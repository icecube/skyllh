# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np
from copy import deepcopy

from skyllh.physics.source import (
    SourceCollection,
    SourceModel
)
from skyllh.core import display
from skyllh.core.model import ModelCollection
from skyllh.core.py import (
    NamedObjectCollection,
    bool_cast,
    classname,
    const,
    float_cast,
    get_number_of_float_decimals,
    issequence,
    issequenceof,
    str_cast
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
    grid = np.arange(low, high+delta, delta)
    return ParameterGrid(name, grid, delta)

def make_params_hash(params):
    """Utility function to create a hash value for a given parameter dictionary.

    Parameters
    ----------
    params : dict | None
        The dictionary holding the parameter (name: value) pairs.
        If set to None, an empty dictionary is used.

    Returns
    -------
    hash : int
        The hash of the parameter dictionary.
    """
    if(params is None):
        params = {}

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
    def __init__(self, name, initial, valmin=None, valmax=None, isfixed=None):
        """Creates a new Parameter instance.

        Parameters
        ----------
        name : str
            The name of the parameter.
        initial : float
            The initial value of the parameter.
        valmin : float | None
            The minimum value of the parameter in case this parameter is
            mutable.
        valmax : float | None
            The maximum value of the parameter in case this parameter is
            mutable.
        isfixed : bool | None
            Flag if the value of this parameter is mutable (False), or not
            (True). If set to `True`, the value of the parameter will always be
            the `initial` value.
            If set to None, the parameter will be mutable if valmin and valmax
            were specified. Otherwise, the parameter is fixed.
            The default is None.
        """
        if(isfixed is None):
            if((valmin is not None) and (valmax is not None)):
                isfixed = False
            else:
                isfixed = True

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
                    v, self.name, self._initial))
        else:
            if((v < self._valmin) or (v > self._valmax)):
                raise ValueError('The value (%f) of parameter "%s" must be '
                    'within the range (%f, %f)!'%(
                    v, self._name, self._valmin, self._valmax))
        self._value = v

    def __eq__(self, other):
        """Implements the equal comparison operator (==).
        By definition two parameters are equal if there property values are
        equal.

        Parameters
        ----------
        other : Parameter instance
            The instance of Parameter which should be used to compare against
            this Parameter instance.

        Returns
        -------
        cmp : bool
            True, if this Parameter instance and the other Parameter instance
            have the same property values.
        """
        if((self.name != other.name) or
           (self.value != other.value) or
           (self.isfixed != other.isfixed)):
            return False

        # If both parameters are floating parameters, also their initial, min,
        # and max values must match.
        if(not self.isfixed):
            if((self.initial != other.initial) or
               (self.valmin != other.valmin) or
               (self.valmax != other.valmax)):
                return False

        return True

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

        Raises
        ------
        ValueError
            If this Parameter instance represents a fixed parameter.
        """
        if(self.isfixed):
            raise ValueError('Cannot create a linear grid from the fixed '
                'parameter "%s". The parameter must be floating!'%(self.name))

        delta = float_cast(delta, 'The delta argument must be castable to type '
            'float!')
        grid = make_linear_parameter_grid_1d(
            self._name, self._valmin, self._valmax, delta)
        return grid

    def change_fixed_value(self, value):
        """Changes the value of this fixed parameter to the given value.

        Parameters
        ----------
        value : float
            The parameter's new value.

        Returns
        -------
        value : float
            The parameter's new value.

        Raises
        ------
        ValueError
            If this parameter is not a fixed parameter.
        """
        if(not self._isfixed):
            raise ValueError('The parameter "%s" is not a fixed parameter!'%(
                self.name))

        self.initial = value
        self.value = value

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
            The parameter's new value.
        """
        self._isfixed = True

        # If no new initial value is given, use the current value.
        if(initial is None):
            self._initial = self._value
            return self._value

        self.initial = initial
        self._value = self._initial

        # Undefine the valmin and valmax values if the parameter's new value is
        # outside the valmin and valmax range.
        if((self._valmin is not None) and (self._valmax is not None) and
           ((self._value < self._valmin) or (self._value > self._valmax))):
            self._valmin = None
            self._valmax = None

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

        Raises
        ------
        ValueError
            If valmin is set to None and this parameter has no valmin defined.
            If valmax is set to None and this parameter has no valmax defined.
        """
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

        self._isfixed = False
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
        self._params = np.empty((0,), dtype=np.object_)
        # Define the (n_params,)-shaped numpy mask array that masks the fixed
        # parameters in the list of all parameters.
        self._params_fixed_mask = np.empty((0,), dtype=np.bool_)

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
        self._fixed_param_values = np.empty((0,), dtype=np.float64)

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
    def fixed_params_mask(self):
        """(read-only) The 1D ndarray holding the mask for the fixed parameters
        of this parameter set.
        """
        return self._params_fixed_mask

    @property
    def floating_params(self):
        """(read-only) The 1D ndarray holding the Parameter instances,
        whose values are floating.
        """
        return self._params[np.invert(self._params_fixed_mask)]

    @property
    def floating_params_mask(self):
        """(read-only) The 1D ndarray holding the mask for the floating
        parameters of this parameter set.
        """
        return np.invert(self._params_fixed_mask)

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
            return np.empty((0,), dtype=np.float64)
        return np.array(
            [ param.initial
             for param in floating_params ], dtype=np.float64)

    @property
    def floating_param_bounds(self):
        """(read-only) The 2D (n_floating_params,2)-shaped ndarray holding the
        boundaries for all the floating parameters.
        """
        floating_params = self.floating_params
        if(len(floating_params) == 0):
            return np.empty((0,2), dtype=np.float64)
        return np.array(
            [ (param.valmin, param.valmax)
             for param in floating_params ], dtype=np.float64)

    def __iter__(self):
        """Returns an iterator over the Parameter instances of this ParameterSet
        instance.
        """
        return iter(self._params)

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

    def make_params_fixed(self, fix_params):
        """Fixes the given parameters to the given values.

        Parameters
        ----------
        fix_params : dict
            The dictionary defining the parameters that should get fixed to the
            given dictionary entry values.

        Raises
        ------
        ValueError
            If one of the given parameters is already a fixed parameter.
        """
        fix_params_keys = fix_params.keys()
        self._fixed_param_name_list = []
        self._floating_param_name_list = []
        self._fixed_param_name_to_idx = dict()
        self._floating_param_name_to_idx = dict()
        self._fixed_param_values = np.empty((0,), dtype=np.float64)
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

    def make_params_floating(self, float_params):
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

        Raises
        ------
        ValueError
            If one of the given parameters is already a floating parameter.
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
        self._fixed_param_values = np.empty((0,), dtype=np.float64)
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
        TypeError
            If param is not an instance of Parameter.
        KeyError
            If given parameter is already present in the set. The check is
            performed based on the parameter name.
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
                self._floating_param_name_to_idx[param.name] = len(
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
            list(zip(self._floating_param_name_list, floating_param_values)) +
            list(zip(self._fixed_param_name_list, self._fixed_param_values)))

        return param_dict


class ParameterSetArray(object):
    """This class provides a data holder for an array of ParameterSet instances.
    Given an array of global floating parameter values, it can split that array
    into floating parameter value sub arrays, one for each ParameterSet instance
    of this ParameterSetArray instance. This functionality is required in
    order to be able to map the global floating parameter values from the
    minimizer to their parameter names.
    """
    def __init__(self, paramsets):
        """Creates a new ParameterSetArray instance, which will hold a list of
        constant ParameterSet instances.

        Parameters
        ----------
        paramsets : const instance of ParameterSet | sequence of const instances
                of ParameterSet
            The sequence of constant ParameterSet instances holding the global
            parameters.

        Raises
        ------
        TypeError
            If the given paramsets argument ist not a sequence of constant
            instances of ParameterSet.
        """
        super(ParameterSetArray, self).__init__()

        if(isinstance(paramsets, ParameterSet)):
            paramsets = [paramsets]
        if(not issequenceof(paramsets, ParameterSet, const)):
            raise TypeError('The paramsets argument must be a constant '
                'instance of ParameterSet or a sequence of constant '
                'ParameterSet instances!')
        self._paramset_list = list(paramsets)

        # Calculate the total number of parameters hold by this
        # ParameterSetArray instance.
        self._n_params = np.sum([paramset.n_params
            for paramset in self._paramset_list])

        # Calculate the total number of fixed parameters hold by this
        # ParameterSetArray instance.
        self._n_fixed_params = np.sum([paramset.n_fixed_params
            for paramset in self._paramset_list])

        # Calculate the total number of floating parameters hold by this
        # ParameterSetArray instance.
        self._n_floating_params = np.sum([paramset.n_floating_params
            for paramset in self._paramset_list])

        # Determine the array of initial values of all floating parameters.
        self._floating_param_initials = np.concatenate([
            paramset.floating_param_initials
            for paramset in self._paramset_list])

        # Determine the array of bounds of all floating parameters.
        self._floating_param_bounds = np.concatenate([
            paramset.floating_param_bounds
            for paramset in self._paramset_list])

    @property
    def paramset_list(self):
        """(read-only) The list of ParameterSet instances holding the global
        parameters.
        """
        return self._paramset_list

    @property
    def n_params(self):
        """(read-only) The total number of parameters hold by this
        ParameterSetArray instance.
        """
        return self._n_params

    @property
    def n_fixed_params(self):
        """(read-only) The total number of fixed parameters hold by this
        ParameterSetArray instance.
        """
        return self._n_fixed_params

    @property
    def n_floating_params(self):
        """(read-only) The total number of floating parameters hold by this
        ParameterSetArray instance.
        """
        return self._n_floating_params

    @property
    def floating_param_initials(self):
        """(read-only) The 1D (n_floating_params,)-shaped ndarray holding the
        initial values of all the floating parameters.
        """
        return self._floating_param_initials

    @property
    def floating_param_bounds(self):
        """(read-only) The 2D (n_floating_params,2)-shaped ndarray holding the
        boundaries for all the floating parameters.
        """
        return self._floating_param_bounds

    def __str__(self):
        """Creates and returns a pretty string representation of this
        ParameterSetArray instance.
        """
        s = '%s: %d parameters (%d floating, %d fixed) {\n'%(
            classname(self), self.n_params, self.n_floating_params,
            self.n_fixed_params)

        for (idx,paramset) in enumerate(self._paramset_list):
            if(idx > 0):
                s += '\n'
            s += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH,
                str(paramset))

        s += '\n}'

        return s

    def generate_random_initials(self, rss):
        """Generates a set of random initials for all global floating
        parameters.
        A new random initial is defined as

            lower_bound + RAND * (upper_bound - lower_bound),

        where RAND is a uniform random variable between 0 and 1.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used for drawing
            random numbers from.
        """
        vb = self.floating_param_bounds
        # Do random_initial = lower_bound + RAND * (upper_bound - lower_bound)
        ri = vb[:,0] + rss.random.uniform(size=vb.shape[0])*(vb[:,1] - vb[:,0])

        return ri

    def split_floating_param_values(self, floating_param_values):
        """Splits the given floating parameter values into their specific
        ParameterSet part.

        Parameters
        ----------
        floating_param_values : (n_floating_params,)-shaped 1D ndarray
            The ndarray holding the values of all the floating parameters for
            all ParameterSet instances. The order must match the order of
            ParameterSet instances and their order of floating parameters.

        Returns
        -------
        floating_param_values_list : list of (n_floating_params,)-shaped 1D
                ndarray
            The list of ndarray objects, where each ndarray holds only the
            floating values of the particular ParameterSet instance. The order
            matches the order of ParameterSet instances defined for this
            ParameterSetArray.
        """
        if(len(floating_param_values) != self.n_floating_params):
            raise ValueError('The number of given floating parameter values '
                '(%d) does not match the total number of defined floating '
                'parameters (%d)!'%(len(floating_param_values),
                self.n_floating_params))

        floating_param_values_list = []

        offset = 0
        for paramset in self._paramset_list:
            n_floating_params = paramset.n_floating_params
            floating_param_values_list.append(floating_param_values[
                offset:offset+n_floating_params])
            offset += n_floating_params

        return floating_param_values_list

    def update_fixed_param_value_cache(self):
        """Updates the internal cache of the fixed parameter values. This method
        has to be called whenever the values of the fixed Parameter instances
        change.
        """
        for paramset in self._paramset_list:
            paramset.update_fixed_param_value_cache()


class ParameterGrid(object):
    """This class provides a data holder for a parameter that has a set of
    discrete values on a grid. Thus, the parameter has a value grid.
    This class represents a one-dimensional grid.
    """
    @staticmethod
    def from_BinningDefinition(binning, delta=None, decimals=None):
        """Creates a ParameterGrid instance from a BinningDefinition instance.

        Parameters
        ----------
        binning : BinningDefinition instance
            The BinningDefinition instance that should be used to create the
            ParameterGrid instance from.
        delta : float | None
            The width between the grid values.
            If set to ``None``, the width is taken from the equal-distant
            ``grid`` values.
        decimals : int | None
            The number of decimals the grid values should get rounded to.
            The maximal number of decimals is 16.
            If set to None, the number of decimals will be the maximum of the
            number of decimals of the first grid value and the number of
            decimals of the delta value.

        Returns
        -------
        param_grid : ParameterGrid instance
            The created ParameterGrid instance.
        """
        return ParameterGrid(
            name=binning.name,
            grid=binning.binedges,
            delta=delta,
            decimals=decimals)

    def __init__(self, name, grid, delta=None, decimals=None):
        """Creates a new parameter grid.

        Parameters
        ----------
        name : str
            The name of the parameter.
        grid : sequence of float
            The sequence of float values defining the discrete grid values of
            the parameter.
        delta : float | None
            The width between the grid values.
            If set to ``None``, the width is taken from the equal-distant
            ``grid`` values.
        decimals : int | None
            The number of decimals the grid values should get rounded to.
            The maximal number of decimals is 16.
            If set to None, the number of decimals will be the maximum of the
            number of decimals of the first grid value and the number of
            decimals of the delta value.
        """
        if(delta is None):
            # We need to take the mean of all the "equal" differences in order
            # to smooth out unlucky rounding issues of a particular difference.
            delta = np.mean(np.diff(grid))

        delta = float_cast(delta, 'The delta argument must be castable to '
            'type float!')
        self._delta = np.float64(delta)

        # Determine the number of decimals of delta.
        if(decimals is None):
            decimals_value = get_number_of_float_decimals(grid[0])
            decimals_delta = get_number_of_float_decimals(delta)
            decimals = int(np.max((decimals_value, decimals_delta)))
        if(not isinstance(decimals, int)):
            raise TypeError('The decimals argument must be an instance of '
                'type int!')
        if(decimals > 16):
            raise ValueError('The maximal number of decimals is 16! Maybe you '
                'should consider log-space!?')

        self.name = name
        self._decimals = decimals
        self._delta = np.around(self._delta, self._decimals)
        self.lower_bound = grid[0]

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
    def decimals(self):
        """(read-only) The number of significant decimals of the grid values.
        """
        return self._decimals

    @property
    def grid(self):
        """The numpy.ndarray with the grid values of the parameter.
        """
        return self._grid
    @grid.setter
    def grid(self, arr):
        if(not issequence(arr)):
            raise TypeError('The grid property must be a sequence!')
        if(not isinstance(arr, np.ndarray)):
            arr = np.array(arr, dtype=np.float64)
        if(arr.ndim != 1):
            raise ValueError('The grid property must be a 1D numpy.ndarray!')
        self._grid = self.round_to_nearest_grid_point(arr)

    @property
    def delta(self):
        """(read-only) The width (float) between the grid values.
        """
        return self._delta

    @property
    def lower_bound(self):
        """The lower bound of the parameter grid.
        """
        return self._lower_bound
    @lower_bound.setter
    def lower_bound(self, v):
        v = float_cast(v, 'The lower_bound property must be castable to type '
            'float!')
        self._lower_bound = np.around(np.float64(v), self._decimals)

    @property
    def ndim(self):
        """The dimensionality of the parameter grid.
        """
        return self._grid.ndim

    def _calc_floatD_and_intD(self, value):
        """Calculates the number of delta intervals of the given values counted
        from the lower bound of the grid. It returns its float and integer
        representation.

        Raises
        ------
        ValueError
            If one of the values are below or above the grid range.
        """
        value = np.atleast_1d(value).astype(np.float64)

        if(hasattr(self, '_grid')):
            m = (value >= self._lower_bound) & (value <= self._grid[-1])
            if(not np.all(m)):
                raise ValueError('The following values are outside the range '
                    'of the parameter grid "%s": %s'%(
                        self.name,
                        ','.join(str(v) for v in value[np.invert(m)])))

        floatD = value/self._delta - self._lower_bound/self._delta
        floatD = np.around(floatD, 9)
        intD = floatD.astype(np.int64)

        return (floatD, intD)

    def add_extra_lower_and_upper_bin(self):
        """Adds an extra lower and upper bin to this parameter grid. This is
        usefull when interpolation or gradient methods require an extra bin on
        each side of the grid.
        """
        newgrid = np.empty((self._grid.size+2,))
        newgrid[1:-1] = self._grid
        newgrid[0] = newgrid[1] - self._delta
        newgrid[-1] = newgrid[-2] + self._delta
        self._lower_bound = newgrid[0]
        del self._grid
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
        scalar_input = np.isscalar(value)

        (floatD, intD) = self._calc_floatD_and_intD(value)
        gp = self._lower_bound + (np.around(floatD % 1, 0) + intD)*self._delta
        gp = np.around(gp, self._decimals)

        if(scalar_input):
            return gp.item()
        return gp

    def round_to_lower_grid_point(self, value):
        """Rounds the given value to the nearest grid point that is lower than
        the given value.

        Note: If the given value is a grid point, that grid point will be
              returned!

        Parameters
        ----------
        value : float | ndarray of float
            The value(s) to round.

        Returns
        -------
        grid_point : float | ndarray of float
            The calculated grid point(s).
        """
        scalar_input = np.isscalar(value)

        (floatD, intD) = self._calc_floatD_and_intD(value)
        gp = self._lower_bound + intD*self._delta
        gp = np.around(gp, self._decimals)

        if(scalar_input):
            return gp.item()
        return gp

    def round_to_upper_grid_point(self, value):
        """Rounds the given value to the nearest grid point that is larger than
        the given value.

        Note: If the given value is a grid point, the next grid point will be
              returned!

        Parameters
        ----------
        value : float | ndarray of float
            The value(s) to round.

        Returns
        -------
        grid_point : ndarray of float
            The calculated grid point(s).
        """
        scalar_input = np.isscalar(value)

        (floatD, intD) = self._calc_floatD_and_intD(value)
        gp = self._lower_bound + (intD + 1)*self._delta
        gp = np.around(gp, self._decimals)

        if(scalar_input):
            return gp.item()
        return gp


class ParameterGridSet(NamedObjectCollection):
    """Describes a set of parameter grids.
    """
    def __init__(self, param_grids=None):
        """Constructs a new ParameterGridSet object.

        Parameters
        ----------
        param_grids : sequence of ParameterGrid instances |
                ParameterGrid instance | None
            The ParameterGrid instances this ParameterGridSet instance should
            get initialized with.
        """
        super(ParameterGridSet, self).__init__(
            objs=param_grids, obj_type=ParameterGrid)

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


class ModelParameterMapper(object, metaclass=abc.ABCMeta):
    """This abstract base class defines the interface of a model parameter
    mapper. A model parameter mapper provides the functionality to map a global
    parameter, usually a fit parameter, to a local parameter of a model, e.g.
    to a source, or a background model parameter.
    """

    def __init__(self, name, models):
        """Constructor of the parameter mapper.

        Parameters
        ----------
        name : str
            The name of the model parameter mapper. In practice this is a
            representative name for the set of global parameters this model
            parameter mapper holds. For a two-component signal-background
            likelihood model, "signal", or "background" could be useful names.
        models : sequence of Model instances.
            The sequence of Model instances the parameter mapper can map global
            parameters to.
        """
        super(ModelParameterMapper, self).__init__()

        self.name = name
        self.models = models

        # Create the parameter set for the global parameters.
        self._global_paramset = ParameterSet()

        # Define a (n_global_params,)-shaped numpy ndarray of str objects that
        # will hold the local parameter names of the global parameters as
        # defined by the models.
        # The local model parameter names are the names used by the internal
        # math objects, like PDFs. Thus, the global parameter names can be
        # aliases of such local model parameter names.
        self._model_param_names = np.empty((0,), dtype=np.object_)

        # (N_params, N_models) shaped boolean ndarray defining what global
        # parameter maps to which model.
        self._global_param_2_model_mask = np.zeros(
            (0, len(self._models)), dtype=np.bool_)

    @property
    def name(self):
        """The name of this ModelParameterMapper instance. In practice this is
        a representative name for the set of global parameters this mapper
        holds.
        """
        return self._name
    @name.setter
    def name(self, name):
        name = str_cast(name, 'The name property must be castable to type str!')
        self._name = name

    @property
    def models(self):
        """The ModelCollection instance defining the models the mapper can
        map global parameters to.
        """
        return self._models
    @models.setter
    def models(self, obj):
        obj = ModelCollection.cast(obj, 'The models property must '
            'be castable to an instance of ModelCollection!')
        self._models = obj

    @property
    def global_paramset(self):
        """(read-only) The ParameterSet instance holding the list of global
        parameters.
        """
        return self._global_paramset

    @property
    def n_models(self):
        """(read-only) The number of models the mapper knows about.
        """
        return len(self._models)

    @property
    def n_global_params(self):
        """(read-only) The number of defined global parameters.
        """
        return self._global_paramset.n_params

    @property
    def n_global_fixed_params(self):
        """(read-only) The number of defined global fixed parameters.
        """
        return self._global_paramset.n_fixed_params

    @property
    def n_global_floating_params(self):
        """(read-only) The number of defined global floating parameters.
        """
        return self._global_paramset.n_floating_params

    def __str__(self):
        """Generates and returns a pretty string representation of this model
        parameter mapper.
        """
        n_global_params = self.n_global_params

        # Determine the number of models that have global parameters assigned.
        # Remember self._global_param_2_model_mask is a
        # (n_global_params, n_models)-shaped 2D ndarray.
        n_models = np.sum(np.sum(self._global_param_2_model_mask, axis=0) > 0)

        s = classname(self) + ' "%s": '%(self._name)
        s += '%d global parameter'%(n_global_params)
        s += '' if n_global_params == 1 else 's'
        s += ', '
        s += '%d model'%(n_models)
        s += '' if n_models == 1 else 's'

        if(n_global_params == 0):
            return s

        s1 = 'Parameters:'
        s += '\n' + display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)
        for (pidx,param) in enumerate(self._global_paramset.params):
            model_names = [ self._models[model_idx].name
                for model_idx in np.nonzero(
                    self._global_param_2_model_mask[pidx])[0]
            ]
            if(param.isfixed):
                pstate = 'fixed (%.3f)'%(
                    param.initial)
            else:
                pstate = 'floating (%.3f <= %.3f <= %.3f)'%(
                    param.valmin, param.initial, param.valmax)
            ps = '\n%s [%s] --> %s\n'%(
                param.name, pstate, self._model_param_names[pidx])
            ps1 = 'in models:\n'
            ps1 += '- '
            ps1 += '\n- '.join(model_names)
            ps += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, ps1)
            s += display.add_leading_text_line_padding(
                2*display.INDENTATION_WIDTH, ps)

        return s

    def finalize(self):
        """Finalizes this ModelParameterMapper instance by declaring its
        ParameterSet instance as constant. No new global parameters can be added
        after calling this method.
        """
        self._global_paramset = const(self._global_paramset)

    @abc.abstractmethod
    def def_param(self, param, model_param_name=None, models=None):
        """This method is supposed to add the given Parameter instance to the
        parameter mapper and maps the global parameter to the given sequence of
        models the parameter mapper knows about.

        Parameters
        ----------
        param : instance of Parameter
            The global parameter which should get mapped to one or more models.
        model_param_name : str | None
            The name of the parameter of the model. Hence, the global
            parameter name can be different to the parameter name of the model.
            If `None`, the name of the global parameter will be used as model
            parameter name.
        models : sequence of Model instances
            The sequence of Model instances the parameter should get
            mapped to. The instances in the sequence must match Model instances
            specified at construction of this mapper.
        """
        pass

    @abc.abstractmethod
    def get_model_param_dict(
            self, global_floating_param_values, model_idx=None):
        """This method is supposed to create a dictionary with the fixed and
        floating parameter names and their values for the given model.

        Parameters
        ----------
        global_floating_param_values : 1D ndarray instance
            The ndarray instance holding the current values of the global
            floating parameters.
        model_idx : int | None
            The index of the model as it was defined at construction
            time of this ModelParameterMapper instance.

        Returns
        -------
        model_param_dict : dict
            The dictionary holding the fixed and floating parameter names and
            values of the specified model.
        """
        pass


class SingleModelParameterMapper(ModelParameterMapper):
    """This class provides a model parameter mapper for a single model, like a
    single source, or a single background model.
    """
    def __init__(self, name, model):
        """Constructs a new model parameter mapper for a single model.

        Parameters
        ----------
        name : str
            The name of the model parameter mapper. In practice this is a
            representative name for the set of global parameters this model
            parameter mapper holds. For a two-component signal-background
            likelihood model, "signal", or "background" could be useful names.
        model : instance of Model
            The instance of Model the parameter mapper can map global
            parameters to.
        """
        super(SingleModelParameterMapper, self).__init__(
            name=name, models=model)

    def def_param(self, param, model_param_name=None):
        """Adds the given Parameter instance to the parameter mapper.

        Parameters
        ----------
        param : instance of Parameter
            The global parameter which should get mapped to the single model.
        model_param_name : str | None
            The parameter name of the model. Hence, the global parameter name
            can be different to the parameter name of the model.
            If set to `None`, the name of the global parameter will be used as
            model parameter name.

        Returns
        -------
        self : SingleModelParameterMapper
            The instance of this SingleModelParameterMapper, so that several
            `def_param` calls can be concatenated.

        Raises
        ------
        KeyError
            If there is already a model parameter with the given name defined.
        """
        if(model_param_name is None):
            model_param_name = param.name
        if(not isinstance(model_param_name, str)):
            raise TypeError('The model_param_name argument must be None or of '
                'type str!')

        if(model_param_name in self._model_param_names):
            raise KeyError('There is already a global parameter defined for '
                'the model parameter name "%s"!'%(model_param_name))

        self._global_paramset.add_param(param)
        self._model_param_names = np.concatenate(
            (self._model_param_names,[model_param_name]))

        mask = np.ones((1,), dtype=np.bool_)
        self._global_param_2_model_mask = np.vstack(
            (self._global_param_2_model_mask, mask))

        return self

    def get_model_param_dict(
            self, global_floating_param_values, model_idx=None):
        """Creates a dictionary with the fixed and floating parameter names and
        their values for the single model.

        Parameters
        ----------
        global_floating_param_values : 1D ndarray instance
            The ndarray instance holding the current values of the global
            floating parameters. The values must be in the same order as the
            floating parameters were defined.
        model_idx : None
            The index of the model as it was defined at construction
            time of this ModelParameterMapper instance. Since this is a
            ModelParameterMapper for a single model, this argument is
            ignored.

        Returns
        -------
        model_param_dict : dict
            The dictionary holding the fixed and floating parameter names and
            values of the single model.
        """
        # Create the list of parameter names such that floating parameters are
        # before the fixed parameters.
        model_param_names = np.concatenate(
            (self._model_param_names[self._global_paramset.floating_params_mask],
             self._model_param_names[self._global_paramset.fixed_params_mask]))

        # Create a 1D (n_global_params,)-shaped ndarray holding the values of
        # the floating and fixed parameters. Since we only have a single model,
        # these values coincide with the parameter values of the single model.
        model_param_values = np.concatenate((
            global_floating_param_values,
            self._global_paramset.fixed_param_values
        ))
        if(len(model_param_values) != len(self._model_param_names)):
            raise ValueError('The number of parameter values (%d) does not '
                'equal the number of parameter names (%d) for model "%s"!'%
                (len(model_param_values), len(self._model_param_names),
                 self._models[0].name))

        model_param_dict = dict(
            zip(model_param_names, model_param_values))

        return model_param_dict


class MultiModelParameterMapper(ModelParameterMapper):
    """This class provides a model parameter mapper for multiple models, like
    multiple sources, or multiple background models.
    """
    def __init__(self, name, models):
        """Constructs a new multi model parameter mapper for mapping global
        parameters to the given models.

        Parameters
        ----------
        name : str
            The name of the model parameter mapper. In practice this is a
            representative name for the set of global parameters this model
            parameter mapper holds. For a two-component signal-background
            likelihood model, "signal", or "background" could be useful names.
        models : sequence of Model instances.
            The sequence of Model instances the parameter mapper can
            map global parameters to.
        """
        super(MultiModelParameterMapper, self).__init__(
            name=name, models=models)

    def def_param(self, param, model_param_name=None, models=None):
        """Adds the given Parameter instance to this parameter mapper and maps
        the parameter to the given sequence of models this model parameter
        mapper knows about.

        Parameters
        ----------
        param : instance of Parameter
            The global parameter which should get mapped to one or multiple
            models.
        model_param_name : str | None
            The parameter name of the models. The parameter name of the models
            must be the same for all the models this global parameter should get
            mapped to. The global parameter name can be different to the
            parameter name of the models.
            If set to `None`, the name of the global parameter will be used as
            model parameter name.
        models : sequence of Model instances | None
            The sequence of Model instances the parameter should get mapped to.
            The instances in the sequence must match Model instances specified
            at construction of this mapper.
            If set to `None` the global parameter will be mapped to all known
            models.

        Returns
        -------
        self : MultiModelParameterMapper
            The instance of this MultiModelParameterMapper, so that several
            `def_param` calls can be concatenated.

        Raises
        ------
        KeyError
            If there is already a model parameter of the same name defined for
            any of the given to-be-applied models.
        """
        if(model_param_name is None):
            model_param_name = param.name
        if(not isinstance(model_param_name, str)):
            raise TypeError('The model_param_name argument must be None or of '
                'type str!')

        if(models is None):
            models = self._models
        models = ModelCollection.cast(models,
            'The models argument must be castable to an instance of '
            'ModelCollection!')
        # Make sure that the user did not provide an empty sequence.
        if(len(models) == 0):
            raise ValueError('The sequence of models, to which the parameter '
                'maps, cannot be empty!')

        # Get the list of model indices to which the parameter maps.
        mask = np.zeros((self.n_models,), dtype=np.bool_)
        for ((midx,model), applied_model) in itertools.product(
                enumerate(self._models), models):
            if(applied_model.id == model.id):
                mask[midx] = True

        # Check that the model parameter name is not already defined for any of
        # the given to-be-mapped models.
        model_indices = np.arange(self.n_models)[mask]
        for midx in model_indices:
            param_mask = self._global_param_2_model_mask[:,midx]
            if(model_param_name in self._model_param_names[param_mask]):
                raise KeyError('The model parameter "%s" is already defined '
                    'for model "%s"!'%(model_param_name,
                    self._models[midx].name))

        self._global_paramset.add_param(param)
        self._model_param_names = np.concatenate(
            (self._model_param_names, [model_param_name]))

        self._global_param_2_model_mask = np.vstack(
            (self._global_param_2_model_mask, mask))

        return self

    def get_model_param_dict(
            self, global_floating_param_values, model_idx):
        """Creates a dictionary with the fixed and floating parameter names and
        their values for the given model.

        Parameters
        ----------
        global_floating_param_values : 1D ndarray instance
            The ndarray instance holding the current values of the global
            floating parameters.
        model_idx : int
            The index of the model as it was defined at construction
            time of this ModelParameterMapper instance.

        Returns
        -------
        model_param_dict : dict
            The dictionary holding the fixed and floating parameter names and
            values of the specified model.
        """
        # Get the model parameter mask that masks the global parameters for
        # the requested model.
        model_mask = self._global_param_2_model_mask[:,model_idx]

        # Create the array of parameter names that belong to the requested
        # model, where floating parameters are before the fixed parameters.
        model_param_names = np.concatenate(
            (self._model_param_names[
                self._global_paramset.floating_params_mask & model_mask],
             self._model_param_names[
                self._global_paramset.fixed_params_mask & model_mask]
            ))

        # Create the array of parameter values that belong to the requested
        # model, where floating parameters are before the fixed parameters.
        model_param_values = np.concatenate((
            global_floating_param_values[
                model_mask[self._global_paramset.floating_params_mask]],
            self._global_paramset.fixed_param_values[
                model_mask[self._global_paramset.fixed_params_mask]]
        ))

        model_param_dict = dict(
            zip(model_param_names, model_param_values))

        return model_param_dict


class HypoParameterDefinition(NamedObjectCollection):
    """This class provides a data holder for a list of model parameter mappers,
    where each parameter mapper defines a set of global parameters for the
    likelihood function, and their mapping to local model parameters.
    In addition this class provides a method to create a copy of itself, where
    floating parameters can get fixed to a certain values.
    """
    def __init__(self, model_param_mappers):
        """Creates a new instance of HypoParameterDefinition with the given list
        of ModelParameterMapper instances.

        Parameters
        ----------
        model_param_mappers : instance of ModelParameterMapper | sequence of
                ModelParameterMapper instances
            The list of ModelParameterMapper instances defining the global
            parameters and their mapping to local parameters of individual
            models.
        """
        super(HypoParameterDefinition, self).__init__(
            model_param_mappers, obj_type=ModelParameterMapper)

        # Finalize all ModelParameterMapper instances, hence no parameters can
        # be added anymore.
        for mapper in self._objects:
            mapper.finalize()

    @property
    def model_param_mapper_list(self):
        """(read-only) The list of ModelParameterMapper instances defining the
        global parameters and their mapping to the individual local model
        parameters.
        """
        return self._objects

    def __str__(self):
        """Creates a pretty string representation of this
        HypoParameterDefinition instance.
        """
        s = '%s:\n'%(classname(self))

        for (idx, mapper) in enumerate(self._objects):
            if(idx > 0):
                s += '\n'
            s1 = str(mapper)
            s += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, s1)

        return s

    def copy(self, fix_params=None):
        """Creates a deep copy of this HypoParameterDefinition instance and
        fixes the given global parameters to the given values.

        Parameters
        ----------
        fix_params : dict | None
            The dictionary defining the global parameters that should get fixed
            in the copy.

        Returns
        -------
        copy : instance of HypoParameterDefinition
            The copy of this HypoParameterDefinition instance with the given
            global parameters fixed to the given values.
        """
        copy = deepcopy(self)

        if(fix_params is not None):
            if(not isinstance(fix_params, dict)):
                raise TypeError('The fix_params argument must be of type dict!')

            for mp_mapper in copy.model_param_mapper_list:
                mp_mapper.global_paramset.make_params_fixed(fix_params)

        return copy

    def create_ParameterSetArray(self):
        """Creates a ParameterSetArray instance for all the ModelParameterMapper
        instances of this HypoParameterDefinition instance.

        Returns
        -------
        paramsetarray : ParameterSetArray
            The instance of ParameterSetArray holding references to the
            ParameterSet instances of all the ModelParameterMapper instances of
            this HypoParameterDefinition instance.
        """
        paramsetarray = ParameterSetArray(
            [mpmapper.global_paramset
             for mpmapper in self._objects])
        return paramsetarray


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
        delta = float_cast(
            delta, 'The delta argument must be castable to type float!')
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
        self._fitparams = np.empty((0,), dtype=np.object_)
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
                         for fitparam in self._fitparams ], dtype=np.float64)

    @property
    def bounds(self):
        """(read-only) The 2D (N_fitparams,2)-shaped ndarray holding the
        boundaries for all the global fit parameters.
        """
        return np.array([ (fitparam.valmin, fitparam.valmax)
                         for fitparam in self._fitparams ], dtype=np.float64)

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
        fitparam_values = np.empty_like(self._fitparams, dtype=np.float64)
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


class SourceFitParameterMapper(object, metaclass=abc.ABCMeta):
    """This abstract base class defines the interface of the source fit
    parameter mapper. This mapper provides the functionality to map a global fit
    parameter to a source fit parameter.
    """

    def __init__(self):
        """Constructor of the source fit parameter mapper.
        """
        self._fitparamset = FitParameterSet()

        # Define the list of source parameter names, which map to the fit
        # parameters.
        # Define the (N_fitparams,)-shaped numpy array of str objects.
        self._src_param_names = np.empty((0,), dtype=np.object_)

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
                                 dtype=[ (name, np.float64)
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
        self._fit_param_2_src_mask = np.zeros(
            (0, len(self.sources)), dtype=np.bool_)

        # Define an array, which will hold the unique source parameter names.
        self._unique_src_param_names = np.empty((0,), dtype=np.object_)

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
        mask = np.zeros((len(self.sources),), dtype=np.bool_)
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
                                 dtype=[ (name, np.float64)
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

