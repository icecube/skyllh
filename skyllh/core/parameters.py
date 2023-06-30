# -*- coding: utf-8 -*-

import itertools
import numpy as np
from copy import deepcopy

from skyllh.core import (
    display,
)
from skyllh.core.model import (
    Model,
    ModelCollection,
)
from skyllh.core.py import (
    NamedObjectCollection,
    bool_cast,
    classname,
    float_cast,
    get_number_of_float_decimals,
    int_cast,
    issequence,
    issequenceof,
)
from skyllh.core.source_model import (
    SourceModel,
)


def make_linear_parameter_grid_1d(
        name,
        low,
        high,
        delta):
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
        if isfixed is None:
            if (valmin is not None) and (valmax is not None):
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
        if not isinstance(name, str):
            raise TypeError(
                'The "name" property must be of type str! '
                f'Its current type is {classname(name)}.')
        self._name = name

    @property
    def initial(self):
        """The initial value of the parameter.
        """
        return self._initial

    @initial.setter
    def initial(self, v):
        v = float_cast(
            v,
            'The "initial" property must be castable to type float!')
        self._initial = v

    @property
    def isfixed(self):
        """The flag if the parameter is mutable (False) or not (True).
        """
        return self._isfixed

    @isfixed.setter
    def isfixed(self, b):
        b = bool_cast(
            b,
            'The "isfixed" property must be castable to type bool!')
        self._isfixed = b

    @property
    def valmin(self):
        """The minimum bound value of the parameter.
        """
        return self._valmin

    @valmin.setter
    def valmin(self, v):
        v = float_cast(
            v,
            'The "valmin" property must be castable to type float!',
            allow_None=True)
        self._valmin = v

    @property
    def valmax(self):
        """The maximum bound value of the parameter.
        """
        return self._valmax

    @valmax.setter
    def valmax(self, v):
        v = float_cast(
            v,
            'The "valmax" property must be castable to type float!',
            allow_None=True)
        self._valmax = v

    @property
    def value(self):
        """The current value of the parameter.
        """
        return self._value

    @value.setter
    def value(self, v):
        v = float_cast(
            v,
            'The "value" property must be castable to type float!')
        if self._isfixed:
            if v != self._initial:
                raise ValueError(
                    f'The value ({v}) of the fixed parameter "{self._name}" '
                    'must be equal to the parameter\'s initial value '
                    f'({self._initial})!')
        else:
            if (v < self._valmin) or (v > self._valmax):
                raise ValueError(
                    f'The value ({v}) of parameter "{self._name}" must be '
                    f'within the range [{self._valmin:g}, {self._valmax:g}]!')
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
        if (self.name != other.name) or\
           (self.value != other.value) or\
           (self.isfixed != other.isfixed):
            return False

        # If both parameters are floating parameters, also their initial, min,
        # and max values must match.
        if not self.isfixed:
            if (self.initial != other.initial) or\
               (self.valmin != other.valmin) or\
               (self.valmax != other.valmax):
                return False

        return True

    def __str__(self):
        """Creates and returns a pretty string representation of this Parameter
        instance.
        """
        indstr = ' ' * display.INDENTATION_WIDTH

        s = f'Parameter: {self._name} = {self._value:g} '

        if self.isfixed:
            s += '[fixed]'
        else:
            s += '[floating] {\n'
            s += indstr + f'initial: {self._initial:g}\n'
            s += indstr + f'range: ({self._valmin:g}, {self._valmax:g})\n'
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
        if self.isfixed:
            raise ValueError(
                'Cannot create a linear grid from the fixed '
                f'parameter "{self._name}". The parameter must be floating!')

        delta = float_cast(
            delta,
            'The delta argument must be castable to type float!')

        grid = make_linear_parameter_grid_1d(
            name=self._name,
            low=self._valmin,
            high=self._valmax,
            delta=delta)

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
        if not self._isfixed:
            raise ValueError(
                f'The parameter "{self._name}" is not a fixed parameter!')

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
        if initial is None:
            self._initial = self._value
            return self._value

        self.initial = initial
        self._value = self._initial

        # Undefine the valmin and valmax values if the parameter's new value is
        # outside the valmin and valmax range.
        if (self._valmin is not None) and (self._valmax is not None) and\
           ((self._value < self._valmin) or (self._value > self._valmax)):
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
        if initial is None:
            initial = self._value
        if valmin is None:
            if self._valmin is None:
                raise ValueError(
                    f'The current minimal value of parameter "{self._name}" '
                    'is not set. So it must be defined through the valmin '
                    'argument!')
            valmin = self._valmin
        if valmax is None:
            if self._valmax is None:
                raise ValueError(
                    f'The current maximal value of parameter "{self._name}" '
                    'is not set. So it must be defined through the valmax '
                    'argument!')
            valmax = self._valmax

        self._isfixed = False
        self.initial = initial
        self.valmin = valmin
        self.valmax = valmax
        self.value = self._initial

        return self._value


class ParameterSet(
        object):
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
        if not issequenceof(paramsets, ParameterSet):
            raise TypeError(
                'The arguments of the union static function must be instances '
                'of ParameterSet!')
        if len(paramsets) == 0:
            raise ValueError(
                'At least 1 ParameterSet instance must be provided to the '
                'union static function!')

        paramset = ParameterSet(params=paramsets[0])
        for paramset_i in paramsets[1:]:
            for param in paramset_i._params:
                if not paramset.has_param(param):
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
        if params is not None:
            if isinstance(params, Parameter):
                params = [params]
            if not issequenceof(params, Parameter):
                raise TypeError(
                    'The params argument must be None, an instance of '
                    'Parameter, or a sequence of Parameter instances!')
            for param in params:
                self.add_param(param)

    @property
    def params(self):
        """(read-only) The 1D ndarray holding the Parameter instances.
        """
        return self._params

    @property
    def params_name_list(self):
        """(read-only) The list of str holding the names of all the parameters.
        """
        return self._fixed_param_name_list + self._floating_param_name_list

    @property
    def fixed_params(self):
        """(read-only) The 1D ndarray holding the Parameter instances, whose
        values are fixed.
        """
        return self._params[self._params_fixed_mask]

    @property
    def fixed_params_name_list(self):
        """(read-only) The list of the fixed parameter names.
        """
        return self._fixed_param_name_list

    @property
    def fixed_params_mask(self):
        """(read-only) The 1D ndarray holding the mask for the fixed parameters
        of this parameter set.
        """
        return self._params_fixed_mask

    @property
    def fixed_params_idxs(self):
        """The numpy ndarray holding the indices of the fixed parameters.
        """
        idxs = np.argwhere(self._params_fixed_mask).flatten()
        return idxs

    @property
    def floating_params(self):
        """(read-only) The 1D ndarray holding the Parameter instances,
        whose values are floating.
        """
        return self._params[np.invert(self._params_fixed_mask)]

    @property
    def floating_params_name_list(self):
        """(read-only) The list of the floating parameter names.
        """
        return self._floating_param_name_list

    @property
    def floating_params_mask(self):
        """(read-only) The 1D ndarray holding the mask for the floating
        parameters of this parameter set.
        """
        return np.invert(self._params_fixed_mask)

    @property
    def floating_params_idxs(self):
        """The numpy ndarray holding the indices of the floating parameters.
        """
        idxs = np.argwhere(self.floating_params_mask).flatten()
        return idxs

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

        if len(floating_params) == 0:
            return np.empty((0,), dtype=np.float64)

        initials = np.array(
            [param.initial for param in floating_params],
            dtype=np.float64)

        return initials

    @property
    def floating_param_bounds(self):
        """(read-only) The 2D (n_floating_params,2)-shaped ndarray holding the
        boundaries for all the floating parameters.
        """
        floating_params = self.floating_params

        if len(floating_params) == 0:
            return np.empty((0, 2), dtype=np.float64)

        bounds = np.array(
            [(param.valmin, param.valmax) for param in floating_params],
            dtype=np.float64)

        return bounds

    def __contains__(self, param_name):
        """Implements the ``param_name in self`` expression. It calls the
        :meth:`has_param` method of this class.

        Parameters
        ----------
        param_name : str
            The name of the parameter.

        Returns
        -------
        check : bool
            Returns ``True`` if the given parameter is part of this ParameterSet
            instance, ``False`` otherwise.
        """
        return self.has_param(param_name)

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
        s = (f'{classname(self)}: {self.n_params} parameters '
             f'({self.n_floating_params} floating, '
             f'{self.n_fixed_params} fixed) ''{')
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

    def generate_random_floating_param_initials(self, rss):
        """Generates a set of random initials for all floating parameters.
        A new random initial is defined as

            lower_bound + RAND * (upper_bound - lower_bound),

        where RAND is a uniform random variable between 0 and 1.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used for drawing
            random numbers from.

        Returns
        -------
        ri : (N_floating_params,)-shaped numpy ndarray
            The numpy 1D ndarray holding the generated random initial values.
        """
        vb = self.floating_param_bounds

        # Do random_initial = lower_bound + RAND * (upper_bound - lower_bound).
        ri = (vb[:, 0] +
              rss.random.uniform(size=vb.shape[0])*(vb[:, 1] - vb[:, 0]))

        return ri

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
            if pname in fix_params_keys:
                # The parameter of name `pname` should get fixed.
                if param.isfixed is True:
                    raise ValueError(
                        f'The parameter "{pname}" is already a fixed '
                        'parameter!')
                initial = fix_params[pname]
                param.make_fixed(initial)
                self._params_fixed_mask[pidx] = True
                self._fixed_param_name_list += [pname]
                self._fixed_param_values = np.concatenate(
                    (self._fixed_param_values, [param.value]))
                self._fixed_param_name_to_idx[pname] = len(
                    self._fixed_param_name_list) - 1
            else:
                if param.isfixed:
                    self._fixed_param_name_list += [pname]
                    self._fixed_param_values = np.concatenate(
                        (self._fixed_param_values, [param.value]))
                    self._fixed_param_name_to_idx[pname] = len(
                        self._fixed_param_name_list) - 1
                else:
                    self._floating_param_name_list += [pname]
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

                ``None``
                    The parameter's initial, minimal and maximal value should be
                    taken from the parameter's current settings.
                initial : float
                    The parameter's initial value should be set to the given
                    value. The minimal and maximal values of the parameter will
                    be taken from the parameter's current settings.
                (initial, valmin, valmax)
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
            if e is None:
                return (None, None, None)
            if issequence(e):
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
            if pname in float_params_keys:
                # The parameter of name `pname` should get set floating.
                if param.isfixed is False:
                    raise ValueError(
                        f'The parameter "{pname}" is already a floating '
                        'parameter!')
                (initial, valmin, valmax) = _parse_float_param_dict_entry(
                    float_params[pname])
                param.make_floating(initial, valmin, valmax)
                self._params_fixed_mask[pidx] = False
                self._floating_param_name_list += [pname]
                self._floating_param_name_to_idx[pname] = len(
                    self._floating_param_name_list) - 1
            else:
                if param.isfixed:
                    self._fixed_param_name_list += [pname]
                    self._fixed_param_values = np.concatenate(
                        (self._fixed_param_values, [param.value]))
                    self._fixed_param_name_to_idx[pname] = len(
                        self._fixed_param_name_list) - 1
                else:
                    self._floating_param_name_list += [pname]
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
        if not isinstance(param, Parameter):
            raise TypeError(
                'The param argument must be an instance of Parameter! '
                f'Its current type is {classname(param)}.')

        if self.has_param(param):
            raise KeyError(
                f'The parameter named "{param.name}" was already added to the '
                'parameter set!')

        param_fixed_mask = True if param.isfixed else False

        if atfront:
            # Add parameter at front of parameter list.
            self._params = np.concatenate(
                ([param], self._params))
            self._params_fixed_mask = np.concatenate(
                ([param_fixed_mask], self._params_fixed_mask))
            if param.isfixed:
                self._fixed_param_name_list = (
                    [param.name] + self._fixed_param_name_list)
                self._fixed_param_values = np.concatenate(
                    ([param.value], self._fixed_param_values))
                # Shift the index of all fixed parameters.
                self._fixed_param_name_to_idx = dict(
                    [(k, v+1)
                     for (k, v) in self._fixed_param_name_to_idx.items()])
                self._fixed_param_name_to_idx[param.name] = 0
            else:
                self._floating_param_name_list = (
                    [param.name] + self._floating_param_name_list)
                # Shift the index of all floating parameters.
                self._floating_param_name_to_idx = dict(
                    [(k, v+1)
                     for (k, v) in self._floating_param_name_to_idx.items()])
                self._floating_param_name_to_idx[param.name] = 0
        else:
            # Add parameter at back of parameter list.
            self._params = np.concatenate(
                (self._params, [param]))
            self._params_fixed_mask = np.concatenate(
                (self._params_fixed_mask, [param_fixed_mask]))
            if param.isfixed:
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
        if (param.name in self._floating_param_name_list) or\
           (param.name in self._fixed_param_name_list):
            return True

        return False

    def get_params_dict(self, floating_param_values):
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
        params_dict : dict
            The dictionary with the floating and fixed parameter names and
            values.
        """
        params_dict = dict(
            list(zip(self._floating_param_name_list, floating_param_values)) +
            list(zip(self._fixed_param_name_list, self._fixed_param_values))
        )

        return params_dict

    def get_floating_params_dict(self, floating_param_values):
        """Converts the given floating parameter values into a dictionary with
        the floating parameter names and values.

        Parameters
        ----------
        floating_param_values : 1D ndarray
            The ndarray holding the values of the floating parameters in the
            order that the floating parameters are defined.

        Returns
        -------
        params_dict : dict
            The dictionary with the floating and fixed parameter names and
            values.
        """
        params_dict = dict(
            list(zip(self._floating_param_name_list, floating_param_values))
        )

        return params_dict


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
        param_grid : instance of ParameterGrid
            The created ParameterGrid instance.
        """
        return ParameterGrid(
            name=binning.name,
            grid=binning.binedges,
            delta=delta,
            decimals=decimals)

    @staticmethod
    def from_range(name, start, stop, delta, decimals=None):
        """Creates a ParameterGrid instance from a range definition. The stop
        value will be the last grid point.

        Parameters
        ----------
        name : str
            The name of the parameter grid.
        start : float
            The start value of the range.
        stop : float
            The end value of the range.
        delta : float
            The width between the grid values.
        decimals : int | None
            The number of decimals the grid values should get rounded to.
            The maximal number of decimals is 16.
            If set to None, the number of decimals will be the maximum of the
            number of decimals of the first grid value and the number of
            decimals of the delta value.

        Returns
        -------
        param_grid : instance of ParameterGrid
            The created ParameterGrid instance.
        """
        start = float_cast(
            start,
            'The start argument must be castable to type float!')
        stop = float_cast(
            stop,
            'The stop argument must be castable to type float!')
        delta = float_cast(
            delta,
            'The delta argument must be castable to type float!')
        decimals = int_cast(
            decimals,
            'The decimals argument must be castable to type int!',
            allow_None=True)

        grid = np.arange(start, stop+delta, delta)

        return ParameterGrid(
            name=name,
            grid=grid,
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
        if delta is None:
            # We need to take the mean of all the "equal" differences in order
            # to smooth out unlucky rounding issues of a particular difference.
            delta = np.mean(np.diff(grid))

        delta = float_cast(
            delta,
            'The delta argument must be castable to type float!')
        self._delta = np.float64(delta)

        # Determine the number of decimals of delta.
        if decimals is None:
            decimals_value = get_number_of_float_decimals(grid[0])
            decimals_delta = get_number_of_float_decimals(delta)
            decimals = int(np.max((decimals_value, decimals_delta)))
        if not isinstance(decimals, int):
            raise TypeError(
                'The decimals argument must be an instance of type int!')
        if decimals > 16:
            raise ValueError(
                'The maximal number of decimals is 16! Maybe you should '
                'consider log-space!?')

        self.name = name
        self._decimals = decimals
        self._delta = np.around(self._delta, self._decimals)
        self.lower_bound = grid[0]

        # Setting the grid, will automatically round the grid values to their
        # next nearest grid value. Hence, we need to set the grid property after
        # setting the delta and offser properties.
        self.grid = grid

    def __str__(self):
        """Pretty string representation.
        """
        return '{:s} = {:s}, decimals = {:d}'.format(
            self._name, str(self._grid), self._decimals)

    @property
    def name(self):
        """The name of the parameter.
        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                'The name property must be of type str!')
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
        if not issequence(arr):
            raise TypeError(
                'The grid property must be a sequence!')
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                'The grid property must be a 1D numpy.ndarray!')
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
        v = float_cast(
            v,
            'The lower_bound property must be castable to type float!')
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
        """
        value = np.atleast_1d(value).astype(np.float64)

        floatD = (value - self._lower_bound)/self._delta
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

        if scalar_input:
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

        if scalar_input:
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

        if scalar_input:
            return gp.item()

        return gp


class ParameterGridSet(
        NamedObjectCollection):
    """Describes a set of parameter grids.
    """
    def __init__(
            self,
            param_grids=None,
            **kwargs):
        """Constructs a new ParameterGridSet object.

        Parameters
        ----------
        param_grids : sequence of instance of ParameterGrid | instance of ParameterGrid | None
            The ParameterGrid instances this instance of ParameterGridSet should
            get initialized with.
        """
        super().__init__(
            objs=param_grids,
            obj_type=ParameterGrid,
            **kwargs)

    @property
    def ndim(self):
        """The dimensionality of this parameter grid set. By definition it's the
        number of parameters of the set.
        """
        return len(self)

    @property
    def params_name_list(self):
        """(read-only) The list of the parameter names.
        """
        return self.name_list

    @property
    def parameter_permutation_dict_list(self):
        """(read-only) The list of parameter dictionaries constructed from all
        permutations of all the parameter values.
        """
        param_grids = [paramgrid.grid for paramgrid in self.objects]

        dict_list = [
            dict([
                (p_i, t_i)
                for (p_i, t_i) in zip(self.name_list, tup)
            ])
            for tup in itertools.product(*param_grids)
        ]

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


class ParameterModelMapper(
        object):
    """This class provides the parameter to model mapper.
    The parameter to model mapper provides the functionality to map a global
    parameter, usually a fit parameter, to a local parameter of a model, e.g.
    to a source, or a background model parameter.
    """

    @staticmethod
    def is_global_fitparam_a_local_param(
            fitparam_id,
            params_recarray,
            local_param_names):
        """Determines if the given global fit parameter is a local parameter of
        the given list of local parameter names.

        Parameters
        ----------
        fitparam_id : int
            The ID of the global fit parameter.
        params_recarray : instance of numpy record ndarray
            The (N_models,)-shaped numpy record ndarray holding the local
            parameter names and values of the models. See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for the format of this record array.
        local_param_names : list of str
            The list of local parameters.

        Returns
        -------
        check : bool
            ``True`` if the global fit parameter translates to a local parameter
            contained in the ``local_param_names`` list, ``False`` otherwise.
        """
        for pname in local_param_names:
            if pname not in params_recarray.dtype.fields:
                continue
            if np.any(params_recarray[f'{pname}:gpidx'] == fitparam_id + 1):
                return True

        return False

    @staticmethod
    def is_local_param_a_fitparam(
            local_param_name,
            params_recarray):
        """Checks if the given local parameter is a (partly) a fit parameter.

        Parameters
        ----------
        local_param_name : str
            The name of the local parameter.
        params_recarray : instance of numpy record ndarray
            The (N_models,)-shaped numpy record ndarray holding the local
            parameter names and values of the models. See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for the format of this record array.

        Returns
        -------
        check : bool
            ``True`` if the given local parameter is (partly) a fit parameter.
        """
        if np.any(params_recarray[f'{local_param_name}:gpidx'] > 0):
            return True

        return False

    def __init__(self, models, **kwargs):
        """Constructor of the parameter mapper.

        Parameters
        ----------
        models : sequence of instance of Model.
            The sequence of Model instances the parameter mapper can map global
            parameters to.
        """
        super().__init__(**kwargs)

        models = ModelCollection.cast(
            models,
            'The models property must be castable to an instance of '
            'ModelCollection!')
        self._models = models

        # Create the parameter set for the global parameters.
        self._global_paramset = ParameterSet()

        # Define the attribute holding the boolean mask of the models that are
        # source models.
        self._source_model_mask = np.array(
            [isinstance(model, SourceModel) for model in self._models],
            dtype=bool)

        # Define a (n_models, n_global_params)-shaped numpy ndarray of str
        # objects that will hold the local model parameter names of the global
        # parameters.
        # The local model parameter names are the names used by the internal
        # math objects, like PDFs. Thus, they can be aliases for the global
        # parameter names. Entries set to None, will indicate masked-out
        # global parameters.
        self._model_param_names = np.empty(
            (len(self._models), 0), dtype=np.object_)

    @property
    def models(self):
        """(read-only) The ModelCollection instance defining the models the
        mapper can map global parameters to.
        """
        return self._models

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

    @property
    def n_sources(self):
        """(read-only) The number of source models the mapper knows about.
        """
        return np.count_nonzero(self._source_model_mask)

    @property
    def unique_model_param_names(self):
        """(read-only) The unique parameters names of all the models.
        """
        m = self._model_param_names != np.array(None)
        return np.unique(self._model_param_names[m])

    @property
    def unique_source_param_names(self):
        """(read-only) The unique parameter names of the sources.
        """
        src_param_names = self._model_param_names[self._source_model_mask, ...]
        m = src_param_names != np.array(None)
        return np.unique(src_param_names[m])

    def __str__(self):
        """Generates and returns a pretty string representation of this
        parameter model mapper.
        """
        n_global_params = self.n_global_params

        # Determine the number of models that have global parameters assigned.
        # Remember self._model_param_names is a (n_models, n_global_params)-
        # shaped 2D ndarray.
        n_models = self.n_models
        n_sources = self.n_sources

        s = f'{classname(self)}: '
        s += f'{n_global_params} global parameter'
        s += '' if n_global_params == 1 else 's'
        s += ', '
        s += f'{n_models} model'
        s += '' if n_models == 1 else 's'
        s += f' ({n_sources} source'
        s += '' if n_sources == 1 else 's'
        s += ')'

        if n_global_params == 0:
            return s

        s1 = 'Parameters:'
        s += '\n' + display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)
        for (pidx, p) in enumerate(self._global_paramset.params):
            if p.isfixed:
                pstate = (
                    f'fixed ({p.initial:g})'
                )
            else:
                pstate = (
                    f'floating ({p.valmin:g} <= {p.initial:g} <= {p.valmax:g})'
                )
            ps = f'\n{p.name} [{pstate}]\n'

            ps1 = 'in models:\n'
            for (midx, mpname) in enumerate(self._model_param_names[:, pidx]):
                if mpname is not None:
                    ps1 += '- ' + self._models[midx].name + ': ' + mpname + "\n"

            ps += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, ps1)
            s += display.add_leading_text_line_padding(
                2*display.INDENTATION_WIDTH, ps)

        return s

    def get_model_param_name(self, model_idx, gp_idx):
        """Retrieves the local parameter name of a given model and global
        parameter index.

        Parameters
        ----------
        model_idx : int
            The index of the model.
        gp_idx : int
            The index of the global parameter.

        Returns
        -------
        param_name : str | None
            The name of the local model parameter. It is ``None``, if the given
            global parameter is not mapped to the given model.
        """
        param_name = self._model_param_names[model_idx, gp_idx]

        return param_name

    def get_gflp_idx(self, name):
        """Gets the index of the global floating parameter of the given name.

        Parameters
        ----------
        name : str
            The global floating parameter's name.

        Returns
        -------
        idx : int
            The index of the global floating parameter.
        """
        return self._global_paramset.get_floating_pidx(
            param_name=name)

    def get_model_idx_by_name(self, name):
        """Determines the index within this ParameterModelMapper instance of
        the model with the given name.

        Parameters
        ----------
        name : str
            The model's name.

        Returns
        -------
        model_idx : int
            The model's index within this ParameterModelMapper instance.

        Raises
        ------
        KeyError
            If there is no model of the given name.
        """
        for (model_idx, model) in enumerate(self._models):
            if model.name == name:
                return model_idx

        raise KeyError(
            f'The model with name "{name}" does not exist within the '
            'ParameterModelMapper instance!')

    def get_src_model_idxs(self, sources=None):
        """Creates a numpy ndarray holding the indices of the requested source
        models.

        Parameters
        ----------
        sources : instance of SourceModel | sequence of SourceModel | None
            The requested sequence of source models.
            If set to ``None``, all source models will be requested.

        Returns
        -------
        src_model_idxs : numpy ndarray
            The (N_sources,)-shaped 1D ndarray holding the indices of the
            requested source models.
        """
        # Get the model indices of all the source models.
        src_model_idxs = np.arange(self.n_models)[self._source_model_mask]

        if sources is None:
            return src_model_idxs

        # Select only the source models of interest.
        if isinstance(sources, SourceModel):
            sources = [sources]
        if not issequenceof(sources, SourceModel):
            raise TypeError(
                'The sources argument must be None, an instance of '
                'SourceModel, or a sequence of SourceModel! '
                f'Its type is {classname(sources)}')

        src_selection_mask = np.zeros((len(src_model_idxs),), dtype=bool)
        for smidx in src_model_idxs:
            src = self._models[smidx]
            if src in sources:
                src_selection_mask[smidx] = True

        src_model_idxs = src_model_idxs[src_selection_mask]

        return src_model_idxs

    def map_param(self, param, models=None, model_param_names=None):
        """Maps the given instance of Parameter to the given sequence of models
        this parameter model mapper knows about. Aliases for the given parameter
        can be specified for each individual model.

        Parameters
        ----------
        param : instance of Parameter
            The global parameter which should get mapped to one or more models.
        models : sequence of Model instances
            The sequence of Model instances the parameter should get
            mapped to. The instances in the sequence must match Model instances
            specified at construction of this mapper.
        model_param_names : str | sequence of str |  None
            The name of the parameter of the model. Hence, the global
            parameter name can be different to the parameter name of the model.
            If `None`, the name of the global parameter will be used as model
            parameter name for all models.

        Returns
        -------
        self : ParameterModelMapper
            The instance of this ParameterModelMapper, so that several
            `map_param` calls can be concatenated.

        Raises
        ------
        KeyError
            If there is already a model parameter of the same name defined for
            any of the given to-be-applied models.
        """
        if model_param_names is None:
            model_param_names = np.array([param.name]*len(self._models))
        if isinstance(model_param_names, str):
            model_param_names = np.array([model_param_names]*len(self._models))
        if not issequenceof(model_param_names, str):
            raise TypeError(
                'The model_param_names argument must be None, an instance of '
                'str, or a sequence of instances of str!')

        if models is None:
            models = self._models
        models = ModelCollection.cast(
            models,
            'The models argument must be castable to an instance of '
            'ModelCollection!')
        # Make sure that the user did not provide an empty sequence.
        if len(models) == 0:
            raise ValueError(
                'The sequence of models, to which the parameter maps, cannot '
                'be empty!')

        # Get the list of model indices to which the parameter maps.
        mask = np.zeros((self.n_models,), dtype=np.bool_)
        for ((midx, model), applied_model) in itertools.product(
                enumerate(self._models), models):
            if applied_model.id == model.id:
                mask[midx] = True

        # Check that the model parameter name is not already defined for any of
        # the given to-be-mapped models.
        for midx in np.arange(self.n_models)[mask]:
            mpnames = self._model_param_names[midx][
                self._model_param_names[midx] != np.array(None)]
            if model_param_names[midx] in mpnames:
                raise KeyError(
                    f'The model parameter "{model_param_names[midx]}" is '
                    f'already defined for model "{self._models[midx].name}"!')

        self._global_paramset.add_param(param)

        entry = np.where(mask, model_param_names, None)
        self._model_param_names = np.hstack(
            (self._model_param_names, entry[np.newaxis, :].T))

        return self

    def create_model_params_dict(self, gflp_values, model):
        """Creates a dictionary with the fixed and floating parameter names and
        their values for the given model.

        Parameters
        ----------
        gflp_values : 1D ndarray of float
            The ndarray instance holding the current values of the global
            floating parameters.
        model : instance of Model | str | int
            The index of the model as it was defined at construction
            time of this ParameterModelMapper instance.

        Returns
        -------
        model_param_dict : dict
            The dictionary holding the fixed and floating parameter names and
            values of the specified model.
        """
        gflp_values = np.atleast_1d(gflp_values)

        if isinstance(model, str):
            midx = self.get_model_idx_by_name(name=model)
        elif isinstance(model, Model):
            midx = self.get_model_idx_by_name(name=model.name)
        else:
            midx = int_cast(
                model,
                'The model argument must be an instance of Model, str, or '
                'castable to int!')
            if midx < 0 or midx >= len(self._models):
                raise IndexError(
                    f'The model index {midx} is out of range '
                    f'[0,{len(self._models)-1}]!')

        # Get the model parameter mask that masks the global parameters for
        # the requested model.
        m_gp_mask = self._model_param_names[midx] != np.array(None)

        _model_param_names = self._model_param_names
        _global_paramset = self._global_paramset
        gflp_mask = _global_paramset.floating_params_mask
        gfxp_mask = _global_paramset.fixed_params_mask

        # Create the array of local parameter names that belong to the
        # requested model, where the floating parameters are before the fixed
        # parameters.
        model_param_names = np.concatenate((
            _model_param_names[
                midx,
                gflp_mask & m_gp_mask],
            _model_param_names[
                midx,
                gfxp_mask & m_gp_mask]
        ))

        # Create the array of parameter values that belong to the requested
        # model, where floating parameters are before the fixed parameters.
        model_param_values = np.concatenate((
            gflp_values[m_gp_mask[gflp_mask]],
            _global_paramset.fixed_param_values[m_gp_mask[gfxp_mask]]
        ))

        model_param_dict = dict(
            zip(model_param_names, model_param_values))

        return model_param_dict

    def create_src_params_recarray(
            self,
            gflp_values=None,
            sources=None):
        """Creates a numpy record ndarray with a field for each local source
        parameter name and parameter's value. In addition each parameter field
        ``<name>`` has a field named ``<<name>:gpidx>`` which holds the index
        plus one of the corresponding global parameter for each source value.
        For values mapping to fixed parameters, the index is negative. Local
        parameter values that do not apply to a particular source are set to
        NaN. The parameter index in such cases is undefined.
        In addition to the parameter fields, the field ``:model_idx`` holds the
        index of the model for which the local parameter values apply.

        Parameters
        ----------
        gflp_values : numpy ndarray | None
            The (N_global_floating_param,)-shaped 1D ndarray holding the global
            floating parameter values. The order must match the order of
            parameter definition in this ParameterModelMapper instance.
            If set to ``None``, the value ``numpy.nan`` will be used as
            parameter value for floating parameters.
        sources : SourceModel | sequence of SourceModel | ndarray of int32 | None
            The sources which should be considered.
            If a ndarray of type int is provides, it must contain the global
            source indices.
            If set to ``None``, all sources are considered.

        Returns
        -------
        recarray : numpy structured ndarray
            The (N_sources,)-shaped numpy structured ndarray holding the local
            parameter names and their values for each requested source.
            It contains the following fields:

                :model_idx
                    The field holding the index of the model to which the set
                    of local parameters apply.
                <name>
                    The field holding the value for the local parameter <name>.
                    Not all local parameters apply to all sources.
                    Example: "gamma".
                <name>:gpidx
                    The field holding the global parameter index plus one for
                    the local parameter <name>. Example: "gamma:gpidx". Indices
                    for values mapping to fixed parameters are negative.
        """
        if gflp_values is None:
            gflp_values = np.full((self.n_global_floating_params,), np.nan)

        gflp_values = np.atleast_1d(gflp_values)

        # Check input.
        n_global_floating_params = self.n_global_floating_params
        if len(gflp_values) != n_global_floating_params:
            raise ValueError(
                f'The gflp_values argument is of length '
                f'{len(gflp_values)}, but must be of length '
                f'{n_global_floating_params}!')

        if isinstance(sources, np.ndarray) and sources.dtype == np.int32:
            # The sources are already specified in terms of their source
            # indices.
            smidxs = sources
        else:
            # Get the source indices of the requested sources.
            smidxs = self.get_src_model_idxs(sources=sources)

        # Create the output record array with nan as default value.
        dtype = [(':model_idx', np.int32)]
        for name in self.unique_source_param_names:
            dtype += [(name, np.float64), (f'{name}:gpidx', np.int32)]

        recarray = np.zeros(
            (len(smidxs),),
            dtype=dtype)
        for name in self.unique_source_param_names:
            recarray[name] = np.nan

        recarray[':model_idx'] = smidxs

        # Loop over the requested sources.
        _model_param_names = self._model_param_names
        _global_paramset = self._global_paramset
        gflp_mask = _global_paramset.floating_params_mask
        gfxp_mask = _global_paramset.fixed_params_mask
        for (i, smidx) in enumerate(smidxs):
            # Get the mask that selects the global parameters for the requested
            # source.
            src_gp_mask = _model_param_names[smidx] != np.array(None)

            # Create the array of local parameter names that belong to the
            # requested model, where the floating parameters are before the
            # fixed parameters.
            model_param_names = np.concatenate((
                _model_param_names[smidx, gflp_mask & src_gp_mask],
                _model_param_names[smidx, gfxp_mask & src_gp_mask]
            ))

            # Create the array of local parameter values that belong to the
            # requested model, where the floating parameters are before the
            # fixed parameters.
            model_param_values = np.concatenate((
                gflp_values[
                    src_gp_mask[gflp_mask]],
                _global_paramset.fixed_param_values[
                    src_gp_mask[gfxp_mask]]
            ))

            # Create the array of the global parameter indices.
            gpidxs = np.arange(len(_global_paramset))
            model_gp_idxs = np.concatenate((
                gpidxs[gflp_mask & src_gp_mask] + 1,
                -gpidxs[gfxp_mask & src_gp_mask] - 1,
            ))

            # Loop over the local parameters of the source and fill the
            # params record array.
            for (name, value, gpidx) in zip(
                    model_param_names, model_param_values, model_gp_idxs):
                recarray[name][i] = value
                recarray[f'{name}:gpidx'][i] = gpidx

        return recarray

    def create_global_params_dict(self, gflp_values):
        """Converts the given global floating parameter values into a dictionary
        holding the names and values of all floating and fixed parameters.

        Parameters
        ----------
        gflp_values : numpy ndarray
            The (n_global_floating_params,)-shaped 1D numpy ndarray holding the
            values of the global floating parameters.

        Returns
        -------
        params_dict : dict
            The dictionary holding the parameter name and values of all
            floating and fixed parameters.
        """
        params_dict = self._global_paramset.get_params_dict(
            floating_param_values=gflp_values)

        return params_dict

    def create_global_floating_params_dict(self, gflp_values):
        """Converts the given global floating parameter values into a dictionary
        holding the names and values of all floating parameters.

        Parameters
        ----------
        gflp_values : numpy ndarray
            The (n_global_floating_params,)-shaped 1D numpy ndarray holding the
            values of the global floating parameters.

        Returns
        -------
        params_dict : dict
            The dictionary holding the parameter name and values of all
            floating parameters.
        """
        params_dict = self._global_paramset.get_floating_params_dict(
            floating_param_values=gflp_values)

        return params_dict

    def get_local_param_is_global_floating_param_mask(
            self,
            local_param_names,
    ):
        """Checks which local parameter name is mapped to a global floating
        parameter.

        Parameters
        ----------
        local_param_names : sequence of str
            The sequence of the local parameter names to test.

        Returns
        -------
        mask : instance of ndarray
            The (N_local_param_names,)-shaped numpy ndarray holding the mask
            for each local parameter name if it is mapped to a global floating
            parameter.
        """
        mask = np.zeros(len(local_param_names), dtype=np.bool_)

        global_floating_params_idxs = self._global_paramset.floating_params_idxs

        # Get the global parameter indices for each local parameter name.
        for (local_param_idx, local_param_name) in enumerate(local_param_names):
            gpidxs = np.unique(
                np.nonzero(self._model_param_names == local_param_name)[1]
            )
            if np.any(np.isin(gpidxs, global_floating_params_idxs)):
                mask[local_param_idx] = True

        return mask
