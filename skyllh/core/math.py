"""The ``math`` module contains classes for pure mathematical objects."""

import abc
from copy import (
    deepcopy,
)

import numpy as np

from skyllh.core.py import (
    classname,
    isproperty,
    issequence,
    issequenceof,
)


class MathFunction(metaclass=abc.ABCMeta):
    """This abstract base class provides an implementation for a mathematical
    function. Such a function has defined parameters, which are implemented as
    class properties. The tuple of parameter names is defined through the
    `param_names` property.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.param_names = ()

    @property
    def param_names(self):
        """The tuple holding the names of the math function's parameters."""
        return self._param_names

    @param_names.setter
    def param_names(self, names):
        if not issequence(names):
            names = (names,)
        if not issequenceof(names, str):
            raise TypeError('The param_names property must be a sequence of str!')
        names = tuple(names)
        # Check if all the given names are actual properties of this
        # MathFunction class.
        for name in names:
            if not hasattr(self, name):
                raise KeyError(f'The "{classname(self)}" class does not have an attribute named "{name}"!')
            if not isproperty(self, name):
                raise TypeError(f'The attribute "{classname(self)}" of "{name}" is not a property!')
        self._param_names = names

    @property
    @abc.abstractmethod
    def math_function_str(self) -> 'str | None':
        """The string showing the mathematical function of this MathFunction."""
        pass

    def __str__(self) -> str:
        """Pretty string representation of this MathFunction instance."""
        return self.math_function_str or ''

    def copy(self, newparams: dict | None = None):
        """Copies this MathFunction object by calling the copy.deepcopy
        function, and sets new parameters if requested.

        Parameters
        ----------
        newparams
            The dictionary with the new parameter values to set, where the
            dictionary key is the parameter name and the dictionary value is the
            new value of the parameter.
        """
        f = deepcopy(self)

        # Set the new parameter values.
        if newparams is not None:
            f.set_params(newparams)

        return f

    def get_param(self, name: str) -> float:
        """Retrieves the value of the given parameter. It returns ``np.nan`` if
        the parameter does not exist.

        Parameters
        ----------
        name
            The name of the parameter.

        Returns
        -------
        value
            The value of the parameter.
        """
        if name not in self._param_names:
            return np.nan

        value = getattr(self, name)

        return value

    def set_params(self, pdict: dict) -> bool:
        """Sets the parameters of the math function to the given parameter
        values.

        Parameters
        ----------
        pdict
            The dictionary holding the names of the parameters and their new
            values.

        Returns
        -------
        updated
            Flag if parameter values were actually updated.
        """
        if not isinstance(pdict, dict):
            raise TypeError('The pdict argument must be of type dict!')

        updated = False

        for pname in self._param_names:
            current_value = getattr(self, pname)
            pvalue = pdict.get(pname, current_value)
            if pvalue != current_value:
                setattr(self, pname, pvalue)
                updated = True

        return updated
