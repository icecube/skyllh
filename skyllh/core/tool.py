# -*- coding: utf-8 -*-

"""The tool module provides functionality to interface with an optional external
python package (tool). The tool can be imported dynamically at run-time when
needed.
"""

import importlib
import importlib.util
import sys

from skyllh.core.py import (
    get_class_of_func,
)


def is_available(name):
    """Checks if the given Python package is available for import.

    Parameters
    ----------
    name : str
        The name of the Python package.

    Returns
    -------
    check : bool
        ``True`` if the given Python package is available, ``False`` otherwise.

    Raises
    ------
    ModuleNotFoundError
        If the package is not a Python package, i.e. lacks a __path__ attribute.
    """
    # Check if module is already imported.
    if name in sys.modules:
        return True

    spec = importlib.util.find_spec(name)
    if spec is not None:
        return True

    return False


def get(name):
    """Returns the module object of the given tool. This will import the Python
    package if it was not yet imported.

    Parameters
    ----------
    name : str
        The name of the Python package.

    Returns
    -------
    module : Python module
        The (imported) Python module object.
    """
    if name in sys.modules:
        return sys.modules[name]

    module = importlib.import_module(name)
    return module


def requires(*tools):
    """This is decorator function that can be used whenever a function requires
    optional tools.

    Parameters
    ----------
    *tools : sequence of str
        The name of the required Python packages.

    Raises
    ------
    ModuleNotFoundError
        If any of the specified tools is not available.
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            for tool in tools:
                if not is_available(tool):
                    raise ModuleNotFoundError(
                        f'The Python module "{tool}" is not available, but is '
                        f'required by "{get_class_of_func(f)}.{f.__name__}"!')
            return f(*args, **kwargs)
        return wrapper
    return decorator
