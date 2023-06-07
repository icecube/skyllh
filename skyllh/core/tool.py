# -*- coding: utf-8 -*-

"""The tool module provides functionality to interface with an optional external
python package (tool). The tool can be imported dynamically at run-time when
needed.
"""

import importlib
import importlib.util
import sys


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
