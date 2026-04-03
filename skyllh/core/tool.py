"""The tool module provides functionality to interface with an optional external
python package (tool). The tool can be imported dynamically at run-time when
needed.
"""

import importlib
import importlib.util
import sys
from types import ModuleType

import numpy as np

from skyllh.core.py import (
    classname,
    get_class_of_func,
)


def assert_tool_version(
    tool: str,
    version: str,
):
    """Asserts the required version of the tool. The tool module must have the
    attribute ``__version__``.

    Parameters
    ----------
    tool
        The name of the tool.
    version
        The required version of the tool in the format ``"<COMP>X.Y.Z"``, where
        ``<COMP>`` is one of ``<=``, ``==``, and ``>=``.

    Raises
    ------
    ImportError
        If the version of the tool does not match the requirements.
    KeyError
        If the tool module has no attribute named ``__version__``.
    ValueError
        If ``<COMP>`` is not supported.
    """
    tool_module = get(tool)
    if not hasattr(tool_module, '__version__'):
        raise KeyError(f'The tool "{tool}" has no attribute "__version__"!')
    tool_version_arr = tool_module.__version__.split('.')
    (comp_op, version) = (version[0:2], version[2:])
    version_arr = version.split('.')
    for i in range(np.min([len(tool_version_arr), len(version_arr)])):
        tool_vers_i = int(tool_version_arr[i])
        vers_i = int(version_arr[i])
        if comp_op == '<=':
            if not (tool_vers_i <= vers_i):
                raise ImportError(
                    f'The version ({".".join(tool_version_arr)}) of the tool '
                    f'"{tool}" is not lower or equal than version '
                    f'{".".join(version_arr)}!'
                )
        elif comp_op == '==':
            if not (tool_vers_i == vers_i):
                raise ImportError(
                    f'The version ({".".join(tool_version_arr)}) of the tool '
                    f'"{tool}" is not equal to the version '
                    f'{".".join(version_arr)}!'
                )
        elif comp_op == '>=':
            if not (tool_vers_i >= vers_i):
                raise ImportError(
                    f'The version ({".".join(tool_version_arr)}) of the tool '
                    f'"{tool}" is not greater or equal than version '
                    f'{".".join(version_arr)}!'
                )
        else:
            raise ValueError(f'The version comparison operator "{comp_op}" for the tool "{tool}" is not supported!')


def is_available(name: str) -> bool:
    """Checks if the given Python package is available for import.

    Parameters
    ----------
    name
        The name of the Python package.

    Returns
    -------
    check
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
    return spec is not None


def get(name: str) -> ModuleType:
    """Returns the module object of the given tool. This will import the Python
    package if it was not yet imported.

    Parameters
    ----------
    name
        The name of the Python package.

    Returns
    -------
    module
        The (imported) Python module object.
    """
    if name in sys.modules:
        return sys.modules[name]

    module = importlib.import_module(name)
    return module


def _get_tool_and_version(
    tool: str | tuple[str, str],
) -> tuple[str, str | None]:
    """Returns the tool and and version based on the input value for the tool.

    Parameters
    ----------
    tool
        Either the tool name or the tuple with the tool name and required
        version string.

    Returns
    -------
    tool
        The name of the tool.
    version
        The tool's version string, or ``None``, if no version was specified.
    """
    if not (isinstance(tool, (str, tuple))):
        raise TypeError(
            f'The tool specification must be an instance of str or tuple! Its current type is {classname(tool)}!'
        )

    if isinstance(tool, tuple):
        if len(tool) != 2:
            raise ValueError(f'The length of the tool tuple must be 2, but it is {len(tool)}!')

        (tool, version) = tool

        if not isinstance(tool, str):
            raise TypeError(
                'The first element of the tool tuple must be an instance '
                'of str! '
                f'Its current type is {classname(tool)}!'
            )
        if not isinstance(version, str):
            raise TypeError(
                'The second element of the tool tuple must be an instance '
                'of str! '
                f'Its current type is {classname(version)}!'
            )

        return (tool, version)

    return (tool, None)


def requires(*tools):
    """This is decorator function that can be used whenever a function requires
    optional tools.

    Parameters
    ----------
    *tools : sequence of str | sequence of (str, str)
        The name(s) of the required Python packages.
        If a 2-element tuple of (str, str) is provided for a tool, the first
        element specifies the name of the tool and the second element its
        version of the form ``"<COMP>X.Y.Z"``, where ``<COMP>`` is one of
        ``<=``, ``==``, and ``>=``.

    Raises
    ------
    ModuleNotFoundError
        If any of the specified tools is not available.
    ImportError
        If the version of a tool does not meet the requirements.
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            for tool in tools:
                (tool, version) = _get_tool_and_version(tool)
                if not is_available(tool):
                    raise ModuleNotFoundError(
                        f'The Python module "{tool}" is not available, but is '
                        f'required by "{get_class_of_func(f)}.{f.__name__}"!'
                    )
                if version is not None:
                    assert_tool_version(tool, version)

            return f(*args, **kwargs)

        return wrapper

    return decorator
