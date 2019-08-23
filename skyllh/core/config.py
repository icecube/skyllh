# -*- coding: utf-8 -*-

"""This file contains global configuration dictionary.
"""

import os.path
import sys

from astropy import units

CFG = {
    'multiproc': {
        # The number of CPUs to use for functions that allow multi-processing.
        # If this setting is set to an int value in the range [1, N] this
        # setting will be used if a function's local ncpu setting is not
        # specified.
        'ncpu': None
    },
    'debugging': {
        # The default log format.
        'log_format': ('%(asctime)s %(processName)s %(name)s %(levelname)s: '
            '%(message)s')
    },
    'project': {
        # The project's working directory.
        'working_directory': '.'
    },
    'repository': {
        # A base path of repository datasets.
        'base_path': None
    },
    # Definition of the internal units to use. These must match with the units
    # from the monto-carlo data files.
    'internal_units': {
        'angle': units.radian,
        'energy': units.GeV,
        'length': units.cm,
        'time': units.s
    }
}

def set_wd(path):
    """Sets the project's working directory configuration variable and adds it
    to the Python path variable.

    Parameters
    ----------
    path : str
        The path of the project's working directory. This can be a path relative
        to the path given by ``os.path.getcwd``, the current working directory
        of the program.

    Returns
    -------
    wd : str
        The project's working directory.
    """
    if(CFG['project']['working_directory'] in sys.path):
        sys.path.remove(CFG['project']['working_directory'])

    wd = os.path.abspath(path)
    CFG['project']['working_directory'] = wd
    sys.path.insert(0, wd)

    return wd
