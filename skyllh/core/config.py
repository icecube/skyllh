# -*- coding: utf-8 -*-

"""This file contains global configuration dictionary.
"""

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
    'physics': {
        'flux': {
            # The internal flux unit to use for the calculation. This unit must
            # match with the units used in the monte-carlo data.
            'internal_unit': 1./units.GeV * 1./units.cm**2 * 1./units.s
        }
    }
}
