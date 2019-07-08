# -*- coding: utf-8 -*-

"""This file contains global configuration dictionary.
"""

CFG = {
    'multiproc': {
        # The number of CPUs to use for functions that allow multi-processing.
        # If this setting is set to an int value in the range [1, N] this
        # setting will be used if a function's local ncpu setting is not
        # specified.
        'ncpu': None
    },
    'project': {
        # The project's working directory.
        'working_directory': '.'
    },
    'repository': {
        # A base path of repository datasets.
        'base_path': None
    }
}
