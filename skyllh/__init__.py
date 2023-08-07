# -*- coding: utf-8 -*-

import logging
import multiprocessing as mp

from skyllh import _version

# Initialize top-level logger with a do-nothing NullHandler. It is required to
# be able to log messages when user has not set up any handler for the logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Change macOS default multiprocessing start method 'spawn' to 'fork'.

try:
    mp.set_start_method("fork")
except Exception:
    # It could be already set by another package.
    if mp.get_start_method() != "fork":
        logging.warning(
            "Couldn't set the multiprocessing start method to 'fork'. "
            "Parallel calculations using 'ncpu' argument != 1 may break."
        )

__version__ = _version.get_versions()['version']
