# -*- coding: utf-8 -*-

# Initialize top-level logger with a do-nothing NullHandler. It is required to
# be able to log messages when user has not set up any handler for the logger.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Change macOS default multiprocessing start method 'spawn' to 'fork'.
import multiprocessing as mp
try:
    mp.set_start_method("fork")
except:
    # It could be already set by another package.
    if mp.get_start_method() != "fork":
        logging.warning(
            "Couldn't set the multiprocessing start method to 'fork'. "
            "Parallel calculations using 'ncpu' argument != 1 may break."
        )

from . import _version
__version__ = _version.get_versions()['version']
