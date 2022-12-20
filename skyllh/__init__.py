# -*- coding: utf-8 -*-

import logging

# Initialize top-level logger with a do-nothing NullHandler. It is required to
# be able to log messages when user has not set up any handler for the logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import _version
__version__ = _version.get_versions()['version']
