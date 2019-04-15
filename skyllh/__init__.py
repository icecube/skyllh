# -*- coding: utf-8 -*-

import logging

# Initialize top-level logger with a do-nothing NullHandler.
logging.getLogger(__name__).addHandler(logging.NullHandler())
