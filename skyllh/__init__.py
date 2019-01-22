# -*- coding: utf-8 -*-

import logging
from skyllh.core.debugging import setup_logging

# Set logging levels of console and file output logs.
CONSOLE_LEVEL = logging.INFO
FILE_LEVEL = logging.DEBUG

# Set log file size and number of backup files.
max_bytes = 10*1024*1024 # 10 MB
backup_count = 1

setup_logging(CONSOLE_LEVEL, FILE_LEVEL, max_bytes, backup_count)
