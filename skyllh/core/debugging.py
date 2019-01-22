# -*- coding: utf-8 -*-

import os.path
import logging
import logging.handlers

def setup_logging(CONSOLE_LEVEL=logging.WARNING, FILE_LEVEL=logging.INFO, maxBytes=10485760, backupCount=1):
    """Setup logging to console and rotating file."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_path, '../log_file.log')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Create root logger.
    logger = logging.getLogger()
    logger.setLevel(min(CONSOLE_LEVEL, FILE_LEVEL))
    
    # Create and add `StreamHandler` handler to the root logger.
    sh = logging.StreamHandler()
    sh.setLevel(CONSOLE_LEVEL)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Create and add `RotatingFileHandler` to the root logger.
    rfh = logging.handlers.RotatingFileHandler(log_file_path,
                                               maxBytes=maxBytes,
                                               backupCount=backupCount)
    rfh.setLevel(FILE_LEVEL)    
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)
