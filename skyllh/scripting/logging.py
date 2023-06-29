# -*- coding: utf-8 -*-

"""This module contains utility functions for logging functionalities of an
analysis script.
"""

import logging

from skyllh.core.debugging import (
    get_logger,
    setup_console_handler,
    setup_file_handler,
)


def setup_logging(
    script_logger_name,
    log_format=None,
    log_level=logging.INFO,
    debug_pathfilename=None,
):
    """Installs console handlers for the ``skyllh`` and ``script_logger_name``
    loggers. If a debug file is specified, file handlers for debug messages
    will be installed as well.

    Parameters
    ----------
    script_logger_name : str
        The name of the logger used by the script.
    log_format : str | None
        The format template of the log message. If set to ``Ǹone``, the format
        will be

            ``'%(asctime)s %(processName)s %(name)s %(levelname)s: %(message)s'``

    log_level : int
        The log level of the loggers. The default is ``logging.INFO``.
    debug_pathfilename : str | None
        If not ``None``, file handlers for DEBUG messages will be installed and
        those messages will be stored in the given file.

    Returns
    -------
    script_logger : instance of logging.Logger
        The logger instance of the script, specified by ``script_logger_name``.
    """
    if log_format is None:
        log_format = (
            "%(asctime)s %(processName)s %(name)s %(levelname)s: " "%(message)s"
        )

    setup_console_handler(
        name="skyllh", log_level=log_level, log_format=log_format
    )

    setup_console_handler(
        name=script_logger_name, log_level=log_level, log_format=log_format
    )

    if debug_pathfilename is not None:
        setup_file_handler(
            name="skyllh",
            filename=debug_pathfilename,
            log_format=log_format,
            log_level=logging.DEBUG,
        )
        setup_file_handler(
            name=script_logger_name,
            filename=debug_pathfilename,
            log_format=log_format,
            log_level=logging.DEBUG,
        )

    script_logger = get_logger(script_logger_name)

    return script_logger
