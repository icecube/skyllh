# -*- coding: utf-8 -*-

import logging
import os.path
import sys


# Initialize the root logger.
logging.root.setLevel(logging.NOTSET)


def get_logger(
        name):
    """Retrieves the logger with the given name from the Python logging system.

    Parameters
    ----------
    name : str
        The name of the logger.
        Logger hierarchy is defined using dots as separators.

    Returns
    -------
    logger : logging.Logger
        The Logger instance.
    """
    logger = logging.getLogger(name)
    return logger


def setup_logger(
        name,
        log_level):
    """Initializes logger with a given name and a log level.

    Parameters
    ----------
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    log_level : int
        The log level.  The ``logging`` module predefines levels, e.g.
        ``logging.DEBUG``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)


def setup_console_handler(
        cfg,
        name,
        log_level=None,
        log_format=None,
        stream=None):
    """Initializes `StreamHandler` for a logger with a given name and sets its
    handling level.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    log_level : int | None
        The log level. The ``logging`` module predefines levels, e.g.
        ``logging.DEBUG``.
        If set to None, the log level of the logger will be used.
    log_format : str | None
        The format of log records in the final output.
        If set to `None`, the log format is taken from the configuration.
    stream : data stream | None
        The stream that the handler should use. Default stream is `sys.stderr`.
    """
    logger = logging.getLogger(name)

    if log_level is None:
        log_level = logger.level

    if log_format is None:
        log_format = cfg['debugging']['log_format']

    if stream is None:
        stream = sys.stderr

    # Create and add `StreamHandler` to the logger.
    sh = logging.StreamHandler(stream=stream)
    sh.setLevel(log_level)
    sh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(sh)


def setup_file_handler(
        cfg,
        name,
        filename,
        log_level=None,
        path=None,
        log_format=None,
        mode='a'):
    """Initializes `FileHandler` for a logger with a given name and sets its
    handling level.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    log_level : int | None
        The log level. There are predefined levels, e.g. ``logging.DEBUG``.
        If set to None, the log level of the logger will be used.
    filename : str
        The filename of the specified file which is opened and used as the
        stream for logging.
    path : str | None
        The path under which the log file should be stored.
        If set to `None`, the project's working directory will be used.
    log_format : str | None
        The format of log records in the final output.
        If set to `None`, the log format is taken from the configuration.
    mode : str
        File opening mode. Default is 'a' for appending.
    """
    logger = logging.getLogger(name)

    if log_level is None:
        log_level = logger.level

    if path is None:
        path = cfg['project']['working_directory']

    if log_format is None:
        log_format = cfg['debugging']['log_format']

    pathfilename = os.path.join(path, filename)

    # Create and add `FileHandler` to the logger.
    fh = logging.FileHandler(pathfilename, mode=mode)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
