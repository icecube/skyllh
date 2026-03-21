# -*- coding: utf-8 -*-

import logging
import os.path
import sys


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


def configure_logging(
        cfg,
        script_logger_name,
        log_format=None,
        log_level=logging.INFO,
        debug_pathfilename=None):
    """Initializes loggers and installs handlers for the ``skyllh`` and
    ``script_logger_name`` loggers.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    script_logger_name : str
        The name of the logger used by the script.
    log_format : str | None
        The format template of the log message. If set to ``None``, the format
        will be taken from ``cfg['debugging']['log_format']``.
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
        log_format = cfg['debugging']['log_format']

    setup_logger('skyllh', log_level)
    setup_logger(script_logger_name, log_level)

    setup_console_handler(
        cfg=cfg,
        name='skyllh',
        log_level=log_level,
        log_format=log_format
    )

    setup_console_handler(
        cfg=cfg,
        name=script_logger_name,
        log_level=log_level,
        log_format=log_format
    )

    if debug_pathfilename is not None:
        setup_file_handler(
            cfg=cfg,
            name='skyllh',
            filename=debug_pathfilename,
            log_format=log_format,
            log_level=logging.DEBUG
        )
        setup_file_handler(
            cfg=cfg,
            name=script_logger_name,
            filename=debug_pathfilename,
            log_format=log_format,
            log_level=logging.DEBUG
        )

    script_logger = get_logger(script_logger_name)

    return script_logger
