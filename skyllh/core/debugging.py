# -*- coding: utf-8 -*-

import logging
import os.path
import sys


def _resolve_log_level(level, default=logging.INFO):
    """Converts a logging level representation into a numeric level.

    Parameters
    ----------
    level : int | str | None
        The level representation.
    default : int
        Default level if ``level`` is None.

    Returns
    -------
    int
        The numeric logging level.
    """
    if level is None:
        return default

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        level_name = level.strip().upper()
        if level_name == '':
            raise ValueError('The logging level string must not be empty!')
        try:
            return int(getattr(logging, level_name))
        except AttributeError as err:
            raise ValueError(
                f'Unknown logging level string "{level}"!') from err

    raise TypeError(
        'The logging level must be int, str, or None! '
        f'Its current type is {type(level)}.')


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
    cfg,
    name,
    log_level=None,
    log_format=None,
    console=False,
    console_level=None,
    stream=None,
    log_file=None,
    file_level=None,
    file_mode="a",
    propagate=False,
    reset_handlers=False,
):
    """Sets up a logger with the given local configuration and a name.

    Parameters
    ----------
    cfg : instance of Config
        Local configuration.
    name : str
        The name of the logger to set up.
        Logger hierarchy is defined using dots as separators.
    log_level : int | str | None
        The log level of the logger.
        If None, the log level is taken from the configuration.
    log_format : str | None
        The format of log records in the final output.
        If None, the log format is taken from the configuration.
    console : bool
        Whether to set up a console handler for the logger. Default: False.
    console_level : int | None
        The log level of the console handler. If None, it uses `log_level`.
    stream : data stream | None
        The stream to which the console handler will write.
        If None, it defaults to `sys.stderr`.
    log_file : str | None
        If not ``None``, file handlers for DEBUG messages will be installed
        and those messages will be stored in the given file.
    file_level : int | None
        The log level of the file handler. If None, it uses `log_level`.
    file_mode : str
        File opening mode. Default is 'a' for appending.
    propagate : bool
        Whether the logger should propagate messages to ancestor loggers.
        Default: False.
    reset_handlers : bool
        Optionally clear handlers before setting up new ones.
        Default: False.
    """
    logger = logging.getLogger(name)
    
    cfg_log_level = cfg['debugging'].get('log_level', logging.INFO)
    resolved_log_level = _resolve_log_level(
        level=log_level,
        default=_resolve_log_level(cfg_log_level, default=logging.INFO))
    
    logger.setLevel(resolved_log_level)
    logger.propagate = propagate

    if log_format is None:
        log_format = cfg["debugging"]["log_format"]
    formatter = logging.Formatter(log_format)

    if reset_handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    if console:
        if console_level is None:
            console_level = resolved_log_level
        else:
            console_level = _resolve_log_level(console_level)
        if stream is None:
            stream = sys.stderr

        # deduplicate StreamHandler targeting same stream
        exists = any(
            isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) is stream
            for h in logger.handlers
        )
        if not exists:
            sh = logging.StreamHandler(stream=stream)
            sh.setLevel(console_level)
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    if log_file is not None:
        log_file = os.path.expanduser(log_file)
        if not os.path.isabs(log_file):
            base = os.path.abspath(
                os.path.expanduser(cfg["project"]["working_directory"]))
            log_file = os.path.join(base, log_file)
        log_file = os.path.abspath(log_file)

        exists = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == log_file
            for h in logger.handlers
        )
        if not exists:
            if file_level is None:
                file_level = resolved_log_level
            else:
                file_level = _resolve_log_level(file_level)
            fh = logging.FileHandler(log_file, mode=file_mode)
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def setup_logging(
        cfg,
        script_logger_name,
        log_format=None,
        log_level=None,
        log_file=None):
    """Initializes package and script loggers and returns the script logger.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    script_logger_name : str
        The name of the logger used by the script.
    log_format : str | None
        The format template of the log message. If ``None``, the format
        is taken from ``cfg['debugging']['log_format']``.
    log_level : int | str | None
        The log level of the loggers. If ``None``, it is taken from the
        configuration.
    log_file : str | None
        If not ``None``, a file handler for log messages will be installed
        for both loggers using this path.

    Returns
    -------
    logging.Logger
        The logger instance specified by ``script_logger_name``.
    """
    if log_format is None:
        log_format = cfg['debugging']['log_format']

    setup_logger(
        cfg=cfg,
        name='skyllh',
        log_level=log_level,
        log_format=log_format,
        console=True,
        log_file=log_file,
        file_level=logging.DEBUG)

    setup_logger(
        cfg=cfg,
        name=script_logger_name,
        log_level=log_level,
        log_format=log_format,
        console=True,
        log_file=log_file,
        file_level=logging.DEBUG)

    return get_logger(script_logger_name)