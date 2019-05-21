# -*- coding: utf-8 -*-

import logging
import os.path
from skyllh.core.config import cfg

def setup_logger(name, logging_level):
    """Initializes logger with a given name and a logging level.

    Parameters
    ----------
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    logging_level : int
        Logging level. There are predefined levels, e.g. ``logging.DEBUG``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

def setup_console_handler(name, handling_level, log_format, stream=None):
    """Initializes `StreamHandler` for a logger with a given name and sets its
    handling level.

    Parameters
    ----------
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    handling_level : int
        Handling level. There are predefined levels, e.g. ``logging.DEBUG``.
    log_format : str
        Specify the layout of log records in the final output.
    stream : data stream, optional
        The stream that the handler should use. Default stream is `sys.stderr`.
    """
    logger = logging.getLogger(name)
    # Create and add `StreamHandler` to the logger.
    sh = logging.StreamHandler(stream=None)
    sh.setLevel(handling_level)
    sh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(sh)

def setup_file_handler(name, handling_level, log_format, filename,
                       mode='a'):
    """Initializes `FileHandler` for a logger with a given name and sets its
    handling level.

    Parameters
    ----------
    name : str
        Logger name. Loggers hierarchy is defined using dots as separators.
    handling_level : int
        Handling level. There are predefined levels, e.g. ``logging.DEBUG``.
    log_format : str
        Specify the layout of log records in the final output.
    filename : str
        Filename of the specified file which is opened and used as the stream
        for logging.
    mode : str, optional
        File opening mode. Default is 'a' for appending.
    """
    logger = logging.getLogger(name)
    # Generate pathfilename.
    wd = cfg['dirs']['wd']
    pathfilename = os.path.join(wd, filename)
    # Create and add `FileHandler` to the logger.
    fh = logging.FileHandler(pathfilename, mode=mode)
    fh.setLevel(handling_level)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.

    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record):
        """
        Enqueue a record.

        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also puts the message into
        # record.message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info attribute, as it's no longer needed and, if not None,
        # will typically not be pickleable.
        self.format(record)
        record.msg = record.message
        record.args = None
        record.exc_info = None
        return record

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)
