import os.path
import logging
import logging.handlers

def setup_logging(console_level=logging.WARNING, file_level=logging.INFO):
    """Setup logging to console and rotating file."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_path, 'log_file.log')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Create root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create and add `StreamHandler` handler to the root logger.
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Create and add `RotatingFileHandler` to the root logger.
    rfh = logging.handlers.RotatingFileHandler(log_file_path,
                                               maxBytes=10485760,
                                               backupCount=1)
    rfh.setLevel(file_level)    
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)

# Set logging levels of console and file output logs.
console_level = logging.WARNING
file_level = logging.DEBUG

setup_logging(console_level, file_level)
