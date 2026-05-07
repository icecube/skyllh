import logging
import multiprocessing as mp

__all__ = [
    'create_datasets',
]

# Initialize top-level logger with a do-nothing NullHandler. It is required to
# be able to log messages when user has not set up any handler for the logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    if name == 'create_datasets':
        from skyllh.datasets import create_datasets

        return create_datasets
    raise AttributeError(f"module 'skyllh' has no attribute {name!r}")


# Change macOS default multiprocessing start method 'spawn' to 'fork'.
try:
    mp.set_start_method('fork')
except Exception:
    # It could be already set by another package.
    if mp.get_start_method() != 'fork':
        logging.warning(
            "Couldn't set the multiprocessing start method to 'fork'. "
            "Parallel calculations using 'ncpu' argument != 1 may break."
        )
