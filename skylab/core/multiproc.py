# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp

# Global setting for the number of CPUs to use for functions that allow
# multi-processing. If this setting is set to an int value in the range [1, N]
# this setting will be used if a function's local ncpu setting is not specified.
NCPU = None

def get_ncpu(local_ncpu):
    """Determines the number of CPUs to use for functions that support
    multi-processing.

    Parameters
    ----------
    local_ncpu : int | None
        The local setting of the number of CPUs to use.

    Returns
    -------
    ncpu : int
        The number of CPUs to use by functions that allow multi-processing.
        If ``local_ncpu`` is set to None, the global NCPU setting is returned.
        If the global NCPU setting is None as well, the default value 1 is
        returned.
    """
    ncpu = local_ncpu
    if(ncpu is None):
        ncpu = NCPU
    if(ncpu is None):
        ncpu = 1
    if(not isinstance(ncpu, int)):
        raise TypeError('The ncpu setting must be of type int!')
    if(ncpu < 1):
        raise ValueError('The ncpu setting must be >= 1!')
    return ncpu

def parallelize(func, args_list, ncpu):
    """Parallelizes the execution of the given function for different arguments.

    Parameters
    ----------
    func : callable
        The function which should be called with different arguments, which are
        given through the args_list argument.
    args_list : list of 2-element tuple
        The list of the different arguments for function ``func``. Each element
        of that list must be a 2-element tuple, where the first element is a
        tuple of the arguments of ``func``, and the second element is a
        dictionary with the keyword arguments of ``func``.
    ncpu : int
        The number of CPUs to use, i.e. the number of subprocesses to spawn.

    Returns
    -------
    result_list : list
        The list of the result values of ``func``, where each element of that
        list corresponds to the arguments element in ``args_list``.
    """
    # Define a wrapper function for the multiprocessing module that evaluates
    # ``func`` for a subset of args_list.
    def wrapper(func, sub_args_list, pid=0, queue=None):
        """Wrapper function for the multiprocessing module that evaluates
        ``func`` for the subset ``sub_args_list`` of ``args_list``.

        Parameters
        ----------
        func : callable
            The function which should be called with different arguments, which
            are given through the sub_args_list argument.
        sub_args_list : list of 2-element tuple
            The list of the different arguments for function ``func``. Each
            element of that list must be a 2-element tuple, where the first
            element is a tuple of the arguments of ``func``, and the second
            element is a dictionary with the keyword arguments of ``func``.
        pid : int
            The process ID that identifies the process in order to sort the
            results to the initial order of the function arguments.
        queue : multiprocessing.Queue | None
            The Queue instance where to put the function result in.
            If set to None, the result list will be returned.

        Returns
        -------
        result_list : list
            The list of the function results for the different function
            arguments, only if ``queue`` is set to None.
        """
        result_list = [func(*args,**kwargs) for (args,kwargs) in sub_args_list]
        if(queue is None):
            return result_list
        queue.put((pid, result_list))

    # Return result list if only one CPU is used.
    if(ncpu == 1):
        return wrapper(func, args_list)

    # Multiple CPUs are used. Split the work across multiple processes.
    # We will use our own process (pid = 0) as a worker too.
    queue = mp.Queue()
    sub_args_list_list = np.array_split(args_list, ncpu)
    processes = [ mp.Process(target=wrapper, args=(func, sub_args_list, pid, queue))
                 for (pid, sub_args_list) in enumerate(sub_args_list_list) if pid > 0]

    # Start the processes.
    for proc in processes:
        proc.start()

    # Compute the first chunk in the main process.
    result_list_0 = wrapper(func, sub_args_list_list[0], pid=0)

    # Gather len(processes) results from the queue.
    pid_result_list_map = {0: result_list_0}
    for proc in processes:
        (pid, result_list) = queue.get()
        pid_result_list_map[pid] = result_list

    # Join all the processes.
    for proc in processes:
        proc.join()

    # Order the result lists.
    result_list = []
    for pid in xrange(len(pid_result_list_map)):
        result_list += pid_result_list_map[pid]

    return result_list

class Parallelizable(object):
    """Abstract base class defining the ncpu property. Classes that derive from
    this class indicate, that they can make use of multi-processing on several
    CPUs at the same time.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @property
    def ncpu(self):
        """The number (int) of CPUs to utilize.
        """
        return self._ncpu
    @ncpu.setter
    def ncpu(self, n):
        if(not isinstance(n, int)):
            raise TypeError('The ncpu property must be of type int!')
        if(n < 1):
            raise ValueError('The ncpu property must be >= 1!')
        self._ncpu = n
