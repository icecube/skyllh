# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp

from skyllh.core.py import range
from skyllh.core.random import RandomStateService
from skyllh.core.timing import TimeLord

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

def parallelize(func, args_list, ncpu, rss=None, tl=None):
    """Parallelizes the execution of the given function for different arguments.

    Parameters
    ----------
    func : callable
        The function which should be called with different arguments, which are
        given through the args_list argument. If the `rss` argument is not None,
        `func` requires an argument named `rss`.
    args_list : list of 2-element tuple
        The list of the different arguments for function ``func``. Each element
        of that list must be a 2-element tuple, where the first element is a
        tuple of the arguments of ``func``, and the second element is a
        dictionary with the keyword arguments of ``func``. If the `rss` argument
        is not None, `func` argument `rss` has to be omitted.
    ncpu : int
        The number of CPUs to use, i.e. the number of subprocesses to spawn.
    rss : RandomStateService | None
        The RandomStateService instance to use for generating random numbers.
    tl : instance of TimeLord | None
        The instance of TimeLord that should be used to time individual tasks.

    Returns
    -------
    result_list : list
        The list of the result values of ``func``, where each element of that
        list corresponds to the arguments element in ``args_list``.
    """
    # Define a wrapper function for the multiprocessing module that evaluates
    # ``func`` for a subset of args_list.
    def wrapper(func, sub_args_list, pid=0, queue=None, rss=None, tl=None):
        """Wrapper function for the multiprocessing module that evaluates
        ``func`` for the subset ``sub_args_list`` of ``args_list``.

        Parameters
        ----------
        func : callable
            The function which should be called with different arguments, which
            are given through the sub_args_list argument. If the `rss` argument
            is not None `func` requires an argument named `rss`.
        sub_args_list : list of 2-element tuple
            The list of the different arguments for function ``func``. Each
            element of that list must be a 2-element tuple, where the first
            element is a tuple of the arguments of ``func``, and the second
            element is a dictionary with the keyword arguments of ``func``.
            If the `rss` argument is not None, `func` argument `rss` has to be
            omitted.
        pid : int
            The process ID that identifies the process in order to sort the
            results to the initial order of the function arguments.
        queue : multiprocessing.Queue | None
            The Queue instance where to put the function result in.
            If set to None, the result list will be returned.
        rss : RandomStateService | None
            The RandomStateService instance to use for generating random numbers.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.

        Returns
        -------
        result_list : list
            The list of the function results for the different function
            arguments, only if ``queue`` is set to None.
        """
        result_list = []
        for (args,kwargs) in sub_args_list:
            if(rss is not None):
                kwargs['rss'] = rss
            if(tl is not None):
                kwargs['tl'] = tl
            result_list.append(func(*args, **kwargs))

        if(queue is None):
            return result_list
        queue.put((pid, result_list, tl))

    # Return result list if only one CPU is used.
    if(ncpu == 1):
        return wrapper(func, args_list, rss=rss, tl=tl)

    # Multiple CPUs are used. Split the work across multiple processes.
    # We will use our own process (pid = 0) as a worker too.
    queue = mp.Queue()
    sub_args_list_list = np.array_split(args_list, ncpu)

    # Create a list of RandomStateService for each process if rss argument is
    # set.
    rss_list = [rss]
    if(rss is None):
        rss_list += [None]*(ncpu-1)
    else:
        if(not isinstance(rss, RandomStateService)):
            raise TypeError('The rss argument must be an instance of '
                'RandomStateService!')
        rss_list.extend([RandomStateService(seed=rss.random.randint(0, 2**32))
            for i in range(1, ncpu)])

    # Create a list of TimeLord instances, one for each process if tl argument
    # is set.
    tl_list = [tl]
    if(tl is None):
        tl_list += [None]*(ncpu-1)
    else:
        if(not isinstance(tl, TimeLord)):
            raise TypeError('The tl argument must be an instance of '
                'TimeLord!')
        tl_list.extend([TimeLord()
            for i in range(1, ncpu)])

    processes = [mp.Process(
        target=wrapper,
        args=(func, sub_args_list, pid, queue, rss_list[pid], tl_list[pid]))
        for (pid, sub_args_list) in enumerate(sub_args_list_list) if pid > 0]

    # Start the processes.
    for proc in processes:
        proc.start()

    # Compute the first chunk in the main process.
    result_list_0 = wrapper(
        func, sub_args_list_list[0], pid=0, rss=rss_list[0], tl=tl_list[0])

    # Gather len(processes) results from the queue and join the process's
    # TimeLord instance with the main TimeLord instance.
    pid_result_list_map = {0: result_list_0}
    for proc in processes:
        (pid, result_list, proc_tl) = queue.get()
        pid_result_list_map[pid] = result_list
        if(tl is not None):
            tl.join(proc_tl)

    # Join all the processes.
    for proc in processes:
        proc.join()

    # Order the result lists.
    result_list = []
    for pid in range(len(pid_result_list_map)):
        result_list += pid_result_list_map[pid]

    return result_list


class IsParallelizable(object):
    """Classifier class defining the ncpu property. Classes that derive from
    this class indicate, that they can make use of multi-processing on several
    CPUs at the same time.
    """
    def __init__(self, ncpu=None, *args, **kwargs):
        super(IsParallelizable, self).__init__(*args, **kwargs)
        self.ncpu = ncpu

    @property
    def ncpu(self):
        """The number (int) of CPUs to utilize. It calls the ``get_ncpu``
        utility function with this property as argument. Hence, if this property
        is set to None, the global NCPU setting will take precidence.
        """
        return get_ncpu(self._ncpu)
    @ncpu.setter
    def ncpu(self, n):
        if(n is not None):
            if(not isinstance(n, int)):
                raise TypeError('The ncpu property must be of type int!')
            if(n < 1):
                raise ValueError('The ncpu property must be >= 1!')
        self._ncpu = n
