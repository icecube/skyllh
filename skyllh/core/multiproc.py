# -*- coding: utf-8 -*-

import logging
import queue
import time

import multiprocessing as mp
import numpy as np

from logging.handlers import (
    QueueHandler,
)

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.timing import (
    TimeLord,
)


def get_ncpu(
        cfg,
        local_ncpu,
):
    """Determines the number of CPUs to use for functions that support
    multi-processing.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
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
    if ncpu is None:
        ncpu = cfg['multiproc']['ncpu']
    if ncpu is None:
        ncpu = 1

    if not isinstance(ncpu, int):
        raise TypeError(
            'The ncpu setting must be of type int!')

    if ncpu < 1:
        raise ValueError(
            'The ncpu setting must be >= 1!')

    return ncpu


def parallelize(  # noqa: C901
        func,
        args_list,
        ncpu,
        rss=None,
        tl=None,
        ppbar=None,
):
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
    ppbar : instance of ProgressBar | None
        The possible parent ProgressBar instance.

    Returns
    -------
    result_list : list
        The list of the result values of ``func``, where each element of that
        list corresponds to the arguments element in ``args_list``.
    """
    # Define a wrapper function for the multiprocessing module that evaluates
    # ``func`` for a subset of `args_list` on a worker process.
    def worker_wrapper(
            func,
            sub_args_list,
            pid,
            rqueue,
            lqueue,
            squeue=None,
            rss=None,
            tl=None,
    ):
        """Wrapper function for the multiprocessing module that evaluates
        ``func`` for the subset ``sub_args_list`` of ``args_list`` on a worker
        process.

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
        rqueue : multiprocessing.Queue
            The Queue instance where to put the function result in.
            If set to None, the result list will be returned.
        lqueue : multiprocessing.Queue
            The queue to hold generated log records by a given function.
        squeue : multiprocessing.Queue | None
            The Queue instance where to put in status information about finished
            tasks. Can be None to skip sending status information.
        rss : RandomStateService | None
            The RandomStateService instance to use for generating random
            numbers.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.
        """
        # Get the `QueueHandler` and update its log records queue.
        logger = logging.getLogger('skyllh')
        queue_handler = list(logger.handlers)[0]
        queue_handler.queue = lqueue

        result_list = []
        for (task_idx, (args, kwargs)) in enumerate(sub_args_list):
            if rss is not None:
                kwargs['rss'] = rss
            if tl is not None:
                kwargs['tl'] = tl
            result_list.append(func(*args, **kwargs))

            if squeue is not None:
                squeue.put((pid, task_idx))

        rqueue.put((pid, result_list, tl))

        # Put None object as the last log records queue item.
        lqueue.put_nowait(None)

    # Define a wrapper function that evaluates ``func`` for a subset of
    # `args_list` on the master process.
    def master_wrapper(
            pbar,
            sarr,
            func,
            sub_args_list,
            squeue=None,
            rss=None,
            tl=None,
    ):
        """This is the wrapper function for the master process.

        Parameters
        ----------
        pbar : instance of ProgressBar | None
            The instance of ProgressBar that should be used to display the
            progress if the current session is interactive.
        sarr : numpy record ndarray
            The status numpy record ndarray for all the processes. The length of
            that array must equal the number of processes, including the master
            process. Hence, the array index is the process id. The array must
            contain the following fields:
                n_finished_tasks : int
                    The number of finished tasks.
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
        squeue : multiprocessing.Queue | None
            The status queue for the worker processes that should be used to
            receive status information about finished tasks.
        rss : RandomStateService | None
            The RandomStateService instance to use for generating random
            numbers.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.
        """
        result_list = []
        for (master_task_idx, (args, kwargs)) in enumerate(sub_args_list):
            if rss is not None:
                kwargs['rss'] = rss
            if tl is not None:
                kwargs['tl'] = tl
            result_list.append(func(*args, **kwargs))

            # Skip the rest, if we are not in an interactive session, hence
            # there is not progress bar.
            if not pbar.is_shown:
                continue

            sarr[0]['n_finished_tasks'] = master_task_idx + 1

            # Get possible status information from the worker processes.
            if squeue is not None:
                while not squeue.empty():
                    (pid, worker_task_idx) = squeue.get()
                    sarr[pid]['n_finished_tasks'] = worker_task_idx + 1

            # Calculate the total number of finished tasks.
            n_finished_tasks = np.sum(sarr['n_finished_tasks'])
            pbar.update(n_finished_tasks)

        return result_list

    # Create the progress bar if we are in an interactive session.
    pbar = ProgressBar(maxval=len(args_list), parent=ppbar).start()

    # Return result list if only one CPU is used.
    if ncpu == 1:
        sarr = np.zeros((1,), dtype=[('n_finished_tasks', np.int64)])
        result_list = master_wrapper(
            pbar, sarr, func, args_list, squeue=None, rss=rss, tl=tl)

        pbar.finish()

        return result_list

    # Multiple CPUs are used. Split the work across multiple processes.
    # We will use our own process (pid = 0) as a worker too.
    rqueue = mp.Queue()
    squeue = None
    if pbar.is_shown:
        squeue = mp.Queue()

    sub_args_list_list = np.array_split(np.array(args_list, dtype=object), ncpu)

    # Create a multiprocessing queue for each worker process.
    # Prepend it with None to be able to use `pid` as the list index.
    lqueue_list = [None] + [mp.Queue() for i in range(ncpu-1)]

    # Create a list of RandomStateService for each process if rss argument is
    # set.
    rss_list = [rss]
    if rss is None:
        rss_list += [None]*(ncpu-1)
    else:
        if not isinstance(rss, RandomStateService):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService!')
        rss_list.extend([
            RandomStateService(seed=rss.random.randint(0, 2**32))
            for i in range(1, ncpu)
        ])

    # Create a list of TimeLord instances, one for each process if tl argument
    # is set.
    tl_list = [tl]
    if tl is None:
        tl_list += [None]*(ncpu-1)
    else:
        if not isinstance(tl, TimeLord):
            raise TypeError(
                'The tl argument must be an instance of TimeLord!')
        tl_list.extend([
            TimeLord()
            for i in range(1, ncpu)
        ])

    # Replace all existing main process handlers with the `QueueHandler`.
    # This allows storing all the log record generated by worker processes at
    # separate `multiprocessing.Queue` instances. After creating
    # worker processes revert handlers to the initial state.
    logger = logging.getLogger('skyllh')
    orig_handlers = list(logger.handlers)
    for orig_handler in orig_handlers:
        logger.removeHandler(orig_handler)
    queue_handler = QueueHandler(lqueue_list[0])
    logger.addHandler(queue_handler)

    processes = [mp.Process(
        target=worker_wrapper,
        args=(func, sub_args_list, pid, rqueue, lqueue_list[pid]),
        kwargs={'squeue': squeue, 'rss': rss_list[pid], 'tl': tl_list[pid]})
        for (pid, sub_args_list) in enumerate(sub_args_list_list) if pid > 0]

    # Start the processes.
    for proc in processes:
        proc.start()

    # Revert main process handlers to the initial state.
    logger.removeHandler(queue_handler)
    for orig_handler in orig_handlers:
        logger.addHandler(orig_handler)

    # Compute the first chunk in the main process.
    sarr = np.zeros((len(processes)+1,), dtype=[('n_finished_tasks', np.int64)])
    result_list_0 = master_wrapper(
        pbar, sarr, func, sub_args_list_list[0], squeue=squeue, rss=rss_list[0],
        tl=tl_list[0])

    # Initialize logger.
    logger = logging.getLogger(__name__)

    # Gather len(processes) results from the rqueue and join the process's
    # TimeLord instance with the main TimeLord instance.
    # Handle log records created by each process.
    pid_result_list_map = {0: result_list_0}
    for proc in processes:
        # Get the result record from the result queue.
        result_received = False
        proc_died = False
        while (result_received is False) and (proc_died is False):
            try:
                (pid, result_list, proc_tl) = rqueue.get(block=False)
                result_received = True
            except queue.Empty:
                # If this exception is raised, either the child process isn't
                # finished yet, or it dies due to an exception.
                if proc.exitcode is None:
                    # Child process hasn't finish yet.
                    # We'll wait a short moment.
                    time.sleep(0.01)
                elif proc.exitcode != 0:
                    proc_died = True
        if proc_died:
            raise RuntimeError(
                f'Child process {proc.pid} did not return with 0! '
                f'Exit code was {proc.exitcode}.')

        pid_result_list_map[pid] = result_list
        if tl is not None:
            tl.join(proc_tl)
        logger.debug(
            f'Beginning of worker process (pid={pid}) log records.')
        lqueue_end = False
        while not lqueue_end:
            record = lqueue_list[pid].get()
            if record is None:
                lqueue_end = True
            else:
                lqueue_logger = logging.getLogger(record.name)
                lqueue_logger.handle(record)
        logger.debug('Ending of worker process (pid=%d) log records.', pid)

    # Join all the processes.
    for proc in processes:
        proc.join()

    # Order the result lists.
    result_list = []
    for pid in range(len(pid_result_list_map)):
        result_list += pid_result_list_map[pid]

    pbar.finish()

    return result_list


class IsParallelizable(
        object,
):
    """Classifier class defining the ncpu property. Classes that derive from
    this class indicate, that they can make use of multi-processing on several
    CPUs at the same time.
    """

    def __init__(
            self,
            *args,
            ncpu=None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(self, HasConfig):
            raise TypeError(
                f'The class "{classname(self)}" is not derived from '
                'skyllh.core.config.HasConfig!')

        self.ncpu = ncpu

    @property
    def ncpu(self):
        """The number (int) of CPUs to utilize. It calls the ``get_ncpu``
        utility function with this property as argument. Hence, if this property
        is set to None, the global NCPU setting will take precedence.
        """
        return get_ncpu(self._cfg, self._ncpu)

    @ncpu.setter
    def ncpu(self, n):
        if n is not None:
            if not isinstance(n, int):
                raise TypeError(
                    'The ncpu property must be of type int!')
            if n < 1:
                raise ValueError(
                    'The ncpu property must be >= 1!')
        self._ncpu = n
