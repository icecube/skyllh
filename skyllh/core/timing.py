# -*- coding: utf-8 -*-

import numpy as np
import time

from skyllh.core import display
from skyllh.core.py import range

"""The timing module provides code execution timing functionalities. The
TimeLord class keeps track of execution times of specific code segments,
called "tasks". The TaskTimer class can be used within a `with`
statement to time the execution of the code within the `with` block.
"""

class TaskRecord(object):
    def __init__(self, name, duration):
        self.name = name
        self.duration = duration


class TimeLord(object):
    def __init__(self):
        self._task_records = []

    def add_task_record(self, name, duration):
        """Adds a task record to the internal list of task records.
        """
        self._task_records.append(TaskRecord(name, duration))

    @property
    def task_name_list(self):
        """(read-only) The list of task names.
        """
        return [ task.name for task in self._task_records ]

    def task_timer(self, name):
        """Creates TaskTimer instance for the given task name.
        """
        return TaskTimer(self, name)

    def __str__(self):
        """Generates a pretty string for this time lord.
        """
        s = 'Executed tasks:'
        task_name_list = self.task_name_list
        task_name_len_list = [ len(task_name) for task_name in task_name_list ]
        max_task_name_len = np.minimum(
            np.max(task_name_len_list), display.PAGE_WIDTH-16)

        n_tasks = len(task_name_list)
        for i in range(n_tasks):
            task_name = task_name_list[i][0:max_task_name_len]
            task_duration = self._task_records[i].duration
            l = '\n[%'+str(max_task_name_len)+'s] %g sec'
            s += l%(task_name, task_duration)

        return s


class TaskTimer(object):
    def __init__(self, time_lord, name):
        """
        Parameters
        ----------
        time_lord : instance of TimeLord
            The TimeLord instance that keeps track of the recorded tasks.
        name : str
            The name of the task.
        """
        self.time_lord = time_lord
        self.name = name

        self._start = None
        self._end = None

    @property
    def time_lord(self):
        """The TimeLord instance that keeps track of the recorded tasks. This
        can be None, which means that the task should not get recorded.
        """
        return self._time_lord
    @time_lord.setter
    def time_lord(self, lord):
        if(lord is not None):
            if(not isinstance(lord, TimeLord)):
                raise TypeError('The time_lord property must be None or an '
                    'instance of TimeLord!')
        self._time_lord = lord

    @property
    def name(self):
        """The name if the task.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be an instance of str!')
        self._name = name

    @property
    def duration(self):
        """The duration in seconds the task was executed.
        """
        return (self._end - self._start)

    def __enter__(self):
        """This gets executed when entering the `with` block.
        """
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """This gets executed when exiting the `with` block.
        """
        self._end = time.time()

        if(self._time_lord is None):
            return

        self._time_lord.add_task_record(self._name, self.duration)
