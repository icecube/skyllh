# -*- coding: utf-8 -*-

import numpy as np
import time

from skyllh.core import display

def get_stopwatch_lap_taker(sw):
    """Returns a function with call signature ``__call__(msg)`` to take a lap
    for the given Stopwatch instance.

    Parameters
    ----------
    sw : Stopwatch instance | None
        The Stopwatch instance for which the lap should be taken. If None, the
        returned function will do nothing.

    Returns
    -------
    func : function with call signature ``__call__(msg)``
        The function that will take the lap. If ``sw`` was set to None, the
        returned function will do nothing.
    """
    if(sw):
        sw_take_lap = sw.take_lap
        return lambda msg: sw_take_lap(msg)

    return lambda msg: False


class Stopwatch(object):
    """This class provides a stop-watch functionality to measure time duration
    within the program.
    """
    def __init__(self):
        self.reset()

    @property
    def started(self):
        """(read-only) Flag if the stop-watch was started.
        """
        return (len(self._start_times) > 0)

    @property
    def stopped(self):
        """(read-only) Flag if the stop-watch was stopped.
        """
        return (self._stop_time is not None)

    @property
    def n_laps(self):
        """(read-only) The number of laps taken.
        """
        n_start_times = len(self._start_times)
        if(n_start_times == 0):
            return 0
        return n_start_times-1

    def reset(self):
        """Resets the stop-watch.
        """
        self._start_times = []
        self._stop_time = None

        self._lap_names = []
        self._max_lap_name_len = 0

    def start(self):
        """Starts the stop-watch for the given key.

        Parameters
        ----------
        name : str | None
            The name of the stop-watch.

        Returns
        -------
        self : Stopwatch
            The Stopwatch instance itself.
        """
        self.reset()

        self._start_times.append(time.time())

        return self

    def take_lap(self, name):
        """Takes a current time and stores it as a lap with the given name.

        Parameters
        ----------
        name : str
            The name of the lap.

        Returns
        -------
        duration : float
            The lap duration in seconds.
        """
        if(not self.started):
            raise RuntimeError('Cannot take lap time, because the stop-watch was not started yet!')

        self._max_lap_name_len = np.maximum(self._max_lap_name_len, len(name))

        self._lap_names.append(name)
        self._start_times.append(time.time())

        duration = self._start_times[-1] - self._start_times[-2]

        return duration

    def stop(self, name):
        """Stops the stop-watch and returns the final duration
        (t_stop - t_start).

        Parameters
        ----------
        name : str
            The name of the final lap.

        Returns
        -------
        duration : float
            The total duration of the stop-watch.
        """
        if(not self.started):
            raise RuntimeError('Cannot stop the stop-watch, because the stop-watch was not started yet!')
        if(self.stopped):
            raise RuntimeError('Connot stop the stop-watch, because it was already stopped!')

        self.take_lap(name)
        self._stop_time = self._start_times[-1]

        duration = self._stop_time - self._start_times[0]
        return duration

    def get_duration(self):
        """Returns the total duration of the stop-watch.

        Returns
        -------
        duration : float
            The total duration of the stop-watch.
        """
        if(not self.started):
            raise RuntimeError('Cannot get final stop-watch duration, because the stop-watch was not started yet!')
        if(not self.stopped):
            raise RuntimeError('Cannot get final stop-watch duration, because the stop-watch was not stopped yet!')
        return self._stop_time - self._start_times[0]

    def __str__(self):
        """Generates a pretty string for this stop-watch.
        """
        s = 'Stopwatch: '
        if(self.stopped):
            s += '%g sec'%(self.get_duration())

        if(not self.started):
            s += 'Not started.'
            return s

        if(not self.stopped):
            s += 'Not stopped yet.'

        max_lap_name = np.minimum(self._max_lap_name_len, display.PAGE_WIDTH-16)
        n_laps = self.n_laps
        for i in range(n_laps):
            lap_name = self._lap_names[i][0:max_lap_name]
            lap_duration = self._start_times[i+1] - self._start_times[i]
            l = '\n[%'+str(max_lap_name)+'s] %g sec'
            s += l%(lap_name, lap_duration)

        return s
