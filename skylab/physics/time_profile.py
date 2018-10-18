# -*- coding: utf-8 -*-

import abc
import numpy as np

from skylab.core.py import float_cast

class TimeProfileModel(object):
    """Abstract base class for an emission time profile of a source.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, t_start, t_end):
        """Creates a new time profile instance.

        Parameters
        ----------
        t_start : float
            The MJD start time of the box profile.
        t_end : float
            The MJD end time of the box profile.
        """
        super(TimeProfileModel, self).__init__()

        self.t_start = t_start
        self.t_end = t_end

    @property
    def t_start(self):
        """The MJD start time of the box profile.
        """
        return self._t_start
    @t_start.setter
    def t_start(self, t):
        t = float_cast(t,
            'The t_start property must be castable to type float!'
        )
        self._t_start = t

    @property
    def t_end(self):
        """The MJD end time of the box profile.
        """
        return self._t_end
    @t_end.setter
    def t_end(self, t):
        t = float_cast(t,
            'The t_end property must be castable to type float!'
        )
        self._t_end = t

    @property
    def duration(self):
        """The duration (in days) of the time profile.
        """
        return self._t_end - self._t_start

    @abc.abstractmethod
    def move(self, dt):
        """Abstract method to move the time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The MJD time difference of how far to move the time profile in time.
            This can be a positive or negative time shift.
        """
        pass

    @abc.abstractmethod
    def get_integral(self, t1, t2):
        """This method is supposed to calculate the integral of the time profile
        from time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The MJD start time of the integration.
        t2 : float | array of float
            The MJD end time of the integration.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile.
        """
        pass


class BoxTimeProfile(TimeProfileModel):
    """The BoxTimeProfile describes a box-shaped emission time profile of a
    source.
    """
    def __init__(self, t_start, t_end):
        """Creates a new box-shaped time profile instance.

        Parameters
        ----------
        t_start : float
            The MJD start time of the box profile.
        t_end : float
            The MJD end time of the box profile.
        """
        super(BoxTimeProfile, self).__init__(t_start, t_end)

    def move(self, dt):
        """Moves the box-shaped time profile by the time difference dt.

        Parameters
        ----------
        dt : float
            The MJD time difference of how far to move the time profile in time.
            This can be a positive or negative time shift.
        """
        self._t_start += dt
        self._t_end += dt

    @property
    def T0(self):
        """The time of the mid point of the box.
        """
        return 0.5*(self._t_start + self._t_end)
    @T0.setter
    def T0(self, t):
        old_T0 = self.T0
        dt = t - old_T0
        self.move(dt)

    @property
    def Tw(self):
        """The time width (in days) of the box.
        """
        return self._t_end - self._t_start
    @Tw.setter
    def Tw(self, w):
        T0 = self.T0
        self._t_start = T0 - 0.5*w
        self._t_end = T0 + 0.5*w

    def get_integral(self, t1, t2):
        """Calculates the integral of the box-shaped time profile from MJD time
        t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The MJD start time(s) of the integration.
        t2 : float | array of float
            The MJD end time(s) of the integration.

        Returns
        -------
        integral : array of float
            The integral value(s).
        """
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

        integrals = np.zeros((t1.shape[0],), dtype=np.float)

        m = (t2 > self._t_start) & (t1 < self._t_end)
        N = np.count_nonzero(m)

        t1 = np.max(np.vstack((t1[m], np.repeat(self._t_start, N))).T, axis=1)
        t2 = np.min(np.vstack((t2[m], np.repeat(self._t_end, N))).T, axis=1)

        f = 1./self.duration

        integrals[m] = f*(t2-t1)

        return integrals
