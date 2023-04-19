# -*- coding: utf-8 -*-

"""The livetime module provides general functionality for detector up-time.
"""
import numpy as np

from skyllh.core.random import RandomStateService
from skyllh.core.py import issequence, classname

class Livetime(object):
    """The ``Livetime`` class defines an interface to query the up-time of the
    detector.
    """
    def __init__(self, uptime_mjd_intervals_arr):
        """Creates a new Livetime object from a (N,2)-shaped ndarray holding
        the uptime intervals.

        Parameters
        ----------
        uptime_mjd_intervals_arr : (N,2)-shaped ndarray
            The (N,2)-shaped ndarray holding the start and end times of each
            up-time interval.

            Note 1: The intervals must be sorted ascedent in time.

            Note 2: By definition the lower edge is included in the interval,
                whereas the upper edge is excluded from the interval.

            Note 3: The intervals must not overlap.

            The integrity of the internal mjd interval array will be ensured by
            the property setter method of ``uptime_mjd_intervals_arr`` by
            calling the ``assert_mjd_intervals_integrity`` method.
        """
        # The internal Nx2 numpy ndarray holding the MJD intervals when the
        # detector was taking data.
        self.uptime_mjd_intervals_arr = uptime_mjd_intervals_arr

    def assert_mjd_intervals_integrity(self):
        """Checks if the internal MJD interval array conforms with all its
        data requirements.

        Raises TypeError if the data array is not a float64 array.
        Raises ValueError if the data integrity is broken.
        """
        if(not isinstance(self._uptime_mjd_intervals_arr, np.ndarray)):
            raise TypeError('The internal MJD interval array must be of type ndarray!')

        if(self._uptime_mjd_intervals_arr.dtype != np.float64):
            raise TypeError('The type of the internal MJD interval array is not float64!')

        # Check the shape of the array.
        if(self._uptime_mjd_intervals_arr.ndim != 2):
            raise ValueError('The dimensionality of the internel MJD interval array must be 2!')
        if(self._uptime_mjd_intervals_arr.shape[1] != 2):
            raise ValueError('The length of the second axis of the internal MJD interval array must be 2!')

        bins = self._onoff_intervals
        # Check if the bin edges are monotonically non decreasing.
        if(not np.all(np.diff(bins) >= 0)):
            raise ValueError('The interval edges of the internal MJD interval array are not monotonically non decreasing!')

    @property
    def uptime_mjd_intervals_arr(self):
        """The Nx2 numpy ndarray holding the up-time intervals of the detector.
        The first and second elements of the second axis is the start and stop
        time of the up-time interval, respectively.
        """
        return self._uptime_mjd_intervals_arr
    @uptime_mjd_intervals_arr.setter
    def uptime_mjd_intervals_arr(self, arr):
        self._uptime_mjd_intervals_arr = arr
        self.assert_mjd_intervals_integrity()

    @property
    def n_uptime_mjd_intervals(self):
        """The number of on-time intervals defined.
        """
        return self._uptime_mjd_intervals_arr.shape[0]

    @property
    def livetime(self):
        """The integrated live-time in days, based on the internal up-time time
        intervals.
        """
        return np.sum(np.diff(self._uptime_mjd_intervals_arr))

    @property
    def time_window(self):
        """(read-only) The two-element tuple holding the time window which is
        spanned by all the MJD uptime intervals.
        By definition this included possible detector down-time periods.
        """
        return (self._uptime_mjd_intervals_arr[0,0],
                self._uptime_mjd_intervals_arr[-1,1])

    @property
    def time_start(self):
        """(read-only) The start of the detector live-time.
        """
        return self._uptime_mjd_intervals_arr[0,0]

    @property
    def time_end(self):
        """(read-only) The end of the detector live-time.
        """
        return self._uptime_mjd_intervals_arr[-1,1]

    @property
    def _onoff_intervals(self):
        """A view on the mjd intervals where each time is a lower bin edge.
        Hence, odd array elements (bins) are on-time intervals, and even array
        elements are off-time intervals.
        """
        return np.reshape(self._uptime_mjd_intervals_arr, (self._uptime_mjd_intervals_arr.size,))

    def __str__(self):
        """Pretty string representation of the Livetime class instance.
        """
        s = '%s(time_window=(%.6f, %.6f))'%(
            classname(self), self.time_window[0], self.time_window[1])
        return s

    def _get_onoff_interval_indices(self, mjds):
        """Retrieves the indices of the on-time and off-time intervals, which
        correspond to the given MJD values.

        Odd indices correspond to detector on-time intervals and even indices
        to detector off-time intervals.

        The indices are in the range (0, 2*N), where N is the number of on-time
        intervals. The index 0 corresponds to a time prior to the first on-time
        interval, whereas 2*N corresponds to a time past the last on-time
        interval.

        Parameters
        ----------
        mjds : numpy array of floats
            The array of MJD values.

        Returns
        -------
        idxs : numpy array of ints
            The array of the on-off-time interval indices that correspond to the
            given MJD values.
        """
        # Get the interval indices.
        # Note: For MJD values outside the total interval range, the np.digitize
        # function will return either 0, or len(bins). Since, there is always
        # an even amount of intervals edges, and 0 is also an 'even' number,
        # those MJDs will correspond to off-time automatically.
        idxs = np.digitize(mjds, self._onoff_intervals)

        return idxs

    def load_from_ontime_mjd_intervals(self, intervals):
        """Loads the internal MJD uptime intervals from the given interval
        array.

        Parameters
        ----------
        intervals : Nx2 ndarray holding the MJD edges of the on-time intervals.

        """
        self._uptime_mjd_intervals = intervals

    def get_ontime_intervals_between(self, t_start, t_end):
        """Creates a (N,2)-shaped ndarray holding the on-time detector intervals
        between the given time range from t_start to t_end.

        Parameters
        ----------
        t_start : float
            The MJD start time of the time range to consider. This might be the
            lower bound of the first on-time interval.
        t_end : float
            The MJD end time of the time range to consider. This might be the
            upper bound of the last on-time interval.

        Returns
        -------
        ontime_intervals : (N,2)-shaped ndarray
            The (N,2)-shaped ndarray holding the on-time detector intervals.
        """
        onoff_intervals = self._onoff_intervals

        (t_start_idx, t_end_idx) = self._get_onoff_interval_indices((t_start, t_end))
        if(t_start_idx % 2 == 0):
            # t_start is during off-time. Use the next on-time lower edge as
            # first on-time edge.
            t_start = onoff_intervals[t_start_idx]
        else:
            t_start_idx -= 1
        if(t_end_idx % 2 == 0):
            # t_end is during off-time. Use the previous on-time upper edge as
            # the last on-time edge.
            t_end = onoff_intervals[t_end_idx-1]
        else:
            t_end_idx += 1

        # The t_start_idx and t_end_idx variables hold even indices.
        N_ontime_intervals = int((t_end_idx - t_start_idx)/2)

        ontime_intervals_flat = np.empty(
            (N_ontime_intervals*2,), dtype=np.float64)
        # Set the first and last on-time interval edges.
        ontime_intervals_flat[0] = t_start
        ontime_intervals_flat[-1] = t_end
        if(N_ontime_intervals > 1):
            # Fill also the interval edges of the intermediate on-time bins.
            ontime_intervals_flat[1:-1] = onoff_intervals[t_start_idx+1:t_end_idx-1]

        ontime_intervals = np.reshape(ontime_intervals_flat, (N_ontime_intervals,2))

        return ontime_intervals

    def get_ontime_upto(self, mjd):
        """Calculates the cumulative detector on-time up to the given time.

        Parameters
        ----------
        mjd : float | array of floats
            The time in MJD up to which the detector on-time should be
            calculated.

        Returns
        -------
        ontimes : float | ndarray of floats
            The ndarray holding the cumulative detector on-time corresponding
            to the the given MJD times.
        """
        mjds = np.atleast_1d(mjd)

        onoff_idxs = self._get_onoff_interval_indices(mjds)

        # Create a mask for all the odd indices, i.e. the MJDs falling into an
        # on-time interval.
        odd_idxs_mask = onoff_idxs & 0x1

        # Map the indices to the cum_ontime_bins array. Off-time indices will
        # be mapped to its prior on-time interval.
        #                               Odd indices.     Even indices.
        idxs = np.where(odd_idxs_mask, (onoff_idxs-1)/2, onoff_idxs/2 - 1)
        # At this point, there could be indices of value -1 from MJD values
        # prior to the first on-time interval. So we just move all the indices
        # by one.
        idxs += 1

        # Create a cumulative on-time array with a leading 0 element for MJDs
        # prior to the first on-time interval.
        ontime_bins = np.diff(self._uptime_mjd_intervals_arr).reshape((self.n_uptime_mjd_intervals,))
        cum_ontime_bins = np.array([0], dtype=np.float64)
        cum_ontime_bins = np.append(cum_ontime_bins, np.cumsum(ontime_bins))

        # For odd (on-time) mjds, use the cumulative value of the previous bin
        # and add the part of the interval bin up to the mjd value.
        ontimes = np.where(odd_idxs_mask, cum_ontime_bins[idxs-1] + mjds - self._onoff_intervals[onoff_idxs-1], cum_ontime_bins[idxs])

        if(not issequence(mjd)):
            return np.asscalar(ontimes)
        return ontimes

    def is_on(self, mjd):
        """Checks if the detector is on at the given MJD time. MJD times
        outside any live-time interval will be masked as False.

        Parameters
        ----------
        mjd : float | sequence of float
            The time in MJD.

        Returns
        -------
        is_on : array of bool
            True if the detector was on at the given time.
        """
        mjd = np.atleast_1d(mjd)

        # Get the on-off-time interval indices corresponding to the given MJD
        # values.
        onoff_idxs = self._get_onoff_interval_indices(mjd)

        # Mask odd indices as on-time (True) MJD values and even indices as
        # off-time (False).
        is_on = np.array(onoff_idxs & 0x1, dtype=np.bool_)

        return is_on

    def draw_ontimes(self, rss, size):
        """Draws random MJD times based on the detector on-time intervals.

        Parameters
        ----------
        rss : RandomStateService
            The skyllh RandomStateService instance to use for drawing random
            numbers from.
        size : int
            The number of random MJD times to generate.

        Returns
        -------
        ontimes : ndarray
            The 1d array holding the generated MJD times.
        """
        # Create bin array with only on-time bins. We have to mask out the
        # off-time bins.
        ontime_bins = np.diff(self._onoff_intervals)
        mask = np.invert(
            np.array(
                np.linspace(0, ontime_bins.size-1, ontime_bins.size)%2,
                dtype=np.bool_))
        ontime_bins = ontime_bins[mask]

        # Create the cumulative array of the on-time bins.
        cum_ontime_bins = np.array([0], dtype=np.float64)
        cum_ontime_bins = np.append(cum_ontime_bins, np.cumsum(ontime_bins))

        #         |<--y->|
        # |----|  |-----------|  |-------|
        # l1   u1 l2     |xL  u2 ul3     u3
        #
        # x \el [0,1]
        # L = \sum (u_i - l_i)

        x = rss.random.uniform(0, 1, size)
        # Get the sum L of all the on-time intervals.
        L = cum_ontime_bins[-1]
        w = x*L
        idxs = np.digitize(w, cum_ontime_bins)
        l = self._uptime_mjd_intervals_arr[:,0]
        y = w - cum_ontime_bins[idxs-1]
        ontimes = l[idxs-1] + y

        return ontimes
