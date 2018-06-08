"""The livetime module provides general functionality for detector up-time.
"""
import abc
import numpy as np

from skylab.core.py import issequence

class LiveTime(object):
    """The abstract base class ``LiveTime`` defines an interface to query the
    up-time of the dector.

    The class holds an internal Nx2 float64 ndarray
    ``_uptime_mjd_intervals_arr``, where
    the first and second elements of the second axis is the start and end time
    of the up-time interval, respectively. This data array needs to be set by
    the derived class by setting the ``_uptime_mjd_intervals`` property with
    the appropriate data array.

        Note 1: The intervals must be sorted ascedent in time.

        Note 2: By definition the lower edge is included in the interval,
            whereas the upper edge is excluded from the interval.

        Note 3: The intervals must not overlap.

    The integrity of the internal mjd interval array will be ensured by the
    property setter method by calling the ``assert_mjd_intervals_integrity``
    method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # The internal Nx2 numpy ndarray holding the MJD intervals when the
        # detector was taking data.
        self._uptime_mjd_intervals_arr = np.ndarray((0,2), dtype=np.float64)

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

        bins = np.reshape(self._uptime_mjd_intervals_arr, (self._uptime_mjd_intervals_arr.size,))
        # Check if the bin edges are monotonically increasing.
        if(not np.all(np.diff(bins) > 0)):
            raise ValueError('The interval edges of the internal MJD interval array are not monotonically increasing!')

    @property
    def _uptime_mjd_intervals(self):
        """The Nx2 numpy ndarray holding the up-time intervals of the detector.
        The first and second elements of the second axis is the start and stop
        time of the up-time interval, respectively.
        """
        return self._uptime_mjd_intervals_arr
    @_uptime_mjd_intervals.setter
    def _uptime_mjd_intervals(self, arr):
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
        """The two-element tuple holding the time window which is spanned by all
        the MJD uptime intervals. By definition this included possible
        dector down-times.
        """
        return (self._uptime_mjd_intervals_arr[0,0],
                self._uptime_mjd_intervals_arr[-1,1])

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
        # Get a view on the mjd intervals where each time is a lower bin edge.
        # Odd bins are on-time intervals, and even bins are off-time intervals.
        on_off_intervals = np.reshape(self._uptime_mjd_intervals_arr, (self._uptime_mjd_intervals_arr.size,))

        # Get the interval indices.
        # Note: For MJD values outside the total interval range, the np.digitize
        # function will return either 0, or len(bins). Since, there is always
        # an even amount of intervals edges, and 0 is also an 'even' number,
        # those MJDs will correspond to off-time automatically.
        idxs = np.digitize(mjds, on_off_intervals)

        return idxs

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

        on_off_intervals = np.reshape(self._uptime_mjd_intervals_arr, (self._uptime_mjd_intervals_arr.size,))

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
        ontimes = np.where(odd_idxs_mask, cum_ontime_bins[idxs-1] + mjds - on_off_intervals[onoff_idxs-1], cum_ontime_bins[idxs])

        if(not issequence(mjd)):
            return np.asscalar(ontimes)
        return ontimes

    def is_live(self, mjd):
        """Checks if the detector is live at the given MJD time. MJD times
        outside any live-time interval will be masked as False.

        Parameters
        ----------
        mjd : float | sequence of float
            The time in MJD.

        Returns
        -------
        is_live : bool | array of bool
            True if the detector was on at the given time.
        """
        mjds = np.atleast_1d(mjd)

        # Get the on-off-time interval indices corresponding to the given MJD
        # values.
        onoff_idxs = self._get_onoff_interval_indices(mjds)

        # Mask odd indices as on-time (True) MJD values.
        is_live = np.array(onoff_idxs & 0x1, dtype=np.bool)

        if(not issequence(mjd)):
            return np.asscalar(is_live)
        return is_live


