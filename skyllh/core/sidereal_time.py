# -*- coding: utf-8 -*-

import numpy as np

from astropy.time import (
    Time,
)

from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.py import (
    classname,
)


class SiderealTimeService(
        object,
):
    """This class provides a sidereal time distribution service for a given
    live-time, i.e. dataset.
    """
    def __init__(
            self,
            detector_model,
            livetime,
            st_bin_width_deg,
    ):
        """Creates a new sidereal time service, that computes the sidereal time
        histogram for the given detector with the given live-time.

        Parameters
        ----------
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the location of the detector.
        livetime : instance of Livetime
            The instance of Livetime defining the detector's on-time intervals.
        st_bin_width_deg : float
            The sidereal time bin width in degree. It must be in the range
            [0, 360].
        """
        if not isinstance(detector_model, DetectorModel):
            raise TypeError(
                'The detector_model argument must be an instance of '
                'DetectorModel! '
                f'Its current type is {classname(detector_model)}!')

        if not isinstance(livetime, Livetime):
            raise TypeError(
                'The livetime argument must be an instance of '
                'Livetime! '
                f'Its current type is {classname(livetime)}!')

        (self._st_hist,
         self._st_hist_binedges) = self.create_sidereal_time_histogram(
            livetime=livetime,
            st_bin_width_deg=st_bin_width_deg,
            longitude=detector_model.location,
        )

    @property
    def st_hist(self):
        """(read-only) The (N_sidereal_time_bins,)-shaped numpy.ndarray holding
        the sidereal time histogram counts.
        """
        return self._st_hist

    @property
    def st_hist_binedges(self):
        """(read-only) The (N_sidereal_time_bins+1,)-shaped numpy.ndarray
        holding the bin edges of the sidereal time histogram.
        """
        return self._st_hist_binedges

    def create_sidereal_time_histogram(
            self,
            livetime,
            st_bin_width_deg,
            longitude,
    ):
        """Creates a histogram which counts how often a sidereal time interval
        has been covered by the on-time live-time intervals. The number of
        sidereal time intervals is specified through a delta angle between
        0 and 360 degree.

        Parameters
        ----------
        livetime : instance of Livetime
            The instance of Livetime providing the on-time MJD time intervals.
        st_bin_width_deg : float
            The sidereal time bin width in degree defining the number of
            sidereal time intervals. This must be a value in the range [0, 360].
        longitude : astropy.coordinates.EarthLocation
            The longitude to which the sidereal times should be calculated.
            This should be the detector's location on Earth.

        Returns
        -------
        hist : instance of numpy.ndarray
            The (N_sidereal_time_bins,)-shaped numpy ndarray holding the counts
            of the sidereal time intervals which are covered by the on-time
            intervals.
        bin_edges : instance of numpy.ndarray
            The (N_sidereal_time_bins+1,)-shaped numpy ndarray holding the
            sidereal time bin edges.
        """
        n_st_bins = int(np.ceil(360 / st_bin_width_deg))
        bin_edges = np.linspace(0, 24, num=n_st_bins+1, endpoint=True)

        sidereal_day = 23.9344696  # hours
        dt_st_bin_sec = (bin_edges[1] - bin_edges[0]) / 24 * sidereal_day * 3600

        mjd_starts = livetime.uptime_mjd_intervals_arr[:, 0]
        mjd_stops = livetime.uptime_mjd_intervals_arr[:, 1]

        hist = None
        for (mjd_start, mjd_stop) in zip(mjd_starts, mjd_stops):

            n_times = int(np.ceil((mjd_stop - mjd_start) / dt_st_bin_sec * 24*3600))
            dt = (mjd_stop - mjd_start) / n_times
            mjd_times = np.minimum(
                mjd_start + np.arange(n_times, dtype=np.int32) * dt,
                mjd_stop
            )

            # Transform MJD times into sidereal times.
            t = Time(mjd_times, format='mjd', scale='utc')
            st_times = t.sidereal_time(
                kind='apparent', longitude=longitude).value

            (hist_, _) = np.histogram(
                st_times,
                bins=bin_edges,
                density=False,
            )
            if hist is None:
                hist = hist_
            else:
                hist += hist_

        return (hist, bin_edges)
