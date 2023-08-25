# -*- coding: utf-8 -*-

import numpy as np

from astropy import (
    units,
)
from astropy.coordinates import (
    AltAz,
    SkyCoord,
)
from astropy.time import (
    Time,
    TimeDelta,
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

    SIDEREAL_DAY = 23.9344696  # hours

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

        self._detector_model = detector_model
        self._livetime = livetime

        (self._st_hist,
         self._st_hist_binedges) = self.create_sidereal_time_histogram(
            livetime=livetime,
            st_bin_width_deg=st_bin_width_deg,
            longitude=detector_model.location,
        )

        dt_st_bin_sec = (
            (self._st_hist_binedges[1] - self._st_hist_binedges[0]) / 24 *
            self.SIDEREAL_DAY * 3600
        )
        self._st_livetime_sec_arr = self._st_hist * dt_st_bin_sec

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

    @property
    def st_livetime_sec_arr(self):
        """(read-only) The (N_sidereal_time_bins,)-shaped numpy.ndarray holding
        the integrated live-time in seconds for each sidereal time bin.
        """
        return self._st_livetime_sec_arr

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

    def create_src_st_zen_array(
            self,
            src_array,
    ):
        """Creates a source sidereal time zenith array for the given sources.

        Parameters
        ----------
        src_array : instance of numpy.ndarray
            The structured numpy.ndarray holding the declination and
            right-ascention of the sources. The following data fields need to
            exist::

                dec : float
                    The declination of the source.
                ra : float
                    The right-ascention of the source.

        Returns
        -------
        src_st_zen_arr : instance of numpy.ndarray
            The (N_st_hist_bins, N_sources)-shaped numpy.ndarray holding the
            local zenith coordinate of the source for the different sidereal
            times.
        """
        # Calculate a reference time which we will take as the midpoint of the
        # dataset's livetime.
        ref_time_mjd = 0.5*(
            self._livetime.time_start +
            self._livetime.time_stop)

        ref_time = Time(ref_time_mjd, format='mjd', scale='utc')
        ref_st = ref_time.sidereal_time(
            kind='apparent',
            longitude=self._detector_model.location).value

        st_hour2sec = 1/24 * self.SIDEREAL_DAY * 3600

        src_dec = np.atleast_1d(src_array['dec'])
        src_ra = np.atleast_1d(src_array['ra'])
        src_skycoord = SkyCoord(
            ra=src_ra*units.radian,
            dec=src_dec*units.radian,
            frame='icrs')

        src_st_zen_arr = np.empty(
            (len(self.st_hist), len(src_array)),
            dtype=np.float32,
        )

        for st_bin_idx in range(len(self.st_hist)):
            st_bc = 0.5*(
                self.st_hist_binedges[st_bin_idx] +
                self.st_hist_binedges[st_bin_idx+1])

            delta_st = st_bc - ref_st
            dt_sec = delta_st * st_hour2sec
            obstime = ref_time + TimeDelta(dt_sec, format='sec')

            src_altaz = src_skycoord.transform_to(AltAz(
                obstime=obstime,
                location=self._detector_model.location,
            ))

            src_zen = src_altaz.zen.to(units.radian).value

            src_st_zen_arr[st_bin_idx] = src_zen

        return src_st_zen_arr
