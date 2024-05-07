# -*- coding: utf-8 -*-

import numpy as np

from scipy import (
    integrate,
    interpolate,
)

from skyllh.core.binning import (
    get_bincenters_from_binedges,
)


class FctSpline1D(object):
    """Class to represent a 1D function spline using the PchipInterpolator
    class from scipy.

    The evaluate the spline, use the ``__call__`` method.
    """

    def __init__(
            self,
            f,
            x_binedges,
            norm=False,
            **kwargs):
        """Creates a new 1D function spline using the PchipInterpolator
        class from scipy.

        Parameters
        ----------
        f : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the function values at the bin centers.
        x_binedges : (n_x+1,)-shaped 1D numpy ndarray
            The numpy ndarray holding the bin edges of the x-axis.
        norm : bool
            Whether to precalculate and save normalization internally.
        """
        super().__init__(**kwargs)

        self.x_binedges = np.copy(x_binedges)

        self.x_min = self.x_binedges[0]
        self.x_max = self.x_binedges[-1]

        x = get_bincenters_from_binedges(self.x_binedges)

        self.spl_f = interpolate.PchipInterpolator(
            x, f, extrapolate=False
        )

        self.norm = None
        if norm:
            self.norm = integrate.quad(
                self.__call__,
                self.x_min,
                self.x_max,
                limit=200,
                full_output=1
            )[0]

    def __call__(
            self,
            x,
            oor_value=0):
        """Evaluates the spline at the given x values. For x-values
        outside the spline's range, the oor_value is returned.

        Parameters
        ----------
        x : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the x values at which the spline should
            get evaluated.
        oor_value : float
            The value for out-of-range (oor) coordinates.

        Returns
        -------
        f : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the evaluated values of the spline.
        """
        f = self.spl_f(x)
        f = np.where(np.isnan(f), oor_value, f)

        return f

    def evaluate(
            self,
            *args,
            **kwargs):
        """Alias for the __call__ method.
        """
        return self(*args, **kwargs)


class FctSpline2D(object):
    """Class to represent a 2D function spline using the RectBivariateSpline
    class from scipy.

    The spline is constructed in the log10 space of the function value to
    ensure a smooth spline.

    The evaluate the spline, use the ``__call__`` method.
    """

    def __init__(
            self,
            f,
            x_binedges,
            y_binedges,
            **kwargs):
        """Creates a new 2D function spline using the RectBivariateSpline
        class from scipy.

        Parameters
        ----------
        f : (n_x, n_y)-shaped 2D numpy ndarray
            he numpy ndarray holding the function values at the bin centers.
        x_binedges : (n_x+1,)-shaped 1D numpy ndarray
            The numpy ndarray holding the bin edges of the x-axis.
        y_binedges : (n_y+1,)-shaped 1D numpy ndarray
            The numpy ndarray holding the bin edges of the y-axis.
        """
        super().__init__(**kwargs)

        self.x_binedges = np.copy(x_binedges)
        self.y_binedges = np.copy(y_binedges)

        self.x_min = self.x_binedges[0]
        self.x_max = self.x_binedges[-1]
        self.y_min = self.y_binedges[0]
        self.y_max = self.y_binedges[-1]

        x = get_bincenters_from_binedges(self.x_binedges)
        y = get_bincenters_from_binedges(self.y_binedges)

        # Note: For simplicity we approximate zero bins with 1000x smaller
        # values than the minimum value. To do this correctly, one should store
        # the zero bins and return zero when those bins are requested.
        z = np.empty(f.shape, dtype=np.double)
        m = f > 0
        z[m] = np.log10(f[m])
        z[np.invert(m)] = np.min(z[m]) - 3

        self.spl_log10_f = interpolate.RectBivariateSpline(
            x, y, z, kx=3, ky=3, s=0)

    def __call__(
            self,
            x,
            y,
            oor_value=0):
        """Evaluates the spline at the given coordinates. For coordinates
        outside the spline's range, the oor_value is returned.

        Parameters
        ----------
        x : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the x values at which the spline should
            get evaluated.
        y : (n_y,)-shaped 1D numpy ndarray
            The numpy ndarray holding the y values at which the spline should
            get evaluated.
        oor_value : float
            The value for out-of-range (oor) coordinates.

        Returns
        -------
        f : (n_x, n_y)-shaped 2D numpy ndarray
            The numpy ndarray holding the evaluated values of the spline.
        """
        m_x_oor = (x < self.x_min) | (x > self.x_max)
        m_y_oor = (y < self.y_min) | (y > self.y_max)

        (m_xx_oor, m_yy_oor) = np.meshgrid(m_x_oor, m_y_oor, indexing='ij')
        m_xy_oor = m_xx_oor | m_yy_oor

        f = np.power(10, self.spl_log10_f(x, y))
        f[m_xy_oor] = oor_value

        return f


def clip_grl_start_times(grl_data):
    """Make sure that the start time of a run is not smaller than the stop time
    of the previous run.

    Parameters
    ----------
    grl_data : instance of numpy structured ndarray
        The numpy structured ndarray of length N_runs, with the following
        fields:

        start : float
            The start time of the run.
        stop : float
            The stop time of the run.
    """
    start = grl_data['start']
    stop = grl_data['stop']

    m = (start[1:] - stop[:-1]) < 0
    new_start = np.where(m, stop[:-1], start[1:])

    grl_data['start'][1:] = new_start


def psi_to_dec_and_ra(
        rss,
        src_dec,
        src_ra,
        psi):
    """Generates random declinations and right-ascension coordinates for the
    given source location and opening angle `psi`.

    Parameters
    ----------
    rss : instance of RandomStateService
        The instance of RandomStateService to use for drawing random numbers.
    src_dec : float
        The declination of the source in radians.
    src_ra : float
        The right-ascension of the source in radians.
    psi : 1d ndarray of float
        The opening-angle values in radians.

    Returns
    -------
    dec : 1d ndarray of float
        The declination values.
    ra : 1d ndarray of float
        The right-ascension values.
    """

    psi = np.atleast_1d(psi)

    # Transform everything in radians and convert the source declination
    # to source zenith angle
    a = psi
    b = np.pi/2 - src_dec
    c = src_ra
    # Random rotation angle for the 2D circle
    t = rss.random.uniform(0, 2*np.pi, size=len(psi))

    # Parametrize the circle
    x = (
        (np.sin(a)*np.cos(b)*np.cos(c)) * np.cos(t) +
        (np.sin(a)*np.sin(c)) * np.sin(t) -
        (np.cos(a)*np.sin(b)*np.cos(c))
    )
    y = (
        -(np.sin(a)*np.cos(b)*np.sin(c)) * np.cos(t) +
        (np.sin(a)*np.cos(c)) * np.sin(t) +
        (np.cos(a)*np.sin(b)*np.sin(c))
    )
    z = (
        (np.sin(a)*np.sin(b)) * np.cos(t) +
        (np.cos(a)*np.cos(b))
    )

    # Convert back to right-ascension and declination.
    # This is to distinguish between diametrically opposite directions.
    zen = np.arccos(z)
    azi = np.arctan2(y, x)

    dec = np.pi/2 - zen
    ra = np.pi - azi

    return (dec, ra)


def create_energy_cut_spline(
        ds,
        exp_data,
        spl_smooth):
    """Create the spline for the declination-dependent energy cut
    that the signal generator needs for injection in the southern sky
    Some special conditions are needed for IC79 and IC86_I, because
    their experimental dataset shows events that should probably have
    been cut by the IceCube selection.
    """
    data_exp = exp_data.copy(keep_fields=['sin_dec', 'log_energy'])
    if ds.name == 'IC79':
        m = np.invert(np.logical_and(
            data_exp['sin_dec'] < -0.75,
            data_exp['log_energy'] < 4.2))
        data_exp = data_exp[m]
    if ds.name == 'IC86_I':
        m = np.invert(np.logical_and(
            data_exp['sin_dec'] < -0.2,
            data_exp['log_energy'] < 2.5))
        data_exp = data_exp[m]

    sin_dec_binning = ds.get_binning_definition('sin_dec')
    sindec_edges = sin_dec_binning.binedges
    min_log_e = np.zeros(len(sindec_edges)-1, dtype=float)
    for i in range(len(sindec_edges)-1):
        mask = np.logical_and(
            data_exp['sin_dec'] >= sindec_edges[i],
            data_exp['sin_dec'] < sindec_edges[i+1])
        min_log_e[i] = np.min(data_exp['log_energy'][mask])
    del data_exp
    sindec_centers = 0.5 * (sindec_edges[1:]+sindec_edges[:-1])

    spline = interpolate.UnivariateSpline(
        sindec_centers, min_log_e, k=2, s=spl_smooth)

    return spline
