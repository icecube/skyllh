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

    def __init__(self, f, x_binedges, norm=False, **kwargs):
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

        self.spl_f = interpolate.PchipInterpolator(x, f, extrapolate=False)

        self.norm = None
        if norm:
            self.norm = integrate.quad(
                self.__call__, self.x_min, self.x_max, limit=200, full_output=1
            )[0]

    def __call__(self, x, oor_value=0):
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

    def evaluate(self, *args, **kwargs):
        """Alias for the __call__ method."""
        return self(*args, **kwargs)


class FctSpline2D(object):
    """Class to represent a 2D function spline using the RectBivariateSpline
    class from scipy.

    The spline is constructed in the log10 space of the function value to
    ensure a smooth spline.

    The evaluate the spline, use the ``__call__`` method.
    """

    def __init__(self, f, x_binedges, y_binedges, **kwargs):
        """Creates a new 2D function spline using the RectBivariateSpline
        class from scipy.

        Parameters
        ----------
        f : (n_x, n_y)-shaped 2D numpy ndarray
            The numpy ndarray holding the function values at the bin centers.
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
        # y = get_bincenters_from_binedges(self.y_binedges)

        # Hack to avoid awkward behaviors due to splining in sin(dec)
        y = np.repeat(self.y_binedges, repeats=2)[1:-1]
        y[1::2] -= 1e-10
        f = np.repeat(f, repeats=2, axis=1)

        # Note: For simplicity we approximate zero bins with 1000x smaller
        # values than the minimum value. To do this correctly, one should store
        # the zero bins and return zero when those bins are requested.
        z = np.empty(f.shape, dtype=np.double)
        m = f > 0
        z[m] = np.log10(f[m])
        z[np.invert(m)] = np.min(z[m]) - 3

        self.spl_log10_f = interpolate.RectBivariateSpline(x, y, z, kx=3, ky=1, s=0)

        # In case we have to renormalize when the evaluation is done...
        self._prepare_quadrature()

    def _prepare_quadrature(self, n=128):
        gx, gw = np.polynomial.legendre.leggauss(n)
        self._qx = 0.5 * (self.x_max - self.x_min) * gx + 0.5 * (
            self.x_max + self.x_min
        )
        self._qw = 0.5 * (self.x_max - self.x_min) * gw

    @staticmethod
    def _pow10(arr):
        return np.power(10.0, arr)

    def _mask_oor_axes(self, x, y):
        m_x = (x < self.x_min) | (x > self.x_max)
        m_y = (y < self.y_min) | (y > self.y_max)
        return m_x, m_y

    def _eval_paired(self, x, y):
        """Evaluate at paired points (grid=False) → 1D array."""
        return self._pow10(self.spl_log10_f(x, y, grid=False))

    def _eval_grid(self, x, y):
        """Evaluate on tensor grid with internal sort/unsort → 2D array."""
        x = np.asarray(x)
        y = np.asarray(y)

        # sort as required by RectBivariateSpline(grid=True)
        ix = np.argsort(x) if (x.size >= 2 and not np.all(np.diff(x) >= 0)) else None
        iy = np.argsort(y) if (y.size >= 2 and not np.all(np.diff(y) >= 0)) else None
        xs = x[ix] if ix is not None else x
        ys = y[iy] if iy is not None else y

        f_sorted = self._pow10(self.spl_log10_f(xs, ys, grid=True))

        # unsort to original order
        if ix is not None:
            f_sorted = f_sorted[np.argsort(ix), :]
        if iy is not None:
            f_sorted = f_sorted[:, np.argsort(iy)]
        return f_sorted

    def _renorm_per_y_grid(self, f2d, y, *, in_user_order=True):
        """Renormalize columns so ∫_x f(x, y) dx = 1 (grid=True)."""
        y = np.asarray(y)
        # For renorm we can evaluate on (qx, y) with grid=True (expects sorted y).
        iy = np.argsort(y) if (y.size >= 2 and not np.all(np.diff(y) >= 0)) else None
        ys = y[iy] if iy is not None else y

        fyq = self._pow10(self.spl_log10_f(self._qx, ys, grid=True))  # (nq, ny)
        Z = fyq.T @ self._qw  # (ny,)
        Z[Z == 0] = np.nan
        if iy is not None:
            Z = Z[np.argsort(iy)]  # back to user order
        f2d /= Z[np.newaxis, :]
        return f2d

    def _renorm_per_y_pairs(self, x, y, f):
        """Renormalize paired evaluations so each value is divided by Z(y_i)."""
        # Compute Z for each distinct y present (no sorting needed for outputs;
        # np.unique returns a sorted list, but we map back via 'inv').
        uniq_y, inv = np.unique(y, return_inverse=True)
        Z = np.empty_like(uniq_y, dtype=float)
        for k, yk in enumerate(uniq_y):
            yq = np.full(self._qx.shape, yk, dtype=float)
            vals = self._pow10(self.spl_log10_f(self._qx, yq, grid=False))  # (nq,)
            Z[k] = np.dot(vals, self._qw)
        Z[Z == 0] = np.nan
        f /= Z[inv]
        return f

    def __call__(self, x, y, oor_value=0, grid=False, renorm=True):
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
        oor_value : float | 0
            The value for out-of-range (oor) coordinates.
        grid : bool | False
            Whether the interpolation should return a 2D numpy array or a
            1D sequence of values.
        renorm_axis : bool | True
        Whether to renormalize the histogram along the x axis for each y-value.
        (This is useful when constructing the background energy PDF.)

        Returns
        -------
        f : numpy ndarray
            The numpy ndarray holding the evaluated values of the spline.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        m_x_oor, m_y_oor = self._mask_oor_axes(x, y)

        if not grid:
            if x.shape != y.shape:
                raise ValueError(
                    "For grid=False, x and y must have the same shape (paired points)."
                )

            m_oor = m_x_oor | m_y_oor
            f = np.empty_like(x, dtype=float)
            inside = ~m_oor

            if np.any(inside):
                f_in = self._eval_paired(x[inside], y[inside])
                if renorm:
                    f_in = self._renorm_per_y_pairs(x[inside], y[inside], f_in)
                f[inside] = f_in

            f[m_oor] = oor_value
            return f

        # grid=True: tensor output
        f2d = self._eval_grid(x, y)
        if renorm:
            f2d = self._renorm_per_y_grid(f2d, y)

        # OOR mask on tensor grid in user order
        mx2d, my2d = np.meshgrid(m_x_oor, m_y_oor, indexing="ij")
        f2d[mx2d | my2d] = oor_value
        return f2d


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
    start = grl_data["start"]
    stop = grl_data["stop"]

    m = (start[1:] - stop[:-1]) < 0
    new_start = np.where(m, stop[:-1], start[1:])

    grl_data["start"][1:] = new_start


def psi_to_dec_and_ra(rss, src_dec, src_ra, psi):
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
    b = np.pi / 2 - src_dec
    c = src_ra
    # Random rotation angle for the 2D circle
    t = rss.random.uniform(0, 2 * np.pi, size=len(psi))

    # Parametrize the circle
    x = (
        (np.sin(a) * np.cos(b) * np.cos(c)) * np.cos(t)
        + (np.sin(a) * np.sin(c)) * np.sin(t)
        - (np.cos(a) * np.sin(b) * np.cos(c))
    )
    y = (
        -(np.sin(a) * np.cos(b) * np.sin(c)) * np.cos(t)
        + (np.sin(a) * np.cos(c)) * np.sin(t)
        + (np.cos(a) * np.sin(b) * np.sin(c))
    )
    z = (np.sin(a) * np.sin(b)) * np.cos(t) + (np.cos(a) * np.cos(b))

    # Convert back to right-ascension and declination.
    # This is to distinguish between diametrically opposite directions.
    zen = np.arccos(z)
    azi = np.arctan2(y, x)

    dec = np.pi / 2 - zen
    ra = np.pi - azi

    return (dec, ra)


def create_energy_cut_spline_old(ds, exp_data, spl_smooth):
    """Create the spline for the declination-dependent energy cut
    that the signal generator needs for injection in the southern sky
    Some special conditions are needed for IC79 and IC86_I, because
    their experimental dataset shows events that should probably have
    been cut by the IceCube selection.
    """
    data_exp = exp_data.copy(keep_fields=["sin_dec", "log_energy"])
    if ds.name == "IC79":
        m = np.invert(
            np.logical_and(data_exp["sin_dec"] < -0.75, data_exp["log_energy"] < 4.2)
        )
        data_exp = data_exp[m]
    if ds.name == "IC86_I":
        m = np.invert(
            np.logical_and(data_exp["sin_dec"] < -0.2, data_exp["log_energy"] < 2.5)
        )
        data_exp = data_exp[m]

    sin_dec_binning = ds.get_binning_definition("sin_dec")
    sindec_edges = sin_dec_binning.binedges
    min_log_e = np.zeros(len(sindec_edges) - 1, dtype=float)
    for i in range(len(sindec_edges) - 1):
        mask = np.logical_and(
            data_exp["sin_dec"] >= sindec_edges[i],
            data_exp["sin_dec"] < sindec_edges[i + 1],
        )
        min_log_e[i] = np.min(data_exp["log_energy"][mask])
    del data_exp
    sindec_centers = 0.5 * (sindec_edges[1:] + sindec_edges[:-1])

    spline = interpolate.UnivariateSpline(sindec_centers, min_log_e, k=2, s=spl_smooth)

    return spline


def create_energy_cut_spline(ds, exp_data, spl_smooth, cumulative_thr):
    """Create the spline for the declination-dependent energy cut
    that the signal generator needs for injection in the southern sky.
    Cut bins which do not exceed the defined `cumulative_thr` threshold
    to exclude isolated, low-statistics bins which would cause extreme
    wiggles in the spline.

    Parameters
    ----------
    ds : instance of Dataset
        The instance of Dataset for which the spline should be calculated.
    exp_data : instance of DataFieldRecordArray
        The array containing the experimental data for dataset `ds`.
    spl_smooth : float

    cumulative_thr : float


    Returns
    -------
    spline : instance of scipy.interpolate.UnivariateSpline

    """
    data_exp = exp_data.copy(keep_fields=["sin_dec", "log_energy"])

    loge_binning = ds.get_binning_definition("log_energy")
    sin_dec_binning = ds.get_binning_definition("sin_dec")
    sindec_edges = sin_dec_binning.binedges

    # Initialize array to store the minumum allowed energy
    # for eaach sin(dec) bin.
    min_log_e = np.zeros(len(sindec_edges) - 1, dtype=float)
    for i in range(len(sindec_edges) - 1):
        # Select events in this declintion bin
        mask = np.logical_and(
            data_exp["sin_dec"] >= sindec_edges[i],
            data_exp["sin_dec"] < sindec_edges[i + 1],
        )

        # Make histogram along energy axis and compute the cum distribution
        counts, _ = np.histogram(
            data_exp["log_energy"][mask], bins=loge_binning.binedges
        )
        cumsum = np.cumsum(counts) / np.sum(counts)

        # Remove low-population energy bins to help the spline smoothness
        remove_isolated_bins = cumsum >= cumulative_thr
        # Define minimum energy for this declination bin
        min_log_e[i] = np.min(loge_binning.bincenters[remove_isolated_bins])

    del data_exp

    # Generate the energy spline as function of sin(dec)
    sindec_centers = 0.5 * (sindec_edges[1:] + sindec_edges[:-1])
    spline = interpolate.UnivariateSpline(sindec_centers, min_log_e, k=2, s=spl_smooth)

    return spline
