# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import interp1d


def make_spline_1d(
        x,
        y,
        kind='linear',
        **kwargs):
    """Creates a 1D spline for the function y(x) using
    :class:`scipy.interpolate.interp1d`.

    Parameters
    ----------
    x : array_like
        The x values.
    y : array_like
        The y values.
    kind : str
        The kind of the spline. See the :class:`scipy.interpolate.interp1d`
        documentation for possible values. Default is ``'linear'``.
    **kwargs
        Additional keyword arguments are passed to the :class:`~scipy.interpolate.interp1d` function.

    Returns
    -------
    spline :
        The created 1D spline instance.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # The interp1d function requires unique x values. So we need to sort x in
    # increasing order and mask out repeating x values.
    xy = np.array(sorted(zip(x, y)), dtype=y.dtype)
    x = xy[:, 0]
    unique_x_mask = np.concatenate(([True], np.invert(
        x[1:] <= x[:-1])))
    x = x[unique_x_mask]
    y = xy[:, 1][unique_x_mask]

    spline = interp1d(
        x,
        y,
        kind=kind,
        copy=False,
        assume_sorted=True,
        **kwargs)

    return spline


class CatmullRomRegular1DSpline(
        object):
    """This class provides a one-dimensional Catmull-Rom spline which is a C^1
    continous spline, where the control points coincide with the data points.
    The x data points need to be equal distant.

    .. note::

        The first and last data point are not part of the splined curve!

    """

    def __init__(
            self,
            x,
            y,
            **kwargs,
    ):
        """Creates a new CatmullRom1DSpline instance.

        Parameters
        ----------
        x : instance of ndarray
            The x values of the data points.
        y : instance of ndarray
            The y values of the data points.
        """
        super().__init__(
            **kwargs)

        if len(x) != len(y):
            raise ValueError(
                f'The number of x ({len(x)}) and y ({len(y)}) data values '
                'need to be equal!')
        if len(x) < 4:
            raise ValueError(
                f'The number of data points ({len(x)}) must be at least 4!')

        unique_delta_x = np.unique(np.diff(x))
        if len(unique_delta_x) != 1:
            raise ValueError(
                'The data points must be equal distant in x!')

        self._delta_x = unique_delta_x[0]
        self._x_start = x[1]
        self._x_stop = x[-2]

        # Calculates the number of segments given the number of data points.
        # Since there are 4 points necessary per segment, we need to subtract 3.
        self._num_segments = len(x) - 3

        # Calculate the required data for each segment.
        self._segment_data = []
        for seg_idx in range(self._num_segments):
            sl = slice(seg_idx, seg_idx+4)
            (t1, t2, t3) = self._calc_segment_coefficients(Px=x[sl], Py=y[sl])
            self._segment_data.append(
                (t1, t2, t3, x[sl], y[sl]))

    def _eval_for_valid_x(
            self,
            x,
    ):
        """Evaluates the spline given valid x-values in data coordinates.

        Parameters
        ----------
        x : instance of ndarray
            The instance of ndarray holding the valid values for which the
            spline should get evaluated.

        Returns
        -------
        y : instance of ndarray
            The instance of ndarray with the spline values at the given x
            values.
        """
        y = np.zeros((len(x),), dtype=np.float64)

        # Determine on which spline segment the data value belongs to.
        seg_idxs = np.empty((len(x),), dtype=np.int64)
        np.floor(
            (x - self._x_start) / self._delta_x,
            out=seg_idxs,
            casting='unsafe')
        m = x == self._x_stop
        seg_idxs[m] = self._num_segments - 1

        # Loop over the unique segments.
        for seg_idx in np.unique(seg_idxs):
            # Create a mask of the values belonging to this segment.
            mask = seg_idxs == seg_idx
            n_points = np.count_nonzero(mask)

            (t1, t2, t3, Px, Py) = self._segment_data[seg_idx]

            # Note: This linear relation between x and t is the reason why the
            # data point x values must be equal distant.
            t = (x[mask] - Px[1]) / (Px[2] - Px[1]) * (t2 - t1) + t1
            t = t.reshape(n_points, 1)

            P0 = (Px[0], Py[0])
            P1 = (Px[1], Py[1])
            P2 = (Px[2], Py[2])
            P3 = (Px[3], Py[3])

            t1_m_t0 = t1
            t2_m_t0 = t2
            t2_m_t1 = t2 - t1
            t3_m_t1 = t3 - t1
            t3_m_t2 = t3 - t2

            t_m_t0 = t
            t_m_t1 = t - t1
            t2_m_t = t2 - t
            t3_m_t = t3 - t

            A1 = (t1 - t) / (t1_m_t0) * P0 + (t_m_t0) / (t1_m_t0) * P1
            A2 = (t2_m_t) / (t2_m_t1) * P1 + (t_m_t1) / (t2_m_t1) * P2
            A3 = (t3_m_t) / (t3_m_t2) * P2 + (t - t2) / (t3_m_t2) * P3
            B1 = (t2_m_t) / (t2_m_t0) * A1 + (t_m_t0) / (t2_m_t0) * A2
            B2 = (t3_m_t) / (t3_m_t1) * A2 + (t_m_t1) / (t3_m_t1) * A3
            seg_points = t2_m_t / t2_m_t1 * B1 + t_m_t1 / t2_m_t1 * B2

            y[mask] = seg_points[:, 1]

        return y

    def __call__(
            self,
            x,
            oor_value=np.nan,
    ):
        """Evaluates the spline given x-values in data coordinates.

        Parameters
        ----------
        x : instance of ndarray
            The instance of ndarray holding the values for which the spline
            should get evaluated.

        Returns
        -------
        y : instance of ndarray
            The instance of ndarray with the spline values at the given x
            values.
        """
        x = np.atleast_1d(x)

        m_valid_x = (x >= self._x_start) & (x <= self._x_stop)

        y = np.full((len(x),), oor_value, dtype=np.float64)
        y[m_valid_x] = self._eval_for_valid_x(x=x[m_valid_x])

        return y

    def _calc_tj(
            self,
            ti,
            Pi_x,
            Pi_y,
            Pj_x,
            Pj_y,
    ):
        """Calculates the next segment coefficient ``tj`` given the previous
        segment coefficient ``ti`` and the previous and next data point
        ``(Pi_x, Pi_y)`` and ``(Pj_x, Pj_y)``, respectively.

        Parameters
        ----------
        ti : float
            The previous segment coefficient.
        Pi_x : float
            The x-value of the previous data point.
        Pi_y : float
            The y-value of the previous data point.
        Pj_x : float
            The x-value of the next data point.
        Pj_y : float
            The y-value of the next data point.

        Returns
        -------
        tj : float
            The next segment coefficient.
        """
        dx = Pj_x - Pi_x
        dy = Pj_y - Pi_y
        tj = ti + np.sqrt(np.sqrt(dx*dx + dy*dy))

        return tj

    def _calc_segment_coefficients(
            self,
            Px,
            Py,
    ):
        """Calculates the segment coefficients t1, t2, and t3 given the 4
        data (control) points of the segment. The coefficient t0 is 0 by
        definition.

        Parameters
        ----------
        Px : instance of ndarray
            The (4,)-shaped numpy ndarray holding the 4 x-values of the
            segment's data points.
        Py : instance of ndarray
            The (4,)-shaped numpy ndarray holding the 4 y-values of the
            segment's data points.
        """
        t0 = 0
        t1 = self._calc_tj(
            ti=t0, Pi_x=Px[0], Pi_y=Py[0], Pj_x=Px[1], Pj_y=Py[1])
        t2 = self._calc_tj(
            ti=t1, Pi_x=Px[1], Pi_y=Py[1], Pj_x=Px[2], Pj_y=Py[2])
        t3 = self._calc_tj(
            ti=t2, Pi_x=Px[2], Pi_y=Py[2], Pj_x=Px[3], Pj_y=Py[3])

        return (t1, t2, t3)
