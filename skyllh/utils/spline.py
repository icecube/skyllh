# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import interp1d


def make_spline_1d(x, y, kind='linear', **kwargs):
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
    x = xy[:,0]
    unique_x_mask = np.concatenate(([True], np.invert(
        x[1:] <= x[:-1])))
    x = x[unique_x_mask]
    y = xy[:,1][unique_x_mask]

    spline = interp1d(x, y, kind=kind, copy=False, assume_sorted=True, **kwargs)

    return spline
