# -*- coding: utf-8 -*-

from __future__ import division

def source_dec_shift_linear(x, w, L, U):
    """Calculates the shift of the sine of the source declination, in order to
    allow the construction of the source sine declination band with
    sin(dec_src) +/- w. This shift function, S(x), is implemented as a line
    with the following points:

        S(L) = w
        S((L+U)/2) = 0
        S(U) = -w

    Parameters
    ----------
    x : 1D numpy ndarray
        The sine of the source declination for each source.
    w : float
        The half size of the sin(dec)-window.
    L : float
        The lower value of the allowed sin(dec) range.
    U : float
        The upper value of the allowed sin(dec) range.

    Returns
    -------
    S : 1D numpy ndarray
        The sin(dec) shift of the sin(dec) values of the given sources, such
        that ``sin(dec_src) + S`` is the new sin(dec) of the source, and
        ``sin(dec_src) + S +/- w`` is always within the sin(dec) range [L, U].
    """
    x = np.atleast_1d(x)

    m = -2*w/(U-L)
    b = w*(L+U)/(U-L)
    S = m*x+b

    return S

def source_dec_shift_cubic(x, w, L, U):
    """Calculates the shift of the sine of the source declination, in order to
    allow the construction of the source sine declination band with
    sin(dec_src) +/- w. This shift function, S(x), is implemented as a cubic
    function with the following points:

        S(L) = w
        S((L+U)/2) = 0
        S(U) = -w

    Parameters
    ----------
    x : 1D numpy ndarray
        The sine of the source declination for each source.
    w : float
        The half size of the sin(dec)-window.
    L : float
        The lower value of the allowed sin(dec) range.
    U : float
        The upper value of the allowed sin(dec) range.

    Returns
    -------
    S : 1D numpy ndarray
        The sin(dec) shift of the sin(dec) values of the given sources, such
        that ``sin(dec_src) + S`` is the new sin(dec) of the source, and
        ``sin(dec_src) + S +/- w`` is always within the sin(dec) range [L, U].
    """
    x = np.atleast_1d(x)

    m = w / (A - 0.5*(L+U))**3
    S = m * np.power(x-0.5*(L+U),3)

    return S
