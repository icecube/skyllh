# -*- coding: utf-8 -*-

"""IceCube specific coordinate utility functions.
"""

import numpy as np


def azi_to_ra_transform(azi, mjd):
    """Rotates the given IceCube azimuth angles into right-ascention angles for
    the given MJD times. This function is IceCube specific and assumes that the
    detector is located excently at the South Pole and neglects all astronomical
    effects like Earth's precession.

    Parameters
    ----------
    azi : instance of numpy.ndarray
        The array with the azimuth angles.
    mjd : instance of numpy.ndarray
        The array with the MJD times for each azimuth angle.

    Returns
    -------
    ra : instance of numpy.ndarray
        The right-ascention values.
    """
    # sidereal day = length * solar day
    _sidereal_length = 0.997269566
    _sidereal_offset = 2.54199002505
    sidereal_day_residuals = (mjd / _sidereal_length) % 1
    ra = _sidereal_offset + 2 * np.pi * sidereal_day_residuals - azi
    ra = np.mod(ra, 2*np.pi)

    return ra


def ra_to_azi_transform(ra, mjd):
    """Rotates the given right-ascention angles to local IceCube azimuth angles.

    Parameters
    ----------
    ra : instance of numpy.ndarray
        The array with the right-ascention angles.
    mjd : instance of numpy.ndarray
        The array with the MJD times for each right-ascention angle.

    Returns
    -------
    azi : instance of numpy.ndarray
        The azimuth angle for each right-ascention angle.
    """
    # Use the azi_to_ra_transform function because it is symmetric.
    azi = azi_to_ra_transform(ra, mjd)

    return azi


def hor_to_equ_transform(azi, zen, mjd):
    """Transforms the coordinate from the horizontal system (azimuth, zenith)
    into the equatorial system (right-ascention, declination) for detector at
    the South Pole and neglecting all astronomical effects like Earth
    precession.

    Parameters
    ----------
    azi : instance of numpy.ndarray
        The azimuth angle.
    zen : instance of numpy.ndarray
        The zenith angle.
    mjd : instance of numpy.ndarray
        The time in MJD.

    Returns
    -------
    ra : instance of numpy.ndarray
        The right-ascention angle.
    dec : instance of numpy.ndarray
        The declination angle.
    """
    ra = azi_to_ra_transform(azi, mjd)
    dec = np.pi - zen
    return (ra, dec)
