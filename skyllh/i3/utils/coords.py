"""IceCube specific coordinate utility functions."""

from typing import overload

import numpy as np


@overload
def azi_to_ra_transform(azi: np.ndarray, mjd: float | np.ndarray) -> np.ndarray: ...
@overload
def azi_to_ra_transform(azi: float, mjd: float) -> float: ...
@overload
def azi_to_ra_transform(azi: float | np.ndarray, mjd: float | np.ndarray) -> float | np.ndarray: ...
def azi_to_ra_transform(azi: float | np.ndarray, mjd: float | np.ndarray) -> float | np.ndarray:
    """Rotates the given IceCube azimuth angles into right-ascention angles for
    the given MJD times. This function is IceCube specific and assumes that the
    detector is located excently at the South Pole and neglects all astronomical
    effects like Earth's precession.

    Parameters
    ----------
    azi
        The array with the azimuth angles.
    mjd
        The array with the MJD times for each azimuth angle.

    Returns
    -------
    ra
        The right-ascention values.
    """
    # sidereal day = length * solar day
    _sidereal_length = 0.997269566
    _sidereal_offset = 2.54199002505
    sidereal_day_residuals = (mjd / _sidereal_length) % 1
    ra = _sidereal_offset + 2 * np.pi * sidereal_day_residuals - azi
    ra = np.mod(ra, 2 * np.pi)

    return ra


def ra_to_azi_transform(ra: float | np.ndarray, mjd: float | np.ndarray) -> float | np.ndarray:
    """Rotates the given right-ascention angles to local IceCube azimuth angles.

    Parameters
    ----------
    ra
        The array with the right-ascention angles.
    mjd
        The array with the MJD times for each right-ascention angle.

    Returns
    -------
    azi
        The azimuth angle for each right-ascention angle.
    """
    # Use the azi_to_ra_transform function because it is symmetric.
    azi = azi_to_ra_transform(ra, mjd)

    return azi


@overload
def hor_to_equ_transform(
    azi: np.ndarray, zen: np.ndarray, mjd: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def hor_to_equ_transform(azi: float, zen: float, mjd: float) -> tuple[float, float]: ...
@overload
def hor_to_equ_transform(
    azi: float | np.ndarray, zen: float | np.ndarray, mjd: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]: ...
def hor_to_equ_transform(
    azi: float | np.ndarray, zen: float | np.ndarray, mjd: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Transforms the coordinate from the horizontal system (azimuth, zenith)
    into the equatorial system (right-ascention, declination) for detector at
    the South Pole and neglecting all astronomical effects like Earth
    precession.

    Parameters
    ----------
    azi
        The azimuth angle.
    zen
        The zenith angle.
    mjd
        The time in MJD.

    Returns
    -------
    ra
        The right-ascention angle.
    dec
        The declination angle.
    """
    ra = azi_to_ra_transform(azi, mjd)
    dec = np.pi - zen
    return (ra, dec)
