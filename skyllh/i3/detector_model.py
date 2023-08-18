# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

"""This module defines the IceCube detector model class.
"""

from astropy import (
    units,
)
from astropy.coordinates import (
    EarthLocation,
)


from skyllh.core.model import (
    DetectorModel,
)


class IceCubeDetectorModel(
        DetectorModel,
):
    def __init__(self, **kwargs):
        super().__init__(
            name='IceCube',
            location=EarthLocation.from_geodetic(
                lon=-62.6081*units.deg,
                lat=-89.9944*units.deg,
                height=883.9*units.m),
            **kwargs)
