# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

"""This module defines the P-ONE detector model class.
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


class PONEDetectorModel(
        DetectorModel,
):
    """Definition of the P-ONE detector.

    For the time being we use the position of STRAW-a. As depth we use the sea
    floor at -2658m and a detector half height of 500m.
    """
    def __init__(self, **kwargs):
        super().__init__(
            name='P-ONE',
            location=EarthLocation.from_geodetic(
                lon=-127.7317*units.deg,
                lat=47.7564*units.deg,
                height=-2158*units.m),
            **kwargs)
