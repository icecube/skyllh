# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the
skyllh.core.services module.
"""

import unittest

import numpy as np

from astropy import (
    units,
)
from astropy.coordinates import (
    EarthLocation,
)

from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.sidereal_time import (
    SiderealTimeService,
)


class SiderealTimeService_TestCase(
        unittest.TestCase,
):
    @classmethod
    def setUpClass(cls):
        cls.livetime = Livetime(
            uptime_mjd_intervals_arr=np.array([
                [54000.0, 54001.0],
                [54020.2, 54023.3],
                [55099.9, 55100.0],
            ]))
        cls.detector_model = DetectorModel(
            name='Detector at (lon=0, lat=0)',
            location=EarthLocation.from_geodetic(
                lon=0*units.deg,
                lat=0*units.deg)
        )
        cls.st_service = SiderealTimeService(
            detector_model=cls.detector_model,
            livetime=cls.livetime,
            st_bin_width_deg=0.1)

    def test_st_hist(self):
        st_hist_binedges = self.st_service.st_hist_binedges
        st_hist = self.st_service.st_hist

        self.assertEqual(st_hist_binedges.shape, (3601,))
        self.assertEqual(st_hist_binedges[0], 0)
        self.assertEqual(st_hist_binedges[-1], 24)
        self.assertEqual(np.sum(st_hist), 15162)


if __name__ == '__main__':
    unittest.main()
