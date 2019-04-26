# -*- coding: utf-8 -*-

import unittest
import numpy as np

from skyllh.i3.coords import (
    azi_to_ra_transform,
    ra_to_azi_transform,
    hor_to_equ_transform
)


class TestCoords(unittest.TestCase):
    """Test IceCube specific coordinate utility functions.
    """
    def setUp(self):
        self.mjd = 58457
        self.azi = 0.5
        self.zen = 0.5
        self.ra = 2.356062973186092
        self.dec = 2.641592653589793

    def test_azi_to_ra_transform(self):
        self.assertEqual(azi_to_ra_transform(self.azi, self.mjd), self.ra)

    def test_ra_to_azi_transform(self):
        self.assertEqual(ra_to_azi_transform(self.ra, self.mjd), self.azi)

    def test_hor_to_equ_transform(self):
        self.assertEqual(hor_to_equ_transform(self.azi, self.zen, self.mjd),
                         (self.ra, self.dec))


if(__name__ == '__main__'):
    unittest.main()
