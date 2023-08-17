# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the
skyllh.core.livetime module.
"""

import unittest

import numpy as np

from skyllh.core.livetime import (
    Livetime,
)


class Livetime_TestCase(
        unittest.TestCase
):
    def setUp(self) -> None:
        self.lt = Livetime(
            uptime_mjd_intervals_arr=np.array([
                [54000.0, 54001.0],
                [54020.2, 54023.3],
                [55099.9, 55100.0],
            ]))

    def test_create_sidereal_time_histogram(self):
        (hist, bin_edges) = self.lt.create_sidereal_time_histogram(
            dangle=0.1,
            longitude=None)

        self.assertEqual(bin_edges.shape, (3601,))
        self.assertEqual(bin_edges[0], 0)
        self.assertEqual(bin_edges[-1], 24)
        self.assertEqual(np.sum(hist), 15121)


if __name__ == '__main__':
    unittest.main()
