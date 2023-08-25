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
        self.livetime = Livetime(
            uptime_mjd_intervals_arr=np.array([
                [54000.0, 54001.0],
                [54020.2, 54023.3],
                [55099.9, 55100.0],
            ]))

    def test_get_integraded_livetime(self):
        integrated_livetime = Livetime.get_integrated_livetime(self.livetime)
        np.testing.assert_allclose(integrated_livetime, (1 + 3.1 + 0.1))


if __name__ == '__main__':
    unittest.main()
