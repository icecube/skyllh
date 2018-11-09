# -*- coding: utf-8 -*-

from __future__ import division

import unittest
import numpy as np

from skyllh.physics.time_profile import BoxTimeProfile


class TestTimeProfile(unittest.TestCase):
    def setUp(self):
        self.T0 = 58430     # MJD time 2018.11.08
        self.Tw = 2         # 1 day width of the box profile
        self.box_time_profile = BoxTimeProfile(self.T0, self.Tw)

    def test_move(self):
        dt = 5
        self.box_time_profile.move(dt)

        self.assertEqual(self.box_time_profile.T0, self.T0 + dt)
        self.assertEqual(self.box_time_profile.Tw, self.Tw)

    def test_T0(self):
        T0 = 0
        self.box_time_profile.T0 = T0

        self.assertEqual(self.box_time_profile.T0, T0)

    def test_Tw(self):
        Tw = 2.5
        self.box_time_profile.Tw = Tw

        self.assertEqual(self.box_time_profile.Tw, Tw)

    def test_str(self):
        s = "BoxTimeProfile(T0={:.6f}, Tw={:.6f})".format(self.T0, self.Tw)

        self.assertEqual(str(self.box_time_profile), s)

    def test_update(self):
        T0 = 0.5
        Tw = 2.5

        # Test update method with the same values
        fitparams = {'T0': self.T0, 'Tw': self.Tw}
        self.box_time_profile.update(fitparams)

        self.assertFalse(self.box_time_profile.update(fitparams))
        self.assertEqual(self.box_time_profile.T0, self.T0)
        self.assertEqual(self.box_time_profile.Tw, self.Tw)

        # Test update method with the new values
        fitparams = {'T0': T0, 'Tw': Tw}

        self.assertTrue(self.box_time_profile.update(fitparams))
        self.assertEqual(self.box_time_profile.T0, T0)
        self.assertEqual(self.box_time_profile.Tw, Tw)

    def test_get_integral(self):
        t1 = self.T0
        t2 = self.T0 + self.Tw/2
        times1 = np.array([self.T0 - self.Tw,
                          self.T0 - self.Tw/2,
                          self.T0])
        times2 = np.array([self.T0 + self.Tw,
                          self.T0 + self.Tw/2,
                          self.T0 + self.Tw/2])
        values = np.array([1,
                           1,
                           0.5])

        self.assertEqual(self.box_time_profile.get_integral(t1, t1), 0)
        self.assertEqual(self.box_time_profile.get_integral(t1, t2), 0.5)
        np.testing.assert_array_equal(self.box_time_profile.get_integral(times1, times2), values)

        # Test cases when t1 > t2
        self.assertEqual(self.box_time_profile.get_integral(t2, t1), 0)
        np.testing.assert_array_equal(self.box_time_profile.get_integral(times2, times1), np.zeros_like(values))

    def test_get_total_integral(self):
        self.assertEqual(self.box_time_profile.get_total_integral(), 1)

    def test_get_value(self):
        value = 1/self.Tw
        times = np.array([self.T0 - self.Tw,
                          self.T0,
                          self.T0 + self.Tw])
        values = np.array([0,
                           value,
                           0])

        self.assertEqual(self.box_time_profile.get_value(self.T0 - self.Tw), 0)
        self.assertEqual(self.box_time_profile.get_value(self.T0), value)
        self.assertEqual(self.box_time_profile.get_value(self.T0 + self.Tw), 0)
        np.testing.assert_array_equal(self.box_time_profile.get_value(times), values)


if(__name__ == '__main__'):
    unittest.main()
