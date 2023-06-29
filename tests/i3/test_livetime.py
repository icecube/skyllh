# -*- coding: utf-8 -*-

import os.path
import unittest

from skyllh.core.storage import (
    create_FileLoader,
)
from skyllh.i3.dataset import (
    I3Dataset,
)
from skyllh.i3.livetime import (
    I3Livetime,
)


class I3LivetimeTestCase(
        unittest.TestCase
):
    """Test I3Livetime class.
    """
    def setUp(self):
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.exp_pathfilenames = os.path.join(self.path, "testdata/exp_testdata.npy")
        self.grl_pathfilenames = os.path.join(self.path, 'testdata/grl_testdata.npy')
        self.mc_pathfilenames = os.path.join(self.path, "testdata/mc_testdata.npy")
        self.livetime_total = 1.6666178000014042
        self.n_uptime_mjd_intervals = 5

    def test_from_grl_data(self):
        grl_fileloader = create_FileLoader(self.grl_pathfilenames)
        grl_data = grl_fileloader.load_data()

        i3livetime = I3Livetime.from_grl_data(grl_data)

        self.assertEqual(
            i3livetime.livetime, self.livetime_total)
        self.assertEqual(
            i3livetime.n_uptime_mjd_intervals, self.n_uptime_mjd_intervals)

    def test_from_grl_files(self):
        i3livetime = I3Livetime.from_grl_files(self.grl_pathfilenames)

        self.assertEqual(
            i3livetime.livetime, self.livetime_total)
        self.assertEqual(
            i3livetime.n_uptime_mjd_intervals, self.n_uptime_mjd_intervals)

    def test_from_I3Dataset(self):
        # Create a test data set.
        i3dataset = I3Dataset(
            name='test_dataset',
            exp_pathfilenames=self.exp_pathfilenames,
            mc_pathfilenames=self.mc_pathfilenames,
            grl_pathfilenames=self.grl_pathfilenames,
            livetime=self.livetime_total,
            default_sub_path_fmt='',
            version=1,
        )

        i3livetime = I3Livetime.from_I3Dataset(i3dataset)

        self.assertEqual(
            i3livetime.livetime, self.livetime_total)
        self.assertEqual(
            i3livetime.n_uptime_mjd_intervals, self.n_uptime_mjd_intervals)


if __name__ == '__main__':
    unittest.main()
