# -*- coding: utf-8 -*-

import os.path
import unittest
import numpy as np

from skyllh.core import storage
from skyllh.core.random import RandomStateService
from skyllh.core.times import (
    LivetimeTimeGenerationMethod,
    TimeGenerator
)
from skyllh.core.livetime import Livetime

from skyllh.i3.coords import (
    azi_to_ra_transform,
    ra_to_azi_transform,
    hor_to_equ_transform
)
from skyllh.i3.scrambling import I3TimeScramblingMethod
from skyllh.i3.dataset import I3Dataset


class TestI3TimeScramblingMethod(unittest.TestCase):
    """Test I3TimeScramblingMethod function.
    """
    def setUp(self):
        # Scrambled exp_testdata data with rss seed=1.
        self.scrambled_data = np.array([
            (57136.7663126 , 3.91628661,  0.49221455, 6.17569364, 2.6493781,
             2.99055817, 2.06154594, 0.00655059),
            (57528.15243901, 1.01047551,  1.1717919 , 5.67417281, 1.96980075,
             5.62599504, 2.74410951, 0.03066161),
            (57136.07148692, 4.95825133,  0.2423321 , 0.75605748, 2.89926055,
             3.85740203, 1.81258875, 0.0147304 ),
            (57136.57516915, 1.13095166, -0.1928237 , 1.47356536, 3.33441635,
             1.42919624, 1.37737515, 0.00438893)],
            dtype=[('time', '<f8'), ('ra', '<f8'), ('dec', '<f8'),
                   ('azi', '<f8'), ('zen', '<f8'), ('ang_err', '<f8'),
                   ('log_energy', '<f8'), ('sin_dec', '<f8')]
        )

    def test_scramble(self):
        path = os.path.abspath(os.path.dirname(__file__))
        exp_pathfilename = os.path.join(path, 'testdata/exp_testdata.npy')
        grl_pathfilename = os.path.join(path, 'testdata/grl_testdata.npy')

        exp_fileloader = storage.create_FileLoader(exp_pathfilename)
        exp_data = exp_fileloader.load_data()
        grl_fileloader = storage.create_FileLoader(grl_pathfilename)
        grl_data = grl_fileloader.load_data()

        uptime_mjd_intervals_arr = np.hstack((
            grl_data['start'].reshape((len(grl_data),1)),
            grl_data['stop'].reshape((len(grl_data),1))
        ))

        livetime = Livetime(uptime_mjd_intervals_arr)
        timegen = TimeGenerator(method=LivetimeTimeGenerationMethod(livetime))

        i3timescramblingmethod = I3TimeScramblingMethod(timegen)
        rss = RandomStateService(seed=1)
        data = i3timescramblingmethod.scramble(rss, exp_data)

        np.testing.assert_allclose(data['time'], self.scrambled_data['time'])
        np.testing.assert_allclose(data['ra'], self.scrambled_data['ra'])


if(__name__ == '__main__'):
    unittest.main()
