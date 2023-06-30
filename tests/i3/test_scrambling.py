# -*- coding: utf-8 -*-

import numpy as np
import os.path
import unittest

from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.storage import (
    create_FileLoader,
)
from skyllh.core.times import (
    LivetimeTimeGenerationMethod,
    TimeGenerator,
)

from skyllh.i3.livetime import (
    I3Livetime,
)
from skyllh.i3.scrambling import (
    I3TimeScramblingMethod,
)


class I3TimeScramblingMethodTestCase(
        unittest.TestCase
):
    """Test I3TimeScramblingMethod class.
    """
    def setUp(self):
        # Scrambled exp_testdata data with rss seed=1.
        self.scrambled_data = np.array(
            [
                (57136.7663126, 3.91628661, 0.49221455, 6.17569364, 2.6493781,
                 2.99055817, 2.06154594, 0.00655059),
                (57528.15243901, 1.01047551, 1.1717919, 5.67417281, 1.96980075,
                 5.62599504, 2.74410951, 0.03066161),
                (57136.07148692, 4.95825133, 0.2423321, 0.75605748, 2.89926055,
                 3.85740203, 1.81258875, 0.0147304),
                (57136.57516915, 1.13095166, -0.1928237, 1.47356536, 3.33441635,
                 1.42919624, 1.37737515, 0.00438893)
            ],
            dtype=[
                ('time', np.float64),
                ('ra', np.float64),
                ('dec', np.float64),
                ('azi', np.float64),
                ('zen', np.float64),
                ('ang_err', np.float64),
                ('log_energy', np.float64),
                ('sin_dec', np.float64)
            ]
        )

    def test_scramble(self):
        path = os.path.abspath(os.path.dirname(__file__))
        exp_pathfilename = os.path.join(path, 'testdata/exp_testdata.npy')
        grl_pathfilename = os.path.join(path, 'testdata/grl_testdata.npy')

        exp_fileloader = create_FileLoader(exp_pathfilename)
        exp_data = exp_fileloader.load_data()
        grl_fileloader = create_FileLoader(grl_pathfilename)
        grl_data = grl_fileloader.load_data()

        livetime = I3Livetime.from_grl_data(grl_data)
        timegen = TimeGenerator(method=LivetimeTimeGenerationMethod(livetime))

        i3timescramblingmethod = I3TimeScramblingMethod(timegen)
        rss = RandomStateService(seed=1)
        data = i3timescramblingmethod.scramble(
            rss=rss,
            dataset=None,
            data=exp_data)

        np.testing.assert_allclose(data['time'], self.scrambled_data['time'])
        np.testing.assert_allclose(data['ra'], self.scrambled_data['ra'])


if __name__ == '__main__':
    unittest.main()
