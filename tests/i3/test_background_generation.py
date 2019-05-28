# -*- coding: utf-8 -*-

import unittest
import numpy as np

from skyllh.core.scrambling import DataScrambler, UniformRAScramblingMethod
from skyllh.i3.background_generation import FixedScrambledExpDataI3BkgGenMethod


class TestFixedScrambledExpDataI3BkgGenMethod(unittest.TestCase):
    def test_data_scrambler(self):
        data_scrambling_method = UniformRAScramblingMethod()
        data_scrambler = DataScrambler(data_scrambling_method)
        test_object = FixedScrambledExpDataI3BkgGenMethod(data_scrambler)

        self.assertIsInstance(test_object.data_scrambler, DataScrambler)


    def test_generate_events(self):
        pass


if(__name__ == '__main__'):
    unittest.main()
