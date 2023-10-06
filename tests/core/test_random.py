# -*- coding: utf-8 -*-

import unittest

import numpy as np

from skyllh.core.random import (
    RandomChoice,
    RandomStateService,
)


class RandomChoice_TestCase(
        unittest.TestCase,
):
    def setUp(self):
        self.size = 100
        self.items = np.arange(self.size)
        self.probs = np.random.uniform(low=0, high=1, size=self.size)
        self.probs /= np.sum(self.probs)

    def test_choice(self):
        rss = RandomStateService(seed=1)
        np_items = rss.random.choice(
            self.items,
            size=5,
            replace=True,
            p=self.probs)

        rss = RandomStateService(seed=1)
        random_choice = RandomChoice(
            items=self.items,
            probabilities=self.probs)
        rc_items = random_choice(
            rss=rss,
            size=5)

        np.testing.assert_equal(np_items, rc_items)


if __name__ == '__main__':
    unittest.main()
