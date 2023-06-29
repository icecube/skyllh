# -*- coding: utf-8 -*-

"""
Example how to use the data scrambling mechanism of skyllh.
"""

import numpy as np

from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.scrambling import (
    DataScrambler,
    UniformRAScramblingMethod,
)


def gen_data(rss, N=100, window=(0, 365)):
    """Create uniformly distributed data on sphere. """
    arr = np.empty((N,), dtype=[("ra", np.float64), ("dec", np.float64)])

    arr["ra"] = rss.random.uniform(0., 2.*np.pi, N)
    arr["dec"] = rss.random.uniform(-np.pi, np.pi, N)

    return arr


if __name__ == '__main__':
    rss = RandomStateService(seed=1)

    # Generate some psydo data.
    data = gen_data(rss, N=10)
    print(data['ra'])

    # Create DataScrambler instance with uniform RA scrambling.
    scr = DataScrambler(method=UniformRAScramblingMethod())

    # Scramble the data.
    scr.scramble_data(
        rss=rss,
        data=data)
    print(data['ra'])
