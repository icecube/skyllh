# -*- coding: utf-8 -*-

"""
Example how to use the data scrambling mechanism of SkyLLH.
"""

import numpy as np

from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.scrambling import (
    DataScrambler,
    UniformRAScramblingMethod,
)
from skyllh.core.times import (
    LivetimeTimeGenerationMethod,
    TimeGenerator,
)

from skyllh.i3.scrambling import (
    I3TimeScramblingMethod,
)


def gen_data(rss, N):
    """Create uniformly distributed data on sphere.
    """
    arr = np.empty(
        (N,),
        dtype=[
            ("azi", np.float64),
            ("zen", np.float64),
            ("ra", np.float64),
            ("dec", np.float64),
            ("time", np.float64),
        ])

    arr["ra"] = rss.random.uniform(0., 2.*np.pi, N)
    arr["dec"] = rss.random.uniform(-np.pi, np.pi, N)

    return arr


def ex1():
    """Data scrambling via right-ascention scrambling.
    """
    print("Example 1")
    print("=========")

    rss = RandomStateService(seed=1)

    # Generate some pseudo data.
    data = gen_data(rss=rss, N=10)
    print(f'before scrambling: data["ra"]={data["ra"]}')

    # Create DataScrambler instance with uniform RA scrambling.
    scrambler = DataScrambler(
        method=UniformRAScramblingMethod())

    # Scramble the data.
    scrambler.scramble_data(
        rss=rss,
        dataset=None,
        data=data)

    print(f'after scrambling: data["ra"]={data["ra"]}')


def ex2():
    """Data scrambling via detector on-time scrambling.
    """
    print("Example 2")
    print("=========")

    rss = RandomStateService(seed=1)

    # Generate some psydo data.
    data = gen_data(rss=rss, N=10)
    print(f'before scrambling: data["ra"]={data["ra"]}')

    # Create a Livetime object, which defines the detector live-time.
    lt = Livetime(uptime_mjd_intervals_arr=np.array(
        [
            [55000, 56000],
            [60000, 69000]
        ],
        dtype=np.float64))

    # Create a TimeGenerator with an on-time time generation method.
    timegen = TimeGenerator(method=LivetimeTimeGenerationMethod(livetime=lt))

    # Create DataScrambler with IceCube time scrambing method.
    scrambler = DataScrambler(
        method=I3TimeScramblingMethod(timegen))

    # Scramble the data.
    scrambler.scramble_data(
        rss=rss,
        dataset=None,
        data=data)

    print(f'after scrambling: data["ra"]={data["ra"]}')


if __name__ == '__main__':
    ex1()
    ex2()
