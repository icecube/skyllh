# Example how to use the data scrambling mechanism of skylab.

import numpy as np

from skylab.core.random import RandomStateService
from skylab.core.scrambling import DataScrambler, RAScrambling

def gen_data(rss, N=100, window=(0,365)):
    """Create uniformly distributed data on sphere. """
    arr = np.empty((N,), dtype=[("ra", np.float), ("dec", np.float)])

    arr["ra"] = rss.random.uniform(0., 2.*np.pi, N)
    arr["dec"] = rss.random.uniform(-np.pi, np.pi, N)

    return arr

rss = RandomStateService(seed=1)

# Generate some psydo data.
data = gen_data(rss, N=10)
print data['ra']

# Create DataScrambler instance with RA scrambling.
scr = DataScrambler(method=RAScrambling(), rss=rss)

# Scramble the data.
scr.scramble(data)
print data['ra']
