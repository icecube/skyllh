# Example how to use the data scrambling mechanism of skylab.

import numpy as np

from skylab.core.scrambling import DataScrambler, RAScrambling

def gen_data(N=100, seed=1, window=(0,365)):
    """Create uniformly distributed data on sphere. """
    np.random.seed(seed)

    arr = np.empty((N,), dtype=[("ra", np.float), ("dec", np.float)])

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["dec"] = np.random.uniform(-np.pi, np.pi, N)

    return arr

seed = 1

# Generate some psydo data.
data = gen_data(N=10, seed=seed)
print data['ra']

# Create DataScrambler instance with RA scrambling.
scr = DataScrambler(RAScrambling(), seed=seed+1)

# Scramble the data.
scr.scramble(data)
print data['ra']
