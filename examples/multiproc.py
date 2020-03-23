"""Example how to use the multiproc module to parallelize the computation of
the function f(x) = x^2 + c
"""

import numpy as np

from skyllh.core.multiproc import parallelize


def f(x, c=0.):
    return x**2 + c

res = parallelize(f, [((x,),{'c':x}) for x in np.arange(1, 10, 1)], ncpu=3)
print(res)
