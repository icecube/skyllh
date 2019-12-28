# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import numpy as np

def isAlmostEqual(a, b, decimals=9):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return np.all(np.around(np.abs(a - b), decimals) == 0)
