# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import unittest

from skyllh.core.py import (
    ConstPyQualifier,
    const
)


class A(object):
    def __init__(self):
        super(A, self).__init__()


class ConstPyQualifier_TestCase(unittest.TestCase):
    def test_call(self):
        a = const(A())
        self.assertTrue(hasattr(a, '__pyqualifiers__'))
        self.assertTrue(ConstPyQualifier in a.__pyqualifiers__)

    def test_check(self):
        a = const(A())
        self.assertTrue(const.check(a))


if(__name__ == '__main__'):
    unittest.main()
