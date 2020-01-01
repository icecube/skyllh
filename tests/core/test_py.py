# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import unittest

from skyllh.core.py import (
    ConstPyQualifier,
    const,
    issequenceof
)


class A(object):
    def __init__(self, name=None):
        super(A, self).__init__()

        self.name = name

class B(object):
    def __init__(self, name=None):
        super(B, self).__init__()

        self.name = name


class ConstPyQualifier_TestCase(unittest.TestCase):
    def test_call(self):
        a = const(A())
        self.assertTrue(hasattr(a, '__pyqualifiers__'))
        self.assertTrue(ConstPyQualifier in a.__pyqualifiers__)

    def test_check(self):
        a = const(A())
        self.assertTrue(const.check(a))


class issequenceof_TestCase(unittest.TestCase):
    def test_type(self):
        seq = [A('a1'), A('a2')]
        self.assertTrue(issequenceof(seq, A))

        seq = [A('a1'), B('b1')]
        self.assertFalse(issequenceof(seq, A))

    def test_pyqualifiers(self):
        """Tests if the issequenceof function works with PyQualifiers.
        """
        seq = [const(A('a1')), const(A('a2'))]
        self.assertTrue(issequenceof(seq, A, const))

        seq = [const(A('a1')), A('a2')]
        self.assertFalse(issequenceof(seq, A, const))


if(__name__ == '__main__'):
    unittest.main()
