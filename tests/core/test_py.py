# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import unittest

from skyllh.core.py import (
    ConstPyQualifier,
    NamedObjectCollection,
    const,
    issequenceof
)


class A(object):
    def __init__(self, name=None):
        super(A, self).__init__()

        self._name = name

    @property
    def name(self):
        return self._name


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


class NamedObjectCollection_TestCase(
        unittest.TestCase):
    def setUp(self):
        self.a1 = A('a1')
        self.a2 = A('a2')
        self.a3 = A('a3')
        self.noc = NamedObjectCollection([self.a1, self.a2, self.a3])

    def test_name_list(self):
        self.assertEqual(self.noc.name_list, ['a1', 'a2', 'a3'])

    def test__contains__(self):
        self.assertTrue('a1' in self.noc)
        self.assertTrue('a2' in self.noc)
        self.assertTrue('a3' in self.noc)
        self.assertFalse('a4' in self.noc)

    def test__getitem__(self):
        self.assertTrue(self.noc['a1'] is self.a1)
        self.assertTrue(self.noc['a2'] is self.a2)
        self.assertTrue(self.noc['a3'] is self.a3)

        self.assertTrue(self.noc[0] is self.a1)
        self.assertTrue(self.noc[1] is self.a2)
        self.assertTrue(self.noc[2] is self.a3)

    def test_get_index_by_name(self):
        self.assertEqual(self.noc.get_index_by_name('a1'), 0)
        self.assertEqual(self.noc.get_index_by_name('a2'), 1)
        self.assertEqual(self.noc.get_index_by_name('a3'), 2)

    def test_add(self):
        a4 = A('a4')
        self.noc.add(a4)

        self.assertEqual(self.noc.name_list, ['a1', 'a2', 'a3', 'a4'])
        self.assertEqual(self.noc.get_index_by_name('a4'), 3)
        self.assertTrue(self.noc['a4'] is a4)

    def test_pop(self):
        obj = self.noc.pop()
        self.assertTrue(obj is self.a3)
        self.assertEqual(self.noc.name_list, ['a1', 'a2'])
        self.assertEqual(self.noc.get_index_by_name('a1'), 0)
        self.assertEqual(self.noc.get_index_by_name('a2'), 1)

    def test_pop_with_int(self):
        obj = self.noc.pop(1)
        self.assertTrue(obj is self.a2)
        self.assertEqual(self.noc.name_list, ['a1', 'a3'])
        self.assertEqual(self.noc.get_index_by_name('a1'), 0)
        self.assertEqual(self.noc.get_index_by_name('a3'), 1)

    def test_pop_with_str(self):
        obj = self.noc.pop('a2')
        self.assertTrue(obj is self.a2)
        self.assertEqual(self.noc.name_list, ['a1', 'a3'])
        self.assertEqual(self.noc.get_index_by_name('a1'), 0)
        self.assertEqual(self.noc.get_index_by_name('a3'), 1)


if __name__ == '__main__':
    unittest.main()
