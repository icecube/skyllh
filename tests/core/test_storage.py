# -*- coding: utf-8 -*-

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from skyllh.core.storage import DataFieldRecordArray


class DataFieldRecordArray_TestCase(unittest.TestCase):
    def setUp(self):
        self.field1 = np.array([1.4, 1.3, 1.5, 1.1, 1.2], dtype=np.float64)
        self.field2 = np.array([2.5, 2.1, 2.3, 2.4, 2.2], dtype=np.float64)
        self.field3 = np.array([3.2, 3.5, 3.1, 3.3, 3.4], dtype=np.float64)
        data = dict(
            field1 = self.field1,
            field2 = self.field2,
            field3 = self.field3
        )
        self.arr = DataFieldRecordArray(data)
        self.arr_len = 5

    def test__contains__(self):
        self.assertFalse('field0' in self.arr)
        self.assertTrue('field1' in self.arr)
        self.assertTrue('field2' in self.arr)
        self.assertTrue('field3' in self.arr)

    def test__getitem__(self):
        # Access a non-existing field.
        with self.assertRaises(KeyError):
            self.arr['field0']

        # Access entire fields.
        assert_array_almost_equal(self.arr['field1'], self.field1)
        assert_array_almost_equal(self.arr['field2'], self.field2)
        assert_array_almost_equal(self.arr['field3'], self.field3)

        # Access rows of the dataset via indices.
        idx = np.array([1,4,2])
        sub_arr = self.arr[idx]
        assert_array_almost_equal(sub_arr['field1'], self.field1[idx])
        assert_array_almost_equal(sub_arr['field2'], self.field2[idx])
        assert_array_almost_equal(sub_arr['field3'], self.field3[idx])

        # Access rows of the dataset via a boolean mask.
        mask = np.array([True, True, False, True, False])
        sub_arr = self.arr[mask]
        assert_array_almost_equal(sub_arr['field1'], self.field1[mask])
        assert_array_almost_equal(sub_arr['field2'], self.field2[mask])
        assert_array_almost_equal(sub_arr['field3'], self.field3[mask])

    def test__setitem__(self):
        # Set an entire field with data of not the same length.
        with self.assertRaises(ValueError):
            new_field2 = np.array([2.51, 2.12, 2.33, 2.44], dtype=np.float64)
            self.arr['field2'] = new_field2

        # Set an entire field with new values.
        new_field2 = np.array([2.51, 2.12, 2.33, 2.44, 2.25], dtype=np.float64)
        self.arr['field2'] = new_field2
        assert_array_almost_equal(self.arr['field2'], new_field2)

        # Reset the array.
        self.setUp()

        # Set selected rows with new values by indices.
        idx = np.array([1,4,2])
        new_data = dict(
            field1 = self.field1[idx],
            field2 = new_field2[idx],
            field3 = self.field3[idx]
        )
        new_arr = DataFieldRecordArray(new_data)
        self.arr[idx] = new_arr
        assert_array_almost_equal(self.arr['field1'], self.field1)
        assert_array_almost_equal(
            self.arr['field2'],
            np.array([2.5, 2.12, 2.33, 2.4, 2.25], dtype=np.float64))
        assert_array_almost_equal(self.arr['field3'], self.field3)

        # Reset the array.
        self.setUp()

        # Set selected rows with new values by mask.
        mask = np.array([True, True, False, True, False])
        new_data = dict(
            field1 = self.field1[mask],
            field2 = new_field2[mask],
            field3 = self.field3[mask]
        )
        new_arr = DataFieldRecordArray(new_data)
        self.arr[mask] = new_arr
        assert_array_almost_equal(self.arr['field1'], self.field1)
        assert_array_almost_equal(
            self.arr['field2'],
            np.array([2.51, 2.12, 2.3, 2.44, 2.2], dtype=np.float64))
        assert_array_almost_equal(self.arr['field3'], self.field3)

        # Reset the array.
        self.setUp()

        # Add a new field.
        new_field = np.array([4.2, 4.5, 4.1, 4.3, 4.4], dtype=np.float64)
        self.arr['field4'] = new_field
        self.assertTrue('field4' in self.arr)
        self.assertTrue('field4' in self.arr.field_name_list)
        assert_array_almost_equal(
            self.arr['field4'],
            new_field)

    def test__str__(self):
        try:
            str(self.arr)
        except:
            self.fail('The __str__ method raised an exception!')

    def test_field_name_list(self):
        self.assertEqual(len(self.arr.field_name_list), 3)
        self.assertTrue('field1' in self.arr.field_name_list)
        self.assertTrue('field2' in self.arr.field_name_list)
        self.assertTrue('field3' in self.arr.field_name_list)

    def test_indices(self):
        assert_array_almost_equal(
            self.arr.indices,
            np.array([0, 1, 2, 3, 4]))

    def test_len(self):
        self.assertEqual(len(self.arr), self.arr_len)

    def test_rename_fields(self):
        self.arr.rename_fields({'field2': 'new_field2'})
        self.assertTrue('field1' in self.arr)
        self.assertFalse('field2' in self.arr)
        self.assertTrue('new_field2' in self.arr)
        self.assertTrue('field3' in self.arr)
        self.assertEqual(len(self.arr.field_name_list), 3)
        self.assertTrue('field1' in self.arr.field_name_list)
        self.assertTrue('new_field2' in self.arr.field_name_list)
        self.assertTrue('field3' in self.arr.field_name_list)

    def test_sort_by_field(self):
        self.arr.sort_by_field('field2')
        assert_array_almost_equal(
            self.arr['field1'],
            np.array([1.3, 1.2, 1.5, 1.1, 1.4], dtype=np.float64))
        assert_array_almost_equal(
            self.arr['field2'],
            np.array([2.1, 2.2, 2.3, 2.4, 2.5], dtype=np.float64))
        assert_array_almost_equal(
            self.arr['field3'],
            np.array([3.5, 3.4, 3.1, 3.3, 3.2], dtype=np.float64))

    def test_tidy_up(self):
        self.arr.tidy_up('field2')
        self.assertEqual(len(self.arr.field_name_list), 1)
        self.assertTrue('field2' in self.arr.field_name_list)
        self.assertFalse('field1' in self.arr)
        self.assertTrue('field2' in self.arr)
        self.assertFalse('field3' in self.arr)

        # Reset the array.
        self.setUp()

        self.arr.tidy_up(('field2','field3'))
        self.assertEqual(len(self.arr.field_name_list), 2)
        self.assertTrue('field2' in self.arr.field_name_list)
        self.assertTrue('field3' in self.arr.field_name_list)
        self.assertFalse('field1' in self.arr)
        self.assertTrue('field2' in self.arr)
        self.assertTrue('field3' in self.arr)

if(__name__ == '__main__'):
    unittest.main()
