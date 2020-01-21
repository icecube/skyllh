# -*- coding: utf-8 -*-
# Author: Martin Wolf <mail@martin-wolf.org>

"""This test module tests classes, methods and functions of the ``core.model``
module.
"""

import unittest

from skyllh.core.model import (
    Model,
    ModelCollection
)


class Model_TestCase(unittest.TestCase):
    def setUp(self):
        self.model = Model('Model1')

    def test_name(self):
        self.assertEqual(self.model.name, 'Model1')

    def test_id(self):
        self.assertTrue(isinstance(self.model.id, int))


class ModelCollection_TestCase(unittest.TestCase):
    def setUp(self):
        self.model1 = Model('Model1')
        self.model2 = Model('Model2')
        self.modelcoll = ModelCollection((self.model1, self.model2))

    def test_cast(self):
        # Test cast function for None type.
        modelcoll = ModelCollection.cast(None)
        self.assertTrue(issubclass(modelcoll.model_type, Model))
        self.assertEqual(len(modelcoll.models), 0)

        # Test cast function for single Model instance.
        modelcoll = ModelCollection.cast(self.model1)
        self.assertTrue(issubclass(modelcoll.model_type, Model))
        self.assertEqual(len(modelcoll.models), 1)
        self.assertEqual(modelcoll.models[0], self.model1)

        # Test cast function for ModelCollection.
        modelcoll = ModelCollection.cast(self.modelcoll)
        self.assertEqual(modelcoll, self.modelcoll)

        # Test sequence of Model instances.
        modelcoll = ModelCollection.cast((self.model1, self.model2))
        self.assertTrue(issubclass(modelcoll.model_type, Model))
        self.assertEqual(len(modelcoll.models), 2)
        self.assertEqual(modelcoll.models[0], self.model1)
        self.assertEqual(modelcoll.models[1], self.model2)

        # Test that non-Model instances raises a TypeError.
        with self.assertRaises(TypeError):
            modelcoll = ModelCollection.cast('A str instance.')
        with self.assertRaises(TypeError):
            modelcoll = ModelCollection.cast(('str1','str2'))

    def test_model_type(self):
        self.assertTrue(issubclass(self.modelcoll.model_type, Model))

    def test_models(self):
        self.assertEqual(len(self.modelcoll.models), 2)
        self.assertEqual(self.modelcoll.models[0], self.model1)
        self.assertEqual(self.modelcoll.models[1], self.model2)


if(__name__ == '__main__'):
    unittest.main()
