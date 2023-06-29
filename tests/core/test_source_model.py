# -*- coding: utf-8 -*-

import unittest

from skyllh.core.catalog import (
    SourceCatalog,
)
from skyllh.core.source_model import (
    PointLikeSource,
    SourceModel,
    SourceModelCollection,
)


class SourceModelTestCase(
        unittest.TestCase
):
    def setUp(self):
        self.name = 'MySource'
        self.classification = 'MyClassification'
        self.weight = 1.1

        self.source_model = SourceModel(
            name=self.name,
            classification=self.classification,
            weight=self.weight)

    def test_name(self):
        self.assertEqual(self.source_model.name, self.name)

    def test_classification(self):
        self.assertEqual(self.source_model.classification, self.classification)

    def test_weight(self):
        self.assertEqual(self.source_model.weight, self.weight)


class SourceModelCollectionTestCase(
        unittest.TestCase,
):
    def setUp(self):
        self.ra = 0
        self.dec = 1

    def test_SourceModelCollection(self):
        source_model1 = SourceModel(self.ra, self.dec)
        source_model2 = SourceModel(self.ra, self.dec)

        source_collection_casted = SourceModelCollection.cast(
            source_model1,
            "Could not cast SourceModel to SourceCollection")
        source_collection = SourceModelCollection(
            source_type=SourceModel,
            sources=[source_model1, source_model2])

        self.assertIsInstance(source_collection_casted, SourceModelCollection)
        self.assertEqual(source_collection.source_type, SourceModel)
        self.assertIsInstance(source_collection.sources[0], SourceModel)
        self.assertIsInstance(source_collection.sources[1], SourceModel)


class SourceCatalogTestCase(
        unittest.TestCase,
):
    def setUp(self):
        self.name = 'MySourceCatalog'
        self.ra = 0.1
        self.dec = 1.1
        self.source1 = SourceModel(self.ra, self.dec)
        self.source2 = SourceModel(self.ra, self.dec)

        self.catalog = SourceCatalog(
            name=self.name,
            sources=[self.source1, self.source2],
            source_type=SourceModel)

    def test_name(self):
        self.assertEqual(self.catalog.name, self.name)

    def test_as_SourceModelCollection(self):
        sc = self.catalog.as_SourceModelCollection()
        self.assertIsInstance(sc, SourceModelCollection)


class PointLikeSourceTestCase(
        unittest.TestCase,
):
    def setUp(self):
        self.name = 'MyPointLikeSource'
        self.ra = 0.1
        self.dec = 1.1
        self.source = PointLikeSource(
            name=self.name,
            ra=self.ra,
            dec=self.dec)

    def test_name(self):
        self.assertEqual(self.source.name, self.name)

    def test_ra(self):
        self.assertEqual(self.source.ra, self.ra)

    def test_dec(self):
        self.assertEqual(self.source.dec, self.dec)


if __name__ == '__main__':
    unittest.main()
