# -*- coding: utf-8 -*-

import unittest
import numpy as np

from skyllh.physics.source import (
    SourceLocation,
    SourceModel,
    SourceCollection,
    Catalog,
    PointLikeSource,
    PointLikeSourceCollection,
    PointLikeSourceCatalog
)


class TestSource(unittest.TestCase):
    def setUp(self):
        self.ra = 0
        self.dec = 1

    def test_SourceLocation(self):
        source_location = SourceLocation(self.ra, self.dec)

        self.assertEqual(source_location.ra, self.ra)
        self.assertEqual(source_location.dec, self.dec)

    def test_SourceModel(self):
        source_model = SourceModel(self.ra, self.dec)

        self.assertEqual(source_model.loc.ra, self.ra)
        self.assertEqual(source_model.loc.dec, self.dec)

    def test_SourceCollection(self):
        source_model1 = SourceModel(self.ra, self.dec)
        source_model2 = SourceModel(self.ra, self.dec)

        source_collection_casted = SourceCollection.cast(source_model1, "Could not cast SourceModel to SourceCollection")
        source_collection = SourceCollection(source_type=SourceModel, sources=[source_model1, source_model2])

        self.assertIsInstance(source_collection_casted, SourceCollection)
        self.assertEqual(source_collection.source_type, SourceModel)
        self.assertIsInstance(source_collection.sources[0], SourceModel)
        self.assertIsInstance(source_collection.sources[1], SourceModel)

    def test_Catalog(self):
        name = "Catalog test"
        source_model1 = SourceModel(self.ra, self.dec)
        source_model2 = SourceModel(self.ra, self.dec)

        catalog = Catalog(name, source_type=SourceModel, sources=[source_model1, source_model2])
        source_collection_from_catalog = catalog.as_source_collection()

        self.assertEqual(catalog.name, name)
        self.assertIsInstance(source_collection_from_catalog, SourceCollection)

    def test_PointLikeSource(self):
        point_like_source = PointLikeSource(self.ra, self.dec)

        self.assertEqual(point_like_source.ra, self.ra)
        self.assertEqual(point_like_source.dec, self.dec)

    def test_PointLikeSourceCollection(self):
        point_like_source1 = PointLikeSource(self.ra, self.dec)
        point_like_source2 = PointLikeSource(self.ra, self.dec)

        point_like_source_collection = PointLikeSourceCollection(sources=[point_like_source1, point_like_source2])
        ra_array = np.array([self.ra, self.ra])
        dec_array = np.array([self.dec, self.dec])

        self.assertIsInstance(point_like_source_collection.sources[0], SourceModel)
        self.assertIsInstance(point_like_source_collection.sources[1], SourceModel)
        np.testing.assert_array_equal(point_like_source_collection.ra, ra_array)
        np.testing.assert_array_equal(point_like_source_collection.dec, dec_array)

    def test_PointLikeSourceCatalog(self):
        name = "Point like source catalog test"
        point_like_source1 = PointLikeSource(self.ra, self.dec)
        point_like_source2 = PointLikeSource(self.ra, self.dec)

        point_like_source_catalog = PointLikeSourceCatalog(name, sources=[point_like_source1, point_like_source2])

        self.assertEqual(point_like_source_catalog.name, name)
        self.assertEqual(point_like_source_catalog.source_type, PointLikeSource)
        self.assertIsInstance(point_like_source_catalog.sources[0], PointLikeSource)
        self.assertIsInstance(point_like_source_catalog.sources[1], PointLikeSource)


if(__name__ == '__main__'):
    unittest.main()
