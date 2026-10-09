import unittest

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from skyllh.core.catalog import (
    SourceCatalog,
)
from skyllh.core.config import (
    Config,
)
from skyllh.core.detsigyield import (
    NullDetSigYieldBuilder,
)
from skyllh.core.flux_model import (
    SteadyPointlikeFFM,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    PointLikeSource,
    SourceModel,
    SourceModelCollection,
)
from skyllh.core.utils.analysis import (
    pointlikesource_to_data_field_array,
)


class SourceModelTestCase(unittest.TestCase):
    def setUp(self):
        self.name = 'MySource'
        self.classification = 'MyClassification'
        self.weight = 1.1

        self.source_model = SourceModel(name=self.name, classification=self.classification, weight=self.weight)

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
        source_model1 = SourceModel(self.ra, self.dec)  # pyright: ignore[reportArgumentType]
        source_model2 = SourceModel(self.ra, self.dec)  # pyright: ignore[reportArgumentType]

        source_collection_casted = SourceModelCollection.cast(
            source_model1, 'Could not cast SourceModel to SourceCollection'
        )
        source_collection = SourceModelCollection(source_type=SourceModel, sources=[source_model1, source_model2])

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
        self.source1 = SourceModel(self.ra, self.dec)  # pyright: ignore[reportArgumentType]
        self.source2 = SourceModel(self.ra, self.dec)  # pyright: ignore[reportArgumentType]

        self.catalog = SourceCatalog(name=self.name, sources=[self.source1, self.source2], source_type=SourceModel)

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
        self.source = PointLikeSource(name=self.name, ra=self.ra, dec=self.dec)

    def test_name(self):
        self.assertEqual(self.source.name, self.name)

    def test_ra(self):
        self.assertEqual(self.source.ra, self.ra)

    def test_dec(self):
        self.assertEqual(self.source.dec, self.dec)

    def test_from_degrees(self):
        source = PointLikeSource.from_degrees(ra=40.67, dec=-0.01, name=self.name, weight=2.0)
        self.assertIsInstance(source, PointLikeSource)
        self.assertAlmostEqual(source.ra, np.radians(40.67))
        self.assertAlmostEqual(source.dec, np.radians(-0.01))
        self.assertEqual(source.name, self.name)
        self.assertEqual(source.weight, 2.0)

    def test_from_degrees_invalid_dec(self):
        with self.assertRaises(ValueError):
            PointLikeSource.from_degrees(ra=10.0, dec=91.0)

    def test_from_skycoord(self):
        coord = SkyCoord(ra=77.35 * u.deg, dec=5.7 * u.deg, frame='icrs')
        source = PointLikeSource.from_skycoord(coord, name=self.name)
        self.assertIsInstance(source, PointLikeSource)
        self.assertIsInstance(source.ra, float)
        self.assertIsInstance(source.dec, float)
        self.assertAlmostEqual(source.ra, np.radians(77.35))
        self.assertAlmostEqual(source.dec, np.radians(5.7))
        self.assertEqual(source.name, self.name)

    def test_from_skycoord_galactic(self):
        coord = SkyCoord(ra=77.35 * u.deg, dec=5.7 * u.deg, frame='icrs')
        source = PointLikeSource.from_skycoord(coord.transform_to('galactic'))
        self.assertAlmostEqual(source.ra, np.radians(77.35))
        self.assertAlmostEqual(source.dec, np.radians(5.7))

    def test_from_skycoord_invalid(self):
        with self.assertRaises(TypeError):
            PointLikeSource.from_skycoord((77.35, 5.7))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PointLikeSource.from_skycoord(SkyCoord(ra=[1, 2] * u.deg, dec=[3, 4] * u.deg))


class PointLikeSourceCatalogTestCase(
    unittest.TestCase,
):
    """Checks that point-like sources created via the alternative constructors
    work within a source catalog and the downstream analysis machinery.
    """

    def setUp(self):
        self.names = ['NGC 1068', 'TXS 0506+056', 'PKS 1424+240']
        self.ra_deg = np.array([40.67, 77.35, 216.76])
        self.dec_deg = np.array([-0.01, 5.7, 23.8])
        self.weights = [1.0, 2.0, 3.0]

        self.ref_sources = [
            PointLikeSource(ra=np.radians(ra), dec=np.radians(dec), name=name, weight=w)
            for (ra, dec, name, w) in zip(self.ra_deg, self.dec_deg, self.names, self.weights, strict=True)
        ]

    def _assert_catalog(self, catalog):
        self.assertIsInstance(catalog, SourceCatalog)
        self.assertEqual(catalog.source_type, PointLikeSource)
        self.assertEqual(len(catalog), len(self.ref_sources))

        cfg = Config()
        shg_mgr = SourceHypoGroupManager(
            SourceHypoGroup(
                sources=catalog.as_SourceModelCollection(),
                fluxmodel=SteadyPointlikeFFM(Phi0=1, energy_profile=None, cfg=cfg),
                detsigyield_builders=NullDetSigYieldBuilder(cfg=cfg),
                sig_gen_method=None,
            )
        )
        arr = pointlikesource_to_data_field_array(tdm=None, shg_mgr=shg_mgr, pmm=None)  # type: ignore[arg-type]

        np.testing.assert_allclose(arr['ra'], [src.ra for src in self.ref_sources])
        np.testing.assert_allclose(arr['dec'], [src.dec for src in self.ref_sources])
        np.testing.assert_allclose(arr['weight'], self.weights)

    def test_catalog_from_degrees(self):
        sources = [
            PointLikeSource.from_degrees(ra=ra, dec=dec, name=name, weight=w)
            for (ra, dec, name, w) in zip(self.ra_deg, self.dec_deg, self.names, self.weights, strict=True)
        ]
        catalog = SourceCatalog(name='MyCatalog', sources=sources, source_type=PointLikeSource)
        self._assert_catalog(catalog)

    def test_catalog_from_skycoord(self):
        sources = [
            PointLikeSource.from_skycoord(SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs'), name=name, weight=w)
            for (ra, dec, name, w) in zip(self.ra_deg, self.dec_deg, self.names, self.weights, strict=True)
        ]
        catalog = SourceCatalog(name='MyCatalog', sources=sources, source_type=PointLikeSource)
        self._assert_catalog(catalog)


if __name__ == '__main__':
    unittest.main()
