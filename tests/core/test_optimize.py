# -*- coding: utf-8 -*-

"""This test module tests classes, methods and functions of the
``core.optimize`` module.

Note: The `PsiFuncEventSelectionMethod` is not currently used/tested.
"""

import unittest
from unittest.mock import Mock

import numpy as np

from skyllh.core.optimize import (
    AllEventSelectionMethod,
    DecBandEventSectionMethod,
    # PsiFuncEventSelectionMethod,
    RABandEventSectionMethod,
    SpatialBoxAndPsiFuncEventSelectionMethod,
    SpatialBoxEventSelectionMethod,
)
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.storage import DataFieldRecordArray
from skyllh.physics.source import PointLikeSource

from tests.core.testdata.testdata_generator import generate_testdata


def shgm_setup(n_sources=1):
    # Mock SourceHypoGroupManager class in order to pass isinstance checks and
    # set its properties used by event selection methods.
    shgm = Mock(spec_set=["__class__", "source_list", "n_sources"])
    shgm.__class__ = SourceHypoGroupManager

    rng = np.random.default_rng(0)
    x = rng.random((n_sources, 2))
    src_ras = 2 * np.pi * x[:, 0]
    src_decs = np.pi * (x[:, 1] - 0.5)
    source_list = [
        PointLikeSource(*src_loc) for src_loc in zip(src_ras, src_decs)
    ]

    shgm.source_list = source_list
    shgm.n_sources = n_sources

    return shgm


def get_func_psi_ang_err(ang_err=0.5):
    def func_psi_ang_err(psi):
        """A dummy function for psi func event selection.

        Parameters
        ----------
        psi : 1d ndarray of floats
            The opening angle between the source position and the event's
            reconstructed position.
        """
        return ang_err * np.ones_like(psi)

    return func_psi_ang_err


class AllEventSelectionMethod_TestCase(unittest.TestCase):
    def setUp(self):
        testdata = generate_testdata()
        self.test_events = DataFieldRecordArray(testdata.get("exp_testdata"))

    def test_change_source_hypo_group_manager(self):
        n_sources = 1
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = AllEventSelectionMethod(src_hypo_group_manager)

        self.assertEqual(
            evt_sel_method.src_hypo_group_manager.source_list,
            src_hypo_group_manager.source_list,
        )
        self.assertEqual(
            evt_sel_method.src_hypo_group_manager.n_sources,
            src_hypo_group_manager.n_sources,
        )

        # Change the SourceHypoGroupManager instance.
        n_sources = 2
        src_hypo_group_manager_new = shgm_setup(n_sources=n_sources)
        evt_sel_method.change_source_hypo_group_manager(
            src_hypo_group_manager_new
        )

        self.assertEqual(
            evt_sel_method.src_hypo_group_manager.source_list,
            src_hypo_group_manager_new.source_list,
        )
        self.assertEqual(
            evt_sel_method.src_hypo_group_manager.n_sources,
            src_hypo_group_manager_new.n_sources,
        )

    def test_select_events_single_source(self):
        n_sources = 1
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = AllEventSelectionMethod(src_hypo_group_manager)

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        np.testing.assert_array_equal(events, self.test_events)
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        np.testing.assert_array_equal(events, self.test_events)
        self.assertEqual(len(src_idxs), n_sources * len(self.test_events))
        self.assertEqual(len(ev_idxs), n_sources * len(self.test_events))
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(self.test_events))
        )

    def test_select_events_multiple_sources(self):
        n_sources = 2
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = AllEventSelectionMethod(src_hypo_group_manager)

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        np.testing.assert_array_equal(events, self.test_events)
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        np.testing.assert_array_equal(events, self.test_events)
        self.assertEqual(len(src_idxs), n_sources * len(self.test_events))
        self.assertEqual(len(ev_idxs), n_sources * len(self.test_events))
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(self.test_events))
        )


class DecBandEventSectionMethod_TestCase(unittest.TestCase):
    def setUp(self):
        testdata = generate_testdata()
        self.test_events = DataFieldRecordArray(testdata.get("events"))

    def test_source_to_array_single_source(self):
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = DecBandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_arr = evt_sel_method.source_to_array(
            src_hypo_group_manager.source_list
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        np.testing.assert_array_equal(src_arr["ra"], src_ras)
        np.testing.assert_array_equal(src_arr["dec"], src_decs)

    def test_source_to_array_multiple_sources(self):
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = DecBandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_arr = evt_sel_method.source_to_array(
            src_hypo_group_manager.source_list
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        np.testing.assert_array_equal(src_arr["ra"], src_ras)
        np.testing.assert_array_equal(src_arr["dec"], src_decs)

    def test_select_events_single_source(self):
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = DecBandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )
        dec_min = src_hypo_group_manager.source_list[0].dec - delta_angle
        dec_max = src_hypo_group_manager.source_list[0].dec + delta_angle

        self.assertTrue(
            np.all(events["dec"] > dec_min),
            msg="Returned selected events below src_dec - delta_angle.",
        )
        self.assertTrue(
            np.all(events["dec"] < dec_max),
            msg="Returned selected events above src_dec + delta_angle.",
        )
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        self.assertTrue(
            np.all(events["dec"] > dec_min),
            msg="Returned selected events below src_dec - delta_angle.",
        )
        self.assertTrue(
            np.all(events["dec"] < dec_max),
            msg="Returned selected events above src_dec + delta_angle.",
        )

        n_expected_events = np.sum(
            (events["dec"] > dec_min) & (events["dec"] < dec_max)
        )
        self.assertEqual(len(src_idxs), n_expected_events)
        self.assertEqual(len(ev_idxs), n_expected_events)
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_multiple_sources(self):
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = DecBandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_decs = [source.dec for source in src_hypo_group_manager.source_list]

        dec_min = np.min(src_decs) - delta_angle
        dec_max = np.max(src_decs) + delta_angle

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        self.assertTrue(
            np.all(events["dec"] > dec_min),
            msg="Returned selected events below minimal src_dec - delta_angle.",
        )
        self.assertTrue(
            np.all(events["dec"] < dec_max),
            msg="Returned selected events above maximal src_dec + delta_angle.",
        )
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        self.assertTrue(
            np.all(events["dec"] > dec_min),
            msg="Returned selected events below minimal src_dec - delta_angle.",
        )
        self.assertTrue(
            np.all(events["dec"] < dec_max),
            msg="Returned selected events above maximal src_dec + delta_angle.",
        )

        for i, src_dec in enumerate(src_decs):
            events_mask = src_idxs == i
            dec_min = src_dec - delta_angle
            dec_max = src_dec + delta_angle

            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] > dec_min),
                msg="Returned selected events below src_dec - delta_angle.",
            )
            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] < dec_max),
                msg="Returned selected events above src_dec + delta_angle.",
            )
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )


class RABandEventSectionMethod_TestCase(unittest.TestCase):
    def setUp(self):
        testdata = generate_testdata()
        self.test_events = DataFieldRecordArray(testdata.get("events"))

    def test_select_events_single_source(self):
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = RABandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi / 2, src_decs - delta_angle)
        src_dec_plus = np.minimum(src_decs + delta_angle, np.pi / 2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2 * np.pi, n_sources), np.fabs(delta_angle / cosfact)],
            axis=0,
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        for i in range(n_sources):
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(np.fabs(events["ra"] - np.pi) < src_ra_max),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        for i in range(n_sources):
            events_mask = src_idxs == i
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(
                    np.fabs(events["ra"][ev_idxs[events_mask]] - np.pi)
                    < src_ra_max
                ),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )

        src_ra_max = src_ras[0] + dRA_half[0] - np.pi
        n_expected_events = np.sum((np.fabs(events["ra"] - np.pi) < src_ra_max))

        self.assertEqual(len(src_idxs), n_expected_events)
        self.assertEqual(len(ev_idxs), n_expected_events)
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_multiple_sources(self):
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = RABandEventSectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi / 2, src_decs - delta_angle)
        src_dec_plus = np.minimum(src_decs + delta_angle, np.pi / 2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2 * np.pi, n_sources), np.fabs(delta_angle / cosfact)],
            axis=0,
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        # TODO: Can't really test events of multiple sources selection without
        # idxs.
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        for i in range(n_sources):
            events_mask = src_idxs == i
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(
                    np.fabs(events["ra"][ev_idxs[events_mask]] - np.pi)
                    < src_ra_max
                ),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )


class SpatialBoxEventSelectionMethod_TestCase(unittest.TestCase):
    def setUp(self):
        testdata = generate_testdata()
        self.test_events = DataFieldRecordArray(testdata.get("events"))

    def test_select_events_single_source(self):
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi / 2, src_decs - delta_angle)
        src_dec_plus = np.minimum(src_decs + delta_angle, np.pi / 2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2 * np.pi, n_sources), np.fabs(delta_angle / cosfact)],
            axis=0,
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        for i in range(n_sources):
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(np.fabs(events["ra"] - np.pi) < src_ra_max),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )

            dec_min = src_decs[i] - delta_angle
            dec_max = src_decs[i] + delta_angle

            self.assertTrue(
                np.all(events["dec"] > dec_min),
                msg="Returned selected events below src_dec - delta_angle.",
            )
            self.assertTrue(
                np.all(events["dec"] < dec_max),
                msg="Returned selected events above src_dec + delta_angle.",
            )
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        for i in range(n_sources):
            events_mask = src_idxs == i
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(
                    np.fabs(events["ra"][ev_idxs[events_mask]] - np.pi)
                    < src_ra_max
                ),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )

            dec_min = src_decs[i] - delta_angle
            dec_max = src_decs[i] + delta_angle

            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] > dec_min),
                msg="Returned selected events below src_dec - delta_angle.",
            )
            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] < dec_max),
                msg="Returned selected events above src_dec + delta_angle.",
            )

        src_ra_max = src_ras[0] + dRA_half[0] - np.pi
        n_expected_events = np.sum((np.fabs(events["ra"] - np.pi) < src_ra_max))
        self.assertEqual(len(src_idxs), n_expected_events)
        self.assertEqual(len(ev_idxs), n_expected_events)
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_multiple_sources(self):
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        evt_sel_method = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        src_ras = np.array(
            [source.ra for source in src_hypo_group_manager.source_list]
        )
        src_decs = np.array(
            [source.dec for source in src_hypo_group_manager.source_list]
        )

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi / 2, src_decs - delta_angle)
        src_dec_plus = np.minimum(src_decs + delta_angle, np.pi / 2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2 * np.pi, n_sources), np.fabs(delta_angle / cosfact)],
            axis=0,
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        dec_min = np.min(src_decs) - delta_angle
        dec_max = np.max(src_decs) + delta_angle

        self.assertTrue(
            np.all(events["dec"] > dec_min),
            msg="Returned selected events below src_dec - delta_angle.",
        )
        self.assertTrue(
            np.all(events["dec"] < dec_max),
            msg="Returned selected events above src_dec + delta_angle.",
        )
        self.assertIsNone(idxs)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )

        for i in range(n_sources):
            events_mask = src_idxs == i
            src_ra_max = src_ras[i] + dRA_half[i] - np.pi

            self.assertTrue(
                np.all(
                    np.fabs(events["ra"][ev_idxs[events_mask]] - np.pi)
                    < src_ra_max
                ),
                msg="Returned selected events above maximal "
                "src_ra + delta_angle/cosfact.",
            )

            dec_min = src_decs[i] - delta_angle
            dec_max = src_decs[i] + delta_angle

            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] > dec_min),
                msg="Returned selected events below src_dec - delta_angle.",
            )
            self.assertTrue(
                np.all(events["dec"][ev_idxs[events_mask]] < dec_max),
                msg="Returned selected events above src_dec + delta_angle.",
            )
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )


class SpatialBoxAndPsiFuncEventSelectionMethod_TestCase(unittest.TestCase):
    def setUp(self):
        testdata = generate_testdata()
        self.test_events = DataFieldRecordArray(testdata.get("events"))

    def test_select_events_single_source(self):
        """Check if the event selection without a psi cut returns an identical
        result to the `SpatialBoxEventSelectionMethod`.
        """
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        func = get_func_psi_ang_err(ang_err=0)
        evt_sel_method = SpatialBoxAndPsiFuncEventSelectionMethod(
            src_hypo_group_manager=src_hypo_group_manager,
            delta_angle=delta_angle,
            psi_name="psi",
            func=func,
            axis_name_list=["psi"],
        )

        evt_sel_method_sb = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )
        (events_sb, idxs_sb) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb.as_numpy_record_array(),
        )
        self.assertIsNone(idxs)
        self.assertIsNone(idxs_sb)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )
        (
            events_sb,
            (src_idxs_sb, ev_idxs_sb),
        ) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=True)

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb.as_numpy_record_array(),
        )
        np.testing.assert_array_equal(src_idxs, src_idxs_sb)
        np.testing.assert_array_equal(ev_idxs, ev_idxs_sb)
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_multiple_sources(self):
        """Check if the event selection without a psi cut returns an identical
        result to the `SpatialBoxEventSelectionMethod`.
        """
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        func = get_func_psi_ang_err(ang_err=0)
        evt_sel_method = SpatialBoxAndPsiFuncEventSelectionMethod(
            src_hypo_group_manager=src_hypo_group_manager,
            delta_angle=delta_angle,
            psi_name="psi",
            func=func,
            axis_name_list=["psi"],
        )

        evt_sel_method_sb = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )
        (events_sb, idxs_sb) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb.as_numpy_record_array(),
        )
        self.assertIsNone(idxs)
        self.assertIsNone(idxs_sb)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )
        (
            events_sb,
            (src_idxs_sb, ev_idxs_sb),
        ) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=True)

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb.as_numpy_record_array(),
        )
        np.testing.assert_array_equal(src_idxs, src_idxs_sb)
        np.testing.assert_array_equal(ev_idxs, ev_idxs_sb)
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_single_source_psi_func(self):
        n_sources = 1
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        ang_err = 3.0

        func = get_func_psi_ang_err(ang_err)
        evt_sel_method = SpatialBoxAndPsiFuncEventSelectionMethod(
            src_hypo_group_manager=src_hypo_group_manager,
            delta_angle=delta_angle,
            psi_name="psi",
            func=func,
            axis_name_list=["psi"],
            psi_floor=0.0,
        )

        evt_sel_method_sb = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )
        (events_sb, idxs_sb) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        mask_psi_cut = events_sb["ang_err"] > ang_err

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb[mask_psi_cut].as_numpy_record_array(),
        )
        self.assertIsNone(idxs)
        self.assertIsNone(idxs_sb)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )
        (
            events_sb,
            (src_idxs_sb, ev_idxs_sb),
        ) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=True)

        mask_psi_cut = events_sb["ang_err"] > ang_err

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb[mask_psi_cut].as_numpy_record_array(),
        )
        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )

    def test_select_events_multiple_sources_psi_func(self):
        n_sources = 2
        delta_angle = np.deg2rad(15)
        src_hypo_group_manager = shgm_setup(n_sources=n_sources)
        ang_err = 3.0

        func = get_func_psi_ang_err(ang_err)
        evt_sel_method = SpatialBoxAndPsiFuncEventSelectionMethod(
            src_hypo_group_manager=src_hypo_group_manager,
            delta_angle=delta_angle,
            psi_name="psi",
            func=func,
            axis_name_list=["psi"],
            psi_floor=0.0,
        )

        evt_sel_method_sb = SpatialBoxEventSelectionMethod(
            src_hypo_group_manager, delta_angle
        )

        # Test with `ret_src_ev_idxs=False`.
        (events, idxs) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=False
        )
        (events_sb, idxs_sb) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=False
        )

        mask_psi_cut = events_sb["ang_err"] > ang_err

        np.testing.assert_array_equal(
            events.as_numpy_record_array(),
            events_sb[mask_psi_cut].as_numpy_record_array(),
        )
        self.assertIsNone(idxs)
        self.assertIsNone(idxs_sb)

        # Test with `ret_src_ev_idxs=True`.
        (events, (src_idxs, ev_idxs)) = evt_sel_method.select_events(
            self.test_events, ret_src_ev_idxs=True
        )
        (
            events_sb,
            (src_idxs_sb, ev_idxs_sb),
        ) = evt_sel_method_sb.select_events(
            self.test_events, ret_src_ev_idxs=True)

        for i in range(n_sources):
            events_mask = src_idxs == i
            events_mask_sb = src_idxs_sb == i
            mask_psi_cut = (
                events_sb[ev_idxs_sb[events_mask_sb]]["ang_err"] > ang_err
            )

            np.testing.assert_array_equal(
                events[ev_idxs[events_mask]].as_numpy_record_array(),
                events_sb[ev_idxs_sb[events_mask_sb]][
                    mask_psi_cut
                ].as_numpy_record_array(),
            )

        np.testing.assert_array_equal(np.unique(src_idxs), np.arange(n_sources))
        np.testing.assert_array_equal(
            np.unique(ev_idxs), np.arange(len(events))
        )


if __name__ == "__main__":
    unittest.main()
