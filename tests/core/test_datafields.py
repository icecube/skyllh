# -*- coding: utf-8 -*-

import unittest

from skyllh.core.datafields import (
    DataFieldStages as DFS,
    DataFields,
)


class DataFieldStagesTestCase(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.stage_and_check = (
            DFS.DATAPREPARATION_EXP |
            DFS.DATAPREPARATION_MC |
            DFS.ANALYSIS_MC
        )
        self.stage_or_check = (
            DFS.DATAPREPARATION_MC |
            DFS.ANALYSIS_MC
        )

    def test_and_check__bitwise_ored(self):
        check = DFS.and_check(
            stage=self.stage_and_check,
            stages=(
                DFS.DATAPREPARATION_EXP |
                DFS.DATAPREPARATION_MC |
                DFS.ANALYSIS_MC
            )
        )
        self.assertTrue(check)

        check = DFS.and_check(
            stage=self.stage_and_check,
            stages=(
                DFS.DATAPREPARATION_EXP |
                DFS.ANALYSIS_EXP |
                DFS.ANALYSIS_MC
            )
        )
        self.assertFalse(check)

    def test_and_check__sequence_int(self):
        check = DFS.and_check(
            stage=self.stage_and_check,
            stages=(
                DFS.DATAPREPARATION_EXP,
                DFS.DATAPREPARATION_MC,
                DFS.ANALYSIS_MC,
            )
        )
        self.assertTrue(check)

        check = DFS.and_check(
            stage=self.stage_and_check,
            stages=(
                DFS.DATAPREPARATION_EXP,
                DFS.ANALYSIS_EXP,
                DFS.ANALYSIS_MC,
            )
        )
        self.assertFalse(check)

    def test_and_check__mixture_bitwise_ored_sequence_int(self):
        check = DFS.and_check(
            stage=self.stage_and_check,
            stages=(
                DFS.DATAPREPARATION_EXP | DFS.DATAPREPARATION_MC,
                DFS.ANALYSIS_MC,
            )
        )
        self.assertTrue(check)

    def test_or_check__bitwise_ored(self):
        check = DFS.or_check(
            stage=self.stage_or_check,
            stages=(
                DFS.DATAPREPARATION_EXP |
                DFS.DATAPREPARATION_MC |
                DFS.ANALYSIS_MC
            )
        )
        self.assertTrue(check)

        check = DFS.or_check(
            stage=self.stage_or_check,
            stages=(
                DFS.ANALYSIS_EXP
            )
        )
        self.assertFalse(check)

    def test_or_check__sequence_int(self):
        check = DFS.or_check(
            stage=self.stage_or_check,
            stages=(
                DFS.DATAPREPARATION_EXP,
                DFS.DATAPREPARATION_MC,
                DFS.ANALYSIS_MC
            )
        )
        self.assertTrue(check)

        check = DFS.or_check(
            stage=self.stage_or_check,
            stages=(
                DFS.ANALYSIS_EXP,
            )
        )
        self.assertFalse(check)

    def test_or_check__mixture_bitwise_ored_sequence_int(self):
        check = DFS.or_check(
            stage=self.stage_or_check,
            stages=(
                DFS.DATAPREPARATION_EXP | DFS.ANALYSIS_EXP,
                DFS.ANALYSIS_MC,
            )
        )
        self.assertTrue(check)


class DataFieldsTestCase(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.datafields = {
            'f0': DFS.DATAPREPARATION_EXP,
            'f1': DFS.DATAPREPARATION_EXP | DFS.DATAPREPARATION_MC,
        }

    def test_get_joint_names(self):
        fieldnames = DataFields.get_joint_names(
            datafields=self.datafields,
            stages=DFS.DATAPREPARATION_EXP)
        self.assertEqual(fieldnames, ['f0', 'f1'])

        fieldnames = DataFields.get_joint_names(
            datafields=self.datafields,
            stages=DFS.DATAPREPARATION_MC)
        self.assertEqual(fieldnames, ['f1'])

        fieldnames = DataFields.get_joint_names(
            datafields=self.datafields,
            stages=DFS.ANALYSIS_EXP)
        self.assertEqual(fieldnames, [])


if __name__ == '__main__':
    unittest.main()
