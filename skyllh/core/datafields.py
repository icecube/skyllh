# -*- coding: utf-8 -*-

"""This module provides functionality for defining which data fields of a data
file is required at what stage.
"""


class DataFieldStages(
    object,
):
    """This class provides the data field stage values, which are individual
    bits of an integer value to allow for multiple stages of a data field.
    """

    DATAPREPARATION_EXP = 1
    DATAPREPARATION_MC = 2
    ANALYSIS_EXP = 4
    ANALYSIS_MC = 8

    @staticmethod
    def and_check(stage, stages):
        """Checks if the given stage matches all of the given stages.
        """
        for stage_ in stages:
            if stage & stage_ == 0:
                return False

        return True

    @staticmethod
    def or_check(stage, stages):
        """Checks if the given stage matches at least one of the given stages.
        """
        for stage_ in stages:
            if stage & stage_ != 0:
                return True

        return False


class DataFields(
    object,
):
    @staticmethod
    def get_joint_names(
            datafields,
            stages,
    ):
        """Returns the list of data field names that match at least one of the
        given stages, i.e. the joint set of data fields given the stages.

        Parameters
        ----------
        datafields : dict
            The dictionary of datafield names as keys and stages as values.
        stages : sequence of int
            The stages for which data field names should get returned.

        Returns
        -------
        datafield_names : list of str
            The list of data field names.
        """
        datafield_names = [
            field
            for (field, stage) in datafields.items()
            if DataFieldStages.or_check(stage, stages)
        ]

        return datafield_names
