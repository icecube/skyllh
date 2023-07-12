# -*- coding: utf-8 -*-

"""This modules defines base types for some of the SkyLLH classes to avoid
circular imports when actively checking for types.
"""


class DataFieldStages_t(
    object,
):

    DATAFILE_EXP = 1
    DATAFILE_MC = 2
    DATAPREPARATION_EXP = 4
    DATAPREPARATION_MC = 8
    ANALYSIS_EXP = 16
    ANALYSIS_MC = 32

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

    @staticmethod
    def get_joint_datafields(datafields, stages):
        """Returns the list of data field names that match at least one of the
        given stages, i.e. the joint set of data fields given the stages.

        Parameters
        ----------
        datafields : dict
            The dictionary of datafield names as keys and stages as values.
        stages : sequence of int
            The stages for which data field names should

        Returns
        -------
        datafield_names : list of str
            The list of data field names.
        """
        datafield_names = [
            field
            for (field, stage) in datafields.items()
            if DataFieldStages_t.or_check(stage, stages)
        ]

        return datafield_names


class SourceHypoGroup_t(
    object,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
