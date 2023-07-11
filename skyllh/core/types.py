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


class SourceHypoGroup_t(
    object,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
