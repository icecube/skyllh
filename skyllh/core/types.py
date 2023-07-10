# -*- coding: utf-8 -*-

"""This modules defines base types for some of the SkyLLH classes to avoid
circular imports when actively checking for types.
"""


class DataFieldStages_t(
    object,
):

    EXP_DATAFILE = 1
    MC_DATAFILE = 2
    EXP_DATAPREPARATION = 4
    MC_DATAPREPARATION = 8
    EXP_ANALYSIS = 16
    MC_ANALYSIS = 32

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
