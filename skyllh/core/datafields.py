"""This module provides functionality for defining which data fields of a data
file is required at what stage.
"""

from collections.abc import Sequence


class DataFieldStages:
    """This class provides the data field stage values, which are individual
    bits of an integer value to allow for multiple stages of a data field.
    """

    DATAPREPARATION_EXP = 1
    DATAPREPARATION_MC = 2
    ANALYSIS_EXP = 4
    ANALYSIS_MC = 8

    @staticmethod
    def and_check(
        stage: int,
        stages: int | Sequence[int],
    ) -> bool:
        """Checks if the given stage matches all of the given stages.

        Parameters
        ----------
        stage
            The stage value, which should get checked.
        stages
            The stage(s) to check for.

        Returns
        -------
        check
            ``True`` if the given stage contains all of the given stages,
            ``False`` otherwise.
        """
        if isinstance(stages, int):
            return stage & stages == stages

        return all(stage & stage_ == stage_ for stage_ in stages)

    @staticmethod
    def or_check(
        stage: int,
        stages: int | Sequence[int],
    ) -> bool:
        """Checks if the given stage matches any of the given stages.

        Parameters
        ----------
        stage
            The stage value, which should get checked.
        stages
            The stage(s) to check for.

        Returns
        -------
        check
            ``True`` if the given stage contains any of the given stages,
            ``False`` otherwise.
        """
        if isinstance(stages, int):
            return stage & stages != 0

        return any(stage & stage_ != 0 for stage_ in stages)


class DataFields:
    @staticmethod
    def get_joint_names(
        datafields: dict,
        stages: int | Sequence[int],
    ) -> list[str]:
        """Returns the list of data field names that match at least one of the
        given stages, i.e. the joint set of data fields given the stages.

        Parameters
        ----------
        datafields
            The dictionary of data field names as keys and stages as values.
        stages
            The stage(s) for which data field names should get returned.

        Returns
        -------
        datafield_names
            The list of data field names.
        """
        datafield_names = [field for (field, stage) in datafields.items() if DataFieldStages.or_check(stage, stages)]

        return datafield_names
