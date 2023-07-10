# -*- coding: utf-8 -*-

"""This file defines IceCube specific global configuration.
"""

from skyllh.core.types import (
    DataFieldStages_t as DFS,
)


def add_icecube_specific_analysis_required_data_fields(cfg):
    """Adds IceCube specific data fields required by an IceCube analysis to
    the given local configuration.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    """
    cfg['datafields']['azi'] = DFS.EXP_ANALYSIS
    cfg['datafields']['zen'] = DFS.EXP_ANALYSIS
    cfg['datafields']['sin_dec'] = DFS.EXP_ANALYSIS
    cfg['datafields']['sin_true_dec'] = DFS.MC_ANALYSIS
