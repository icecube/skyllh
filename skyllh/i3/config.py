# -*- coding: utf-8 -*-

"""This file defines IceCube specific global configuration.
"""

from skyllh.core.datafields import (
    DataFieldStages as DFS,
)


def add_icecube_specific_analysis_required_data_fields(cfg):
    """Adds IceCube specific data fields required by an IceCube analysis to
    the given local configuration.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    """
    cfg['datafields']['azi'] = DFS.ANALYSIS_EXP
    cfg['datafields']['zen'] = DFS.ANALYSIS_EXP
    cfg['datafields']['sin_dec'] = DFS.ANALYSIS_EXP
    cfg['datafields']['sin_true_dec'] = DFS.ANALYSIS_MC
