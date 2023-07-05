# -*- coding: utf-8 -*-

"""This file defines IceCube specific global configuration.
"""


def add_icecube_specific_analysis_required_data_fields(cfg):
    """Adds IceCube specific data fields required by an IceCube analysis to
    the given local configuration.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    """
    cfg.add_analysis_required_exp_data_field_names([
        'azi',
        'zen',
        'sin_dec'])

    cfg.add_analysis_required_mc_data_field_names([
        'sin_true_dec'])
