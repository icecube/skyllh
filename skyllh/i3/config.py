# -*- coding: utf-8 -*-

"""This file defines IceCube specific global configuration.
"""

from skyllh.core.config import (
    add_analysis_required_exp_data_field_names,
    add_analysis_required_mc_data_field_names,
)


def add_icecube_specific_analysis_required_data_fields():
    add_analysis_required_exp_data_field_names(['azi', 'zen', 'sin_dec'])
    add_analysis_required_mc_data_field_names(['sin_true_dec'])
