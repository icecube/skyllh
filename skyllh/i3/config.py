# -*- coding: utf-8 -*-

"""This file defines IceCube specific global configuration.
"""

from skyllh.core.config import CFG

# Add default analysis required data fields for experimental and monte-carlo
# data that are IceCube specific.
CFG['dataset']['analysis_required_exp_field_names'] += ['azi', 'zen', 'sin_dec']
CFG['dataset']['analysis_required_mc_field_names'] += ['sin_true_dec']
