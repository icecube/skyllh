# -*- coding: utf-8 -*-
# TODO: File out of date, tell flake8 to ignore
# flake8: noqa

from __future__ import division

import os.path
import unittest
import numpy as np

from skyllh.core.analysis import Analysis
from skyllh.core.random import RandomStateService

# Classes to define the source hypothesis.
from skyllh.physics.source import PointLikeSource
from skyllh.physics.flux import PowerLawFlux
from skyllh.core.source_hypo_group import SourceHypoGroup
from skyllh.core.source_hypothesis import SourceHypoGroupManager

# Classes to define the fit parameters.
from skyllh.core.parameters import (
    SingleSourceFitParameterMapper,
    FitParameter
)

# Classes for the minimizer.
from skyllh.core.minimizer import Minimizer, LBFGSMinimizerImpl

# Classes for defining the analysis.
from skyllh.core.test_statistic import TestStatisticWilks
#from skyllh.core.analysis import (
#    SpacialEnergyTimeIntegratedMultiDatasetSingleSourceAnalysis as Analysis
#)

"""
class TestAnalysis(unittest.TestCase):
    def setUp(self):
        # path = os.path.abspath(os.path.dirname(__file__))
        # self.exp_data = np.load(os.path.join(path, 'testdata/exp_testdata.npy'))
        # self.mc_data = np.load(os.path.join(path, 'testdata/mc_testdata.npy'))
        # self.livetime_data = np.load(os.path.join(path, 'testdata/livetime_testdata.npy'))

        # Create the minimizer instance.
        minimizer = Minimizer(LBFGSMinimizerImpl())

        # Create a source hypothesis group manager.
        src_hypo_group_manager = SourceHypoGroupManager(SourceHypoGroup(
            source, fluxmodel, detsigeff_implmethod, sig_gen_method))

        # Create a source fit parameter mapper and define the fit parameters.
        src_fitparam_mapper = SingleSourceFitParameterMapper(rss)
        src_fitparam_mapper.def_fit_parameter(fitparam_gamma)
        # Define the test statistic.
        test_statistic = TestStatisticWilks()

        self.analysis = Analysis()
        self.rss = RandomStateService(seed=0)

        # Define the data scrambler with its data scrambling method, which is used
        # for background generation.
        data_scrambler = DataScrambler(UniformRAScramblingMethod(),
                                       inplace_scrambling=True)

        # Create background generation method.
        bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(data_scrambler)

    def tearDown(self):
        # self.exp_data.close()
        # self.mc_data.close()
        # self.livetime_data.close()
        pass

    def test_do_trials(self):
        N = 10
        ncpu = None

        self.analysis.do_trials(N, self.rss, ncpu=ncpu)
"""

if(__name__ == '__main__'):
    unittest.main()
