# -*- coding: utf-8 -*-

"""The unit tests in this module test classes of the skyllh.core.signalpdf
module.
"""

import numpy as np

import unittest
from unittest.mock import Mock

from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.model import (
    SourceModel,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.signalpdf import (
    SignalTimePDF,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)
from skyllh.physics.flux_model import (
    BoxTimeFluxProfile,
)


def create_tdm(n_sources, n_selected_events):
    """Creates a Mock instance mimicing a TrialDataManager instance with a
    given number of sources and selected events.
    """
    tdm = Mock(spec_set=[
        '__class__',
        'trial_data_state_id',
        'get_n_values',
        'src_evt_idxs',
        'n_sources',
        'n_selected_events',
        'get_data'])

    def tdm_get_data(key):
        if n_selected_events == 3:
            return np.array([0, 5, 9.7])
        raise ValueError(
            f'Value n_selected_events={n_selected_events} is not supported!')

    tdm.__class__ = TrialDataManager
    tdm.trial_data_state_id = 1
    tdm.get_n_values = lambda: n_sources*n_selected_events
    tdm.src_evt_idxs = (
        np.repeat(np.arange(n_sources), n_selected_events),
        np.tile(np.arange(n_selected_events), n_sources)
    )
    tdm.n_sources = n_sources
    tdm.n_selected_events = n_selected_events
    tdm.get_data = tdm_get_data

    return tdm


class TestSignalTimePDF(unittest.TestCase):
    def setUp(self):
        self.pmm = ParameterModelMapper(
            models=[
                SourceModel(),
                SourceModel()])

        self.livetime = Livetime(np.array([
            [0, 1],
            [1.3, 4.6],
            [7.7, 10],
        ]))

        self.integral = 10
        self.S = (1-0) + (4.6-1.3) + (10-7.7)

        self.time_flux_profile = BoxTimeFluxProfile(
            t0=5,
            tw=10)

        self.sig_time_pdf = SignalTimePDF(
            pmm=self.pmm,
            livetime=self.livetime,
            time_flux_profile=self.time_flux_profile)

    def test__str__(self):
        str(self.sig_time_pdf)

    def test_calculate_time_flux_profile_integrals(self):
        (integral, S) = self.sig_time_pdf._calculate_time_flux_profile_integrals()
        self.assertEqual(integral, 10)
        self.assertEqual(S, self.S)

    def test_get_pd(self):
        tdm = create_tdm(n_sources=self.pmm.n_sources, n_selected_events=3)
        src_params_recarray = self.pmm.create_src_params_recarray(gflp_values=[])

        (pd, grads) = self.sig_time_pdf.get_pd(
            tdm=tdm,
            src_params_recarray=src_params_recarray)

        np.testing.assert_almost_equal(
            pd,
            np.array([
                1*self.S/self.integral**2,
                0.,
                1*self.S/self.integral**2,
                1*self.S/self.integral**2,
                0.,
                1*self.S/self.integral**2
            ]))

        self.assertEqual(grads, {})


if __name__ == '__main__':
    unittest.main()
