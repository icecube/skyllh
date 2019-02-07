# -*- coding: utf-8 -*-

import numpy as np

def estimate_sensitivity(analysis, rss, eps=0.03, p=0.9,
                         h0_ts_quantile=0.5, ns_range=[0, 5]):
    """Estimates number of signal events that should be injected to reach
    test statistic value higher than `h0_ts_quantile` part of null-hypothesis
    test statistic in `p` part of the performed trials. Default `p` and
    `h0_ts_quantile` values define standard sensitivity.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    eps : float, optional
        Precision in `p` for execution to break.
    p : float, optional
        Probability of signal test statistic larger than `h0_ts_quantile`
        part of null-hypothesis test statistic.
    h0_ts_quantile : float, optional
        Null-hypothesis test statistic quantile.
    ns_range : list, optional
        Initial range to search for number of injected signal events.

    Returns
    -------
    median_signal : float
        Estimated number of injected signal events to reach desired
        sensitivity.
    median_signal_sigma : float
        Estimated error of `median_signal`.
    """

    median_signal, median_signal_sigma = _calc_median_signal_for_ts_quantile(analysis, rss, eps, p, h0_ts_quantile, ns_range)
    return (median_signal, median_signal_sigma)

def estimate_discovery_potential(analysis, rss, eps=0.03, p=0.5,
                                 h0_ts_quantile=5.733e-7, ns_range=[0, 5]):
    """Estimates number of signal events that should be injected to reach
    test statistic value higher than `h0_ts_quantile` part of null-hypothesis
    test statistic in `p` part of the performed trials. Default `p` and
    `h0_ts_quantile` values define standard discovery potential.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for discovery potential estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    eps : float, optional
        Precision in `p` for execution to break.
    p : float, optional
        Probability of signal test statistic larger than `h0_ts_quantile`
        part of null-hypothesis test statistic.
    h0_ts_quantile : float, optional
        Null-hypothesis test statistic quantile.
    ns_range : list, optional
        Initial range to search for number of injected signal events.

    Returns
    -------
    median_signal : float
        Estimated number of injected signal events to reach desired
        discovery potential.
    median_signal_sigma : float
        Estimated error of `median_signal`.
    """
    median_signal, median_signal_sigma = _calc_median_signal_for_ts_quantile(analysis, rss, eps, p, h0_ts_quantile, ns_range)
    return (median_signal, median_signal_sigma)

def _calc_median_signal_for_ts_quantile(analysis, rss, eps, p, h0_ts_quantile, ns_range):
    """ Calculates median signal events needed to be injected to reach test
    statistic distribution with defined properties for a given analysis.
    """
    N=1000
    range_min = ns_range[0]
    range_max = ns_range[1]
    N_scaling = 1./100
    bkg_TS = analysis.do_trials(N*10, rss, sig_mean=0)['TS']
    bkg_TS_percentile = np.percentile(bkg_TS, (1-h0_ts_quantile)*100)

    # Initialization.
    p_trial_max = 1
    p_trial_max_sigma = 2*eps
    p_trial_min = 0
    p_trial_min_sigma = 2*eps
    while (p_trial_min_sigma + p_trial_max_sigma)/2. > eps:
        # Left part of signal range.
        p_trial_min = _estimate_p_trial(analysis, N*N_scaling, rss, range_min, bkg_TS_percentile)
        p_trial_min_sigma = 1./np.sqrt(N*N_scaling)
        range_length = range_max - range_min
        if (p_trial_min + p_trial_min_sigma) < p:
            range_min += 0.5*range_length
        elif p_trial_min > p:
            range_min -= 0.5*range_length
        else:
            N_scaling *= 2

        # Right part of signal range.
        p_trial_max = _estimate_p_trial(analysis, N*N_scaling, rss, range_max, bkg_TS_percentile)
        p_trial_max_sigma = 1./np.sqrt(N*N_scaling)
        range_length = range_max - range_min
        if (p_trial_max - p_trial_max_sigma) > p:
            range_max -= 0.5*range_length
        elif p_trial_max < p:
            range_max += 0.5*range_length
        else:
            N_scaling *= 2
    median_signal = (range_min + range_max)/2.
    median_signal_sigma = (range_max - range_min)/2.

    return (median_signal, median_signal_sigma)

def _estimate_p_trial(analysis, N, rss, sig_mean, bkg_TS_percentile):
    """Estimates trial sensitivity value for injected `sig_mean` signal
    events.
    """
    sig_TS = analysis.do_trials(int(N), rss, sig_mean=sig_mean)['TS']
    p_trial = sig_TS[sig_TS > bkg_TS_percentile].size/float(sig_TS.size)
    return p_trial
