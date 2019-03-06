# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.lib import recfunctions as np_rfn

from skyllh.core.py import issequenceof, range
from skyllh.core.storage import NPYFileLoader
from skyllh.physics.source import PointLikeSource

from scipy.interpolate import interp1d

"""This module contains common utility functions useful for an analysis.
"""

def pointlikesource_to_data_field_array(tdm, src_hypo_group_manager):
    """Function to transform a list of PointLikeSource sources into a numpy
    record ndarray. The resulting numpy record ndarray contains the following
    fields:

        `ra`: float
            The right-ascention of the point-like source.
        `dec`: float
            The declination of the point-like source.

    Parameters
    ----------
    tdm : instance of TrialDataManager
        The TrialDataManager instance.
    src_hypo_group_manager : instance of SourceHypoGroupManager
        The instance of SourceHypoGroupManager that defines the sources.

    Returns
    -------
    arr : (N_sources,)-shaped numpy record ndarray
        The numpy record ndarray holding the source parameters `ra` and `dec`.
    """
    sources = src_hypo_group_manager.source_list

    if(not issequenceof(sources, PointLikeSource)):
        raise TypeError('The sources of the SourceHypoGroupManager must be '
            'PointLikeSource instances!')

    arr = np.empty(
        (len(sources),),
        dtype=[('ra', np.float), ('dec', np.float)],
        order='F')

    for (i, src) in enumerate(sources):
        arr['ra'][i] = src.ra
        arr['dec'][i] = src.dec

    return arr

def estimate_sensitivity(analysis, rss, eps_p=0.03, p=0.9,
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
    eps_p : float, optional
        Precision in `p` for execution to break.
    p : float, optional
        Desired probability of signal test statistic for exceeding
        `h0_ts_quantile` part of null-hypothesis test statistic threshold.
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

    median_signal, median_signal_sigma = _calc_median_signal_for_ts_quantile(analysis, rss, eps_p, p, h0_ts_quantile, ns_range)
    return (median_signal, median_signal_sigma)

def estimate_discovery_potential(analysis, rss, eps_p=0.03, p=0.5,
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
    eps_p : float, optional
        Precision in `p` for execution to break.
    p : float, optional
        Desired probability of signal test statistic for exceeding
        `h0_ts_quantile` part of null-hypothesis test statistic threshold.
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
    median_signal, median_signal_sigma = _calc_median_signal_for_ts_quantile(analysis, rss, eps_p, p, h0_ts_quantile, ns_range)
    return (median_signal, median_signal_sigma)

def _calc_median_signal_for_ts_quantile(analysis, rss, eps_p, p, h0_ts_quantile, ns_range):
    """Calculates median signal events needed to be injected to reach test
    statistic distribution with defined properties for a given analysis.
    Calculation is done by calculating `p_trial` values at lower and upper range
    points and comparing if `p_trial` values interval is around given `p` value.
    If `p_trial` value is within 1 sigma to `p` we increase the number of trials
    (resolution) until the desired `eps_p` precision is reached, otherwise the
    interval bounds are changed to minimize ns_range while keeping
    `p_trial_min` <= `p` <= `p_trial_max` inequality correct.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    eps_p : float
        Precision in `p` for execution to break.
    p : float
        Desired probability of signal test statistic for exceeding
        `h0_ts_quantile` part of null-hypothesis test statistic threshold.
    h0_ts_quantile : float
        Null-hypothesis test statistic quantile.
    ns_range : list
        Initial range to search for number of injected signal events.

    Returns
    -------
    median_signal : float
        Estimated number of injected signal events to reach desired
        sensitivity.
    median_signal_sigma : float
        Estimated error of `median_signal`.
    """
    N = 1000
    range_min = ns_range[0]
    range_max = ns_range[1]
    N_scaling = 1/100
    bkg_TS = analysis.do_trials(rss, N*10, sig_mean=0)['TS']
    bkg_TS_percentile = np.percentile(bkg_TS, (1-h0_ts_quantile)*100)

    # Initialization.
    p_trial_max = 1
    p_trial_max_sigma = 2*eps_p
    p_trial_min = 0
    p_trial_min_sigma = 2*eps_p
    while (p_trial_min_sigma + p_trial_max_sigma)/2 > eps_p:
        # Left part of signal range.
        N_scaled = int(N*N_scaling)
        p_trial_min = _estimate_p_trial(analysis, N_scaled, rss, range_min, bkg_TS_percentile)
        p_trial_min_sigma = 1/np.sqrt(N_scaled)
        range_length = range_max - range_min
        if (p_trial_min + p_trial_min_sigma) < p:
            range_min += 0.5*range_length
        elif p_trial_min > p:
            range_min -= 0.5*range_length
        else:
            N_scaling *= 2

        # Right part of signal range.
        N_scaled = int(N*N_scaling)
        p_trial_max = _estimate_p_trial(analysis, N_scaled, rss, range_max, bkg_TS_percentile)
        p_trial_max_sigma = 1/np.sqrt(N_scaled)
        range_length = range_max - range_min
        if (p_trial_max - p_trial_max_sigma) > p:
            range_max -= 0.5*range_length
        elif p_trial_max < p:
            range_max += 0.5*range_length
        else:
            N_scaling *= 2
    median_signal = (range_min + range_max)/2
    median_signal_sigma = (range_max - range_min)/2

    return (median_signal, median_signal_sigma)

def _estimate_p_trial(analysis, N, rss, sig_mean, bkg_TS_percentile):
    """Estimates trial with injected `sig_mean` signal probability for exceeding
    background test statistic threshold.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    N : int
        Number of times to perform analysis trial.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    sig_mean : float
        The mean number of signal events that should be generated for the
        trial. The actual number of generated events will be drawn from a
        Poisson distribution with this given signal mean as mean.
    bkg_TS_percentile : float
        Background test statistic value at defined percentile.

    Returns
    -------
    p_trial : float
        Estimated trial with injected `sig_mean` signal probability for
        exceeding background test statistic threshold.
    """
    sig_TS = analysis.do_trials(rss, N, sig_mean=sig_mean)['TS']
    p_trial = sig_TS[sig_TS > bkg_TS_percentile].size/sig_TS.size
    return p_trial

def create_trial_data_file(analysis, rss, pathfilename, ns_max=30, N=2000):
    """Creates and fills a trial data file with `N` generated trials for each
    mean number of injected signal events up to `ns_max` for a given analysis. 

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    pathfilename : string
        Trial data file path including the filename.
    ns_max : int, optional
        Maximum number of injected signal events. 
    N : int
        Number of times to perform analysis trial.
    """
    # Initialize empty `trial_data` array.
    trial_data = np.array([])
    for ns in range(0, ns_max):
        trials = analysis.do_trials(rss, N, sig_mean=ns)
        names = ['sig_mean', 'seed']
        data = [[ns]*N, [rss.seed]*N]
        trials = np_rfn.append_fields(trials, names, data)
        trial_data = np_rfn.stack_arrays([trial_data, trials], usemask=False,
                                         asrecarray=True)
    # Save trial data to file.
    np.save(pathfilename, trial_data)

def extend_trial_data_file(analysis, rss, pathfilename, ns_max=30, N=1000):
    """Appends the trial data file with `N` generated trials for each mean
    number of injected signal events up to `ns_max` for a given analysis. 

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    pathfilename : string
        Trial data file path including the filename.
    ns_max : int, optional
        Maximum number of injected signal events. 
    N : int
        Number of times to perform analysis trial.
    """
    # Load trial data file.
    trial_data = NPYFileLoader(pathfilename).load_data()

    # Use unique seed to generate non identical trials.
    if rss.seed in trial_data['seed']:
        seed = next(i for i, e in enumerate(sorted(trial_data['seed']) + [None],
                    1) if i != e)
        rss.reseed(seed)
    for ns in range(0, ns_max):
        trials = analysis.do_trials(rss, N, sig_mean=ns)
        names = ['sig_mean', 'seed']
        data = [[ns]*N, [rss.seed]*N]
        trials = np_rfn.append_fields(trials, names, data)
        trial_data = np_rfn.stack_arrays([trial_data, trials], usemask=False,
                                         asrecarray=True)
    # Save trial data to the file.
    np.save(pathfilename, trial_data)

def calculate_upper_limit_distribution(analysis, rss, pathfilename, N_bkg=5000,
                                       n_bins=100):
    """Function to calculate upper limit distribution. It loads the trial data
    file containing test statistic distribution and calculates 10 percentile
    value for each mean number of injected signal event. Then it finds upper
    limit values which correspond to generated background trials test statistic
    values by linearly interpolated curve of 10 percentile values distribution.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    pathfilename : string
        Trial data file path including the filename.
    N_bkg : int, optional
        Number of times to perform background analysis trial.
    n_bins : int, optional
        Number of returned test statistic histograms bins.

    Returns
    -------
    result : dict
        Result dictionary which contains the following fields:
        - ul : list of float
            List of upper limit values.
        - mean : float
            Mean of upper limit values.
        - median : float
            Median of upper limit values.
        - var : float
            Variance of upper limit values.
        - ts_hist : numpy ndarray
            2D array of test statistic histograms calculated by axis 1.
        - extent : list of float
            Test statistic histogram boundaries.
        - q_values : list of float
            `q` percentile values of test statistic for different injected
            events means.
    """
    # Load trial data file.
    trial_data = NPYFileLoader(pathfilename).load_data()
    ns_max = max(trial_data['sig_mean']) + 1
    ts_bins_range = (min(trial_data['TS']), max(trial_data['TS']))

    q = 10 # Upper limit criterion.
    trial_data_q_values = np.empty((ns_max,))
    trial_data_ts_hist = np.empty((ns_max, n_bins))
    for ns in range(ns_max):
        trial_data_q_values[ns] = np.percentile(
            trial_data['TS'][trial_data['sig_mean'] == ns], q)
        (trial_data_ts_hist[ns, :], bin_edges) = np.histogram(
            trial_data['TS'][trial_data['sig_mean'] == ns],
            bins=n_bins, range=ts_bins_range)

    ts_inv_f = interp1d(trial_data_q_values, range(ns_max), kind='linear')
    ts_bkg = analysis.do_trials(rss, N_bkg, sig_mean=0)['TS']

    # Cut away lower background test statistic values than the minimal
    # `ts_inv_f` interpolation boundary.
    ts_bkg = ts_bkg[ts_bkg >= min(trial_data_q_values)]

    ul_list = map(ts_inv_f, ts_bkg)
    ul_mean = np.mean(ul_list)
    ul_median = np.median(ul_list)
    ul_var = np.var(ul_list)
    
    result = {}
    result['ul'] = ul_list
    result['mean'] = ul_mean
    result['median'] = ul_median
    result['var'] = ul_var
    result['ts_hist'] = trial_data_ts_hist
    result['extent'] = [0, ns_max, ts_bins_range[0], ts_bins_range[1]]
    result['q_values'] = trial_data_q_values

    return result
