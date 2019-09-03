# -*- coding: utf-8 -*-

from __future__ import division

import logging
import numpy as np
from numpy.lib import recfunctions as np_rfn
import itertools

from skyllh.core.progressbar import ProgressBar
from skyllh.core.py import issequenceof, range
from skyllh.core.session import is_interactive_session
from skyllh.core.storage import NPYFileLoader
from skyllh.physics.source import PointLikeSource

from scipy.interpolate import interp1d

"""This module contains common utility functions useful for an analysis.
"""

def pointlikesource_to_data_field_array(
        tdm, src_hypo_group_manager):
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


def calculate_pval_from_trials(
        ts_vals, ts_threshold):
    """Calculates the percentage (p-value) of test-statistic trials that are
    above the given test-statistic critical value.
    In addition it calculates the standard deviation of the p-value assuming
    binomial statistics.

    Parameters
    ----------
    ts_vals : (n_trials,)-shaped 1D ndarray of float
        The ndarray holding the test-statistic values of the trials.
    ts_threshold : float
        The critical test-statistic value.
    """
    p = ts_vals[ts_vals > ts_threshold].size / ts_vals.size
    p_sigma = np.sqrt(p * (1 - p) / ts_vals.size)

    return (p, p_sigma)


def estimate_mean_nsignal_for_ts_quantile(
        ana, rss, h0_ts_vals, h0_ts_quantile, p, eps_p, mu_range, min_dmu=0.5,
        bkg_kwargs=None, sig_kwargs=None, ppbar=None):
    """Calculates the mean number of signal events needed to be injected to
    reach a test statistic distribution with defined properties for the given
    analysis.

    Parameters
    ----------
    ana : Analysis instance
        The Analysis instance to use for the calculation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random numbers.
    h0_ts_vals : (n_h0_ts_vals,)-shaped 1D ndarray | None
        The 1D ndarray holding the test-statistic values for the
        null-hypothesis. If set to `None`, the number of trials is calculated
        from binomial statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
        where `eps` is `min(5e-3, h0_ts_quantile/10)`.
    h0_ts_quantile : float
        Null-hypothesis test statistic quantile.
    p : float
        Desired probability of signal test statistic for exceeding
        `h0_ts_quantile` part of null-hypothesis test statistic threshold.
    eps_p : float
        Precision in `p` as stopping condition for the calculation.
    mu_range : 2-element sequence
        The range of mu (lower,upper) to search for mean number of signal
        events.
    min_dmu : float
        The minimum delta mu to use for calculating the derivative dmu/dp.
        The default is ``0.5``.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the mean number of signal events, mu.
    ppbar : instance of ProgressBar | None
        The possible parent ProgressBar instance.

    Returns
    -------
    mu : float
        Estimated mean number of signal events.
    mu_err : float
        The uncertainty on the mean number of signal events.
    """
    logger = logging.getLogger(__name__)

    n_total_generated_trials = 0

    if(h0_ts_vals is None):
        eps = min(0.005, h0_ts_quantile/10)
        n_trials = int(h0_ts_quantile*(1-h0_ts_quantile)/eps**2 + 0.5)
        logger.debug(
            'Generate %d null-hypothesis trials',
            n_trials)
        h0_ts_vals = ana.do_trials(
            rss, n_trials, mean_n_sig=0, bkg_kwargs=bkg_kwargs,
            sig_kwargs=sig_kwargs, ppbar=ppbar)['ts']
        n_total_generated_trials += n_trials

    h0_ts_vals = h0_ts_vals[np.isfinite(h0_ts_vals)]
    logger.debug(
        'Number of trials after finite cut: %d',
        len(h0_ts_vals))
    logger.debug(
        'Min / Max h0 TS value: %e / %e',
        np.min(h0_ts_vals), np.max(h0_ts_vals))

    c = np.percentile(h0_ts_vals, (1 - h0_ts_quantile) * 100)
    logger.debug(
        'Critical ts value for bkg ts quantile %g: %e',
        h0_ts_quantile, c)

    # Make sure ns_range is mutable.
    ns_range_ = list(mu_range)

    ns_lower_bound = 0
    ns_upper_bound = +np.inf

    # The number of required trials per mu point for the desired uncertainty in
    # probability can be estimated via binomial statistics.
    n_trials = int(p*(1-p)/eps_p**2 + 0.5)

    while True:
        logger.debug(
            'Doing new loop for nsignal range %s',
            str(ns_range_))

        ns0 = (ns_range_[1] + ns_range_[0]) / 2

        # Generate statistics (trials) for the current point ns0 as long as
        # the we are only 5sigma away from the desired propability and the
        # uncertainty of the probability is still larger than the desired
        # uncertainty ``eps_p``.
        # Initially generate trials for a 5-times larger uncertainty ``eps_p``
        # to catch ns0 points far away from the desired propability quicker.
        dn_trials = max(100, int(n_trials/5**2 + 0.5))
        (ts_vals0, p0_sigma, delta_p) = ([], 2*eps_p, 0)
        while (delta_p < p0_sigma*5) and (p0_sigma > eps_p):
            ts_vals0 = np.concatenate((
                ts_vals0, ana.do_trials(
                    rss, dn_trials, mean_n_sig=ns0, bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs, ppbar=ppbar)['ts']))
            (p0, p0_sigma) = calculate_pval_from_trials(ts_vals0, c)
            n_total_generated_trials += dn_trials

            delta_p = np.abs(p0 - p)

            logger.debug(
                'n_trials: %d, ns0: %.6f, p0: %.6f, p0_sigma: %.6f, '
                'delta_p: %.6f',
                ts_vals0.size, ns0, p0, p0_sigma, delta_p)

            # After the initial number of trials generated the number of trials
            # to generate, dn_trials, for the next iteration of trial generation
            # to decrease p0_sigma can be set to the number of remaining trials
            # to reach n_trials. But do at least 100 trials more, in case the
            # n_trials estimate was initially too low.
            dn_trials = max(100, n_trials - ts_vals0.size)

            if((p0_sigma < eps_p) and (delta_p < eps_p)):
                # We found the ns0 value that corresponds to the desired
                # probability within the desired uncertainty.

                logger.debug(
                    'Found mu value %g with p value %g within uncertainty +-%g',
                    ns0, p0, p0_sigma)

                if(p0 > p):
                    ns1 = ns_range_[0]
                    if(np.abs(ns0 - ns1) < min_dmu):
                        ns1 = ns0 - min_dmu
                else:
                    ns1 = ns_range_[1]
                    if(np.abs(ns0 - ns1) < min_dmu):
                        ns1 = ns0 + min_dmu

                ts_vals1 = ana.do_trials(
                    rss, ts_vals0.size, mean_n_sig=ns1, bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs, ppbar=ppbar)['ts']
                n_total_generated_trials += ts_vals0.size

                (p1, p1_sigma) = calculate_pval_from_trials(ts_vals1, c)
                logger.debug(
                    'Final mu value is supposed to be within mu range (%g,%g) '
                    'corresponding to p=(%g +-%g, %g +-%g)',
                    ns0, ns1, p0, p0_sigma, p1, p1_sigma)

                # Check if p1 and p0 are equal, which would result in a divison
                # by zero.
                if(p0 == p1):
                    mu = 0.5*(ns0 + ns1)
                    mu_err = 0.5*np.abs(ns1 - ns0)

                    logger.debug(
                        'Probability for mu=%g and mu=%g has the same value %g',
                        ns0, ns1, p0)
                else:
                    dns_dp = np.abs((ns1 - ns0) / (p1 - p0))

                    logger.debug(
                        'Estimated |dmu/dp| = %g within mu range (%g,%g) '
                        'corresponding to p=(%g +-%g, %g +-%g)',
                        dns_dp, ns0, ns1, p0, p0_sigma, p1, p1_sigma)

                    if(p0 > p):
                        mu = ns0 - dns_dp * delta_p
                    else:
                        mu = ns0 + dns_dp * delta_p
                    mu_err = dns_dp * delta_p

                logger.debug(
                    'Estimated final mu to be %g +- %g',
                    mu, mu_err)

                logger.debug(
                    'Generated %d trials in total',
                    n_total_generated_trials)

                return (mu, mu_err)

        if(delta_p < p0_sigma*5):
            # The desired probability is within the 5 sigma region of the
            # current probability. So we use a linear approximation to find the
            # next ns range.
            # For the current ns0 the uncertainty of p0 is smaller than the
            # required uncertainty, hence p0_sigma <= eps_p.

            # Store ns0 for the new lower or upper bound depending on where the
            # p0 lies.
            if(p0+p0_sigma+eps_p <= p):
                ns_lower_bound = ns0
            elif(p0-p0_sigma-eps_p >= p):
                ns_upper_bound = ns0

            ns1 = ns0 * (1 - np.sign(p0 - p) * 0.05)
            if(np.abs(ns0 - ns1) < min_dmu):
                if((p0 - p) < 0):
                    ns1 = ns0 + min_dmu
                else:
                    ns1 = ns0 - min_dmu

            logger.debug(
                'Do interpolation between ns=(%.3f, %.3f)',
                ns0, ns1)

            ts_vals1 = ana.do_trials(
                rss, ts_vals0.size, mean_n_sig=ns1, bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs, ppbar=ppbar)['ts']
            n_total_generated_trials += ts_vals0.size

            (p1, p1_sigma) = calculate_pval_from_trials(ts_vals1, c)

            # Check if p0 and p1 are equal, which would result into a division
            # by zero.
            if(p0 == p1):
                dp = 0.5*(p0_sigma + p1_sigma)
                logger.debug(
                    'p1 and p0 are equal to %g, causing division by zero. '
                    'p0_sigma=%g, p1_sigma=%g. Calculating dns/dp with dp=%g.',
                    p0, p0_sigma, p1_sigma, dp)
                dns_dp = np.abs((ns1 - ns0) / dp)
            else:
                dns_dp = np.abs((ns1 - ns0) / (p1 - p0))
                # p0 and p1 might be very similar, resulting into a numerically
                # infitite slope.
                if(np.isinf(dns_dp)):
                    dp = 0.5*(p0_sigma + p1_sigma)
                    logger.debug(
                        'Infinite dns/dp dedected: ns0=%g, ns1=%g, p0=%g, '
                        'p0_sigma=%g, p1=%g, p1_sigma=%g. Recalculating dns/dp '
                        'with deviation %g.',
                        ns0, ns1, p0, p0_sigma, p1, p1_sigma, dp)
                    dns_dp = np.abs((ns1 - ns0) / dp)
            logger.debug('dns/dp = %g', dns_dp)

            if(p0 > p):
                ns_range_[0] = ns0 - dns_dp * (delta_p + p0_sigma)
                ns_range_[1] = ns0 + dns_dp * p0_sigma
            else:
                ns_range_[0] = ns0 - dns_dp * p0_sigma
                ns_range_[1] = ns0 + dns_dp * (delta_p + p0_sigma)

            # Restrict the range to ns values we already know well.
            ns_range_[0] = np.max((ns_range_[0], ns_lower_bound))
            ns_range_[1] = np.min((ns_range_[1], ns_upper_bound))

            # In case the new calculated mu range is smaller than the minimum
            # delta mu, the mu range gets widened by half of the minimum delta
            # mu on both sides.
            if(np.abs(ns_range_[1] - ns_range_[0]) < min_dmu):
                ns_range_[0] -= 0.5*min_dmu
                ns_range_[1] += 0.5*min_dmu
        else:
            # The current ns corresponds to a probability p0 that is at least
            # 5 sigma away from the desired probability p, hence
            # delta_p >= p0_sigma*5.
            if(p0 < p):
                ns_range_[0] = ns0
            else:
                ns_range_[1] = ns0

            if(np.abs(ns_range_[1] - ns_range_[0]) < min_dmu):
                # The mu range became smaller than the minimum delta mu and
                # still beeing far away from the desired probability.
                # So move the mu range towards the desired probability.
                if(p0 < p):
                    ns_range_[1] += 10*min_dmu
                else:
                    ns_range_[0] -= 10*min_dmu


def estimate_sensitivity(
        ana, rss, h0_ts_vals=None, h0_ts_quantile=0.5, p=0.9, eps_p=0.005,
        mu_range=None, min_dmu=0.5, bkg_kwargs=None, sig_kwargs=None,
        ppbar=None):
    """Estimates the mean number of signal events that whould have to be
    injected into the data such that the test-statistic value of p*100% of all
    trials are larger than the critical test-statistic value c, which
    corresponds to the test-statistic value where h0_ts_quantile*100% of all
    null hypothesis test-statistic values are larger than c.

    For sensitivity h0_ts_quantile, and p are usually set to 0.5, and 0.9,
    respectively.

    Parameters
    ----------
    ana : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    h0_ts_vals : (n_h0_ts_vals,)-shaped 1D ndarray | None
        The 1D ndarray holding the test-statistic values for the
        null-hypothesis. If set to `None`, the number of trials is calculated
        from binomial statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
        where `eps` is `min(5e-3, h0_ts_quantile/10)`.
    h0_ts_quantile : float, optional
        Null-hypothesis test statistic quantile that defines the critical value.
    p : float, optional
        Desired probability of the signal test statistic value to exceed
        the null-hypothesis test statistic value threshold, which is defined
        through the `h0_ts_quantile` value.
    eps_p : float, optional
        Precision in `p` for execution to break.
    mu_range : 2-element sequence | None
        Range to search for the mean number of signal events.
        If set to None, the range (0, 10) will be used.
    min_dmu : float
        The minimum delta mu to use for calculating the derivative dmu/dp.
        The default is ``0.5``.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the mean number of signal events, mu.
    ppbar : instance of ProgressBar | None
        The possible parent ProgressBar instance.

    Returns
    -------
    mu : float
        Estimated median number of signal events to reach desired sensitivity.
    mu_err : float
        The uncertainty of the estimated mean number of signal events.
    """
    if(mu_range is None):
        mu_range = (0, 10)

    (mu, mu_err) = estimate_mean_nsignal_for_ts_quantile(
        ana=ana,
        rss=rss,
        h0_ts_vals=h0_ts_vals,
        h0_ts_quantile=h0_ts_quantile,
        p=p,
        eps_p=eps_p,
        mu_range=mu_range,
        min_dmu=min_dmu,
        bkg_kwargs=bkg_kwargs,
        sig_kwargs=sig_kwargs,
        ppbar=ppbar)

    return (mu, mu_err)


def estimate_discovery_potential(
        ana, rss, h0_ts_vals=None, h0_ts_quantile=5.733e-7, p=0.5, eps_p=0.005,
        mu_range=None, min_dmu=0.5, bkg_kwargs=None, sig_kwargs=None,
        ppbar=None):
    """Estimates the mean number of signal events that whould have to be
    injected into the data such that the test-statistic value of p*100% of all
    trials are larger than the critical test-statistic value c, which
    corresponds to the test-statistic value where h0_ts_quantile*100% of all
    null hypothesis test-statistic values are larger than c.

    For the 5 sigma discovery potential `h0_ts_quantile`, and `p` are usually
    set to 5.733e-7, and 0.5, respectively.

    Parameters
    ----------
    ana : Analysis
        The Analysis instance to use for discovery potential estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    h0_ts_vals : (n_h0_ts_vals,)-shaped 1D ndarray | None
        The 1D ndarray holding the test-statistic values for the
        null-hypothesis. If set to `None`, the number of trials is calculated
        from binomial statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
        where `eps` is `min(5e-3, h0_ts_quantile/10)`.
    h0_ts_quantile : float, optional
        Null-hypothesis test statistic quantile that defines the critical value.
    p : float, optional
        Desired probability of the signal test statistic value to exceed the
        critical value.
    eps_p : float, optional
        Precision in `p` for execution to break.
    mu_range : 2-element sequence | None
        Range to search for the mean number of signal events.
        If set to None, the range (0, 10) will be used.
    min_dmu : float
        The minimum delta mu to use for calculating the derivative dmu/dp.
        The default is ``0.5``.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the mean number of signal events, mu.
    ppbar : instance of ProgressBar | None
        The possible parent ProgressBar instance.

    Returns
    -------
    mu : float
        Estimated mean number of injected signal events to reach the desired
        discovery potential.
    mu_err : float
        Estimated error of `mu`.
    """
    if(mu_range is None):
        mu_range = (0, 10)

    (mu, mu_err) = estimate_mean_nsignal_for_ts_quantile(
        ana=ana,
        rss=rss,
        h0_ts_vals=h0_ts_vals,
        h0_ts_quantile=h0_ts_quantile,
        p=p,
        eps_p=eps_p,
        mu_range=mu_range,
        bkg_kwargs=bkg_kwargs,
        sig_kwargs=sig_kwargs,
        ppbar=ppbar)

    return (mu, mu_err)


def generate_mu_of_p_spline_interpolation(
        ana, rss, h0_ts_vals, h0_ts_quantile, eps_p, mu_range, mu_step,
        kind='cubic', bkg_kwargs=None, sig_kwargs=None, ppbar=None):
    """Generates a spline interpolation for mu(p) function for a pre-defined
    range of mu, where mu is the mean number of injected signal events and p the
    probability for the ts value larger than the ts value corresponding to the
    given quantile, h0_ts_quantile, of the null hypothesis test-statistic
    distribution.

    Parameters
    ----------
    ana : instance of Analysis
        The Analysis instance to use for the calculation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random numbers.
    h0_ts_vals : (n_h0_ts_vals,)-shaped 1D ndarray | None
        The 1D ndarray holding the test-statistic values for the
        null-hypothesis. If set to `None`, 100/(1-h0_ts_quantile)
        null-hypothesis trials will be generated.
    h0_ts_quantile : float
        Null-hypothesis test statistic quantile, which should be exceeded by
        the alternative hypothesis ts value.
    eps_p : float
        The one sigma precision in `p` as stopping condition for the
        calculation for a single mu value.
    mu_range : 2-element sequence
        The range (lower,upper) of mean number of injected signal events to
        create the interpolation spline for.
    mu_step : float
        The step size of the mean number of signal events.
    kind : str
        The kind of spline to generate. Possble values are 'linear' and 'cubic'
        (default).
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the mean number of signal events, mu.
    ppbar : instance of ProgressBar | None
        The possible parent ProgressBar instance.

    Returns
    -------
    spline : callable
        The spline function mu(p).
    """
    logger = logging.getLogger(__name__)

    n_total_generated_trials = 0

    if(h0_ts_vals is None):
        n_bkg = int(100/(1 - h0_ts_quantile))
        logger.debug('Generate %d null-hypothesis trials', n_bkg)
        h0_ts_vals = ana.do_trials(
            rss, n_bkg, mean_n_sig=0, bkg_kwargs=bkg_kwargs,
            sig_kwargs=sig_kwargs)['ts']
        n_total_generated_trials += n_bkg

    n_h0_ts_vals = len(h0_ts_vals)
    h0_ts_vals = h0_ts_vals[np.isfinite(h0_ts_vals)]
    logger.debug(
        'Number of trials after finite cut: %d (%g%% of total)',
        len(h0_ts_vals), (len(h0_ts_vals)/n_h0_ts_vals)*100)

    c = np.percentile(h0_ts_vals, (1 - h0_ts_quantile) * 100)
    logger.debug(
        'Critical ts value for bkg ts quantile %g: %g',
        h0_ts_quantile, c)

    n_mu = int((mu_range[1]-mu_range[0])/mu_step) + 1
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_mu)
    p_vals = np.empty_like(mu_vals)

    logger.debug(
        'Generate trials for %d mu values',
        n_mu)

    # Create the progress bar if we are in an interactive session.
    pbar = None
    if(is_interactive_session()):
        pbar = ProgressBar(len(mu_vals), parent=ppbar).start()

    for (idx,mu) in enumerate(mu_vals):
        p = None
        (ts_vals, p_sigma) = ([], 2*eps_p)
        while (p_sigma > eps_p):
            ts_vals = np.concatenate(
                (ts_vals,
                 ana.do_trials(
                     rss, 100, mean_n_sig=mu, bkg_kwargs=bkg_kwargs,
                     sig_kwargs=sig_kwargs, ppbar=pbar)['ts']))
            (p, p_sigma) = calculate_pval_from_trials(ts_vals, c)
            n_total_generated_trials += 100
        logger.debug(
            'mu: %g, n_trials: %d, p: %g, p_sigma: %g',
            mu, ts_vals.size, p, p_sigma)
        p_vals[idx] = p

        if(pbar is not None):
            pbar.increment()

    # Make a mu(p) spline via interp1d.
    # The interp1d function requires unique x values. So we need to sort the
    # p_vals in increasing order and mask out repeating p values.
    p_mu_vals = np.array(sorted(zip(p_vals, mu_vals)), dtype=np.float)
    p_vals = p_mu_vals[:,0]
    unique_pval_mask = np.concatenate(([True], np.invert(
        p_vals[1:] <= p_vals[:-1])))
    p_vals = p_vals[unique_pval_mask]
    mu_vals = p_mu_vals[:,1][unique_pval_mask]

    spline = interp1d(p_vals, mu_vals, kind=kind, copy=False,
        assume_sorted=True)

    if(pbar is not None):
        pbar.finish()

    return spline


def create_trial_data_file(
        analysis, rss, pathfilename, n_trials, mean_n_sig_min=0,
        mean_n_sig_max=10, mean_n_sig_0_min=0, mean_n_sig_0_max=10,
        mean_n_bkg_list=None, bkg_kwargs=None, sig_kwargs=None,
        ncpu=None, tl=None):
    """Creates and fills a trial data file with `n_trials` generated trials for
    each mean number of injected signal events from `ns_min` up to `ns_max` for
    a given analysis.

    Parameters
    ----------
    analysis : Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    pathfilename : string
        Trial data file path including the filename.
    n_trials : int
        The number of trials to perform for each hypothesis test.
    mean_n_sig_min : int
        The minimum number of mean injected signal events.
    mean_n_sig_max : int
        The maximum number of mean injected signal events.
    mean_n_sig_0_min : int
        The minimum number of fixed mean signal events for the null-hypothesis.
    mean_n_ns_0_max : int
        The maximum number of fixed mean signal events for the null-hypothesis.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`.
    ncpu : int | None
        The number of CPUs to use.
    tl: instance of TimeLord | None
        The instance of TimeLord that should be used to measure individual
        tasks.
    """
    trial_data = None
    for (mean_n_sig, mean_n_sig_0) in itertools.product(
            range(mean_n_sig_min, mean_n_sig_max+1),
            range(mean_n_sig_0_min, mean_n_sig_0_max+1)):

        trials = analysis.do_trials(
            rss, n=n_trials, mean_n_bkg_list=mean_n_bkg_list,
            mean_n_sig=mean_n_sig, mean_n_sig_0=mean_n_sig_0,
            bkg_kwargs=bkg_kwargs, sig_kwargs=sig_kwargs, ncpu=ncpu)

        trials = np_rfn.append_fields(
            trials, 'seed', np.repeat(rss.seed, n_trials))

        if(trial_data is None):
            trial_data = trials
        else:
            trial_data = np_rfn.stack_arrays(
                [trial_data, trials], usemask=False, asrecarray=True)

    if(trial_data is None):
        raise RuntimeError('No trials have been generated! Check your '
            'generation boundaries!')

    # Save the trial data to file.
    np.save(pathfilename, trial_data)


def extend_trial_data_file(
        analysis, rss, pathfilename, ns_max=30, N=1000):
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


def calculate_upper_limit_distribution(
        analysis, rss, pathfilename, N_bkg=5000, n_bins=100):
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
