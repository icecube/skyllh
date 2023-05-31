# -*- coding: utf-8 -*-

import itertools
import logging
import numpy as np
from numpy.lib import (
    recfunctions as np_rfn,
)
from os import (
    makedirs,
)
import os.path
from scipy.interpolate import (
    interp1d,
)
from scipy.stats import (
    gamma,
)

try:
    from iminuit import minimize
except Exception:
    IMINUIT_LOADED = False
else:
    IMINUIT_LOADED = True

from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    float_cast,
    int_cast,
    issequence,
    issequenceof,
)
from skyllh.core.session import (
    is_interactive_session,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.storage import (
    NPYFileLoader,
)
from skyllh.core.utils.spline import (
    make_spline_1d,
)


"""This module contains common utility functions useful for an analysis.
"""


def pointlikesource_to_data_field_array(
        tdm, shg_mgr, pmm):
    """Function to transform a list of PointLikeSource sources into a numpy
    record ndarray. The resulting numpy record ndarray contains the following
    fields:

        `ra`: float
            The right-ascention of the point-like source.
        `dec`: float
            The declination of the point-like source.
        `weight`: float
            The weight of the point-like source.

    Parameters
    ----------
    tdm : instance of TrialDataManager
        The TrialDataManager instance.
    shg_mgr : instance of SourceHypoGroupManager
        The instance of SourceHypoGroupManager that defines the sources.
    pmm : instance of ParameterModelMapper
        The instance of ParameterModelMapper that defines the mapping of global
        parameters to local model parameters.

    Returns
    -------
    arr : (N_sources,)-shaped numpy record ndarray
        The numpy record ndarray holding the source parameters.
    """
    sources = shg_mgr.source_list

    if not issequenceof(sources, PointLikeSource):
        raise TypeError(
            'The sources of the SourceHypoGroupManager must be '
            'PointLikeSource instances!')

    arr = np.empty(
        (len(sources),),
        dtype=[
            ('ra', np.float64),
            ('dec', np.float64),
            ('weight', np.float64),
        ],
        order='F')

    for (i, src) in enumerate(sources):
        arr['ra'][i] = src.ra
        arr['dec'][i] = src.dec
        arr['weight'][i] = src.weight

    return arr


def calculate_pval_from_trials(
        ts_vals, ts_threshold, comp_operator='greater'):
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
    comp_operator: string, optional
        The comparison operator for p-value calculation. It can be set to one of
        the following options: 'greater' or 'greater_equal'.

    Returns
    -------
    p, p_sigma: tuple(float, float)
    """
    if comp_operator == 'greater':
        p = ts_vals[ts_vals > ts_threshold].size / ts_vals.size
    elif comp_operator == 'greater_equal':
        p = ts_vals[ts_vals >= ts_threshold].size / ts_vals.size
    else:
        raise ValueError(
            f"The comp_operator={comp_operator} is not an"
            "available option ('greater' or 'greater_equal')."
        )

    p_sigma = np.sqrt(p * (1 - p) / ts_vals.size)

    return (p, p_sigma)


def calculate_pval_from_gammafit_to_trials(
        ts_vals,
        ts_threshold,
        eta=3.0,
        n_max=500000):
    """Calculates the probability (p-value) of test-statistic exceeding
    the given test-statistic threshold. This calculation relies on fitting
    a gamma distribution to a list of ts values.

    Parameters
    ----------
    ts_vals : (n_trials,)-shaped 1D ndarray of float
        The ndarray holding the test-statistic values of the trials.
    ts_threshold : float
        The critical test-statistic value.
    eta : float, optional
        Test-statistic value at which the gamma function is truncated
        from below. Default = 3.0.
    n_max : int, optional
        The maximum number of trials that should be used during
        fitting. Default = 500,000

    Returns
    -------
    p, p_sigma: tuple(float, float)
    """
    if not IMINUIT_LOADED:
        raise ImportError(
            'The iminuit module was not imported! '
            'This module is a requirement for the function '
            '"calculate_pval_from_gammafit_to_trials"!')

    if ts_threshold < eta:
        raise ValueError(
            'ts threshold value = %e, eta = %e. The calculation of the p-value'
            'from the fit is correct only for ts threshold larger than '
            'the truncation threshold eta.',
            ts_threshold, eta)

    if len(ts_vals) > n_max:
        ts_vals = ts_vals[:n_max]

    Ntot = len(ts_vals)
    ts_eta = ts_vals[ts_vals > eta]
    N_prime = len(ts_eta)
    alpha = N_prime/Ntot

    def obj(x):
        return truncated_gamma_logpdf(
            x[0],
            x[1],
            eta=eta,
            ts_above_eta=ts_eta,
            N_above_eta=N_prime)

    x0 = [0.75, 1.8]  # Initial values of function parameters.
    bounds = [[0.1, 10], [0.1, 10]]  # Ranges for the minimization fitter.
    r = minimize(obj, x0, bounds=bounds)
    pars = r.x

    norm = alpha/gamma.sf(eta, a=pars[0], scale=pars[1])
    p = norm * gamma.sf(ts_threshold, a=pars[0], scale=pars[1])

    # a correct calculation of the error in pvalue due to
    # fitting uncertainty remains to be implemented
    # return p_sigma = 0 for now for consistentcy with
    # calculate_pval_from_trials()
    p_sigma = 0.0
    return (p, p_sigma)


def calculate_pval_from_trials_mixed(
        ts_vals,
        ts_threshold,
        switch_at_ts=3.0,
        eta=None,
        n_max=500000,
        comp_operator='greater_equal'):
    """Calculates the probability (p-value) of test-statistic exceeding
    the given test-statistic threshold. This calculation relies on fitting
    a gamma distribution to a list of ts values if ts_threshold is larger than
    switch_at_ts. If ts_threshold is smaller then the pvalue will be taken
    from the trials directly.

    Parameters
    ----------
    ts_vals : (n_trials,)-shaped 1D ndarray of float
        The ndarray holding the test-statistic values of the trials.
    ts_threshold : float
        The critical test-statistic value.
    switch_at_ts : float, optional
        Test-statistic value below which p-value is computed from trials
        directly. For thresholds greater than switch_at_ts the pvalue is
        calculated using a gamma fit.
    eta : float, optional
        Test-statistic value at which the gamma function is truncated
        from below. Default is None.
    n_max : int, optional
        The maximum number of trials that should be used during
        fitting. Default = 500,000
    comp_operator: string, optional
        The comparison operator for p-value calculation. It can be set to one of
        the following options: 'greater' or 'greater_equal'.

    Returns
    -------
    p, p_sigma: tuple(float, float)
    """
    # Set `eta` to `switch_at_ts` as a default.
    # It makes sure that both functions return the same pval at `switch_at_ts`.
    if eta is None:
        eta = switch_at_ts

    if ts_threshold < switch_at_ts:
        return calculate_pval_from_trials(
            ts_vals,
            ts_threshold,
            comp_operator=comp_operator)
    else:
        return calculate_pval_from_gammafit_to_trials(
            ts_vals,
            ts_threshold,
            eta=eta,
            n_max=n_max)


def truncated_gamma_logpdf(
        a,
        scale,
        eta,
        ts_above_eta,
        N_above_eta):
    """Calculates the -log(likelihood) of a sample of random numbers
    generated from a gamma pdf truncated from below at x=eta.

    Parameters
    ----------
    a : float
        Shape parameter.
    scale : float
        Scale parameter.
    eta : float
        Test-statistic value at which the gamma function is truncated
        from below.
    ts_above_eta : (n_trials,)-shaped 1D ndarray
        The ndarray holding the test-statistic values falling in
        the truncated gamma pdf.
    N_above_eta : int
        Number of test-statistic values falling in the truncated
        gamma pdf.

    Returns
    -------
    -logl : float
    """
    c0 = 1. - gamma.cdf(eta, a=a, scale=scale)
    c0 = 1./c0
    logl = N_above_eta*np.log(c0)
    logl += np.sum(
        gamma.logpdf(
            ts_above_eta,
            a=a,
            scale=scale))

    return -logl


def calculate_critical_ts_from_gamma(
        ts,
        h0_ts_quantile,
        eta=3.0):
    """Calculates the critical test-statistic value corresponding
    to h0_ts_quantile by fitting the ts distribution with a truncated
    gamma function.

    Parameters
    ----------
    ts : (n_trials,)-shaped 1D ndarray
        The ndarray holding the test-statistic values of the trials.
    h0_ts_quantile : float
        Null-hypothesis test statistic quantile.
    eta : float, optional
        Test-statistic value at which the gamma function is truncated
        from below.

    Returns
    -------
    critical_ts : float
    """
    if not IMINUIT_LOADED:
        raise ImportError(
            'The iminuit module was not imported! '
            'This module is a requirement of the function '
            '"calculate_critical_ts_from_gamma"!')

    Ntot = len(ts)
    ts_eta = ts[ts > eta]
    N_prime = len(ts_eta)
    alpha = N_prime/Ntot

    def obj(x):
        return truncated_gamma_logpdf(
            x[0],
            x[1],
            eta=eta,
            ts_above_eta=ts_eta,
            N_above_eta=N_prime)

    x0 = [0.75, 1.8]  # Initial values of function parameters.
    bounds = [[0.1, 10], [0.1, 10]]  # Ranges for the minimization fitter.
    r = minimize(obj, x0, bounds=bounds)
    pars = r.x

    norm = alpha/gamma.sf(eta, a=pars[0], scale=pars[1])
    critical_ts = gamma.ppf(1 - 1./norm*h0_ts_quantile, a=pars[0], scale=pars[1])

    if critical_ts < eta:
        raise ValueError(
            'Critical ts value = %e, eta = %e. The calculation of the critical '
            'ts value from the fit is correct only for critical ts larger than '
            'the truncation threshold eta.',
            critical_ts, eta)

    return critical_ts


def polynomial_fit(
        ns,
        p,
        p_weight,
        deg,
        p_thr):
    """Performs a polynomial fit on the p-values of test-statistic trials
    associated to each ns..
    Using the fitted parameters it computes the number of signal events
    correponding to the given p-value critical value.

    Parameters
    ----------
    ns : 1D array_like object
        x-coordinates of the sample.
    p : 1D array_like object
        y-coordinates of the sample.
    p_weight : 1D array_like object
        Weights to apply to the y-coordinates of the sample points. For gaussian
        uncertainties, use 1/sigma.
    deg : int
        Degree of the fitting polynomial function.
    p_thr : float within [0,1]
        The critical p-value.

    Returns
    -------
    ns : float
    """
    (params, cov) = np.polyfit(ns, p, deg, w=p_weight, cov=True)

    # Check if the second order coefficient is positive and eventually
    # change to a polynomial fit of order 1 to avoid to overestimate
    # the mean number of signal events for the chosen ts quantile.
    if deg == 2 and params[0] > 0:
        deg = 1
        (params, cov) = np.polyfit(ns, p, deg, w=p_weight, cov=True)

    if deg == 1:
        (a, b) = (params[0], params[1])
        ns = (p_thr - b)/a
        return ns

    elif deg == 2:
        (a, b, c) = (params[0], params[1], params[2])
        ns = (- b + np.sqrt((b**2)-4*a*(c-p_thr))) / (2*a)
        return ns

    else:
        raise ValueError(
            'deg = %g is not valid. The order of the polynomial function '
            'must be 1 or 2.',
            deg)


def estimate_mean_nsignal_for_ts_quantile(  # noqa: C901
        ana,
        rss,
        p,
        eps_p,
        mu_range,
        critical_ts=None,
        h0_trials=None,
        h0_ts_quantile=None,
        min_dmu=0.5,
        bkg_kwargs=None,
        sig_kwargs=None,
        ppbar=None,
        tl=None,
        pathfilename=None):
    """Calculates the mean number of signal events needed to be injected to
    reach a test statistic distribution with defined properties for the given
    analysis.

    Parameters
    ----------
    ana : Analysis instance
        The Analysis instance to use for the calculation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random numbers.
    p : float
        Desired probability of signal test statistic for exceeding
        `h0_ts_quantile` part of null-hypothesis test statistic threshold.
    eps_p : float
        Precision in `p` as stopping condition for the calculation.
    mu_range : 2-element sequence
        The range of mu (lower,upper) to search for mean number of signal
        events.
    critical_ts : float | None
        The critical test-statistic value that should be overcome by the signal
        distribution. If set to None, the null-hypothesis test-statistic
        distribution will be used to compute the critical TS value.
    h0_trials : (n_h0_trials,)-shaped ndarray | None
        The structured ndarray holding the trials for the null-hypothesis.
        If set to `None`, the number of trials is calculated
        from binomial statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
        where `eps` is `min(5e-3, h0_ts_quantile/10)`.
    h0_ts_quantile : float | None
        Null-hypothesis test statistic quantile.
        If set to None, the critical test-statistic value that should be
        overcome by the signal distribution MUST be given.
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
    tl: instance of TimeLord | None
        The optional TimeLord instance that should be used to collect timing
        information about this function.
    pathfilename: string | None
        Trial data file path including the filename.
        If set to None, generatedtrials won't be saved.

    Returns
    -------
    mu : float
        Estimated mean number of signal events.
    mu_err : None
        Error estimate needs to be implemented.
    """
    logger = logging.getLogger(__name__)

    n_total_generated_trials = 0

    if (critical_ts is None) and (h0_ts_quantile is None):
        raise RuntimeError(
            "Both the critical test-statistic value and the null-hypothesis "
            "test-statistic quantile are set to None. One of the two is "
            "needed to have the critical test-statistic value that defines "
            "the type of test to run.")
    elif critical_ts is None:
        n_trials_max = int(5.e5)
        # Via binomial statistics, calcuate the minimum number of trials
        # needed to get the required precision on the critial TS value.
        eps = min(0.005, h0_ts_quantile/10)
        n_trials_min = int(h0_ts_quantile*(1-h0_ts_quantile)/eps**2 + 0.5)

        # Compute either n_trials_max or n_trials_min trials depending on
        # which one is smaller. If n_trials_max trials are computed, a
        # fit to the ts distribution is performed to get the critial TS.
        n_trials_total = min(n_trials_min, n_trials_max)
        if h0_trials is None:
            h0_ts_vals = ana.do_trials(
                rss=rss,
                n=n_trials_total,
                mean_n_sig=0,
                bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs,
                ppbar=ppbar,
                tl=tl)['ts']

            logger.debug(
                'Generate %d null-hypothesis trials',
                n_trials_total)
            n_total_generated_trials += n_trials_total

            if pathfilename is not None:
                makedirs(os.path.dirname(pathfilename), exist_ok=True)
                np.save(pathfilename, h0_ts_vals)
        else:
            if h0_trials.size < n_trials_total:
                if not ('seed' in h0_trials.dtype.names):
                    logger.debug(
                        'Uploaded trials miss the rss_seed field. '
                        'Will not be possible to extend the trial file '
                        'safely. Uploaded trials will *not* be used.')
                    n_trials = n_trials_total
                    h0_ts_vals = ana.do_trials(
                        rss=rss,
                        n=n_trials,
                        mean_n_sig=0,
                        bkg_kwargs=bkg_kwargs,
                        sig_kwargs=sig_kwargs,
                        ppbar=ppbar,
                        tl=tl)['ts']
                else:
                    n_trials = n_trials_total - h0_trials.size
                    h0_ts_vals = extend_trial_data_file(
                        ana,
                        rss,
                        n_trials,
                        trial_data=h0_trials,
                        mean_n_sig=0,
                        pathfilename=pathfilename)['ts']
                logger.debug(
                    'Generate %d null-hypothesis trials',
                    n_trials)
                n_total_generated_trials += n_trials
            else:
                h0_ts_vals = h0_trials['ts']

        h0_ts_vals = h0_ts_vals[np.isfinite(h0_ts_vals)]
        logger.debug(
            'Number of trials after finite cut: %d',
            len(h0_ts_vals))
        logger.debug(
            'Min / Max h0 TS value: %e / %e',
            np.min(h0_ts_vals), np.max(h0_ts_vals))

        # If the minimum number of trials needed to get the required precision
        # on the critical TS value is smaller then 500k, compute the critical ts
        # value directly from trials; otherwise calculate it from the gamma
        # function fitted to the ts distribution.
        if n_trials_min <= n_trials_max:
            c = np.percentile(h0_ts_vals, (1 - h0_ts_quantile)*100)
        else:
            c = calculate_critical_ts_from_gamma(h0_ts_vals, h0_ts_quantile)
        logger.debug(
            'Critical ts value for bkg ts quantile %g: %e',
            h0_ts_quantile, c)
    elif h0_ts_quantile is None:
        # Make sure that the critical ts is a float.
        if not isinstance(critical_ts, float):
            raise TypeError(
                "The critical test-statistic value must be a float, not "
                f"{type(critical_ts)}!"
            )
        c = critical_ts
        logger.debug(
            'Critical ts value for upper limit: %e',
            c)
    else:
        raise RuntimeError(
            "Both a critical ts value and a null-hypothesis test_statistic "
            "quantile were given. If you want to use your critical_ts "
            "value, set h0_ts_quantile to None; if you want to compute the "
            "critical ts from the background distribution, set critical_ts "
            "to None.")

    # Make sure ns_range is mutable.
    ns_range_ = list(mu_range)

    ns_lower_bound = 0
    ns_upper_bound = +np.inf

    # The number of required trials per mu point for the desired uncertainty in
    # probability can be estimated via binomial statistics.
    n_trials = int(p*(1-p)/eps_p**2 + 0.5)

    # Define the range of p-values that will be possible to fit with a
    # polynomial function of order not larger than 2.
    min_fit_p, max_fit_p = p - 0.35, p + 0.35
    if min_fit_p < 0.5:
        min_fit_p = 0.5
    if max_fit_p > 0.985:
        max_fit_p = 0.985

    (n_sig, p_vals, p_val_weights) = ([], [], [])

    while True:
        ns_range_[0] = np.max([ns_range_[0], 0])
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
                    rss=rss,
                    n=dn_trials,
                    mean_n_sig=ns0,
                    bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs,
                    ppbar=ppbar,
                    tl=tl)['ts']))
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

            if (p0_sigma < eps_p) and (delta_p < eps_p):
                # We found the ns0 value that corresponds to the desired
                # probability within the desired uncertainty.

                if (p0 < max_fit_p) and (p0 > min_fit_p):
                    n_sig.append(ns0)
                    p_vals.append(p0)
                    p_val_weights.append(1. / p0_sigma)

                logger.debug(
                    'Found mu value %g with p value %g within uncertainty +-%g',
                    ns0, p0, p0_sigma)

                if p0 > p:
                    ns1 = ns_range_[0]
                    if np.abs(ns0 - ns1) > 1.0:
                        ns1 = ns0 - 1.0
                    if np.abs(ns0 - ns1) < min_dmu:
                        ns1 = ns0 - min_dmu
                else:
                    ns1 = ns_range_[1]
                    if np.abs(ns0 - ns1) > 1.0:
                        ns1 = ns0 + 1.0
                    if np.abs(ns0 - ns1) < min_dmu:
                        ns1 = ns0 + min_dmu

                ts_vals1 = ana.do_trials(
                    rss=rss,
                    n=ts_vals0.size,
                    mean_n_sig=ns1,
                    bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs,
                    ppbar=ppbar,
                    tl=tl)['ts']
                n_total_generated_trials += ts_vals0.size

                (p1, p1_sigma) = calculate_pval_from_trials(ts_vals1, c)
                logger.debug(
                    'Final mu value is supposed to be within mu range (%g,%g) '
                    'corresponding to p=(%g +-%g, %g +-%g)',
                    ns0, ns1, p0, p0_sigma, p1, p1_sigma)

                if (p1 < max_fit_p) and (p1 > min_fit_p):
                    n_sig.append(ns1)
                    p_vals.append(p1)
                    p_val_weights.append(1. / p1_sigma)

                if len(n_sig) > 2:
                    scanned_range = np.max(n_sig) - np.min(n_sig)

                    if (len(n_sig) < 5) or (scanned_range < 1.5):
                        deg = 1
                    else:
                        deg = 2

                    logger.debug(
                        'Scanned mu range: [%g , %g]\nPoints to fit: %g\n '
                        'Using polynomial fit of order %g',
                        np.min(n_sig), np.max(n_sig), len(n_sig), deg)

                    mu = polynomial_fit(n_sig, p_vals, p_val_weights, deg, p)
                    mu_err = None

                else:
                    # If the points in the scanned range are only two, we
                    # calculate the final mu value with a linear interpolation
                    # between those points.

                    logger.debug(
                        'Scanned mu range: [%g , %g]\nPoints to fit: %g\n '
                        'Doing a linear interpolation.',
                        np.min(n_sig), np.max(n_sig), len(n_sig))

                    # Check if p1 and p0 are equal, which would result in a
                    # divison by zero.
                    if p0 == p1:
                        mu = 0.5*(ns0 + ns1)
                        mu_err = 0.5*np.abs(ns1 - ns0)

                        logger.debug(
                            'Probability for mu=%g and mu=%g has the same '
                            'value %g',
                            ns0, ns1, p0)
                    else:
                        dns_dp = np.abs((ns1 - ns0) / (p1 - p0))

                        logger.debug(
                            'Estimated |dmu/dp| = %g within mu range (%g,%g) '
                            'corresponding to p=(%g +-%g, %g +-%g)',
                            dns_dp, ns0, ns1, p0, p0_sigma, p1, p1_sigma)
                        if p0 > p:
                            mu = ns0 - dns_dp * delta_p
                        else:
                            mu = ns0 + dns_dp * delta_p
                        mu_err = dns_dp * delta_p

                logger.debug(
                    'Estimated final mu to be %g (error estimation to be '
                    'implemented)',
                    mu)

                return (mu, mu_err)

        if delta_p < p0_sigma*5:
            # The desired probability is within the 5 sigma region of the
            # current probability. So we use a linear approximation to find the
            # next ns range.
            # For the current ns0 the uncertainty of p0 is smaller than the
            # required uncertainty, hence p0_sigma <= eps_p.

            # Store ns0 for the new lower or upper bound depending on where the
            # p0 lies.

            if (p0 < max_fit_p) and (p0 > min_fit_p):
                n_sig.append(ns0)
                p_vals.append(p0)
                p_val_weights.append(1. / p0_sigma)

            if p0+p0_sigma+eps_p <= p:
                ns_lower_bound = ns0
            elif p0-p0_sigma-eps_p >= p:
                ns_upper_bound = ns0

            ns1 = ns0 * (1 - np.sign(p0 - p) * 0.05)
            if np.abs(ns0 - ns1) < min_dmu:
                if (p0 - p) < 0:
                    ns1 = ns0 + min_dmu
                else:
                    ns1 = ns0 - min_dmu

            logger.debug(
                'Do interpolation between ns=(%.3f, %.3f)',
                ns0, ns1)

            ts_vals1 = ana.do_trials(
                rss=rss,
                n=ts_vals0.size,
                mean_n_sig=ns1,
                bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs,
                ppbar=ppbar,
                tl=tl)['ts']
            n_total_generated_trials += ts_vals0.size

            (p1, p1_sigma) = calculate_pval_from_trials(ts_vals1, c)

            if (p1 < max_fit_p) and (p1 > min_fit_p):
                n_sig.append(ns1)
                p_vals.append(p1)
                p_val_weights.append(1. / p1_sigma)

            # Check if p0 and p1 are equal, which would result into a division
            # by zero.
            if p0 == p1:
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
                if np.isinf(dns_dp):
                    dp = 0.5*(p0_sigma + p1_sigma)
                    logger.debug(
                        'Infinite dns/dp dedected: ns0=%g, ns1=%g, p0=%g, '
                        'p0_sigma=%g, p1=%g, p1_sigma=%g. Recalculating dns/dp '
                        'with deviation %g.',
                        ns0, ns1, p0, p0_sigma, p1, p1_sigma, dp)
                    dns_dp = np.abs((ns1 - ns0) / dp)
            logger.debug('dns/dp = %g', dns_dp)

            if p0 > p:
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
            if np.abs(ns_range_[1] - ns_range_[0]) < min_dmu:
                ns_range_[0] -= 0.5*min_dmu
                ns_range_[1] += 0.5*min_dmu
        else:
            # The current ns corresponds to a probability p0 that is at least
            # 5 sigma away from the desired probability p, hence
            # delta_p >= p0_sigma*5.

            if (p0 < max_fit_p) and (p0 > min_fit_p):
                n_sig.append(ns0)
                p_vals.append(p0)
                p_val_weights.append(1. / p0_sigma)

            if p0 < p:
                ns_range_[0] = ns0
            else:
                ns_range_[1] = ns0

            if np.abs(ns_range_[1] - ns_range_[0]) < min_dmu:
                # The mu range became smaller than the minimum delta mu and
                # still beeing far away from the desired probability.
                # So move the mu range towards the desired probability.
                if p0 < p:
                    ns_range_[1] += 10*min_dmu
                else:
                    ns_range_[0] -= 10*min_dmu


def estimate_sensitivity(
        ana,
        rss,
        h0_trials=None,
        h0_ts_quantile=0.5,
        p=0.9,
        eps_p=0.005,
        mu_range=None,
        min_dmu=0.5,
        bkg_kwargs=None,
        sig_kwargs=None,
        ppbar=None,
        tl=None,
        pathfilename=None):
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
    h0_trials : (n_h0_ts_vals,)-shaped ndarray | None
        The strutured ndarray holding the trials for the null-hypothesis.
        If set to `None`, the number of trials is calculated from binomial
        statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
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
    tl: instance of TimeLord | None
        The optional TimeLord instance that should be used to collect timing
        information about this function.
    pathfilename : string | None
        Trial data file path including the filename.
        If set to None, generated trials won't be saved.

    Returns
    -------
    mu : float
        Estimated median number of signal events to reach desired sensitivity.
    mu_err : float
        The uncertainty of the estimated mean number of signal events.
    """
    if mu_range is None:
        mu_range = (0, 10)

    (mu, mu_err) = estimate_mean_nsignal_for_ts_quantile(
        ana=ana,
        rss=rss,
        h0_trials=h0_trials,
        h0_ts_quantile=h0_ts_quantile,
        p=p,
        eps_p=eps_p,
        mu_range=mu_range,
        min_dmu=min_dmu,
        bkg_kwargs=bkg_kwargs,
        sig_kwargs=sig_kwargs,
        ppbar=ppbar,
        tl=tl,
        pathfilename=pathfilename)

    return (mu, mu_err)


def estimate_discovery_potential(
        ana,
        rss,
        h0_trials=None,
        h0_ts_quantile=2.8665e-7,
        p=0.5,
        eps_p=0.005,
        mu_range=None,
        min_dmu=0.5,
        bkg_kwargs=None,
        sig_kwargs=None,
        ppbar=None,
        tl=None,
        pathfilename=None):
    """Estimates the mean number of signal events that whould have to be
    injected into the data such that the test-statistic value of p*100% of all
    trials are larger than the critical test-statistic value c, which
    corresponds to the test-statistic value where h0_ts_quantile*100% of all
    null hypothesis test-statistic values are larger than c.

    For the 5 sigma discovery potential `h0_ts_quantile`, and `p` are usually
    set to 2.8665e-7, and 0.5, respectively.

    Parameters
    ----------
    ana : Analysis
        The Analysis instance to use for discovery potential estimation.
    rss : RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    h0_trials : (n_h0_ts_vals,)-shaped ndarray | None
        The structured ndarray holding the trials for the null-hypothesis.
        If set to `None`, the number of trials is calculated from binomial
        statistics via `h0_ts_quantile*(1-h0_ts_quantile)/eps**2`,
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
    tl: instance of TimeLord | None
        The optional TimeLord instance that should be used to collect timing
        information about this function.
    pathfilename : string | None
        Trial data file path including the filename.
        If set to None, generated trials won't be saved.

    Returns
    -------
    mu : float
        Estimated mean number of injected signal events to reach the desired
        discovery potential.
    mu_err : float
        Estimated error of `mu`.
    """
    if mu_range is None:
        mu_range = (0, 10)

    (mu, mu_err) = estimate_mean_nsignal_for_ts_quantile(
        ana=ana,
        rss=rss,
        p=p,
        eps_p=eps_p,
        mu_range=mu_range,
        h0_trials=h0_trials,
        h0_ts_quantile=h0_ts_quantile,
        min_dmu=min_dmu,
        bkg_kwargs=bkg_kwargs,
        sig_kwargs=sig_kwargs,
        ppbar=ppbar,
        tl=tl,
        pathfilename=pathfilename)

    return (mu, mu_err)


def generate_mu_of_p_spline_interpolation(
        ana,
        rss,
        h0_ts_vals,
        h0_ts_quantile,
        eps_p,
        mu_range,
        mu_step,
        kind='cubic',
        bkg_kwargs=None,
        sig_kwargs=None,
        ppbar=None,
        tl=None):
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
    tl: instance of TimeLord | None
        The optional TimeLord instance that should be used to collect timing
        information about this function.

    Returns
    -------
    spline : callable
        The spline function mu(p).
    """
    logger = logging.getLogger(__name__)

    n_total_generated_trials = 0

    if h0_ts_vals is None:
        n_bkg = int(100/(1 - h0_ts_quantile))
        logger.debug('Generate %d null-hypothesis trials', n_bkg)
        h0_ts_vals = ana.do_trials(
            rss=rss,
            n=n_bkg,
            mean_n_sig=0,
            bkg_kwargs=bkg_kwargs,
            sig_kwargs=sig_kwargs,
            ppbar=ppbar,
            tl=tl)['ts']
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
    if is_interactive_session():
        pbar = ProgressBar(len(mu_vals), parent=ppbar).start()

    for (idx, mu) in enumerate(mu_vals):
        p = None
        (ts_vals, p_sigma) = ([], 2*eps_p)
        while (p_sigma > eps_p):
            ts_vals = np.concatenate(
                (ts_vals,
                 ana.do_trials(
                     rss=rss,
                     n=100,
                     mean_n_sig=mu,
                     bkg_kwargs=bkg_kwargs,
                     sig_kwargs=sig_kwargs,
                     ppbar=pbar,
                     tl=tl)['ts']))
            (p, p_sigma) = calculate_pval_from_trials(ts_vals, c)
            n_total_generated_trials += 100
        logger.debug(
            'mu: %g, n_trials: %d, p: %g, p_sigma: %g',
            mu, ts_vals.size, p, p_sigma)
        p_vals[idx] = p

        if pbar is not None:
            pbar.increment()

    # Make a mu(p) spline via interp1d.
    spline = make_spline_1d(
        p_vals,
        mu_vals,
        kind=kind,
        copy=False,
        assume_sorted=True)

    if pbar is not None:
        pbar.finish()

    return spline


def create_trial_data_file(  # noqa: C901
        ana,
        rss,
        n_trials,
        mean_n_sig=0,
        mean_n_sig_null=0,
        mean_n_bkg_list=None,
        bkg_kwargs=None,
        sig_kwargs=None,
        pathfilename=None,
        ncpu=None,
        ppbar=None,
        tl=None):
    """Creates and fills a trial data file with `n_trials` generated trials for
    each mean number of injected signal events from `ns_min` up to `ns_max` for
    a given analysis.

    Parameters
    ----------
    ana : instance of Analysis
        The Analysis instance to use for the trial generation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    n_trials : int
        The number of trials to perform for each hypothesis test.
    mean_n_sig : ndarray of float | float | 2- or 3-element sequence of float
        The array of mean number of injected signal events (MNOISEs) for which
        to generate trials. If this argument is not a ndarray, an array of
        MNOISEs is generated based on this argument.
        If a single float is given, only this given MNOISEs are injected.
        If a 2-element sequence of floats is given, it specifies the range of
        MNOISEs with a step size of one.
        If a 3-element sequence of floats is given, it specifies the range plus
        the step size of the MNOISEs.
    mean_n_sig_null : ndarray of float | float | 2- or 3-element sequence of float
        The array of the fixed mean number of signal events (FMNOSEs) for the
        null-hypothesis for which to generate trials. If this argument is not a
        ndarray, an array of FMNOSEs is generated based on this argument.
        If a single float is given, only this given FMNOSEs are used.
        If a 2-element sequence of floats is given, it specifies the range of
        FMNOSEs with a step size of one.
        If a 3-element sequence of floats is given, it specifies the range plus
        the step size of the FMNOSEs.
    mean_n_bkg_list : list of float | None
        The mean number of background events that should be generated for
        each dataset. This parameter is passed to the ``do_trials`` method of
        the ``Analysis`` class. If set to None (the default), the background
        generation method needs to obtain this number itself.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`.
    pathfilename : string | None
        Trial data file path including the filename.
        If set to None generated trials won't be saved.
    ncpu : int | None
        The number of CPUs to use.
    ppbar : instance of ProgressBar | None
        The optional instance of the parent progress bar.
    tl: instance of TimeLord | None
        The instance of TimeLord that should be used to measure individual
        tasks.

    Returns
    -------
    seed : int
        The seed used to generate the trials.
    mean_n_sig : 1d ndarray
        The array holding the mean number of signal events used to generate the
        trials.
    mean_n_sig_null : 1d ndarray
        The array holding the fixed mean number of signal events for the
        null-hypothesis used to generate the trials.
    trial_data : structured numpy ndarray
        The generated trial data.
    """
    n_trials = int_cast(
        n_trials,
        'The n_trials argument must be castable to type int!')

    if not isinstance(mean_n_sig, np.ndarray):
        if not issequence(mean_n_sig):
            mean_n_sig = float_cast(
                mean_n_sig,
                'The mean_n_sig argument must be castable to type float!')
            mean_n_sig_min = mean_n_sig
            mean_n_sig_max = mean_n_sig
            mean_n_sig_step = 1
        else:
            mean_n_sig = float_cast(
                mean_n_sig,
                'The sequence elements of the mean_n_sig argument must be '
                'castable to float values!')
            if len(mean_n_sig) == 2:
                (mean_n_sig_min, mean_n_sig_max) = mean_n_sig
                mean_n_sig_step = 1
            elif len(mean_n_sig) == 3:
                (mean_n_sig_min, mean_n_sig_max, mean_n_sig_step) = mean_n_sig

        mean_n_sig = np.arange(
            mean_n_sig_min, mean_n_sig_max+1, mean_n_sig_step,
            dtype=np.float64)

    if not isinstance(mean_n_sig_null, np.ndarray):
        if not issequence(mean_n_sig_null):
            mean_n_sig_null = float_cast(
                mean_n_sig_null,
                'The mean_n_sig_null argument must be castable to type float!')
            mean_n_sig_null_min = mean_n_sig_null
            mean_n_sig_null_max = mean_n_sig_null
            mean_n_sig_null_step = 1
        else:
            mean_n_sig_null = float_cast(
                mean_n_sig_null,
                'The sequence elements of the mean_n_sig_null argument must '
                'be castable to float values!')
            if len(mean_n_sig_null) == 2:
                (mean_n_sig_null_min, mean_n_sig_null_max) = mean_n_sig_null
                mean_n_sig_null_step = 1
            elif len(mean_n_sig_null) == 3:
                (mean_n_sig_null_min, mean_n_sig_null_max,
                 mean_n_sig_null_step) = mean_n_sig_null

        mean_n_sig_null = np.arange(
            mean_n_sig_null_min, mean_n_sig_null_max+1, mean_n_sig_null_step,
            dtype=np.float64)

    pbar = ProgressBar(
        len(mean_n_sig)*len(mean_n_sig_null), parent=ppbar).start()
    trial_data = None
    for (mean_n_sig_, mean_n_sig_null_) in itertools.product(
            mean_n_sig, mean_n_sig_null):

        trials = ana.do_trials(
            rss=rss,
            n=n_trials,
            mean_n_bkg_list=mean_n_bkg_list,
            mean_n_sig=mean_n_sig_,
            mean_n_sig_0=mean_n_sig_null_,
            bkg_kwargs=bkg_kwargs,
            sig_kwargs=sig_kwargs,
            ncpu=ncpu,
            tl=tl,
            ppbar=pbar)

        if trial_data is None:
            trial_data = trials
        else:
            trial_data = np_rfn.stack_arrays(
                [trial_data, trials],
                usemask=False,
                asrecarray=True)

        pbar.increment()
    pbar.finish()

    if trial_data is None:
        raise RuntimeError(
            'No trials have been generated! Check your generation boundaries!')

    if pathfilename is not None:
        # Save the trial data to file.
        makedirs(os.path.dirname(pathfilename), exist_ok=True)
        np.save(pathfilename, trial_data)

    return (rss.seed, mean_n_sig, mean_n_sig_null, trial_data)


def extend_trial_data_file(
        ana,
        rss,
        n_trials,
        trial_data,
        mean_n_sig=0,
        mean_n_sig_null=0,
        mean_n_bkg_list=None,
        bkg_kwargs=None,
        sig_kwargs=None,
        pathfilename=None,
        **kwargs):
    """Appends to the trial data file `n_trials` generated trials for each
    mean number of injected signal events up to `ns_max` for a given analysis.

    Parameters
    ----------
    ana : instance of Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    n_trials : int
        The number of trials the trial data file needs to be extended by.
    trial_data : structured numpy ndarray
        The structured numpy ndarray holding the trials.
    mean_n_sig : ndarray of float | float | 2- or 3-element sequence of float
        The array of mean number of injected signal events (MNOISEs) for which
        to generate trials. If this argument is not a ndarray, an array of
        MNOISEs is generated based on this argument.
        If a single float is given, only this given MNOISEs are injected.
        If a 2-element sequence of floats is given, it specifies the range of
        MNOISEs with a step size of one.
        If a 3-element sequence of floats is given, it specifies the range plus
        the step size of the MNOISEs.
    mean_n_sig_null : ndarray of float | float | 2- or 3-element sequence of float
        The array of the fixed mean number of signal events (FMNOSEs) for the
        null-hypothesis for which to generate trials. If this argument is not a
        ndarray, an array of FMNOSEs is generated based on this argument.
        If a single float is given, only this given FMNOSEs are used.
        If a 2-element sequence of floats is given, it specifies the range of
        FMNOSEs with a step size of one.
        If a 3-element sequence of floats is given, it specifies the range plus
        the step size of the FMNOSEs.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`.
    pathfilename : string | None
        Trial data file path including the filename.

    Additional keyword arguments
    ----------------------------
    Additional keyword arguments are passed-on to the ``create_trial_data_file``
    function.

    Returns
    -------
    trial_data :
        Trial data file extended by the required number of trials for each
        mean number of injected signal events..
    """
    # Use unique seed to generate non identical trials.
    if rss.seed in trial_data['seed']:
        seed = next(
            i
            for (i, e) in enumerate(
                sorted(np.unique(trial_data['seed'])) + [None], 1)
            if i != e)
        rss.reseed(seed)

    (seed, mean_n_sig, mean_n_sig_null, trials) = create_trial_data_file(
        ana=ana,
        rss=rss,
        n_trials=n_trials,
        mean_n_sig=mean_n_sig,
        mean_n_sig_null=mean_n_sig_null,
        mean_n_bkg_list=mean_n_bkg_list,
        bkg_kwargs=bkg_kwargs,
        sig_kwargs=sig_kwargs,
        **kwargs
    )
    trial_data = np_rfn.stack_arrays(
        [trial_data, trials],
        usemask=False,
        asrecarray=True)

    if pathfilename is not None:
        # Save the trial data to file.
        makedirs(os.path.dirname(pathfilename), exist_ok=True)
        np.save(pathfilename, trial_data)

    return trial_data


def calculate_upper_limit_distribution(
        ana,
        rss,
        pathfilename,
        n_bkg=5000,
        n_bins=100):
    """Function to calculate upper limit distribution. It loads the trial data
    file containing test statistic distribution and calculates 10 percentile
    value for each mean number of injected signal event. Then it finds upper
    limit values which correspond to generated background trials test statistic
    values by linearly interpolated curve of 10 percentile values distribution.

    Parameters
    ----------
    ana : instance of Analysis
        The Analysis instance to use for sensitivity estimation.
    rss : instance of RandomStateService
        The RandomStateService instance to use for generating random
        numbers.
    pathfilename : string
        Trial data file path including the filename.
    n_bkg : int, optional
        Number of times to perform background analysis trial.
    n_bins : int, optional
        Number of returned test statistic histograms bins.

    Returns
    -------
    result : dict
        Result dictionary which contains the following fields:

        ul : list of float
            List of upper limit values.
        mean : float
            Mean of upper limit values.
        median : float
            Median of upper limit values.
        var : float
            Variance of upper limit values.
        ts_hist : numpy ndarray
            2D array of test statistic histograms calculated by axis 1.
        extent : list of float
            Test statistic histogram boundaries.
        q_values : list of float
            `q` percentile values of test statistic for different injected
            events means.
    """
    # Load trial data file.
    trial_data = NPYFileLoader(pathfilename).load_data()
    ns_max = max(trial_data['sig_mean']) + 1
    ts_bins_range = (min(trial_data['TS']), max(trial_data['TS']))

    q = 10  # Upper limit criterion.
    trial_data_q_values = np.empty((ns_max,))
    trial_data_ts_hist = np.empty((ns_max, n_bins))
    for ns in range(ns_max):
        trial_data_q_values[ns] = np.percentile(
            trial_data['TS'][trial_data['sig_mean'] == ns], q)
        (trial_data_ts_hist[ns, :], bin_edges) = np.histogram(
            trial_data['TS'][trial_data['sig_mean'] == ns],
            bins=n_bins, range=ts_bins_range)

    ts_inv_f = interp1d(trial_data_q_values, range(ns_max), kind='linear')
    ts_bkg = ana.do_trials(
        rss=rss,
        n=n_bkg,
        mean_n_sig=0)['TS']

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
