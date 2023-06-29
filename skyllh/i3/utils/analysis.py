# -*- coding: utf-8 -*-

import logging
import numpy as np

from skyllh.core.utils.analysis import (
    estimate_discovery_potential,
    estimate_sensitivity,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.source_model import (
    PointLikeSource,
)


def generate_ps_sin_dec_h0_ts_values(
        ana, rss, sin_dec_min, sin_dec_max, sin_dec_step, n_bkg_trials=10000,
        n_iter=1, bkg_kwargs=None, ppbar=None):
    """Generates sets of null-hypothesis, i.e. background-only trial data
    events, test-statistic values for the given point-source analysis for a
    grid of sin(dec) values.
    These ts values can the be used for the
    ``estimate_ps_sin_dec_sensitivity_curve`` and
    ``estimate_ps_sin_dec_discovery_potential_curve`` functions.

    Parameters
    ----------
    ana : Analysis instance
        The instance of Analysis to use for generating trials.
    rss : RandomStateService instance
        The instance of RandomStateService to use for generating random numbers
        from.
    sin_dec_min : float
        The minimum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_max : float
        The maximum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_step : float
        The step size in sin(dec) for creating the sin(dec) value grid.
    n_bkg_trials : int, optional
        The number of background trials to generate.
        Default is 10000.
    n_iter : int, optional
        Each set of ts values can be calculated several times to be able to
        estimate the variance of the sensitivity / discovery potential.
        This parameter specifies the number of iterations to perform.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    ppbar : ProgressBar instance | None
        The optional parent progress bar.

    Returns
    -------
    sin_dec_arr : (n_sin_dec,)-shaped 1D ndarray of float
        The numpy ndarray holding the sin(dec) values for which the ts values
        have been calculated.
    h0_ts_vals_arr : (n_sin_dec,n_iter,n_bkg_trials)-shaped 3D ndarray of float
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    """
    logger = logging.getLogger(__name__)

    sin_dec_arr = np.linspace(
        sin_dec_min, sin_dec_max,
        int((sin_dec_max-sin_dec_min)/sin_dec_step)+1, endpoint=True)

    h0_ts_vals_arr = np.empty(
        (len(sin_dec_arr), n_iter, n_bkg_trials), dtype=np.float64)

    logger.debug(
        'Generating %d null-hypothesis trials for %d sin(dec) values, '
        '%d times each, that is %d trials in total',
        n_bkg_trials, len(sin_dec_arr), n_iter,
        n_bkg_trials*len(sin_dec_arr)*n_iter)

    pbar_iter = ProgressBar(n_iter, parent=ppbar).start()
    for iter_idx in range(n_iter):
        pbar_sin_dec = ProgressBar(len(sin_dec_arr), parent=pbar_iter).start()
        for (sin_dec_idx, sin_dec) in enumerate(sin_dec_arr):
            source = PointLikeSource(np.pi, np.arcsin(sin_dec))
            ana.change_source(source)

            h0_ts_vals_arr[sin_dec_idx, iter_idx] = ana.do_trials(
                rss, n_bkg_trials, mean_n_sig=0, bkg_kwargs=bkg_kwargs,
                ppbar=pbar_sin_dec)['ts']

            pbar_sin_dec.increment()
        pbar_sin_dec.finish()

        pbar_iter.increment()

        rss.reseed(rss.seed+1)
    pbar_iter.finish()

    return (sin_dec_arr, h0_ts_vals_arr)


def estimate_ps_sin_dec_sensitivity_curve(
        ana, rss, fitparam_values, sin_dec_arr, h0_ts_vals_arr,
        eps_p=0.0075, mu_min=0, mu_max=20, n_iter=1,
        bkg_kwargs=None, sig_kwargs=None, ppbar=None, **kwargs):
    """Estimates the point-source sensitivity of the given analysis as a
    function of sin(dec). This function places a PointLikeSource source on the
    given declination values and estimates its sensitivity.

    Parameters
    ----------
    ana : Analysis instance
        The instance of Analysis to use for generating trials.
    rss : RandomStateService instance
        The instance of RandomStateService to use for generating random numbers
        from.
    fitparam_values : array_like
        The fit parameter values for which to convert the mean number of signal
        events into a flux.
    sin_dec_arr : (n_sin_dec,)-shaped 1D ndarray
        The ndarray holding the sin(dec) values for which to estimate the
        point-source sensitivity.
    h0_ts_vals_arr : (n_sin_dec,n_iter,n_bkg_trials)-shaped 3D ndarray of float
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    eps_p : float, optional
        The precision in probability used for the `estimate_sensitivity`
        function.
        Default is 0.0075.
    mu_min : float, optional
        The minimum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 0.
    mu_max : float, optional
        The maximum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 20.
    n_iter : int, optional
        Each sensitivity can be estimated several times to be able to estimate
        the variance of the sensitivity. This parameter specifies the number of
        iterations to perform for each sensitivity.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the given mean number of signal events.
    ppbar : ProgressBar instance | None
        The optional parent progress bar.

    Additional Keyword Arguments
    ----------------------------
    Any additional keyword arguments are passed to the ``estimate_sensitivity``
    function.

    Returns
    -------
    sin_dec_arr : (n_sin_dec,)-shaped 1D ndarray
        The ndarray holding the sin(dec) values for which the sensitivities have
        been estimated.
    mean_ns_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the mean number of signal events corresponding to
        the sensitivity for each sin(dec) value and iteration.
    mean_ns_err_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the estimated error in `mean_ns` for each sin(dec)
        value and iteration.
    flux_scaling_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the scaling factor the reference flux needs to get
        scaled to obtain the flux for the estimated sensitivity.
    """
    logger = logging.getLogger(__name__)

    mu_min_arr = np.repeat(mu_min, len(sin_dec_arr))
    mu_max_arr = np.repeat(mu_max, len(sin_dec_arr))

    mean_ns_arr = np.empty((len(sin_dec_arr), n_iter))
    mean_ns_err_arr = np.empty((len(sin_dec_arr), n_iter))
    flux_scaling_arr = np.empty((len(sin_dec_arr), n_iter))

    pbar_sin_dec = ProgressBar(n_iter, parent=ppbar).start()
    for (sin_dec_idx, sin_dec) in enumerate(sin_dec_arr):
        logger.debug(
            'Estimate point-source sensitivity for sin(dec) = %g, %d times',
            sin_dec, n_iter)
        source = PointLikeSource(np.pi, np.arcsin(sin_dec))
        ana.change_source(source)

        pbar_iter = ProgressBar(len(sin_dec_arr), parent=pbar_sin_dec).start()
        for iter_idx in range(n_iter):
            h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

            mu_min = mu_min_arr[sin_dec_idx]
            mu_max = mu_max_arr[sin_dec_idx]

            (mean_ns, mean_ns_err) = estimate_sensitivity(
                ana, rss, mu_range=(mu_min, mu_max), eps_p=eps_p,
                h0_ts_vals=h0_ts_vals, bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs, ppbar=pbar_iter)

            mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
            mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
            flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor(
                mean_ns=mean_ns, fitparam_values=np.array(fitparam_values))

            # A new iteration is done, update the mu range using the previous
            # results.
            mu_min_arr = np.mean(mean_ns_arr[:, 0:iter_idx+1]*0.8, axis=1)
            mu_max_arr = np.mean(mean_ns_arr[:, 0:iter_idx+1]*1.2, axis=1)

            rss.reseed(rss.seed+1)

            pbar_iter.increment()
        pbar_iter.finish()

        # It could happen that the first estimation wasn't very accurate due to
        # the unknown seed range for mu. We check for that by calculating the
        # variance of the further iterations and check if the first estimation
        # deviates more than 2 times that variance. If so, recalculate the first
        # estimation.
        if n_iter >= 5:
            mean_ns = mean_ns_arr[sin_dec_idx, 0]
            mean_ns_mean = np.mean(mean_ns_arr[sin_dec_idx, 1:])
            mean_ns_std = np.std(mean_ns_arr[sin_dec_idx, 1:])
            if np.abs(mean_ns - mean_ns_mean) >= 2*mean_ns_std:
                logger.debug(
                    'Detected unprecise estimate for first iteration (mu=%g) '
                    'for sin(dec)=%g: (|%g - %g| >= 2*%g). Recalculating ...',
                    mean_ns, sin_dec, mean_ns, mean_ns_mean, mean_ns_std)
                iter_idx = 0

                h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

                mu_min = mu_min_arr[sin_dec_idx]
                mu_max = mu_max_arr[sin_dec_idx]

                (mean_ns, mean_ns_err) = estimate_sensitivity(
                    ana, rss, mu_range=(mu_min, mu_max), eps_p=eps_p,
                    h0_ts_vals=h0_ts_vals, bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs, ppbar=pbar_sin_dec)

                mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
                mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
                flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor(
                    mean_ns=mean_ns, fitparam_values=np.array(fitparam_values))

        pbar_sin_dec.increment()
    pbar_sin_dec.finish()

    return (sin_dec_arr, mean_ns_arr, mean_ns_err_arr, flux_scaling_arr)


def estimate_ps_sin_dec_discovery_potential_curve(
        ana, rss, fitparam_values, sin_dec_arr, h0_ts_vals_arr,
        h0_ts_quantile=2.7e-3, eps_p=0.0075, mu_min=0, mu_max=20, n_iter=1,
        bkg_kwargs=None, sig_kwargs=None, ppbar=None, **kwargs):
    """Estimates the point-source discovery potential of the given analysis as a
    function of sin(dec). This function places a PointLikeSource source on the
    given declination values and estimates its discovery potential.

    Parameters
    ----------
    ana : Analysis instance
        The instance of Analysis to use for generating trials.
    rss : RandomStateService instance
        The instance of RandomStateService to use for generating random numbers
        from.
    fitparam_values : array_like
        The fit parameter values for which to convert the mean number of signal
        events into a flux.
    sin_dec_arr : (n_sin_dec,)-shaped 1D ndarray
        The ndarray holding the sin(dec) values for which to estimate the
        point-source sensitivity.
    h0_ts_vals_arr : (n_sin_dec,n_iter,n_bkg_trials)-shaped 3D ndarray of float
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    h0_ts_quantile : float, optional
        Null-hypothesis test statistic quantile that defines the critical value.
        For a 5sigma discovery potential that value is 5.733e-7. For a 3sigma
        discovery potential this value is 2.7e-3.
        Default is 2.7e-3.
    eps_p : float, optional
        The precision in probability used for the `estimate_discovery_potential`
        function.
        Default is 0.0075.
    mu_min : float, optional
        The minimum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 0.
    mu_max : float, optional
        The maximum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 20.
    n_iter : int, optional
        Each discovery potential can be estimated several times to be able to
        estimate its variance. This parameter specifies the number of
        iterations to perform for each discovery potential estimation.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
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
    ppbar : ProgressBar instance | None
        The optional parent progress bar.

    Additional Keyword Arguments
    ----------------------------
    Any additional keyword arguments are passed to the
    ``estimate_discovery_potential`` function.

    Returns
    -------
    sin_dec_arr : (n_sin_dec,)-shaped 1D ndarray
        The ndarray holding the sin(dec) values for which the discovery
        potential have been estimated.
    mean_ns_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the mean number of signal events corresponding to
        the discovery potential for each sin(dec) value and iteration.
    mean_ns_err_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the estimated error in `mean_ns` for each sin(dec)
        value and iteration.
    flux_scaling_arr : (n_sin_dec,n_iter)-shaped 2D ndarray
        The ndarray holding the scaling factor the reference flux needs to get
        scaled to obtain the flux for the estimated discovery potential.
    """
    mu_min_arr = np.repeat(mu_min, len(sin_dec_arr))
    mu_max_arr = np.repeat(mu_max, len(sin_dec_arr))

    mean_ns_arr = np.empty((len(sin_dec_arr), n_iter))
    mean_ns_err_arr = np.empty((len(sin_dec_arr), n_iter))
    flux_scaling_arr = np.empty((len(sin_dec_arr), n_iter))

    pbar_iter = ProgressBar(n_iter, parent=ppbar).start()
    for iter_idx in range(n_iter):
        pbar = ProgressBar(len(sin_dec_arr), parent=pbar_iter).start()
        for (sin_dec_idx, sin_dec) in enumerate(sin_dec_arr):
            source = PointLikeSource(np.pi, np.arcsin(sin_dec))
            ana.change_source(source)

            h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

            mu_min = mu_min_arr[sin_dec_idx]
            mu_max = mu_max_arr[sin_dec_idx]

            (mean_ns, mean_ns_err) = estimate_discovery_potential(
                ana, rss, h0_ts_quantile=h0_ts_quantile,
                mu_range=(mu_min, mu_max), eps_p=eps_p,
                h0_ts_vals=h0_ts_vals, bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs, ppbar=pbar, **kwargs)

            mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
            mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
            flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor(
                mean_ns=mean_ns, fitparam_values=np.array(fitparam_values))

            pbar.increment()
        pbar.finish()

        # One iteration is done, update the mu range using the previous results.
        mu_min_arr = np.mean(mean_ns_arr[:, 0:iter_idx+1]*0.8, axis=1)
        mu_max_arr = np.mean(mean_ns_arr[:, 0:iter_idx+1]*1.2, axis=1)

        pbar_iter.increment()

        rss.reseed(rss.seed+1)
    pbar_iter.finish()

    return (sin_dec_arr, mean_ns_arr, mean_ns_err_arr, flux_scaling_arr)
