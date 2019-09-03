# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.analysis_utils import estimate_sensitivity
from skyllh.core.progressbar import ProgressBar
from skyllh.physics.source import PointLikeSource


def estimate_ps_sin_dec_sensitivity_curve(
        ana, rss, fitparam_values, sin_dec_min, sin_dec_max, sin_dec_step,
        n_bkg_trials=10000, eps_p=0.0075, mu_min=0, mu_max=20, n_iter=1,
        bkg_kwargs=None, sig_kwargs=None, ppbar=None):
    """Estimates the point-source sensitivity of the given analysis as a
    function of sin(dec). This function creates a grid in sin(dec) from
    `sin_dec_min` to `sin_dec_max``in steps of `sin_dec_step`, and places a
    PointLikeSource source on that declination.

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
    sin_dec_min : float
        The minimum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_max : float
        The maximum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_step : float
        The step size in sin(dec) for creating the sin(dec) value grid.
    n_bkg_trials : int, optional
        The number of background trials to generate.
        Default is 10000.
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
    sin_dec_arr = np.linspace(
        sin_dec_min, sin_dec_max,
        int((sin_dec_max-sin_dec_min)/sin_dec_step)+1, endpoint=True)

    mu_min_arr = np.repeat(mu_min, len(sin_dec_arr))
    mu_max_arr = np.repeat(mu_max, len(sin_dec_arr))

    mean_ns_arr = np.empty((len(sin_dec_arr), n_iter))
    mean_ns_err_arr = np.empty((len(sin_dec_arr), n_iter))
    flux_scaling_arr = np.empty((len(sin_dec_arr), n_iter))

    pbar_iter = ProgressBar(n_iter, parent=ppbar).start()
    for iter_idx in range(n_iter):
        pbar = ProgressBar(len(sin_dec_arr), parent=pbar_iter).start()
        for (idx,sin_dec) in enumerate(sin_dec_arr):
            source = PointLikeSource(np.pi, np.arcsin(sin_dec))
            ana.change_source(source)

            h0_ts_vals = ana.do_trials(
                rss, n_bkg_trials, mean_n_sig=0, bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs, ppbar=pbar)['ts']

            mu_min = mu_min_arr[idx]
            mu_max = mu_max_arr[idx]

            (mean_ns, mean_ns_err) = estimate_sensitivity(
                ana, rss, mu_range=(mu_min,mu_max), eps_p=eps_p,
                h0_ts_vals=h0_ts_vals, bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs, ppbar=pbar)

            mean_ns_arr[idx,iter_idx] = mean_ns
            mean_ns_err_arr[idx,iter_idx] = mean_ns_err
            flux_scaling_arr[idx,iter_idx] = ana.calculate_fluxmodel_scaling_factor(
                mean_ns=mean_ns, fitparam_values=np.array(fitparam_values))

            pbar.increment()
        pbar.finish()

        # One iteration is done, update the mu range using the previous results.
        mu_min_arr = np.mean(mean_ns_arr[:,0:iter_idx+1]*0.8, axis=1)
        mu_max_arr = np.mean(mean_ns_arr[:,0:iter_idx+1]*1.2, axis=1)

        pbar_iter.increment()

        rss.reseed(rss.seed+1)
    pbar_iter.finish()

    return (sin_dec_arr, mean_ns_arr, mean_ns_err_arr, flux_scaling_arr)
