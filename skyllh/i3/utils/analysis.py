import numpy as np

from skyllh.core.analysis import SingleSourceMultiDatasetLLHRatioAnalysis
from skyllh.core.logging import (
    get_logger,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.utils.analysis import (
    estimate_discovery_potential,
    estimate_sensitivity,
)


def generate_ps_sin_dec_h0_ts_values(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    rss: RandomStateService,
    sin_dec_min: float,
    sin_dec_max: float,
    sin_dec_step: float,
    n_bkg_trials: int = 10000,
    n_iter: int = 1,
    bkg_kwargs: dict | None = None,
    ppbar: ProgressBar | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates sets of null-hypothesis, i.e. background-only trial data
    events, test-statistic values for the given point-source analysis for a
    grid of sin(dec) values.
    These ts values can the be used for the
    ``estimate_ps_sin_dec_sensitivity_curve`` and
    ``estimate_ps_sin_dec_discovery_potential_curve`` functions.

    Parameters
    ----------
    ana
        The instance of Analysis to use for generating trials.
    rss
        The instance of RandomStateService to use for generating random numbers
        from.
    sin_dec_min
        The minimum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_max
        The maximum sin(dec) value for creating the sin(dec) value grid.
    sin_dec_step
        The step size in sin(dec) for creating the sin(dec) value grid.
    n_bkg_trials
        The number of background trials to generate.
        Default is 10000.
    n_iter
        Each set of ts values can be calculated several times to be able to
        estimate the variance of the sensitivity / discovery potential.
        This parameter specifies the number of iterations to perform.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
    bkg_kwargs
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    ppbar
        The optional parent progress bar.

    Returns
    -------
    sin_dec_arr
        The numpy ndarray holding the sin(dec) values for which the ts values
        have been calculated.
    h0_ts_vals_arr
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    """
    logger = get_logger(__name__)

    sin_dec_arr = np.linspace(
        sin_dec_min, sin_dec_max, int((sin_dec_max - sin_dec_min) / sin_dec_step) + 1, endpoint=True
    )

    h0_ts_vals_arr = np.empty((len(sin_dec_arr), n_iter, n_bkg_trials), dtype=np.float64)

    logger.debug(
        'Generating %d null-hypothesis trials for %d sin(dec) values, %d times each, that is %d trials in total',
        n_bkg_trials,
        len(sin_dec_arr),
        n_iter,
        n_bkg_trials * len(sin_dec_arr) * n_iter,
    )

    pbar_iter = ProgressBar(n_iter, parent=ppbar).start()
    for iter_idx in range(n_iter):
        pbar_sin_dec = ProgressBar(len(sin_dec_arr), parent=pbar_iter).start()
        for sin_dec_idx, sin_dec in enumerate(sin_dec_arr):
            source = PointLikeSource(np.pi, np.arcsin(sin_dec))
            ana.change_source(source)

            h0_ts_vals_arr[sin_dec_idx, iter_idx] = ana.do_trials(
                rss, n_bkg_trials, mean_n_sig=0, bkg_kwargs=bkg_kwargs, ppbar=pbar_sin_dec
            )['ts']

            pbar_sin_dec.increment()
        pbar_sin_dec.finish()

        pbar_iter.increment()

        assert rss.seed is not None
        rss.reseed(rss.seed + 1)
    pbar_iter.finish()

    return (sin_dec_arr, h0_ts_vals_arr)


def estimate_ps_sin_dec_sensitivity_curve(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    rss: RandomStateService,
    sin_dec_arr: np.ndarray,
    h0_ts_vals_arr: np.ndarray,
    eps_p: float = 0.0075,
    mu_min: float = 0,
    mu_max: float = 20,
    n_iter: int = 1,
    bkg_kwargs: dict | None = None,
    sig_kwargs: dict | None = None,
    ppbar: ProgressBar | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the point-source sensitivity of the given analysis as a
    function of sin(dec). This function places a PointLikeSource source on the
    given declination values and estimates its sensitivity.

    Parameters
    ----------
    ana
        The instance of Analysis to use for generating trials.
    rss
        The instance of RandomStateService to use for generating random numbers
        from.
    sin_dec_arr
        The ndarray holding the sin(dec) values for which to estimate the
        point-source sensitivity.
    h0_ts_vals_arr
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    eps_p
        The precision in probability used for the `estimate_sensitivity`
        function.
        Default is 0.0075.
    mu_min
        The minimum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 0.
    mu_max
        The maximum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 20.
    n_iter
        Each sensitivity can be estimated several times to be able to estimate
        the variance of the sensitivity. This parameter specifies the number of
        iterations to perform for each sensitivity.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
    bkg_kwargs
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the given mean number of signal events.
    ppbar
        The optional parent progress bar.

    Additional Keyword Arguments
    ----------------------------
    Any additional keyword arguments are passed to the ``estimate_sensitivity``
    function.

    Returns
    -------
    sin_dec_arr
        The ndarray holding the sin(dec) values for which the sensitivities have
        been estimated.
    mean_ns_arr
        The ndarray holding the mean number of signal events corresponding to
        the sensitivity for each sin(dec) value and iteration.
    mean_ns_err_arr
        The ndarray holding the estimated error in `mean_ns` for each sin(dec)
        value and iteration.
    flux_scaling_arr
        The ndarray holding the scaling factor the reference flux needs to get
        scaled to obtain the flux for the estimated sensitivity.
    """
    logger = get_logger(__name__)

    mu_min_arr = np.repeat(mu_min, len(sin_dec_arr))
    mu_max_arr = np.repeat(mu_max, len(sin_dec_arr))

    mean_ns_arr = np.empty((len(sin_dec_arr), n_iter))
    mean_ns_err_arr = np.empty((len(sin_dec_arr), n_iter))
    flux_scaling_arr = np.empty((len(sin_dec_arr), n_iter))

    pbar_sin_dec = ProgressBar(n_iter, parent=ppbar).start()
    for sin_dec_idx, sin_dec in enumerate(sin_dec_arr):
        logger.debug('Estimate point-source sensitivity for sin(dec) = %g, %d times', sin_dec, n_iter)
        source = PointLikeSource(np.pi, np.arcsin(sin_dec))
        ana.change_source(source)

        pbar_iter = ProgressBar(len(sin_dec_arr), parent=pbar_sin_dec).start()
        for iter_idx in range(n_iter):
            h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

            mu_min = mu_min_arr[sin_dec_idx]
            mu_max = mu_max_arr[sin_dec_idx]

            (mean_ns, mean_ns_err) = estimate_sensitivity(
                ana,
                rss,
                mu_range=(mu_min, mu_max),
                eps_p=eps_p,
                h0_trials=h0_ts_vals,
                bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs,
                ppbar=pbar_iter,
            )

            mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
            mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
            flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor() * mean_ns

            # A new iteration is done, update the mu range using the previous
            # results.
            mu_min_arr = np.mean(mean_ns_arr[:, 0 : iter_idx + 1] * 0.8, axis=1)
            mu_max_arr = np.mean(mean_ns_arr[:, 0 : iter_idx + 1] * 1.2, axis=1)

            assert rss.seed is not None
            rss.reseed(rss.seed + 1)

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
            if np.abs(mean_ns - mean_ns_mean) >= 2 * mean_ns_std:
                logger.debug(
                    'Detected unprecise estimate for first iteration (mu=%g) '
                    'for sin(dec)=%g: (|%g - %g| >= 2*%g). Recalculating ...',
                    mean_ns,
                    sin_dec,
                    mean_ns,
                    mean_ns_mean,
                    mean_ns_std,
                )
                iter_idx = 0

                h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

                mu_min = mu_min_arr[sin_dec_idx]
                mu_max = mu_max_arr[sin_dec_idx]

                (mean_ns, mean_ns_err) = estimate_sensitivity(
                    ana,
                    rss,
                    mu_range=(mu_min, mu_max),
                    eps_p=eps_p,
                    h0_trials=h0_ts_vals,
                    bkg_kwargs=bkg_kwargs,
                    sig_kwargs=sig_kwargs,
                    ppbar=pbar_sin_dec,
                )

                mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
                mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
                flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor() * mean_ns

        pbar_sin_dec.increment()
    pbar_sin_dec.finish()

    return (sin_dec_arr, mean_ns_arr, mean_ns_err_arr, flux_scaling_arr)


def estimate_ps_sin_dec_discovery_potential_curve(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    rss: RandomStateService,
    sin_dec_arr: np.ndarray,
    h0_ts_vals_arr: np.ndarray,
    h0_ts_quantile: float = 2.7e-3,
    eps_p: float = 0.0075,
    mu_min: float = 0,
    mu_max: float = 20,
    n_iter: int = 1,
    bkg_kwargs: dict | None = None,
    sig_kwargs: dict | None = None,
    ppbar: ProgressBar | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the point-source discovery potential of the given analysis as a
    function of sin(dec). This function places a PointLikeSource source on the
    given declination values and estimates its discovery potential.

    Parameters
    ----------
    ana
        The instance of Analysis to use for generating trials.
    rss
        The instance of RandomStateService to use for generating random numbers
        from.
    sin_dec_arr
        The ndarray holding the sin(dec) values for which to estimate the
        point-source sensitivity.
    h0_ts_vals_arr
        The numpy ndarray holding the null-hypothesis ts values for all
        sin(dec) values and iterations.
    h0_ts_quantile
        Null-hypothesis test statistic quantile that defines the critical value.
        For a 5sigma discovery potential that value is 5.733e-7. For a 3sigma
        discovery potential this value is 2.7e-3.
        Default is 2.7e-3.
    eps_p
        The precision in probability used for the `estimate_discovery_potential`
        function.
        Default is 0.0075.
    mu_min
        The minimum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 0.
    mu_max
        The maximum value for the mean number of injected signal events as a
        seed for the mu range in which the sensitivity is located.
        Default is 20.
    n_iter
        Each discovery potential can be estimated several times to be able to
        estimate its variance. This parameter specifies the number of
        iterations to perform for each discovery potential estimation.
        For each iteration the RandomStateService is re-seeded with a seed that
        is incremented by 1.
        Default is 1.
    bkg_kwargs
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`. If `poisson` is set to True, the actual number of
        generated signal events will be drawn from a Poisson distribution
        with the mean number of signal events, mu.
    ppbar
        The optional parent progress bar.

    Additional Keyword Arguments
    ----------------------------
    Any additional keyword arguments are passed to the
    ``estimate_discovery_potential`` function.

    Returns
    -------
    sin_dec_arr
        The ndarray holding the sin(dec) values for which the discovery
        potential have been estimated.
    mean_ns_arr
        The ndarray holding the mean number of signal events corresponding to
        the discovery potential for each sin(dec) value and iteration.
    mean_ns_err_arr
        The ndarray holding the estimated error in `mean_ns` for each sin(dec)
        value and iteration.
    flux_scaling_arr
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
        for sin_dec_idx, sin_dec in enumerate(sin_dec_arr):
            source = PointLikeSource(np.pi, np.arcsin(sin_dec))
            ana.change_source(source)

            h0_ts_vals = h0_ts_vals_arr[sin_dec_idx, iter_idx]

            mu_min = mu_min_arr[sin_dec_idx]
            mu_max = mu_max_arr[sin_dec_idx]

            (mean_ns, mean_ns_err) = estimate_discovery_potential(
                ana,
                rss,
                h0_ts_quantile=h0_ts_quantile,
                mu_range=(mu_min, mu_max),
                eps_p=eps_p,
                h0_trials=h0_ts_vals,
                bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs,
                ppbar=pbar,
                **kwargs,
            )

            mean_ns_arr[sin_dec_idx, iter_idx] = mean_ns
            mean_ns_err_arr[sin_dec_idx, iter_idx] = mean_ns_err
            flux_scaling_arr[sin_dec_idx, iter_idx] = ana.calculate_fluxmodel_scaling_factor() * mean_ns

            pbar.increment()
        pbar.finish()

        # One iteration is done, update the mu range using the previous results.
        mu_min_arr = np.mean(mean_ns_arr[:, 0 : iter_idx + 1] * 0.8, axis=1)
        mu_max_arr = np.mean(mean_ns_arr[:, 0 : iter_idx + 1] * 1.2, axis=1)

        pbar_iter.increment()

        assert rss.seed is not None
        rss.reseed(rss.seed + 1)
    pbar_iter.finish()

    return (sin_dec_arr, mean_ns_arr, mean_ns_err_arr, flux_scaling_arr)
