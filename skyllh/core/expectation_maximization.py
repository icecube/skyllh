import numpy as np
from scipy.stats import norm


def em_expectation_step(
    ns: np.ndarray | list,
    mu: np.ndarray | list,
    sigma: np.ndarray | list,
    t: np.ndarray,
    sob: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Expectation step of expectation maximization algorithm.

    Parameters
    ----------
    ns
        The (n_flares,)-shaped numpy ndarray holding the number of signal
        neutrinos, as weight for each gaussian flare.
    mu
        The (n_flares,)-shaped numpy ndarray holding the mean for each gaussian
        flare.
    sigma
        The (n_flares,)-shaped numpy ndarray holding the sigma for each gaussian
        flare.
    t
        The (n_events,)-shaped numpy ndarray holding the time of each event.
    sob
        The (n_events,)-shaped numpy ndarray holding the signal-over-background
        values of each event.

    Returns
    -------
    expectations
        The (n_flares, n_events)-shaped numpy ndarray holding the expectation
        of each flare and event.
    llh
        The log-likelihood value, which is the sum of log of the signal and
        background expectations.
    """
    n_flares = len(ns)

    b_term = (1 - np.cos(10 / 180 * np.pi)) / 2
    N = len(t)
    e_sig = np.empty((n_flares, N), dtype=np.float64)
    for i in range(n_flares):
        e_sig[i] = norm(loc=mu[i], scale=sigma[i]).pdf(t)  # type: ignore[attr-defined]
        e_sig[i] *= sob
        e_sig[i] *= ns[i]
    e_bkg = (N - np.sum(ns)) / (np.max(t) - np.min(t)) / b_term
    denom = np.sum(e_sig, axis=0) + e_bkg

    expectations = e_sig / denom
    llh = np.sum(np.log(denom))

    return (expectations, llh)


def em_maximization_step(
    e: np.ndarray,
    t: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    """The maximization step of the expectation maximization algorithm.

    Parameters
    ----------
    e
        The (n_flares, n_events)-shaped numpy ndarray holding the expectation
        for each event and flare.
    t
        The times of each event.

    Returns
    -------
    mu
        Best fit mean time of the gaussian flare.
    sigma
        Best fit sigma of the gaussian flare.
    ns
        Best fit number of signal neutrinos, as weight for the gaussian flare.
    """
    mu: list[float] = []
    sigma: list[float] = []
    ns: list[float] = []
    for i in range(e.shape[0]):
        mu.append(float(np.average(t, weights=e[i])))
        sigma.append(float(np.sqrt(np.average(np.square(t - mu[i]), weights=e[i]))))
        ns.append(float(np.sum(e[i])))
    sigma = [max(1, s) for s in sigma]

    return (mu, sigma, ns)


def em_fit(
    x: np.ndarray,
    weights: np.ndarray,
    n: int = 1,
    tol: float = 1.0e-200,
    iter_max: int = 500,
    weight_thresh: float = 0,
    initial_width: float = 5000,
    remove_x: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Perform the expectation maximization fit.

    Parameters
    ----------
    x
        The quantity to run EM on (e.g. the time if EM should find time flares).
    weights
        The weights for each x value (e.g. the signal over background ratio).
    n
        How many Gaussians flares we are looking for.
    tol
        The stopping criteria for the expectation maximization. This is the
        difference in the normalized likelihood over the last 20 iterations.
    iter_max
        The maximum number of iterations, even if stopping criteria tolerance
        (``tol``) is not yet reached.
    weight_thresh
        Set a minimum threshold for event weights. Events with smaller weights
        will be removed.
    initial_width
        The starting width for the gaussian flare in days.
    remove_x
        Specific x of event that should be removed.

    Returns
    -------
    mu
        The list of size ``n`` with the determined mean values.
    sigma
        The list of size ``n`` with the standard deviation values.
    ns
        The list of size ``n`` with the normalization factor values.
    """
    if weight_thresh > 0:
        # Remove events below threshold.
        for i in range(len(weights)):
            mask = weights > weight_thresh
            weights[i] = weights[i][mask]
            x[i] = x[i][mask]

    if remove_x is not None:
        # Remove data point.
        mask = x == remove_x
        weights = weights[~mask]
        x = x[~mask]

    # Do the expectation maximization.
    mu = np.linspace(x[0], x[-1], n + 2)[1:-1]
    sigma = np.full((n,), initial_width)
    ns = np.full((n,), 10)

    llh_diff = 100
    llh_old = 0
    llh_diff_list = [100] * 20

    # Run until convergence or maximum number of iterations is reached.
    iteration = 0
    while (iteration < iter_max) and (llh_diff > tol):
        iteration += 1

        (e, llh_new) = em_expectation_step(ns=ns, mu=mu, sigma=sigma, t=x, sob=weights)

        tmp_diff = np.abs(llh_old - llh_new) / llh_new
        llh_diff_list = llh_diff_list[:-1]
        llh_diff_list.insert(0, tmp_diff)
        llh_diff = np.max(llh_diff_list)

        llh_old = llh_new

        (mu, sigma, ns) = em_maximization_step(e=e, t=x)

    return (list(map(float, mu)), list(map(float, sigma)), list(map(float, ns)))
