import numpy as np
from scipy.stats import norm

from skyllh.core.analysis import TimeIntegratedMultiDatasetSingleSourceAnalysis
from skyllh.core.backgroundpdf import BackgroundUniformTimePDF
from skyllh.core.pdf import TimePDF
from skyllh.core.pdfratio import SigOverBkgPDFRatio
from skyllh.core.random import RandomStateService
from skyllh.core.signalpdf import (
    SignalBoxTimePDF,
    SignalGaussTimePDF,
)


def expectation_em(ns, mu, sigma, t, sob):
    """
    Expectation step of expectation maximization.

    Parameters
    ----------
    ns : float | 1d ndarray of float
        The number of signal neutrinos, as weight for the gaussian flare.
    mu : float | 1d ndarray of float
        The mean time of the gaussian flare.
    sigma: float | 1d ndarray of float
        Sigma of the gaussian flare.
    t : 1d ndarray of float
        Times of the events.
    sob : 1d ndarray of float
        The signal over background values of events, or weights of events

    Returns
    -------
    expectation : list of 1d ndarray of float
        Weighted "responsibility" function of each event to belong to the flare.
    sum_log_denom : float
        Sum of log of denominators.
    """
    ns = np.atleast_1d(ns)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    b_term = (1 - np.cos(10 / 180 * np.pi)) / 2
    N = len(t)
    e_sig = []
    for i in range(len(ns)):
        e_sig.append(norm(loc=mu[i], scale=sigma[i]).pdf(t) * sob * ns[i])
    e_bg = (N - np.sum(ns)) / (np.max(t) - np.min(t)) / b_term
    denom = sum(e_sig) + e_bg

    return [e / denom for e in e_sig], np.sum(np.log(denom))


def maximization_em(e_sig, t):
    """
    Maximization step of expectation maximization.

    Parameters
    ----------
    e_sig : list of 1d ndarray of float
        The weights for each event from the expectation step.
    t : 1d ndarray of float
        The times of each event.

    Returns
    -------
    mu : list of float
        Best fit mean time of the gaussian flare.
    sigma : list of float
        Best fit sigma of the gaussian flare.
    ns : list of float
        Best fit number of signal neutrinos, as weight for the gaussian flare.
    """
    mu = []
    sigma = []
    ns = []
    for i in range(len(e_sig)):
        mu.append(np.average(t, weights=e_sig[i]))
        sigma.append(np.sqrt(np.average(np.square(t - mu[i]), weights=e_sig[i])))
        ns.append(np.sum(e_sig[i]))
    sigma = [max(1, s) for s in sigma]

    return mu, sigma, ns


def em_fit(x, weights, n=1, tol=1.e-200, iter_max=500, weight_thresh=0, initial_width=5000,
            remove_x=None):
    """Run expectation maximization.
    
    Parameters
    ----------
    x : array[float]
        Quantity to run EM on (e.g. the time if EM should find time flares)
    weights :
        weights for each event (e.g. the signal over background ratio)
    fitparams : dict
        Dictionary with value for gamma, e.g. {'gamma': 2}.
    n : int
        How many Gaussians flares we are looking for.
    tol : float
        the stopping criteria for expectation maximization. This is the difference in the normalized likelihood over the
        last 20 iterations.
    iter_max : int
        The maximum number of iterations, even if stopping criteria tolerance (`tol`) is not yet reached.
    weight_thresh : float
        Set a minimum threshold for event weights. Events with smaller weights will be removed.
    initial_width : float
        Starting width for the gaussian flare in days.
    remove_x : float | None
        Specific x of event that should be removed.
    
    Returns
    -------
    Mean, width, normalization factor
    """

    if weight_thresh > 0: # remove events below threshold
        for i in range(len(weights)):
            mask = weights > weight_thresh
            weights[i] = weights[i][mask]
            x[i] = x[i][mask]

    # in case, remove event
    if remove_x is not None:
        mask = x == remove_x
        weights = weights[~mask]
        x = x[~mask]

    # expectation maximization
    mu = np.linspace(x[0], x[-1], n+2)[1:-1]
    sigma = np.ones(n) * initial_width
    ns = np.ones(n) * 10
    llh_diff = 100
    llh_old = 0
    llh_diff_list = [100] * 20

    iteration = 0

    while iteration < iter_max and llh_diff > tol: # run until convergence or maximum number of iterations
        iteration += 1

        e, logllh = expectation_em(ns, mu, sigma, x, weights)

        llh_new = np.sum(logllh)
        tmp_diff = np.abs(llh_old - llh_new) / llh_new
        llh_diff_list = llh_diff_list[:-1]
        llh_diff_list.insert(0, tmp_diff)
        llh_diff = np.max(llh_diff_list)
        llh_old = llh_new
        mu, sigma, ns = maximization_em(e, x)

    return mu, sigma, ns
