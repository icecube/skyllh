import numpy as np
from scipy.stats import norm

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
        The signal over background values of events.

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