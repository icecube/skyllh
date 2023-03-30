import numpy as np
from scipy.stats import norm

def expectation_em(ns, mu, sigma, t, sob):
    """
    Expectation step of expectation maximization

    Parameters
    ----------
    ns: the number of signal neutrinos, as weight for the gaussian flare
    mu: the mean of the gaussian flare
    sigma: sigma of gaussian flare
    t: [array] times of the events
    sob: [array] the signal over background values of events

    Returns
    -------
    array, weighted "responsibility" function of each event to belong to the flare
    """
    b_term = (1 - np.cos(10 / 180 * np.pi)) / 2
    N = len(t)
    e_sig = []
    for i in range(len(ns)):
        e_sig.append(norm(loc=mu[i], scale=sigma[i]).pdf(t) * sob * ns[i])
    e_bg = (N - np.sum(ns)) / (np.max(t) - np.min(t)) / b_term  # 2198.918456004788
    denom = sum(e_sig) + e_bg

    return [e / denom for e in e_sig], np.sum(np.log(denom))


def maximization_em(e_sig, t):
    """
    maximization step of expectation maximization

    Parameters
    ----------

    e_sig: [array] the weights for each event form the expectation step
    t: [array] the times of each event

    Returns
    -------
    mu (float) : best fit mean 
    sigma (float) : best fit width
    ns (float) : scaling of gaussian 

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