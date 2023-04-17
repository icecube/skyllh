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
   

class ExpectationMaximizationTimefit():
    """ Helper class for using expectation maximization for fitting the flare time
    """

    def __init__(self, ana):
        """
        Parameters
        ----------
        ana : analysis instance

        """
        self.ana = ana


    def get_energy_spatial_signal_over_background(self, fitparams):
        """Returns the signal over background ratio for 
        (spatial_signal * energy_signal) / (spatial_background * energy_background).
        
        Parameter
        ---------
        fitparams : dict
            Dictionary with {"gamma": float} for energy pdf.
        
        Returns
        -------
        ratio : 1d ndarray
            Product of spatial and energy signal over background pdfs.
        """

        ratio = self.ana._llhratio.llhratio_list[0].pdfratio_list[0].get_ratio(self.ana._tdm_list[0], fitparams)
        ratio *= self.ana._llhratio.llhratio_list[0].pdfratio_list[1].get_ratio(self.ana._tdm_list[0], fitparams)

        return ratio


    def em_fit(self, fitparams, n=1, tol=1.e-200, iter_max=500, sob_thresh=0, initial_width=5000,
            remove_time=None):
        """Run expectation maximization.
        
        Parameters
        ----------
        fitparams : dict
            Dictionary with value for gamma, e.g. {'gamma': 2}.
        n : int
            How many Gaussians flares we are looking for.
        tol : float
            the stopping criteria for expectation maximization. This is the difference in the normalized likelihood over the
            last 20 iterations.
        iter_max : int
            The maximum number of iterations, even if stopping criteria tolerance (`tol`) is not yet reached.
        sob_thresh : float
            Set a minimum threshold for signal over background ratios. Ratios below this threshold will be removed.
        initial_width : float
            Starting width for the gaussian flare in days.
        remove_time : float | None
            Time information of event that should be removed.
        
        Returns
        -------
        mean flare time, flare width, normalization factor for time pdf
        """
        ana = self.ana

        ratio = self.get_energy_spatial_signal_over_background(fitparams)
        time = ana._tdm_list[0].get_data("time")

        if sob_thresh > 0: # remove events below threshold
            for i in range(len(ratio)):
                mask = ratio > sob_thresh
                ratio[i] = ratio[i][mask]
                time[i] = time[i][mask]

        # in case, remove event
        if remove_time is not None:
            mask = time == remove_time
            ratio = ratio[~mask]
            time = time[~mask]

        # expectation maximization
        mu = np.linspace(ana._data_list[0].grl["start"][0], ana._data_list[-1].grl["stop"][-1], n+2)[1:-1]
        sigma = np.ones(n) * initial_width
        ns = np.ones(n) * 10
        llh_diff = 100
        llh_old = 0
        llh_diff_list = [100] * 20

        iteration = 0

        while iteration < iter_max and llh_diff > tol: # run until convergence or maximum number of iterations
            iteration += 1

            e, logllh = expectation_em(ns, mu, sigma, time, ratio)

            llh_new = np.sum(logllh)
            tmp_diff = np.abs(llh_old - llh_new) / llh_new
            llh_diff_list = llh_diff_list[:-1]
            llh_diff_list.insert(0, tmp_diff)
            llh_diff = np.max(llh_diff_list)
            llh_old = llh_new
            mu, sigma, ns = maximization_em(e, time)

        return mu, sigma, ns


    def run_gamma_scan_single_flare(self, remove_time=None, gamma_min=1, gamma_max=5, n_gamma=51):
        """Run em for different gammas in the signal energy pdf

        Parameters
        ----------
        remove_time : float
            Time information of event that should be removed.
        gamma_min : float
            Lower bound for gamma scan.
        gamma_max : float
            Upper bound for gamma scan.
        n_gamma : int
            Number of steps for gamma scan.
        
        Returns 
        -------
        array with "gamma", "mu", "sigma", and scaling factor for flare "ns_em"
        """
        dtype = [("gamma", "f8"), ("mu", "f8"), ("sigma", "f8"), ("ns_em", "f8")]
        results = np.empty(n_gamma, dtype=dtype)

        for index, g in enumerate(np.linspace(gamma_min, gamma_max, n_gamma)):
            mu, sigma, ns = self.em_fit({"gamma": g}, n=1, tol=1.e-200, iter_max=500, sob_thresh=0,
                                initial_width=5000, remove_time=remove_time)
            results[index] = (g, mu[0], sigma[0], ns[0])

        return results


    def unblind_flare(self, remove_time=None):
        """Run EM on unscrambled data. Similar to the original analysis, remove the alert event.

        Parameters
        ----------
        remove_time : float
            Time information of event that should be removed.
            In the case of the TXS analysis: remove_time=58018.8711856

        Returns 
        -------
        array with "gamma", "mu", "sigma", and scaling factor for flare "ns_em"
        """
        ana = self.ana

        # get the original unblinded data
        rss = RandomStateService(seed=1)

        ana.unblind(rss)

        time_results = self.run_gamma_scan_single_flare(remove_time=remove_time)

        return time_results
    
