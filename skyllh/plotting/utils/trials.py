# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import text as mpl_text
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plot_ns_fit_vs_mean_ns_inj(
        trials,
        title='',
        axis_fontsize=16,
        title_fontsize=16,
        return_hist=False,
        mean_n_sig_key='mean_n_sig',
        ns_key='ns',
        xlabel=None,
        ylabel=None):
    """Creates a 2D histogram plot showing the fit number of signal events vs.
    the mean number of injected signal events.

    Parameters
    ----------
    trials : numpy record array
        The record array holding the results if the trials.
    title : str
        The title of the plot.
    axis_fontsize : float
        The size of the font for axis labels.
    title_fontsize : float
        The size of the font for the title.
    return_hist : bool
        If set to ``True``, the histogram data will be return as well.
        Default is ``False``.
    mean_n_sig_key : str
        The name of the key for the mean number of injected signal events in the
        given trials record array.
        Default is ``'mean_n_sig'``.
    ns_key : str
        The name of the key for the fitted number of signal events in the
        given trials record array.
        Default is ``'ns'``.
    xlabel : str | None
        The label of the x-axis in math syntax.
        Default is ``r'<n>_{\mathrm{sig,inj}}}'``.
    ylabel : str | None
        The label of if y-axis in math syntax.
        Default is ``r'n_\mathrm{sig,fit}'``.

    Returns
    -------
    fig : MPL Figure instance
        The created matplotlib Figure instance.
    hist : 2d ndarray
        If the ``return_hist`` option was set to ``True``, this will be
        returned as well. It contains the histogram bin content.

    """
    if(xlabel is None):
        xlabel = r'<n>_{\mathrm{sig,inj}}'
    if(ylabel is None):
        ylabel = r'n_\mathrm{sig,fit}'

    mean_n_sig_min = np.min(trials[mean_n_sig_key])
    mean_n_sig_max = np.max(trials[mean_n_sig_key])
    mean_n_sig_step = np.diff(np.sort(np.unique(trials[mean_n_sig_key])))[0]
    x_bins = np.arange(
        mean_n_sig_min-mean_n_sig_step/2,
        mean_n_sig_max+mean_n_sig_step/2+1,
        mean_n_sig_step)

    reco_ns_min = np.min(trials[ns_key])
    reco_ns_max = np.max(trials[ns_key])
    dy = np.min((mean_n_sig_step, np.abs(reco_ns_max-reco_ns_min)/100))
    y_bins = np.arange(np.floor(reco_ns_min), reco_ns_max+dy, dy)

    hist_weights = np.ones_like(trials[mean_n_sig_key])
    (mean_n_sig, n_trials) = np.unique(
        trials[mean_n_sig_key], return_counts=True)
    ns_median = np.empty_like(mean_n_sig, dtype=np.float)
    ns_uq = np.empty_like(mean_n_sig, dtype=np.float)
    ns_lq = np.empty_like(mean_n_sig, dtype=np.float)
    for (idx, (mean_n_sig_,n_trials_)) in enumerate(zip(mean_n_sig, n_trials)):
        m = trials[mean_n_sig_key] == mean_n_sig_
        hist_weights[m] /= n_trials_
        ns_median[idx] = np.median(trials[m][ns_key])
        ns_uq[idx] = np.percentile(trials[m][ns_key], 84.1)
        ns_lq[idx] = np.percentile(trials[m][ns_key], 15.9)

    (fig, ax) = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [3,1]}, sharex=True, figsize=(12,10))

    ax_divider = make_axes_locatable(ax[0])
    # Add an axes above the main axes for the colorbar.
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    (hist, xedges, yedges, image) = ax[0].hist2d(
        trials[mean_n_sig_key], trials['ns'],
        bins=[x_bins, y_bins],
        weights=hist_weights,
        norm=mpl.colors.LogNorm(),
        cmap=plt.get_cmap('GnBu'))

    ax[0].set_ylabel('$'+ylabel+'$', fontsize=axis_fontsize)

    # Add diagonal line.
    ax[0].plot(
        x_bins, x_bins,
        color='black', alpha=0.4, linestyle='-', linewidth=2)
    # Plot the median fitted ns.
    ax[0].plot(
        mean_n_sig, ns_median,
        color='orange', linestyle='--', linewidth=2)

    cb = fig.colorbar(image, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.text(
        0.5, 1, title, horizontalalignment='center', verticalalignment='bottom',
        transform=fig.transFigure, fontsize=title_fontsize)

    ax[1].hlines(0, x_bins[0], x_bins[-1])
    m = mean_n_sig != 0
    ax[1].plot(
        mean_n_sig[m], (ns_median[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        linewidth=2)
    ax[1].fill_between(
        mean_n_sig[m],
        (ns_uq[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        (ns_lq[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        alpha=0.2, color='gray', label=r'$1\sigma$')

    ax[1].legend()

    ax[1].set_xlabel('$'+xlabel+'$', fontsize=axis_fontsize)
    ratio_ylabel = r'$\frac{%s - %s}{%s}$'%(ylabel, xlabel, xlabel)+' [%]'
    ax[1].set_ylabel(ratio_ylabel, fontsize=axis_fontsize)
    ax[1].set_xlim(x_bins[0], x_bins[-1])
    ax[1].set_ylim([-100, 100])

    plt.tight_layout()

    if(return_hist):
        return (fig, hist, xedges, yedges)

    return fig
