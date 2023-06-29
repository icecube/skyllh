# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import matplotlib as mpl
import numpy as np

from matplotlib import (
    pyplot as plt,
)
from mpl_toolkits.axes_grid1.axes_divider import (
    make_axes_locatable,
)


def plot_ns_fit_vs_mean_ns_inj(  # noqa: C901
        trials,
        mean_n_sig_key='mean_n_sig',
        ns_fit_key='ns',
        rethist=False,
        title='',
        figsize=None,
        line_color=None,
        axis_fontsize=16,
        title_fontsize=16,
        tick_fontsize=16,
        xlabel=None,
        ylabel=None,
        ylim=None,
        ratio_ylim=None,
):
    r"""Creates a 2D histogram plot showing the fit number of signal events vs.
    the mean number of injected signal events.

    Parameters
    ----------
    trials : numpy record array
        The record array holding the results of the trials.
    mean_n_sig_key : str
        The name of the key for the mean number of injected signal events in the
        given trials record array.
        Default is ``'mean_n_sig'``.
    ns_fit_key : str
        The name of the key for the fitted number of signal events in the
        given trials record array.
        Default is ``'ns'``.
    rethist : bool
        If set to ``True``, the histogram data along with the histogram bin
        edges will be return as well.
        Default is ``False``.
    title : str
        The title of the plot.
    figsize : tuple | None
        The two-element tuple (width,height) specifying the size of the figure.
        If set to None, the default size (12,10) will be used.
    line_color : str | None
        The color of the lines.
        The default is '#E37222'.
    axis_fontsize : float
        The font size of the axis labels.
    title_fontsize : float
        The font size of the plot title.
    tick_fontsize : float
        The font size of the tick labels.
    xlabel : str | None
        The label of the x-axis in math syntax.
        Default is ``r'<n>_{\mathrm{sig,inj}}}'``.
    ylabel : str | None
        The label of if y-axis in math syntax.
        Default is ``r'n_\mathrm{sig,fit}'``.
    ylim : tuple | None
        The (low,high)-two-element tuple specifying the y-axis limits of the
        main plot.
    ratio_ylim : tuple | None
        The (low,high)-two-element tuple specifying the y-axis limits of the
        ratio plot in percentage.
        If set to None, the default (-100,100) will be used.

    Returns
    -------
    fig : MPL Figure instance
        The created matplotlib Figure instance.
    hist : 2d ndarray
        The histogram bin content.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    xedges : 1d ndarray
        The histogram x-axis bin edges.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    yedges : 1d ndarray
        The histogram y-axis bin edges.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    """
    if figsize is None:
        figsize = (12, 10)
    if line_color is None:
        line_color = '#E37222'
    if xlabel is None:
        xlabel = r'<n>_{\mathrm{sig,inj}}'
    if ylabel is None:
        ylabel = r'n_\mathrm{sig,fit}'
    if ratio_ylim is None:
        ratio_ylim = (-100, 100)

    # Create the x-axis binning.
    mean_n_sig_min = np.min(trials[mean_n_sig_key])
    mean_n_sig_max = np.max(trials[mean_n_sig_key])
    mean_n_sig_step = np.diff(np.sort(np.unique(trials[mean_n_sig_key])))[0]
    x_bins = np.arange(
        mean_n_sig_min-mean_n_sig_step/2,
        mean_n_sig_max+mean_n_sig_step/2+1,
        mean_n_sig_step)

    # Create the y-axis binning.
    ns_fit_min = np.min(trials[ns_fit_key])
    ns_fit_max = np.max(trials[ns_fit_key])
    dy = np.min((mean_n_sig_step, np.abs(ns_fit_max-ns_fit_min)/100))
    y_bins = np.arange(np.floor(ns_fit_min), ns_fit_max+dy, dy)

    # Calculate the weight of each trial for the histogram so that the trials
    # are normalized separately for each mean number of injected signal events.
    # Also calculate the median and upper and lower 68% quantile of ns_fit.
    hist_weights = np.ones_like(trials[mean_n_sig_key])
    (mean_n_sig, n_trials) = np.unique(
        trials[mean_n_sig_key], return_counts=True)
    ns_fit_median = np.empty_like(mean_n_sig, dtype=np.float64)
    ns_fit_uq = np.empty_like(mean_n_sig, dtype=np.float64)
    ns_fit_lq = np.empty_like(mean_n_sig, dtype=np.float64)
    for (idx, (mean_n_sig_, n_trials_)) in enumerate(zip(mean_n_sig, n_trials)):
        m = trials[mean_n_sig_key] == mean_n_sig_
        hist_weights[m] /= n_trials_
        ns_fit_median[idx] = np.median(trials[m][ns_fit_key])
        ns_fit_uq[idx] = np.percentile(trials[m][ns_fit_key], 84.1)
        ns_fit_lq[idx] = np.percentile(trials[m][ns_fit_key], 15.9)

    # Create two Axes objects, one for the histogram and one for the ratio.
    (fig, ax) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
        figsize=figsize)

    # Add an axes above the main axes for the colorbar.
    ax_divider = make_axes_locatable(ax[0])
    cax = ax_divider.append_axes("top", size="7%", pad="2%")

    # Create and plot the 2D histogram.
    (hist, xedges, yedges, image) = ax[0].hist2d(
        trials[mean_n_sig_key], trials[ns_fit_key],
        bins=[x_bins, y_bins],
        weights=hist_weights,
        norm=mpl.colors.LogNorm(),
        cmap=plt.get_cmap('GnBu'))

    ax[0].set_ylabel('$'+ylabel+'$', fontsize=axis_fontsize)

    # Add the diagonal expectation line.
    ax[0].plot(
        x_bins, x_bins,
        color='black',
        alpha=0.4,
        linestyle='-',
        linewidth=2)

    # Plot the lower quantile.
    ax[0].plot(
        mean_n_sig, ns_fit_lq,
        color=line_color,
        linestyle='-.',
        linewidth=2)

    # Plot the median fitted ns.
    ax[0].plot(
        mean_n_sig, ns_fit_median,
        color=line_color,
        linestyle='-',
        linewidth=2,
        label=r'median')

    # Plot the upper quantile.
    ax[0].plot(
        mean_n_sig, ns_fit_uq,
        color=line_color,
        linestyle='-.',
        linewidth=2,
        label=r'$1\sigma$')

    ax[0].legend()

    if ylim is not None:
        ax[0].set_ylim(ylim)

    # Create the color bar.
    cb = fig.colorbar(image, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.text(
        0.5, 1, title, horizontalalignment='center', verticalalignment='bottom',
        transform=fig.transFigure, fontsize=title_fontsize)

    # Plot the ratio.
    ax[1].hlines(0, x_bins[0], x_bins[-1])
    m = mean_n_sig != 0
    ax[1].plot(
        mean_n_sig[m], (ns_fit_median[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        color=line_color, linewidth=2, label=r'median')
    ax[1].fill_between(
        mean_n_sig[m],
        (ns_fit_uq[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        (ns_fit_lq[m]-mean_n_sig[m])/mean_n_sig[m]*100,
        alpha=0.2, color='gray', label=r'$1\sigma$')

    ax[1].legend()

    ax[1].set_xlabel('$'+xlabel+'$', fontsize=axis_fontsize)
    ratio_ylabel = r'$\frac{%s - %s}{%s}$' % (ylabel, xlabel, xlabel)+' [%]'
    ax[1].set_ylabel(ratio_ylabel, fontsize=axis_fontsize)
    ax[1].set_xlim(x_bins[0], x_bins[-1])
    ax[1].set_ylim(ratio_ylim)

    # Set the font size of the tick labels.
    for tick in ax[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)

    plt.tight_layout()

    if rethist:
        return (fig, hist, xedges, yedges)

    return fig


def plot_gamma_fit_vs_mean_ns_inj(  # noqa: C901
        trials,
        gamma_inj=2,
        mean_n_sig_key='mean_n_sig',
        gamma_fit_key='gamma',
        rethist=False,
        title='',
        figsize=None,
        line_color=None,
        axis_fontsize=16,
        title_fontsize=16,
        tick_fontsize=16,
        xlabel=None,
        ylabel=None,
        ratio_ylim=None,
):
    r"""Creates a 2D histogram plot showing the fit spectral index gamma vs.
    the mean number of injected signal events.

    Parameters
    ----------
    trials : numpy record array
        The record array holding the results of the trials.
    gamma_inj : float
        The spectral index with which signal events got injected into tha trial
        data set.
    mean_n_sig_key : str
        The name of the key for the mean number of injected signal events in the
        given trials record array.
        Default is ``'mean_n_sig'``.
    gamma_fit_key : str
        The name of the key for the fitted spectral index in the given trials
        record array.
        Default is ``'gamma'``.
    rethist : bool
        If set to ``True``, the histogram data along with the histogram bin
        edges will be return as well.
        Default is ``False``.
    title : str
        The title of the plot.
    figsize : tuple | None
        The two-element tuple (width,height) specifying the size of the figure.
        If set to None, the default size (12,10) will be used.
    line_color : str | None
        The color of the lines.
        The default is '#E37222'.
    axis_fontsize : float
        The font size of the axis labels.
    title_fontsize : float
        The font size of the plot title.
    tick_fontsize : float
        The font size of the tick labels.
    xlabel : str | None
        The label of the x-axis in math syntax.
        Default is ``r'<n>_{\mathrm{sig,inj}}}'``.
    ylabel : str | None
        The label of if y-axis in math syntax.
        Default is ``r'\gamma_\mathrm{fit}'``.
    ratio_ylim : tuple | None
        The (low,high)-two-element tuple specifying the y-axis limits of the
        ratio plot in percentage.
        If set to None, the default (-100,100) will be used.

    Returns
    -------
    fig : MPL Figure instance
        The created matplotlib Figure instance.
    hist : 2d ndarray
        The histogram bin content.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    xedges : 1d ndarray
        The histogram x-axis bin edges.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    yedges : 1d ndarray
        The histogram y-axis bin edges.
        This will only be returned, when the ``rethist`` argument was set to
        ``True``.
    """
    if figsize is None:
        figsize = (12, 10)
    if line_color is None:
        line_color = '#E37222'
    if xlabel is None:
        xlabel = r'<n>_{\mathrm{sig,inj}}'
    if ylabel is None:
        ylabel = r'\gamma_\mathrm{fit}'
    if ratio_ylim is None:
        ratio_ylim = (-100, 100)

    # Create the x-axis binning.
    mean_n_sig_min = np.min(trials[mean_n_sig_key])
    mean_n_sig_max = np.max(trials[mean_n_sig_key])
    mean_n_sig_step = np.diff(np.sort(np.unique(trials[mean_n_sig_key])))[0]
    x_bins = np.arange(
        mean_n_sig_min-mean_n_sig_step/2,
        mean_n_sig_max+mean_n_sig_step/2+1,
        mean_n_sig_step)

    # Create the y-axis binning.
    gamma_fit_min = np.min(trials[gamma_fit_key])
    gamma_fit_max = np.max(trials[gamma_fit_key])
    dy = np.min((mean_n_sig_step, np.abs(gamma_fit_max-gamma_fit_min)/100))
    y_bins = np.arange(np.floor(gamma_fit_min), gamma_fit_max+dy, dy)

    # Calculate the weight of each trial for the histogram so that the trials
    # are normalized separately for each mean number of injected signal events.
    # Also calculate the median and upper and lower 68% quantile of gamma_fit.
    hist_weights = np.ones_like(trials[mean_n_sig_key])
    (mean_n_sig, n_trials) = np.unique(
        trials[mean_n_sig_key], return_counts=True)
    gamma_fit_median = np.empty_like(mean_n_sig, dtype=np.float64)
    gamma_fit_uq = np.empty_like(mean_n_sig, dtype=np.float64)
    gamma_fit_lq = np.empty_like(mean_n_sig, dtype=np.float64)
    for (idx, (mean_n_sig_, n_trials_)) in enumerate(zip(mean_n_sig, n_trials)):
        m = trials[mean_n_sig_key] == mean_n_sig_
        hist_weights[m] /= n_trials_
        gamma_fit_median[idx] = np.median(trials[m][gamma_fit_key])
        gamma_fit_uq[idx] = np.percentile(trials[m][gamma_fit_key], 84.1)
        gamma_fit_lq[idx] = np.percentile(trials[m][gamma_fit_key], 15.9)

    # Create two Axes objects, one for the histogram and one for the ratio.
    (fig, ax) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
        figsize=figsize)

    # Add an axes above the main axes for the colorbar.
    ax_divider = make_axes_locatable(ax[0])
    cax = ax_divider.append_axes('top', size='7%', pad='2%')

    # Create and plot the 2D histogram.
    (hist, xedges, yedges, image) = ax[0].hist2d(
        trials[mean_n_sig_key], trials[gamma_fit_key],
        bins=[x_bins, y_bins],
        weights=hist_weights,
        norm=mpl.colors.LogNorm(),
        cmap=plt.get_cmap('GnBu'))

    ax[0].set_ylabel('$'+ylabel+'$', fontsize=axis_fontsize)

    # Add the horizontal expectation line.
    ax[0].hlines(
        gamma_inj, x_bins[0], x_bins[-1],
        color='black',
        alpha=0.4,
        linestyle='-',
        linewidth=2)

    # Plot the upper quantile curve.
    ax[0].plot(
        mean_n_sig, gamma_fit_uq,
        color=line_color,
        linestyle='-.',
        linewidth=2)

    # Plot the median fitted gamma.
    ax[0].plot(
        mean_n_sig, gamma_fit_median,
        color=line_color,
        linestyle='-',
        linewidth=2)

    # Plot the lower quantile curve.
    ax[0].plot(
        mean_n_sig, gamma_fit_lq,
        color=line_color,
        linestyle='-.',
        linewidth=2)

    # Create the color bar.
    cb = fig.colorbar(image, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.text(
        0.5, 1, title,
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=fig.transFigure,
        fontsize=title_fontsize)

    # Plot the ratio.
    ax[1].hlines(0, x_bins[0], x_bins[-1])
    m = mean_n_sig != 0
    ax[1].plot(
        mean_n_sig[m], (gamma_fit_median[m]-gamma_inj)/gamma_inj*100,
        linewidth=2)
    ax[1].fill_between(
        mean_n_sig[m],
        (gamma_fit_uq[m]-gamma_inj)/gamma_inj*100,
        (gamma_fit_lq[m]-gamma_inj)/gamma_inj*100,
        alpha=0.2,
        color='gray',
        label=r'$1\sigma$')

    ax[1].legend()

    ax[1].set_xlabel('$'+xlabel+'$', fontsize=axis_fontsize)
    gamma_inj_label = r'\gamma_{\mathrm{inj}}'
    ratio_ylabel = r'$\frac{<%s> - %s}{%s}$' % (
        ylabel, gamma_inj_label, gamma_inj_label)+' [%]'
    ax[1].set_ylabel(ratio_ylabel, fontsize=axis_fontsize)
    ax[1].set_xlim(x_bins[0], x_bins[-1])
    ax[1].set_ylim(ratio_ylim)

    # Set the font size of the tick labels.
    for tick in ax[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)

    plt.tight_layout()

    if rethist:
        return (fig, hist, xedges, yedges)

    return fig
