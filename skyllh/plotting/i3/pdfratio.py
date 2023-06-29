# -*- coding: utf-8 -*-

"""Plotting module to plot IceCube specific PDF ratio objects.
"""

import numpy as np
import itertools

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from skyllh.core.py import classname
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.trialdata import TrialDataManager
from skyllh.i3.pdfratio import SplinedI3EnergySigSetOverBkgPDFRatio


class SplinedI3EnergySigSetOverBkgPDFRatioPlotter(object):
    """Plotter class to plot an I3EnergySigSetOverBkgPDFRatioSpline object.
    """
    def __init__(self, tdm, pdfratio):
        """Creates a new plotter object for plotting an
        I3EnergySigSetOverBkgPDFRatioSpline object.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that provides the data for the
            PDF ratio evaluation.
        pdfratio : I3EnergySigSetOverBkgPDFRatioSpline
            The PDF ratio object to plot.
        """
        self.tdm = tdm
        self.pdfratio = pdfratio

    @property
    def pdfratio(self):
        """The PDF ratio object to plot.
        """
        return self._pdfratio

    @pdfratio.setter
    def pdfratio(self, pdfratio):
        if not isinstance(pdfratio, SplinedI3EnergySigSetOverBkgPDFRatio):
            raise TypeError(
                'The pdfratio property must be an instance of '
                'SplinedI3EnergySigSetOverBkgPDFRatio!')
        self._pdfratio = pdfratio

    @property
    def tdm(self):
        """The TrialDataManager that provides the data for the PDF evaluation.
        """
        return self._tdm

    @tdm.setter
    def tdm(self, obj):
        if not isinstance(obj, TrialDataManager):
            raise TypeError(
                'The tdm property must be an instance of TrialDataManager!')
        self._tdm = obj

    def plot(self, src_hypo_group_manager, axes, fitparams, **kwargs):
        """Plots the PDF ratio for the given set of fit paramater values.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypotheses.
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.
        fitparams : dict
            The dictionary with the set of fit paramater values.

        Additional Keyword Arguments
        ----------------------------
        Any additional keyword arguments will be passed to the `mpl.imshow`
        function.

        Returns
        -------
        img : instance of mpl.AxesImage
            The AxesImage instance showing the PDF ratio image.
        """
        if not isinstance(src_hypo_group_manager, SourceHypoGroupManager):
            raise TypeError(
                'The src_hypo_group_manager argument must be an '
                'instance of SourceHypoGroupManager!')
        if not isinstance(axes, Axes):
            raise TypeError(
                'The axes argument must be an instance of '
                'matplotlib.axes.Axes!')
        if not isinstance(fitparams, dict):
            raise TypeError(
                'The fitparams argument must be an instance of dict!')

        # Get the binning for the axes. We use the background PDF to get it
        # from. By construction, all PDFs use the same binning. We know that
        # the PDFs are 2-dimensional.
        (xbinning, ybinning) = self._pdfratio.backgroundpdf.binnings

        # Create a 2D array with the ratio values. We put one event into each
        # bin.
        ratios = np.zeros((xbinning.nbins, ybinning.nbins), dtype=np.float64)
        events = DataFieldRecordArray(np.zeros(
            (ratios.size,),
            dtype=[('ix', np.int64), (xbinning.name, np.float64),
                   ('iy', np.int64), (ybinning.name, np.float64)]))
        for (i, ((ix, x), (iy, y))) in enumerate(itertools.product(
                enumerate(xbinning.bincenters),
                enumerate(ybinning.bincenters))):
            events['ix'][i] = ix
            events[xbinning.name][i] = x
            events['iy'][i] = iy
            events[ybinning.name][i] = y

        self._tdm.initialize_for_new_trial(src_hypo_group_manager, events)

        event_ratios = self.pdfratio.get_ratio(self._tdm, fitparams)
        for i in range(len(events)):
            ratios[events['ix'][i], events['iy'][i]] = event_ratios[i]

        (left, right, bottom, top) = (xbinning.lower_edge, xbinning.upper_edge,
                                      ybinning.lower_edge, ybinning.upper_edge)
        img = axes.imshow(
            ratios.T,
            extent=(left, right, bottom, top),
            origin='lower',
            norm=LogNorm(),
            interpolation='none',
            **kwargs)
        axes.set_xlabel(xbinning.name)
        axes.set_ylabel(ybinning.name)
        axes.set_title(classname(self._pdfratio))

        return img
