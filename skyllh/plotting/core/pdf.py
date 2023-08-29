# -*- coding: utf-8 -*-

"""Plotting module to plot PDF objects.
"""

import itertools

import numpy as np

from matplotlib.axes import (
    Axes,
)
from matplotlib.colors import (
    LogNorm,
)

from skyllh.core.pdf import (
    SingleConditionalEnergyPDF,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)


class SingleConditionalEnergyPDFPlotter(
        object,
):
    """Plotter class to plot an SingleConditionalEnergyPDF object.
    """
    def __init__(self, tdm, pdf):
        """Creates a new plotter object for plotting a
        SingleConditionalEnergyPDF instance.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that provides the data for the
            PDF evaluation.
        pdf : instance of SingleConditionalEnergyPDF
            The PDF object to plot.
        """
        self.tdm = tdm
        self.pdf = pdf

    @property
    def pdf(self):
        """The PDF object to plot.
        """
        return self._pdf

    @pdf.setter
    def pdf(self, obj):
        if not isinstance(obj, SingleConditionalEnergyPDF):
            raise TypeError(
                'The pdf property must be an object of instance '
                'SingleConditionalEnergyPDF!')
        self._pdf = obj

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

    def plot(
            self,
            shg_mgr,
            axes,
            **kwargs,
    ):
        """Plots the PDF object.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypotheses.
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.

        Additional Keyword Arguments
        ----------------------------
        Any additional keyword arguments will be passed to the `mpl.imshow`
        function.

        Returns
        -------
        img : instance of mpl.AxesImage
            The AxesImage instance showing the PDF ratio image.
        """
        if not isinstance(shg_mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr argument must be an instance of '
                'SourceHypoGroupManager!')
        if not isinstance(axes, Axes):
            raise TypeError(
                'The axes argument must be an instance of '
                'matplotlib.axes.Axes!')

        # The SingleConditionalEnergyPDF instance has two axes, one for
        # log10_energy and param.
        (xbinning, ybinning) = self._pdf.binnings

        pdf_values = np.zeros(
            (xbinning.nbins, ybinning.nbins),
            dtype=np.float64)
        events = DataFieldRecordArray(np.zeros(
            (pdf_values.size,),
            dtype=[('ix', np.int64), (xbinning.name, np.float64),
                   ('iy', np.int64), (ybinning.name, np.float64)]))
        for (i, ((ix, x), (iy, y))) in enumerate(itertools.product(
                enumerate(xbinning.bincenters),
                enumerate(ybinning.bincenters))):
            events['ix'][i] = ix
            events[xbinning.name][i] = x
            events['iy'][i] = iy
            events[ybinning.name][i] = y

        self._tdm.initialize_for_new_trial(shg_mgr, events)

        event_pdf_values = self._pdf.get_prob(self._tdm)
        pdf_values[events['ix'], events['iy']] = event_pdf_values

        (left, right, bottom, top) = (xbinning.lower_edge, xbinning.upper_edge,
                                      ybinning.lower_edge, ybinning.upper_edge)
        img = axes.imshow(
            pdf_values.T,
            extent=(left, right, bottom, top),
            origin='lower',
            norm=LogNorm(),
            interpolation='none',
            **kwargs)
        axes.set_xlabel(xbinning.name)
        axes.set_ylabel(ybinning.name)
        axes.set_title(classname(self._pdf))

        return img
