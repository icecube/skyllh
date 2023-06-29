# -*- coding: utf-8 -*-

"""Plotting module to plot IceCube specific background PDF objects.
"""

import numpy as np

from matplotlib.axes import (
    Axes,
)
from matplotlib.colors import (
    LogNorm,
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
from skyllh.i3.backgroundpdf import (
    BackgroundI3SpatialPDF,
)


class BackgroundI3SpatialPDFPlotter(object):
    """Plotter class to plot an BackgroundI3SpatialPDF object.
    """
    def __init__(self, tdm, pdf):
        """Creates a new plotter object for plotting an BackgroundI3SpatialPDF
        object.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that provides the data for the PDF
            evaluation.
        pdf : instance of BackgroundI3SpatialPDF
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
    def pdf(self, pdf):
        if not isinstance(pdf, BackgroundI3SpatialPDF):
            raise TypeError(
                'The pdf property must be an object of instance '
                'BackgroundI3SpatialPDF!')
        self._pdf = pdf

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

    def plot(self, src_hypo_group_manager, axes):
        """Plots the spatial PDF. It uses the sin(dec) binning of the PDF to
        propperly represent the resolution of the PDF in the drawing.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypotheses.
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF should get drawn to.

        Returns
        -------
        img : instance of mpl.AxesImage
            The AxesImage instance showing the PDF image.
        """
        if not isinstance(src_hypo_group_manager, SourceHypoGroupManager):
            raise TypeError(
                'The src_hypo_group_manager argument must be an '
                'instance of SourceHypoGroupManager!')
        if not isinstance(axes, Axes):
            raise TypeError(
                'The axes argument must be an instance of '
                'matplotlib.axes.Axes!')

        # By construction the BackgroundI3SpatialPDF does not depend on
        # right-ascention. Hence, we only need a single bin for the
        # right-ascention.
        sin_dec_binning = self.pdf.get_binning('sin_dec')
        pdfprobs = np.zeros((1, sin_dec_binning.nbins))

        sin_dec_points = sin_dec_binning.bincenters
        events = DataFieldRecordArray(np.zeros(
            (pdfprobs.size,),
            dtype=[('sin_dec', np.float64)]))
        for (i, sin_dec) in enumerate(sin_dec_points):
            events['sin_dec'][i] = sin_dec

        self._tdm.initialize_for_new_trial(src_hypo_group_manager, events)

        event_probs = self._pdf.get_prob(self._tdm)

        for i in range(len(events)):
            pdfprobs[0, i] = event_probs[i]

        ra_axis = self.pdf.axes['ra']
        (left, right, bottom, top) = (
            ra_axis.vmin, ra_axis.vmax,
            sin_dec_binning.lower_edge, sin_dec_binning.upper_edge)
        img = axes.imshow(
            pdfprobs.T,
            extent=(left, right, bottom, top),
            origin='lower',
            norm=LogNorm(),
            interpolation='none')
        axes.set_xlabel('ra')
        axes.set_ylabel('sin_dec')
        axes.set_title(classname(self.pdf))

        return img
