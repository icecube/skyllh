# -*- coding: utf-8 -*-

import numpy as np
import itertools

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from skyllh.core.pdf import (
    IsSignalPDF,
    SpatialPDF,
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


class SignalSpatialPDFPlotter(
        object,
):
    """Plotter class to plot spatial signal PDF object.
    """
    def __init__(
            self,
            tdm,
            pdf,
            **kwargs,
    ):
        """Creates a new plotter object for plotting a spatial signal PDF
        object.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that provides the data for the
            PDF evaluation.
        pdf : class instance derived from SpatialPDF and IsSignalPDF
            The PDF object to plot.
        """
        super().__init__(**kwargs)
        self.tdm = tdm
        self.pdf = pdf

    @property
    def pdf(self):
        """The PDF object to plot.
        """
        return self._pdf

    @pdf.setter
    def pdf(self, pdf):
        if not isinstance(pdf, SpatialPDF):
            raise TypeError(
                'The pdf property must be an object of instance SpatialPDF!')
        if not isinstance(pdf, IsSignalPDF):
            raise TypeError(
                'The pdf property must be an object of instance IsSignalPDF!')
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

    def plot(
            self,
            src_hypo_group_manager,
            axes,
            source_idx=None,
            sin_dec=True,
            log=True,
            **kwargs,
    ):
        """Plots the signal spatial PDF for the specified source.

        Parameters
        ----------
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.
        source_idx : int | None
            The index of the source for which the PDF ratio should get plotted.
            If set to None and the signal PDF depends on the source, index 0
            will be used.
        sin_dec : bool
            Flag if the plot should be made in right-ascention vs. declination
            (False), or in right-ascention vs. sin(declination) (True).

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

        if source_idx is None:
            source_idx = 0

        # Define the binning for ra, dec, and sin_dec.
        delta_ra_deg = 0.5
        delta_dec_deg = 0.5
        delta_sin_dec = 0.01
        # Define the event spatial uncertainty.
        sigma_deg = 0.5

        # Create a grid of signal probabilities in right-ascention and
        # declination/sin(declination) and fill it with probabilities from
        # events that fall into these bins.
        raaxis = self.pdf.axes['ra']
        rabins = int(np.ceil(raaxis.length / np.deg2rad(delta_ra_deg)))
        ra_binedges = np.linspace(raaxis.vmin, raaxis.vmax, rabins+1)
        ra_bincenters = 0.5*(ra_binedges[:-1] + ra_binedges[1:])

        decaxis = self.pdf.axes['dec']
        if sin_dec is True:
            (dec_min, dec_max) = (np.sin(decaxis.vmin), np.sin(decaxis.vmax))
            decbins = int(np.ceil((dec_max-dec_min) / delta_sin_dec))
        else:
            (dec_min, dec_max) = (decaxis.vmin, decaxis.vmax)
            decbins = int(np.ceil(decaxis.length / np.deg2rad(delta_dec_deg)))
        dec_binedges = np.linspace(dec_min, dec_max, decbins+1)
        dec_bincenters = 0.5*(dec_binedges[:-1] + dec_binedges[1:])

        probs = np.zeros((rabins, decbins), dtype=np.float64)

        # Generate events that fall into the probability bins.
        events = DataFieldRecordArray(
            np.zeros(
                (probs.size,),
                dtype=[
                    ('ira', np.int64), ('ra', np.float64),
                    ('idec', np.int64), ('dec', np.float64),
                    ('ang_err', np.float64)
                ]))
        for (i, ((ira, ra), (idec, dec))) in enumerate(itertools.product(
                enumerate(ra_bincenters),
                enumerate(dec_bincenters))):
            events['ira'][i] = ira
            events['ra'][i] = ra
            events['idec'][i] = idec
            if sin_dec is True:
                events['dec'][i] = np.arcsin(dec)
            else:
                events['dec'][i] = dec
            events['ang_err'][i] = np.deg2rad(sigma_deg)

        self._tdm.initialize_for_new_trial(src_hypo_group_manager, events)

        event_probs = self._pdf.get_prob(self._tdm)

        # Select only the probabilities for the requested source.
        if event_probs.ndim == 2:
            event_probs = event_probs[source_idx]

        # Fill the probs grid array.
        probs[events['ira'], events['idec']] = event_probs

        (left, right, bottom, top) = (raaxis.vmin, raaxis.vmax,
                                      dec_min, dec_max)
        norm = None
        if log:
            norm = LogNorm()
        img = axes.imshow(
            probs.T,
            extent=(left, right, bottom, top),
            origin='lower',
            norm=norm,
            interpolation='none',
            **kwargs)
        axes.set_xlabel(raaxis.name)
        if sin_dec is True:
            axes.set_ylabel('sin('+decaxis.name+')')
        else:
            axes.set_ylabel(decaxis.name)
        axes.set_title(classname(self._pdf))

        return img
