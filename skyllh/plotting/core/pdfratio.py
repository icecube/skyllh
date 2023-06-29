# -*- coding: utf-8 -*-

"""Plotting module for core PDF ratio objects.
"""

import numpy as np
import itertools

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from skyllh.core.py import classname
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.trialdata import TrialDataManager
from skyllh.core.pdfratio import SigOverBkgPDFRatio


class SigOverBkgPDFRatioPlotter(object):
    """Plotter class to plot a SigOverBkgPDFRatio object.
    """
    def __init__(self, tdm, pdfratio):
        """Creates a new plotter object for plotting a
        SpatialSigOverBkgPDFRatio object.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that provides the data for the
            PDF ratio evaluation.
        pdfratio : SpatialSigOverBkgPDFRatio
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
        if not isinstance(pdfratio, SigOverBkgPDFRatio):
            raise TypeError(
                'The pdfratio property must be an instance of '
                'SigOverBkgPDFRatio!')
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

    def plot(
            self,
            src_hypo_group_manager,
            axes,
            source_idx=None,
            log=True,
            **kwargs
    ):
        """Plots the spatial PDF ratio. If the signal PDF depends on the source,
        source_idx specifies the index of the source for which the PDF should
        get plotted.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypotheses.
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.
        source_idx : int | None
            The index of the source for which the PDF ratio should get plotted.
            If set to None and the signal PDF depends on the source, index 0
            will be used.

        Additional Keyword Arguments
        ----------------------------
        Any additional keyword arguments will be passed to the `mpl.imshow`
        function.

        Returns
        -------
        img : instance of mpl.AxesImage
            The AxesImage instance showing the PDF ratio image.
        """
        if not isinstance(axes, Axes):
            raise TypeError(
                'The axes argument must be an instance of '
                'matplotlib.axes.Axes!')

        if source_idx is None:
            source_idx = 0

        # Define the binning for ra, dec, and sin_dec.
        delta_ra_deg = 0.5
        delta_dec_deg = 0.5

        raaxis = self._pdfratio.signalpdf.axes['ra']
        decaxis = self._pdfratio.signalpdf.axes['dec']

        # Create a grid of ratio in right-ascention and declination and fill it
        # with PDF ratio values from events that fall into these bins.
        # Use a binning for 1/2 degree.
        rabins = int(np.ceil(raaxis.length / np.deg2rad(delta_ra_deg)))
        decbins = int(np.ceil(decaxis.length / np.deg2rad(delta_dec_deg)))

        ratios = np.zeros((rabins, decbins), dtype=np.float64)

        ra_binedges = np.linspace(raaxis.vmin, raaxis.vmax, rabins+1)
        ra_bincenters = 0.5*(ra_binedges[:-1] + ra_binedges[1:])

        dec_binedges = np.linspace(decaxis.vmin, decaxis.vmax, decbins+1)
        dec_bincenters = 0.5*(dec_binedges[:-1] + dec_binedges[1:])

        # Generate events that fall into the ratio bins.
        events = DataFieldRecordArray(
            np.zeros(
                (ratios.size,),
                dtype=[('ira', np.int64),
                       ('ra', np.float64),
                       ('idec', np.int64),
                       ('dec', np.float64),
                       ('sin_dec', np.float64),
                       ('ang_err', np.float64)]))
        for (i, ((ira, ra), (idec, dec))) in enumerate(itertools.product(
                                                enumerate(ra_bincenters),
                                                enumerate(dec_bincenters))):
            events['ira'][i] = ira
            events['ra'][i] = ra
            events['idec'][i] = idec
            events['dec'][i] = dec
            events['sin_dec'][i] = np.sin(dec)
            events['ang_err'][i] = np.deg2rad(0.5)

        self._tdm.initialize_for_new_trial(src_hypo_group_manager, events)

        event_ratios = self._pdfratio.get_ratio(self._tdm)

        # Select only the ratios for the requested source.
        if event_ratios.ndim == 2:
            event_ratios = event_ratios[source_idx]

        ratios[events['ira'], events['idec']] = event_ratios

        (left, right, bottom, top) = (raaxis.vmin, raaxis.vmax,
                                      decaxis.vmin, decaxis.vmax)
        norm = LogNorm() if log else None
        img = axes.imshow(
            ratios.T,
            extent=(left, right, bottom, top),
            origin='lower',
            norm=norm,
            interpolation='none', **kwargs)
        axes.set_xlabel(raaxis.name)
        axes.set_ylabel(decaxis.name)
        axes.set_title(classname(self._pdfratio))

        return img
