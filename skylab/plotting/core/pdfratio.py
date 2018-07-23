# -*- coding: utf-8 -*-

"""Plotting module for core PDF ratio objects.
"""

import numpy as np

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from skylab.core.py import classname
from skylab.core.pdfratio import BasicSpatialSigOverBkgPDFRatio

class BasicSpatialSigOverBkgPDFRatioPlotter(object):
    """Plotter class to plot a BasicSpatialSigOverBkgPDFRatio object.
    """
    def __init__(self, pdfratio):
        """Creates a new plotter object for plotting a
        BasicSpatialSigOverBkgPDFRatio object.

        Parameters
        ----------
        pdfratio : BasicSpatialSigOverBkgPDFRatio
            The PDF ratio object to plot.
        """
        self.pdfratio = pdfratio

    @property
    def pdfratio(self):
        """The PDF ratio object to plot.
        """
        return self._pdfratio
    @pdfratio.setter
    def pdfratio(self, pdfratio):
        if(not isinstance(pdfratio, BasicSpatialSigOverBkgPDFRatio)):
            raise TypeError('The pdfratio property must be an object of instance I3EnergySigOverBkgPDFRatioSpline!')
        self._pdfratio = pdfratio

    def plot(self, axes, source_idx=None):
        """Plots the spatial PDF ratio. If the signal PDF depends on the source,
        source_idx specifies the index of the source for which the PDF should
        get plotted.

        Parameters
        ----------
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.
        source_idx : int | None
            The index of the source for which the PDF ratio should get plotted.
            If set to None and the signal PDF depends on the source, index 0
            will be used.
        """
        if(not isinstance(axes, Axes)):
            raise TypeError('The axes argument must be an instance of matplotlib.axes.Axes!')

        if(source_idx is None):
            source_idx = 0

        raaxis = self.pdfratio.signalpdf.axes.get_axis('ra')
        decaxis = self.pdfratio.signalpdf.axes.get_axis('dec')

        # Create a grid of ratio in right-ascention and declination and fill it
        # with PDF ratio values from events that fall into these bins.
        # Use a binning for 1/2 degree.
        rabins = 360/0.5
        decbins = 180/0.5
        ratios = np.zeros((rabins,decbins), dtype=np.float)

        ra_binedges = np.linspace(raaxis.vmin, raaxis.vmax, rabins+1)
        dec_binedges = np.linspace(decaxis.vmin, decaxis.vmax, decbins+1)

        ra_bincenters = 0.5*(ra_binedges[:-1] + ra_binedges[1:])
        dec_bincenters = 0.5*(dec_binedges[:-1] + dec_binedges[1:])

        # Generate events that fall into the ratio bins.
        events = np.zeros((ratios.size,),
                          dtype=[('ira', np.int), ('ra', np.float),
                                 ('idec', np.int), ('dec', np.float),
                                 ('sigma', np.float)])
        for (i, ((ira,ra),(idec,dec))) in enumerate(itertools.product(
                                                enumerate(ra_bincenters),
                                                enumerate(dec_bincenters))):
            events['ira'][i] = ira
            events['ra'][i] = ra
            events['idec'][i] = idec
            events['dec'][i] = dec
            events['sigma'][i] = np.deg2rad(0.5)

        event_ratios = self.pdfratio.get_ratio(events)

        # Select only the ratios for the requested source.
        if(event_ratios.ndim == 2):
            event_ratios = event_ratios[:,source_idx]

        for i in range(len(events)):
            ratios[events['ira'][i],events['idec'][i]] = event_ratios[i]

        (left, right, bottom, top) = (raaxis.vmin, raaxis.vmax,
                                      decaxis.vmin, decaxis.vmax)
        axes.imshow(ratios.T, extent=(left, right, bottom, top), origin='lower',
                    norm=LogNorm(), interpolation='none')
        axes.set_xlabel(raaxis.name)
        axes.set_ylabel(decaxis.name)
        axes.set_title(classname(self.pdfratio))
