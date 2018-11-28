# -*- coding: utf-8 -*-

"""Plotting module to plot IceCube specific PDF ratio objects.
"""

import numpy as np
import itertools

from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from skyllh.core.py import classname
from skyllh.i3.pdfratio import I3EnergySigSetOverBkgPDFRatioSpline

class I3EnergySigSetOverBkgPDFRatioSplinePlotter(object):
    """Plotter class to plot an I3EnergySigSetOverBkgPDFRatioSpline object.
    """
    def __init__(self, pdfratio):
        """Creates a new plotter object for plotting an
        I3EnergySigSetOverBkgPDFRatioSpline object.

        Parameters
        ----------
        pdfratio : I3EnergySigSetOverBkgPDFRatioSpline
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
        if(not isinstance(pdfratio, I3EnergySigSetOverBkgPDFRatioSpline)):
            raise TypeError('The pdfratio property must be an object of instance I3EnergySigSetOverBkgPDFRatioSpline!')
        self._pdfratio = pdfratio

    def plot(self, axes, fitparams):
        """Plots the PDF ratio for the given set of fit paramater values.

        Parameters
        ----------
        axes : mpl.axes.Axes
            The matplotlib Axes object on which the PDF ratio should get drawn
            to.
        fitparams : dict
            The dictionary with the set of fit paramater values.
        """
        if(not isinstance(axes, Axes)):
            raise TypeError('The axes argument must be an instance of matplotlib.axes.Axes!')

        # Get the binning for the axes. We use the background PDF to get it
        # from. By construction, all PDFs use the same binning. We know that
        # the PDFs are 2-dimensional.
        (xbinning, ybinning) = self.pdfratio.backgroundpdf.binnings

        # Create a 2D array with the ratio values. We put one event into each
        # bin.
        ratios = np.zeros((xbinning.nbins, ybinning.nbins), dtype=np.float)
        events = np.zeros((ratios.size,),
                          dtype=[('ix', np.int), (xbinning.name, np.float),
                                 ('iy', np.int), (ybinning.name, np.float)])
        for (i, ((ix,x),(iy,y))) in enumerate(itertools.product(
                                                enumerate(xbinning.bincenters),
                                                enumerate(ybinning.bincenters))):
            events['ix'][i] = ix
            events[xbinning.name][i] = x
            events['iy'][i] = iy
            events[ybinning.name][i] = y

        event_ratios = self.pdfratio.get_ratio(events, fitparams)
        for i in range(len(events)):
            ratios[events['ix'][i],events['iy'][i]] = event_ratios[i]

        (left, right, bottom, top) = (xbinning.lower_edge, xbinning.upper_edge,
                                      ybinning.lower_edge, ybinning.upper_edge)
        axes.imshow(ratios.T, extent=(left, right, bottom, top), origin='lower',
                    norm=LogNorm(), interpolation='none')
        axes.set_xlabel(xbinning.name)
        axes.set_ylabel(ybinning.name)
        axes.set_title(classname(self.pdfratio))
