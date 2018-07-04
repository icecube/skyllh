# -*- coding: utf-8 -*-

import numpy as np

from skylab.core.analysis import UsesBinning
from skylab.core.pdf import EnergyPDF

class I3EnergyPDF(EnergyPDF, UsesBinning):
    """This is the base class for all IceCube specific energy PDF models.
    IceCube energy PDFs depend soley on the energy and the
    zenith angle, and hence, on the declination of the event.

    The IceCube energy PDF is modeled as a 1d histogram in energy,
    but for different sin(declination) bins, hence, stored as a 2d histogram.
    """
    def __init__(self, data_logE, data_sinDec, data_weights, logE_binning, sinDec_binning):
        """Creates a new IceCube energy PDF object.

        Parameters
        ----------
        data_logE : 1d ndarray
            The array holding the log10(E) values of the events.
        data_sinDec : 1d ndarray
            The array holding the sin(dec) values of the events.
        data_weights : 1d ndarray
            The array holding the weight of each event used for histogramming.
        logE_binning : BinningDefinition
            The binning definition for the log(E) axis.
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination) axis.
        """
        super(I3EnergyBackgroundPDF, self).__init__()

        self.add_binning(logE_binning, 'log_energy')
        self.add_binning(sinDec_binning, 'sin_dec')

        # Create a 2D histogram. We will do the normalization along the logE
        # axis manually.
        (h, bins) = np.histogram2d(data_logE, data_sinDec,
            bins = [logE_binning.binedges, sinDec_binning.binedges],
            weights = data_weights,
            range = [logE_binning.range, sinDec_binning.range],
            normed = False)

        # Calculate the normalization for each sin(dec) bin, i.e. along the
        # logE-axis (axis 0).
        norms = np.sum(h, axis=(0,)) * np.diff(logE_binning.binedges)
        h /= norms

        self._hist_logE_sinDec = h

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this energy PDF is valid for all the given experimental
        data.
        It checks if all the data is within the logE and sin(dec) binning range.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:
            'log_energy' : float
                The logarithm of the energy value of the data event.
            'dec' : float
                The declination of the data event.

        Errors
        ------
        ValueError
            If some of the data is outside the logE or sin(dec) binning range.
        """
        logE_binning = self.get_binning('log_energy')
        sinDec_binning = self.get_binning('sin_dec')

        exp_logE = data_exp['log_energy']
        exp_sinDec = np.sin(data_exp['dec'])

        # Check if all the data is within the binning range.
        if(logE_binning.any_data_out_of_binning_range(exp_logE)):
            raise ValueError('Some data is outside the logE range (%.3f, %.3f)!'%(logE_binning.lower_edge, logE_binning.upper_edge))
        if(sinDec_binning.any_data_out_of_binning_range(exp_sinDec)):
            raise ValueError('Some data is outside the sin(dec) range (%.3f, %.3f)!'%(sinDec_binning.lower_edge, sinDec_binning.upper_edge))

    def get_prob(self, events, params=None):
        """Calculates the energy probability (in logE) of each event.

        Parameters
        ----------
        events : numpy record ndarray
            The array holding the event data. The following data fields must
            exist:
            'log_energy' : float
                The logarithm of the energy value of the event.
            'sinDec' : float
                The sin(declination) value of the event.
        params : None
            Unused interface parameter.

        Returns
        -------
        prob : 1d ndarray
            The array with the energy probability for each event.
        """
        logE_binning = self.get_binning('log_energy')
        sinDec_binning = self.get_binning('sin_dec')

        logE_idx = np.digitize(events['log_energy'], logE_binning.binedges)
        sinDec_idx = np.digitize(events['sin_dec'], sinDec_binning.binedges)

        prob = self._hist_logE_sinDec[(logE_idx,sinDec_idx)]
        return prob
