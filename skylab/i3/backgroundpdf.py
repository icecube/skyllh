# -*- coding: utf-8 -*-

import numpy as np

import scipy.interpolate

from skylab.core.analysis import BinningDefinition, UsesBinning
from skylab.core.pdf import SpatialPDF, EnergyPDF

class I3SpatialBackgroundPDF(SpatialPDF, UsesBinning):
    """This is the base class for all IceCube specific spatial background PDF
    models. IceCube spatial background PDFs depend soley on the zenith angle,
    and hence, on the declination of the event.

    The IceCube spatial background PDF is modeled as a 1d spline function in
    sin(declination).
    """
    def __init__(self, data_sinDec, data_weights, sinDec_binning, spline_order_sinDec):
        """Creates a new IceCube spatial background PDF object.

        Parameters
        ----------
        data_sinDec : 1d ndarray
            The array holding the sin(dec) values of the events.
        data_weights : 1d ndarray
            The array holding the weight of each event used for histogramming.
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination) axis.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
        """
        super(I3SpatialBackgroundPDF, self).__init__()

        self.add_binning(sinDec_binning, 'sin_dec')
        self.spline_order_sinDec = spline_order_sinDec

        (h, bins) = np.histogram(data_sinDec,
                                 bins = sinDec_binning.binedges,
                                 weights = data_weights,
                                 range = sinDec_binning.range,
                                 density = True)

        # Check if there are any NaN values.
        if(np.any(np.isnan(h))):
            raise ValueError('The declination histogram contains NaN values! Check your sin(dec) binning! The bins with NaN values are: {0}'.format(sinDec_binning.bincenters[np.isnan(h)]))

        if(np.any(h <= 0.)):
            raise ValueError('Some declination histogram bins for the spatial background PDF are empty, this must not happen! The empty bins are: {0}'.format(sinDec_binning.bincenters[h <= 0.]))

        # Create the logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sinDec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

    @property
    def spline_order_sinDec(self):
        """The order (int) of the logarithmic spline function, that splines the
        background PDF, along the sin(dec) axis.
        """
        return self._spline_order_sinDec
    @spline_order_sinDec.setter
    def spline_order_sinDec(self, order):
        if(not isinstance(order, int)):
            raise TypeError('The spline_order_sinDec property must be of type int!')
        self._spline_order_sinDec = order

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this spatial background PDF is valid for all the given
        experimental data.
        It checks if all the data is within the sin(dec) binning range.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            need to exist:
            'dec' : float
                The declination of the data event.

        Errors
        ------
        ValueError
            If some of the data is outside the sin(dec) binning range.
        """
        sinDec_binning = self.get_binning('sin_dec')
        exp_sinDec = np.sin(data_exp['dec'])

        # Check if all the data is within the binning range.
        if(np.any((exp_sinDec < sinDec_binning.lower_edge) |
                  (exp_sinDec > sinDec_binning.upper_edge))):
            raise ValueError('Some data is outside the sin(dec) range (%.3f, %.3f)!'%(sinDec_binning.lower_edge, sinDec_binning.upper_edge))

    def get_prob(self, events, params=None):
        """Calculates the spatial background probability on the sphere of each
        event.

        Parameters
        ----------
        events : numpy record ndarray
            The array holding the event data. The following data fields must
            exist:
            'sin_dec' : float
                The sin(declination) value of the event.
        params : None
            Unused interface parameter.

        Returns
        -------
        prob : 1d ndarray
            The spherical probability of each data event.
        """
        prob = 0.5 / np.pi * np.exp(self._log_spline(events['sin_dec']))
        return prob

class I3DataSpatialBackgroundPDF(I3SpatialBackgroundPDF):
    """This is the IceCube spatial background PDF, which gets constructed from
    experimental data.
    """
    def __init__(self, data_exp, sinDec_binning, spline_order_sinDec=2):
        """Constructs a new IceCube spatial background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:
            'dec' : float
                The declination of the data event.
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
            The default is 2.
        """
        if(not isinstance(data_exp, np.ndarray)):
            raise TypeError('The data_exp argument must be of type numpy.ndarray!')

        data_sinDec = np.sin(data_exp['dec'])
        data_weights = np.ones((data_exp.size,))

        # Create the PDF using the base class.
        super(I3DataSpatialBackgroundPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec
        )

class I3MCSpatialBackgroundPDF(I3SpatialBackgroundPDF):
    """This is the IceCube spatial background PDF, which gets constructed from
    monte-carlo data.
    """
    def __init__(self, data_mc, mc_weight_field_names, sinDec_binning, spline_order_sinDec=2):
        """Constructs a new IceCube spatial background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The array holding the monte-carlo data. The following data fields
            must exist:
            'dec' : float
                The declination of the data event.
        mc_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as event weights. If a list is given, the weight
            values of all the fields will be summed to construct the final event
            weight.
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
            The default is 2.
        """
        if(not isinstance(data_mc, np.ndarray)):
            raise TypeError('The data_mc argument must be of type numpy.ndarray!')

        if(isinstance(mc_weight_field_names, str)):
            mc_weight_field_names = [mc_weight_field_names]
        if(not issequenceof(mc_weight_field_names, str)):
            raise TypeError('The mc_weight_field_names argument must be of type str or a sequence of type str!')

        data_sinDec = np.sin(data_mc['dec'])

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_weights = np.zeros(data_mc.size, dtype=np.float64)
        for name in mc_weight_field_names:
            if(name not in data_mc.dtype.names):
                raise KeyError('The field "%s" does not exist in the MC data!'%(name))
            data_weights += data_mc[name]

        # Create the PDF using the base class.
        super(I3MCSpatialBackgroundPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec
        )

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

class I3DataEnergyBackgroundPDF(I3EnergyPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    experimental data.
    """
    def __init__(self, data_exp, logE_binning, sinDec_binning):
        """Constructs a new IceCube energy background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:
            'log_energy' : float
                The logarithm of the energy value of the data event.
            'dec' : float
                The declination of the data event.
        logE_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        """
        if(not isinstance(data_exp, np.ndarray)):
            raise TypeError('The data_exp argument must be of type numpy.ndarray!')

        data_logE = data_exp['log_energy']
        data_sinDec = np.sin(data_exp['dec'])
        data_weights = np.ones((data_exp.size,))

        # Create the PDF using the base class.
        super(I3DataEnergyBackgroundPDF, self).__init__(
            data_logE, data_sinDec, data_weights,
            logE_binning, sinDec_binning
        )

class I3MCEnergyBackgroundPDF(I3EnergyPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    monte-carlo data.
    """
    def __init__(self, data_mc, mc_weight_field_names, logE_binning, sinDec_binning):
        """Constructs a new IceCube energy background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The array holding the monte-carlo data. The following data fields
            must exist:
            'log_energy' : float
                The logarithm of the energy value of the data event.
            'dec' : float
                The declination of the data event.
        mc_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as event weights. If a list is given, the weight
            values of all the fields will be summed to construct the final event
            weight.
        logE_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        """
        if(not isinstance(data_mc, np.ndarray)):
            raise TypeError('The data_mc argument must be of type numpy.ndarray!')

        if(isinstance(mc_weight_field_names, str)):
            mc_weight_field_names = [mc_weight_field_names]
        if(not issequenceof(mc_weight_field_names, str)):
            raise TypeError('The mc_weight_field_names argument must be of type str or a sequence of type str!')

        data_logE = data_mc['log_energy']
        data_sinDec = np.sin(data_mc['dec'])

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_weights = np.zeros(data_mc.size, dtype=np.float64)
        for name in mc_weight_field_names:
            if(name not in data_mc.dtype.names):
                raise KeyError('The field "%s" does not exist in the MC data!'%(name))
            data_weights += data_mc[name]

        # Create the PDF using the base class.
        super(I3MCEnergyBackgroundPDF, self).__init__(
            data_logE, data_sinDec, data_weights,
            logE_binning, sinDec_binning
        )
