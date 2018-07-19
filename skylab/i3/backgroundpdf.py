# -*- coding: utf-8 -*-

import numpy as np

import scipy.interpolate

from skylab.core.analysis import BinningDefinition, UsesBinning
from skylab.core.pdf import SpatialPDF, EnergyPDF, IsBackgroundPDF
from skylab.i3.pdf import I3EnergyPDF

class I3BackgroundSpatialPDF(SpatialPDF, UsesBinning, IsBackgroundPDF):
    """This is the base class for all IceCube specific spatial background PDF
    models. IceCube spatial background PDFs depend solely on the zenith angle,
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
        super(I3BackgroundSpatialPDF, self).__init__()

        # Define the PDF axes.
        self.add_axis(PDFAxis(name='sin_dec',
            vmin=sinDec_binning.lower_edge,
            vmax=sinDec_binning.upper_edge))

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

class I3DataBackgroundSpatialPDF(I3BackgroundSpatialPDF):
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
        super(I3DataBackgroundSpatialPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec
        )

class I3MCBackgroundSpatialPDF(I3BackgroundSpatialPDF):
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
        super(I3MCBackgroundSpatialPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec
        )

class DataBackgroundI3EnergyPDF(I3EnergyPDF, IsBackgroundPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    experimental data. This class is derived from I3EnergyPDF.
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
                The logarithm of the reconstructed energy value of the data
                event.
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
        # For experimental data, the MC and physics weight are unity.
        data_mcweight = np.ones((data_exp.size,))
        data_physicsweight = data_mcweight

        # Create the PDF using the base class.
        super(DataBackgroundI3EnergyPDF, self).__init__(
            data_logE, data_sinDec, data_mcweight, data_physicsweight,
            logE_binning, sinDec_binning
        )

class MCBackgroundI3EnergyPDF(I3EnergyPDF, IsBackgroundPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    monte-carlo data. This class is derived from I3EnergyPDF.
    """
    def __init__(self, data_mc, physics_weight_field_names, logE_binning, sinDec_binning):
        """Constructs a new IceCube energy background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The array holding the monte-carlo data. The following data fields
            must exist:
            'log_energy' : float
                The logarithm of the reconstructed energy value of the data
                event.
            'dec' : float
                The declination of the data event.
            'mcweight': float
                The monte-carlo weight of the event.
        physics_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as physics event weights. If a list is given, the
            weight values of all the fields will be summed to construct the
            final event physics weight.
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
        data_mcweight = data_mc['mcweight']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_physicsweight = np.zeros(data_mc.size, dtype=np.float64)
        for name in physics_weight_field_names:
            if(name not in data_mc.dtype.names):
                raise KeyError('The field "%s" does not exist in the MC data!'%(name))
            data_physicsweight += data_mc[name]

        # Create the PDF using the base class.
        super(MCBackgroundI3EnergyPDF, self).__init__(
            data_logE, data_sinDec, data_mcweight, data_physicsweight,
            logE_binning, sinDec_binning
        )
