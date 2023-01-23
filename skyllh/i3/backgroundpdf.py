# -*- coding: utf-8 -*-

import numpy as np

import scipy.interpolate

from skyllh.core.binning import (
    BinningDefinition,
    UsesBinning
)
from skyllh.core.pdf import (
    EnergyPDF,
    IsBackgroundPDF,
    SpatialPDF
)
from skyllh.core.py import issequenceof
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.timing import TaskTimer
from skyllh.i3.pdf import I3EnergyPDF


class BackgroundI3SpatialPDF(SpatialPDF, UsesBinning, IsBackgroundPDF):
    """This is the base class for all IceCube specific spatial background PDF
    models. IceCube spatial background PDFs depend solely on the zenith angle,
    and hence, on the declination of the event.

    The IceCube spatial background PDF is modeled as a 1d spline function in
    sin(declination).
    """
    def __init__(self, data_sinDec, data_weights, sinDec_binning,
                 spline_order_sinDec):
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
        super(BackgroundI3SpatialPDF, self).__init__(
            ra_range=(0, 2*np.pi),
            dec_range=(np.arcsin(sinDec_binning.lower_edge),
                       np.arcsin(sinDec_binning.upper_edge)))

        self.add_binning(sinDec_binning, 'sin_dec')
        self.spline_order_sinDec = spline_order_sinDec

        (h, bins) = np.histogram(data_sinDec,
                                 bins = sinDec_binning.binedges,
                                 weights = data_weights,
                                 range = sinDec_binning.range)

        # Save original histogram.
        self._orig_hist = h

        # Normalize histogram to get PDF.
        h = h / h.sum() / (bins[1:] - bins[:-1])

        # Check if there are any NaN values.
        if(np.any(np.isnan(h))):
            raise ValueError('The declination histogram contains NaN values! Check your sin(dec) binning! The bins with NaN values are: {0}'.format(sinDec_binning.bincenters[np.isnan(h)]))

        if(np.any(h <= 0.)):
            raise ValueError('Some declination histogram bins for the spatial background PDF are empty, this must not happen! The empty bins are: {0}'.format(sinDec_binning.bincenters[h <= 0.]))

        # Create the logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sinDec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

        # Save original spline.
        self._orig_log_spline = self._log_spline

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

    def add_events(self, events):
        """Add events to spatial background PDF object and recalculate
        logarithmic spline function.

        Parameters
        ----------
        events : numpy record ndarray
            The array holding the event data. The following data fields must
            exist:

            - 'sin_dec' : float
                The sin(declination) value of the event.
        """
        data = events['sin_dec']

        sinDec_binning = self.get_binning('sin_dec')

        (h_upd, bins) = np.histogram(data,
                         bins = sinDec_binning.binedges,
                         range = sinDec_binning.range)

        # Construct histogram with added events.
        h = self._orig_hist + h_upd

        # Normalize histogram to get PDF.
        h = h / h.sum() / (bins[1:] - bins[:-1])

        # Create the updated logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sinDec_binning.bincenters, np.log(h), k=self.spline_order_sinDec)

    def reset(self):
        """Reset the logarithmic spline to the original function, which was
        calculated when the object was initialized.
        """
        self._log_spline = self._orig_log_spline

    def get_prob(self, tdm, fitparams=None, tl=None):
        """Calculates the spatial background probability on the sphere of each
        event.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            to calculate the PDF values. The following data fields must exist:

            - 'sin_dec' : float
                The sin(declination) value of the event.
        fitparams : None
            Unused interface parameter.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : 1d ndarray
            The spherical background probability of each data event.
        """
        with TaskTimer(tl, 'Evaluating bkg log-spline.'):
            log_spline_val = self._log_spline(tdm.get_data('sin_dec'))

        prob = 0.5 / np.pi * np.exp(log_spline_val)

        grads = np.array([], dtype=np.float64)

        return (prob, grads)


class DataBackgroundI3SpatialPDF(BackgroundI3SpatialPDF):
    """This is the IceCube spatial background PDF, which gets constructed from
    experimental data.
    """
    def __init__(self, data_exp, sinDec_binning, spline_order_sinDec=2):
        """Constructs a new IceCube spatial background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the experimental data.
            The following data fields must exist:

            - 'dec' : float
                The declination of the data event.

        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
            The default is 2.
        """
        if(not isinstance(data_exp, DataFieldRecordArray)):
            raise TypeError('The data_exp argument must be of type '
                'numpy.ndarray!')

        data_sinDec = np.sin(data_exp['dec'])
        data_weights = np.ones((len(data_exp),))

        # Create the PDF using the base class.
        super(DataBackgroundI3SpatialPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec)


class MCBackgroundI3SpatialPDF(BackgroundI3SpatialPDF):
    """This is the IceCube spatial background PDF, which gets constructed from
    monte-carlo data.
    """
    def __init__(self, data_mc, physics_weight_field_names, sinDec_binning,
                 spline_order_sinDec=2):
        """Constructs a new IceCube spatial background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

            - 'sin_dec' : float
                The sine of the reconstructed declination of the data event.

        physics_weight_field_names : str | list of str
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
        if(not isinstance(data_mc, DataFieldRecordArray)):
            raise TypeError('The data_mc argument must be and instance of '
                'DataFieldRecordArray!')

        if(isinstance(physics_weight_field_names, str)):
            physics_weight_field_names = [physics_weight_field_names]
        if(not issequenceof(physics_weight_field_names, str)):
            raise TypeError('The physics_weight_field_names argument must be '
                'of type str or a sequence of type str!')

        data_sinDec = data_mc['sin_dec']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_weights = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if(name not in data_mc.field_name_list):
                raise KeyError('The field "%s" does not exist in the MC '
                    'data!'%(name))
            data_weights += data_mc[name]

        # Create the PDF using the base class.
        super(MCBackgroundI3SpatialPDF, self).__init__(
            data_sinDec, data_weights, sinDec_binning, spline_order_sinDec
        )


class DataBackgroundI3EnergyPDF(I3EnergyPDF, IsBackgroundPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    experimental data. This class is derived from I3EnergyPDF.
    """
    def __init__(self, data_exp, logE_binning, sinDec_binning,
                 smoothing_filter=None):
        """Constructs a new IceCube energy background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The array holding the experimental data. The following data fields
            must exist:

            - 'log_energy' : float
                The logarithm of the reconstructed energy value of the data
                event.
            - 'sin_dec' : float
                The sine of the reconstructed declination of the data event.

        logE_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        smoothing_filter : SmoothingFilter instance | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if(not isinstance(data_exp, DataFieldRecordArray)):
            raise TypeError('The data_exp argument must be an instance of '
                'DataFieldRecordArray!')

        data_logE = data_exp['log_energy']
        data_sinDec = data_exp['sin_dec']
        # For experimental data, the MC and physics weight are unity.
        data_mcweight = np.ones((len(data_exp),))
        data_physicsweight = data_mcweight

        # Create the PDF using the base class.
        super(DataBackgroundI3EnergyPDF, self).__init__(
            data_logE, data_sinDec, data_mcweight, data_physicsweight,
            logE_binning, sinDec_binning, smoothing_filter
        )
        # Check if this PDF is valid for all the given experimental data.
        self.assert_is_valid_for_exp_data(data_exp)


class MCBackgroundI3EnergyPDF(I3EnergyPDF, IsBackgroundPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    monte-carlo data. This class is derived from I3EnergyPDF.
    """
    def __init__(self, data_mc, physics_weight_field_names, logE_binning,
                 sinDec_binning, smoothing_filter=None):
        """Constructs a new IceCube energy background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

            - 'log_energy' : float
                The logarithm of the reconstructed energy value of the data
                event.
            - 'sin_dec' : float
                The sine of the reconstructed declination of the data event.
            - 'mcweight': float
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
        smoothing_filter : SmoothingFilter instance | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if(not isinstance(data_mc, DataFieldRecordArray)):
            raise TypeError('The data_mc argument must be an instance of '
                'DataFieldRecordArray!')

        if(isinstance(physics_weight_field_names, str)):
            physics_weight_field_names = [physics_weight_field_names]
        if(not issequenceof(physics_weight_field_names, str)):
            raise TypeError('The physics_weight_field_names argument must be '
                'of type str or a sequence of type str!')

        data_logE = data_mc['log_energy']
        data_sinDec = data_mc['sin_dec']
        data_mcweight = data_mc['mcweight']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_physicsweight = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if(name not in data_mc.field_name_list):
                raise KeyError('The field "%s" does not exist in the MC '
                    'data!'%(name))
            data_physicsweight += data_mc[name]

        # Create the PDF using the base class.
        super(MCBackgroundI3EnergyPDF, self).__init__(
            data_logE, data_sinDec, data_mcweight, data_physicsweight,
            logE_binning, sinDec_binning, smoothing_filter
        )
