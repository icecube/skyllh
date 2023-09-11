# -*- coding: utf-8 -*-

import scipy.interpolate

import numpy as np

from skyllh.core.binning import (
    UsesBinning,
)
from skyllh.core.pdf import (
    IsBackgroundPDF,
    SpatialPDF,
)
from skyllh.core.py import (
    classname,
    int_cast,
    issequence,
    issequenceof,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.i3.pdf import (
    I3EnergyPDF,
)


class BackgroundI3SpatialPDF(
        SpatialPDF,
        UsesBinning,
        IsBackgroundPDF,
):
    """This is the base class for all IceCube specific spatial background PDF
    models. IceCube spatial background PDFs depend solely on the zenith angle,
    and hence, on the declination of the event.

    The IceCube spatial background PDF is modeled as a 1d spline function in
    sin(declination).
    """
    def __init__(
            self,
            data_sin_dec,
            data_weights,
            sin_dec_binning,
            spline_order_sin_dec,
            **kwargs,
    ):
        """Creates a new IceCube spatial background PDF object.

        Parameters
        ----------
        data_sin_dec : 1d ndarray
            The array holding the sin(dec) values of the events.
        data_weights : 1d ndarray
            The array holding the weight of each event used for histogramming.
        sin_dec_binning : BinningDefinition
            The binning definition for the sin(declination) axis.
        spline_order_sin_dec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
        """
        super().__init__(
            pmm=None,
            ra_range=(0, 2*np.pi),
            dec_range=(
                np.arcsin(sin_dec_binning.lower_edge),
                np.arcsin(sin_dec_binning.upper_edge)),
            **kwargs)

        self.add_binning(sin_dec_binning, 'sin_dec')
        self.spline_order_sin_dec = spline_order_sin_dec

        (h, bins) = np.histogram(
            data_sin_dec,
            bins=sin_dec_binning.binedges,
            weights=data_weights,
            range=sin_dec_binning.range)

        # Save original histogram.
        self._orig_hist = h

        # Normalize histogram to get PDF.
        h = h / h.sum() / (bins[1:] - bins[:-1])

        # Check if there are any NaN values.
        if np.any(np.isnan(h)):
            nan_bcs = sin_dec_binning.bincenters[np.isnan(h)]
            raise ValueError(
                'The declination histogram contains NaN values! Check your '
                'sin(dec) binning! The bins with NaN values are: '
                f'{nan_bcs}')

        if np.any(h <= 0.):
            empty_bcs = sin_dec_binning.bincenters[h <= 0.]
            raise ValueError(
                'Some declination histogram bins for the spatial background '
                'PDF are empty, this must not happen! The empty bins are: '
                f'{empty_bcs}')

        # Create the logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sin_dec_binning.bincenters, np.log(h), k=self.spline_order_sin_dec)

        # Save original spline.
        self._orig_log_spline = self._log_spline

    @property
    def spline_order_sin_dec(self):
        """The order (int) of the logarithmic spline function, that splines the
        background PDF, along the sin(dec) axis.
        """
        return self._spline_order_sin_dec

    @spline_order_sin_dec.setter
    def spline_order_sin_dec(self, order):
        self._spline_order_sin_dec = int_cast(
            order,
            'The spline_order_sin_dec property must be castable to type int!')

    def add_events(self, events):
        """Add events to spatial background PDF object and recalculate
        logarithmic spline function.

        Parameters
        ----------
        events : numpy record ndarray
            The array holding the event data. The following data fields must
            exist:

                sin_dec : float
                    The sin(declination) value of the event.

        """
        data_sin_dec = events['sin_dec']

        sin_dec_binning = self.get_binning('sin_dec')

        (h_upd, bins) = np.histogram(
            data_sin_dec,
            bins=sin_dec_binning.binedges,
            range=sin_dec_binning.range)

        # Construct histogram with added events.
        h = self._orig_hist + h_upd

        # Normalize histogram to get PDF.
        h = h / h.sum() / (bins[1:] - bins[:-1])

        # Create the updated logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sin_dec_binning.bincenters, np.log(h), k=self.spline_order_sin_dec)

    def reset(self):
        """Reset the logarithmic spline to the original function, which was
        calculated when the object was initialized.
        """
        self._log_spline = self._orig_log_spline

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Pre-cumputes the probability density values when new trial data is
        available.
        """
        with TaskTimer(tl, 'Evaluating bkg log-spline.'):
            log_spline_val = self._log_spline(tdm.get_data('sin_dec'))

        self._pd = 0.5 / np.pi * np.exp(log_spline_val)

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the spatial background probability on the sphere of each
        event.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            to calculate the PDF values. The following data fields must exist:

                sin_dec : float
                    The sin(declination) value of the event.

        params_recarray : None
            Unused interface parameter.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_events,)-shaped numpy ndarray holding the background
            probability density value for each event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter.
            The background PDF does not depend on any global fit parameter,
            hence, this is an empty dictionary.
        """
        return (self._pd, dict())


class DataBackgroundI3SpatialPDF(
        BackgroundI3SpatialPDF,
):
    """This is the IceCube spatial background PDF, which gets constructed from
    experimental data.
    """
    def __init__(
            self,
            data_exp,
            sin_dec_binning,
            spline_order_sin_dec=2,
            **kwargs,
    ):
        """Constructs a new IceCube spatial background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the experimental data.
            The following data fields must exist:

                sin_dec : float
                    The sin(declination) of the data event.

        sin_dec_binning : BinningDefinition
            The binning definition for the sin(declination).
        spline_order_sin_dec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
            The default is 2.
        """
        if not isinstance(data_exp, DataFieldRecordArray):
            raise TypeError(
                'The data_exp argument must be an instance of '
                'DataFieldRecordArray! '
                f'It is of type "{classname(data_exp)}"!')

        data_sin_dec = data_exp['sin_dec']
        data_weights = np.ones((len(data_exp),))

        # Create the PDF using the base class.
        super().__init__(
            data_sin_dec=data_sin_dec,
            data_weights=data_weights,
            sin_dec_binning=sin_dec_binning,
            spline_order_sin_dec=spline_order_sin_dec,
            **kwargs)


class MCBackgroundI3SpatialPDF(
        BackgroundI3SpatialPDF,
):
    """This is the IceCube spatial background PDF, which gets constructed from
    monte-carlo data.
    """
    def __init__(
            self,
            data_mc,
            physics_weight_field_names,
            sin_dec_binning,
            spline_order_sin_dec=2,
            **kwargs,
    ):
        """Constructs a new IceCube spatial background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

                sin_dec : float
                    The sine of the reconstructed declination of the data event.

        physics_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as event weights. If a list is given, the weight
            values of all the fields will be summed to construct the final event
            weight.
        sin_dec_binning : BinningDefinition
            The binning definition for the sin(declination).
        spline_order_sin_dec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(dec) axis.
            The default is 2.
        """
        if not isinstance(data_mc, DataFieldRecordArray):
            raise TypeError(
                'The data_mc argument must be and instance of '
                'DataFieldRecordArray! '
                f'It is of type {classname(data_mc)}')

        if not issequence(physics_weight_field_names):
            physics_weight_field_names = [physics_weight_field_names]
        if not issequenceof(physics_weight_field_names, str):
            raise TypeError(
                'The physics_weight_field_names argument must be of type str '
                'or a sequence of type str! It is of type '
                f'"{classname(physics_weight_field_names)}"!')

        data_sin_dec = data_mc['sin_dec']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_weights = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if name not in data_mc:
                raise KeyError(
                    f'The field "{name}" does not exist in the MC data!')
            data_weights += data_mc[name]

        # Create the PDF using the base class.
        super().__init__(
            data_sin_dec=data_sin_dec,
            data_weights=data_weights,
            sin_dec_binning=sin_dec_binning,
            spline_order_sin_dec=spline_order_sin_dec,
            **kwargs)


class DataBackgroundI3EnergyPDF(
        I3EnergyPDF,
        IsBackgroundPDF,
):
    """This is the IceCube energy background PDF, which gets constructed from
    experimental data. This class is derived from I3EnergyPDF.
    """
    def __init__(
            self,
            data_exp,
            log10_energy_binning,
            sin_dec_binning,
            smoothing_filter=None,
            **kwargs,
    ):
        """Constructs a new IceCube energy background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The array holding the experimental data. The following data fields
            must exist:

                log_energy : float
                    The logarithm of the reconstructed energy value of the data
                    event.
                sin_dec : float
                    The sine of the reconstructed declination of the data event.

        log10_energy_binning : instance of BinningDefinition
            The binning definition for the binning in log10(E).
        sin_dec_binning : instance of BinningDefinition
            The binning definition for the sin(declination).
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if not isinstance(data_exp, DataFieldRecordArray):
            raise TypeError(
                'The data_exp argument must be an instance of '
                'DataFieldRecordArray! '
                f'It is of type "{classname(data_exp)}"!')

        data_log10_energy = data_exp['log_energy']
        data_sin_dec = data_exp['sin_dec']
        # For experimental data, the MC and physics weight are unity.
        data_mcweight = np.ones((len(data_exp),))
        data_physicsweight = data_mcweight

        # Create the PDF using the base class.
        super().__init__(
            pmm=None,
            data_log10_energy=data_log10_energy,
            data_sin_dec=data_sin_dec,
            data_mcweight=data_mcweight,
            data_physicsweight=data_physicsweight,
            log10_energy_binning=log10_energy_binning,
            sin_dec_binning=sin_dec_binning,
            smoothing_filter=smoothing_filter,
            **kwargs)


class MCBackgroundI3EnergyPDF(
        I3EnergyPDF,
        IsBackgroundPDF,
):
    """This is the IceCube energy background PDF, which gets constructed from
    monte-carlo data. This class is derived from I3EnergyPDF.
    """
    def __init__(
            self,
            data_mc,
            physics_weight_field_names,
            log10_energy_binning,
            sin_dec_binning,
            smoothing_filter=None,
            **kwargs,
    ):
        """Constructs a new IceCube energy background PDF from monte-carlo
        data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

                log_energy : float
                    The logarithm of the reconstructed energy value of the data
                    event.
                sin_dec : float
                    The sine of the reconstructed declination of the data event.
                mcweight: float
                    The monte-carlo weight of the event.

        physics_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as physics event weights. If a list is given, the
            weight values of all the fields will be summed to construct the
            final event physics weight.
        log10_energy_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sin_dec_binning : BinningDefinition
            The binning definition for the sin(declination).
        smoothing_filter : SmoothingFilter instance | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if not isinstance(data_mc, DataFieldRecordArray):
            raise TypeError(
                'The data_mc argument must be an instance of '
                'DataFieldRecordArray! '
                f'It is of type "{classname(data_mc)}"!')

        if not issequence(physics_weight_field_names):
            physics_weight_field_names = [physics_weight_field_names]
        if not issequenceof(physics_weight_field_names, str):
            raise TypeError(
                'The physics_weight_field_names argument must be '
                'of type str or a sequence of type str! '
                f'It is of type {classname(physics_weight_field_names)}')

        data_log10_energy = data_mc['log_energy']
        data_sin_dec = data_mc['sin_dec']
        data_mcweight = data_mc['mcweight']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_physicsweight = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if name not in data_mc:
                raise KeyError(
                    f'The field "{name}" does not exist in the MC data!')
            data_physicsweight += data_mc[name]

        # Create the PDF using the base class.
        super().__init__(
            pmm=None,
            data_log10_energy=data_log10_energy,
            data_sin_dec=data_sin_dec,
            data_mcweight=data_mcweight,
            data_physicsweight=data_physicsweight,
            log10_energy_binning=log10_energy_binning,
            sin_dec_binning=sin_dec_binning,
            smoothing_filter=smoothing_filter,
            **kwargs)
