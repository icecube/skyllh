# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains background PDF classes for the
likelihood function.
"""

import scipy.interpolate

import numpy as np

from skyllh.core.binning import (
    UsesBinning,
)
from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF,
    SingleConditionalEnergyPDF,
    SpatialPDF,
    TimePDF,
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


class BackgroundAltUniformAziSpatialPDF(
        SpatialPDF,
        UsesBinning,
        IsBackgroundPDF,
):
    """This is the base class for a spatial background PDF in the
    altitude-azimuth coordinate system. It is modeled as a 1d spline function in
    sin(reco_altitude). The reco_azimuth dimension is assumed to be uniform.
    """

    def __init__(
            self,
            data_sin_alt,
            data_weights,
            sin_alt_binning,
            spline_order_sin_alt,
            **kwargs,
    ):
        """Creates a new background spatial PDF in the altitude-azimuth
        coordinate system.

        Parameters
        ----------
        data_sin_alt : instance of numpy.ndarray
            The 1d numpy.ndarray holding the sin(reco_altitude) values of the
            events.
        data_weights : instance of numpy.ndarray
            The 1d numpy.ndarray holding the weight of each event used for
            histogramming.
        sin_alt_binning : instance of BinningDefinition
            The binning definition for the sin(reco_altitude) axis.
        spline_order_sin_alt : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(reco_altitude) axis.
        """
        super().__init__(
            pmm=None,
            dim1_name='alt',
            dim1_range=(
                np.arcsin(sin_alt_binning.lower_edge),
                np.arcsin(sin_alt_binning.upper_edge),
            ),
            dim2_name='azi',
            dim2_range=(0, 2*np.pi),
            **kwargs)

        self.add_binning(sin_alt_binning)
        self.spline_order_sin_alt = spline_order_sin_alt

        (h, bins) = np.histogram(
            data_sin_alt,
            bins=sin_alt_binning.binedges,
            weights=data_weights,
            range=sin_alt_binning.range)

        # Normalize histogram to get PDF.
        h = h / h.sum() / (bins[1:] - bins[:-1])

        # Check if there are any NaN values.
        if np.any(np.isnan(h)):
            nan_bcs = sin_alt_binning.bincenters[np.isnan(h)]
            raise ValueError(
                'The altitude histogram contains NaN values! Check your '
                'sin(reco_altitude) binning! The bins with NaN values are: '
                f'{nan_bcs}')

        if np.any(h <= 0.):
            empty_bcs = sin_alt_binning.bincenters[h <= 0.]
            raise ValueError(
                'Some altitude histogram bins for the spatial background '
                'PDF are empty, this must not happen! The empty bins are: '
                f'{empty_bcs}')

        # Create the logarithmic spline.
        self._log_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            sin_alt_binning.bincenters, np.log(h), k=self.spline_order_sin_alt)

    @property
    def spline_order_sin_alt(self):
        """The order (int) of the logarithmic spline function, that splines the
        background PDF, along the sin(reco_altitude) axis.
        """
        return self._spline_order_sin_alt

    @spline_order_sin_alt.setter
    def spline_order_sin_alt(self, order):
        self._spline_order_sin_alt = int_cast(
            order,
            'The spline_order_sin_alt property must be castable to type int!')

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """Pre-cumputes the probability density values when new trial data is
        available. The following data fields need to be present in the trial
        data:

            sin_alt : float
                The sin(reco_altitude) value of the trial data event.

        """
        with TaskTimer(tl, 'Evaluating bkg log-spline.'):
            log_spline_val = self._log_spline(tdm.get_data('sin_alt'))

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

                sin_alt : float
                    The sin(reco_altitude) value of the trial data event.

        params_recarray : None
            Unused interface parameter.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_events,)-shaped numpy ndarray holding the background
            probability density value for each trial data event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter.
            The background PDF does not depend on any global fit parameter,
            hence, this is an empty dictionary.
        """
        return (self._pd, dict())


class DataBackgroundAltUniformAziSpatialPDF(
        BackgroundAltUniformAziSpatialPDF,
):
    """This class provides a spatial background PDF in the altitude-azimuth
    coordinate system, constructed from the experimental data.
    """

    def __init__(
            self,
            data_exp,
            sin_alt_binning,
            spline_order_sin_alt=2,
            **kwargs,
    ):
        """Constructs a new spatial background PDF in the altitude-azimuth
        coordinate system, constructed from the experimental data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the experimental data.
            The following data fields must exist:

                sin_alt : float
                    The sin(reco_altitude) of the trial data event.

        sin_alt_binning : instance of BinningDefinition
            The binning definition for the sin(reco_altitude).
        spline_order_sin_dec : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(reco_altitude) axis.
            The default is 2.
        """
        if not isinstance(data_exp, DataFieldRecordArray):
            raise TypeError(
                'The data_exp argument must be an instance of '
                'DataFieldRecordArray! '
                f'Its current type is "{classname(data_exp)}"!')

        data_sin_alt = data_exp['sin_alt']
        data_weights = None

        super().__init__(
            data_sin_alt=data_sin_alt,
            data_weights=data_weights,
            sin_alt_binning=sin_alt_binning,
            spline_order_sin_alt=spline_order_sin_alt,
            **kwargs)


class MCBackgroundAltUniformAziSpatialPDF(
        BackgroundAltUniformAziSpatialPDF,
):
    """This class provides a spatial background PDF in the altitude-azimuth
    coordinate system, constructed from the monte-carlo data.
    """

    def __init__(
            self,
            data_mc,
            physics_weight_field_names,
            sin_alt_binning,
            spline_order_sin_alt=2,
            **kwargs,
    ):
        """Constructs a new spatial background PDF in the altitude-azimuth
        coordinate system, using the monte-carlo data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

                sin_alt : float
                    The sin(reco_altitude) value of the trial data event.

        physics_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as event weights. If a list is given, the weight
            values of all the fields will be summed to construct the final event
            weight.
        sin_alt_binning : instance of BinningDefinition
            The binning definition for the sin(reco_altitude).
        spline_order_sin_alt : int
            The order of the spline function for the logarithmic values of the
            spatial background PDF along the sin(reco_altitude) axis.
            The default is 2.
        """
        if not isinstance(data_mc, DataFieldRecordArray):
            raise TypeError(
                'The data_mc argument must be and instance of '
                'DataFieldRecordArray! '
                f'Its current type is {classname(data_mc)}')

        if not issequence(physics_weight_field_names):
            physics_weight_field_names = [physics_weight_field_names]
        if not issequenceof(physics_weight_field_names, str):
            raise TypeError(
                'The physics_weight_field_names argument must be of type str '
                'or a sequence of type str! Its current type is '
                f'"{classname(physics_weight_field_names)}"!')

        data_sin_alt = data_mc['sin_alt']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_weights = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if name not in data_mc:
                raise KeyError(
                    f'The field "{name}" does not exist in the MC data!')
            data_weights += data_mc[name]

        super().__init__(
            data_sin_alt=data_sin_alt,
            data_weights=data_weights,
            sin_alt_binning=sin_alt_binning,
            spline_order_sin_alt=spline_order_sin_alt,
            **kwargs)


class DataBackgroundSinAltEnergyPDF(
        SingleConditionalEnergyPDF,
        IsBackgroundPDF,
):
    """This class provides an energy background PDF in log10(E_reco) conditional
    on sin(altitude_reco), constructed from experimental data.
    """

    def __init__(
            self,
            data_exp,
            log10_energy_binning,
            sin_alt_binning,
            smoothing_filter=None,
            **kwargs,
    ):
        """Constructs a new energy background PDF in log10(E_reco), conditional
        on sin(altitude_reco) from experimental data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The array holding the experimental data. The following data fields
            must exist:

                ``log10_energy_binning.name`` : float
                    The logarithm of the reconstructed energy value of the data
                    event.
                ``sin_alt_binning.name`` : float
                    The sine of the reconstructed altitude of the data event.

        log10_energy_binning : instance of BinningDefinition
            The binning definition for the binning in log10(E_reco). The name
            of this binning definition defines the field name in the
            experimental and trial data.
        sin_dec_binning : instance of BinningDefinition
            The binning definition for the sin(declination_reco). The name
            of this binning definition defines the field name in the
            experimental and trial data.
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If ``None``, no smoothing will be applied.
        """
        if not isinstance(data_exp, DataFieldRecordArray):
            raise TypeError(
                'The data_exp argument must be an instance of '
                'DataFieldRecordArray! '
                f'Its current type is "{classname(data_exp)}"!')

        data_log10_energy = data_exp[log10_energy_binning.name]
        data_sin_alt = data_exp[sin_alt_binning.name]

        # For experimental data, the MC and physics weight are unity and we can
        # use None.
        data_mcweight = None
        data_physicsweight = None

        super().__init__(
            pmm=None,
            data_log10_energy=data_log10_energy,
            data_param=data_sin_alt,
            data_mcweight=data_mcweight,
            data_physicsweight=data_physicsweight,
            log10_energy_binning=log10_energy_binning,
            param_binning=sin_alt_binning,
            smoothing_filter=smoothing_filter,
            **kwargs)


class MCBackgroundSinAltEnergyPDF(
        SingleConditionalEnergyPDF,
        IsBackgroundPDF,
):
    """This class provides an energy background PDF in log10(E_reco) conditional
    on sin(altitude_reco), constructed from monte-carlo data.
    """

    def __init__(
            self,
            data_mc,
            physics_weight_field_names,
            log10_energy_binning,
            sin_alt_binning,
            smoothing_filter=None,
            **kwargs,
    ):
        """Constructs a new energy background PDF in log10(E_reco), conditional
        on sin(altitude_reco) from monte-carlo data.

        Parameters
        ----------
        data_mc : instance of DataFieldRecordArray
            The array holding the monte-carlo data. The following data fields
            must exist:

                ``log10_energy_binning.name`` : float
                    The logarithm of the reconstructed energy value of the data
                    event.
                ``sin_alt_binning.name`` : float
                    The sine of the reconstructed altitude of the data event.
                mcweight: float
                    The monte-carlo weight of the event.

        physics_weight_field_names : str | list of str
            The name or the list of names of the monte-carlo data fields, which
            should be used as physics event weights. If a list is given, the
            weight values of all the fields will be summed to construct the
            final event physics weight.
        log10_energy_binning : instance of BinningDefinition
            The binning definition for the binning in log10(E_reco).
            The name of this binning definition defines the field name in the
            MC and trial data.
        sin_alt_binning : instance of BinningDefinition
            The binning definition for the sin(altitude_reco).
            The name of this binning definition defines the field name in the
            MC and trial data.
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if not isinstance(data_mc, DataFieldRecordArray):
            raise TypeError(
                'The data_mc argument must be an instance of '
                'DataFieldRecordArray! '
                f'Its current type is "{classname(data_mc)}"!')

        if not issequence(physics_weight_field_names):
            physics_weight_field_names = [physics_weight_field_names]
        if not issequenceof(physics_weight_field_names, str):
            raise TypeError(
                'The physics_weight_field_names argument must be '
                'of type str or a sequence of type str! Its current type is'
                f'"{classname(physics_weight_field_names)}"!')

        data_log10_energy = data_mc[log10_energy_binning.name]
        data_sin_alt = data_mc[sin_alt_binning.name]
        data_mcweight = data_mc['mcweight']

        # Calculate the event weights as the sum of all the given data fields
        # for each event.
        data_physicsweight = np.zeros(len(data_mc), dtype=np.float64)
        for name in physics_weight_field_names:
            if name not in data_mc:
                raise KeyError(
                    f'The field "{name}" does not exist in the MC data!')
            data_physicsweight += data_mc[name]

        super().__init__(
            pmm=None,
            data_log10_energy=data_log10_energy,
            data_param=data_sin_alt,
            data_mcweight=data_mcweight,
            data_physicsweight=data_physicsweight,
            log10_energy_binning=log10_energy_binning,
            param_binning=sin_alt_binning,
            smoothing_filter=smoothing_filter,
            **kwargs)


class BackgroundMultiDimGridPDF(
        MultiDimGridPDF,
        IsBackgroundPDF):
    """This class provides a multi-dimensional background PDF defined on a grid.
    The PDF is created from pre-calculated PDF data on a grid. The grid data is
    interpolated using a :class:`scipy.interpolate.RegularGridInterpolator`
    instance.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new :class:`~skyllh.core.pdf.MultiDimGridPDF` instance that
        is also derived from :class:`~skyllh.core.pdf.IsBackgroundPDF`.

        For the documentation of arguments see the documentation of the
        :meth:`~skyllh.core.pdf.MultiDimGridPDF.__init__` method.
        """
        super().__init__(*args, **kwargs)


class BackgroundTimePDF(
        TimePDF,
        IsBackgroundPDF):
    """This class provides a background time PDF class.
    """

    def __init__(
            self,
            livetime,
            time_flux_profile,
            **kwargs):
        """Creates a new signal time PDF instance for a given time flux profile
        and detector live time.

        Parameters
        ----------
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_flux_profile : instance of TimeFluxProfile
            The signal's time flux profile.
        """
        super().__init__(
            pmm=None,
            livetime=livetime,
            time_flux_profile=time_flux_profile,
            **kwargs)

        self._pd = None

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Initializes the background time PDF with new trial data. Because this
        PDF does not depend on any parameters, the probability density values
        can be pre-computed here.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            ``'time'`` : float
                The MJD time of the event.

        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.
        """
        times = tdm.get_data('time')

        self._pd = np.zeros((len(times),), dtype=np.float64)

        # Get a mask of the event times which fall inside a detector on-time
        # interval.
        on = self._livetime.is_on(times)

        self._pd[on] = self._time_flux_profile(t=times[on]) / self._S

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """
        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            ``'time'`` : float
                The MJD time of the event.

        params_recarray : None
            Unused interface argument.
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
        if self._pd is None:
            raise RuntimeError(
                f'The {classname(self)} was not initialized with trial data!')

        grads = dict()

        return (self._pd, grads)
