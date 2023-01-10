# -*- coding: utf-8 -*-

"""The ``signalpdf`` module contains possible signal PDF models for the
likelihood function.
"""

import numpy as np
import scipy as scp

from skyllh.core import display
from skyllh.core.py import (
    classname,
    issequenceof,
    str_cast,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.pdf import (
    PDFAxis,
    IsSignalPDF,
    MultiDimGridPDF,
    MultiDimGridPDFSet,
    MappedMultiDimGridPDFSet,
    NDPhotosplinePDF,
    SpatialPDF,
    TimePDF,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.utils.coords import (
    angular_separation,
)
from skyllh.physics.source_model import (
    PointLikeSource,
)
from skyllh.physics.flux_model import (
    TimeFluxProfile,
)


class GaussianPSFPointLikeSourceSignalSpatialPDF(SpatialPDF, IsSignalPDF):
    r"""This spatial signal PDF model describes the spatial PDF for a point
    source smeared with a 2D gaussian point-spread-function (PSF).
    Mathematically, it's the convolution of a point in the sky, i.e. the source
    location, with the PSF. The result of this convolution has the gaussian form

    .. math::

        \frac{1}{2\pi\sigma^2} \exp(-\frac{r^2}{2\sigma^2}),

    where :math:`\sigma` is the spatial uncertainty of the event and :math:`r`
    the distance on the sphere between the source and the data event.

    This PDF requires the `src_array` data field, that is numpy record ndarray
    with the data fields `ra` and `dec` holding the right-ascention and
    declination of the point-like sources, respectively.
    """

    def __init__(
            self,
            ra_range=None,
            dec_range=None,
            pd_event_data_field_name=None,
            **kwargs):
        """Creates a new spatial signal PDF for point-like sources with a
        gaussian point-spread-function (PSF).

        Parameters
        ----------
        ra_range : 2-element tuple | None
            The range in right-ascention this spatial PDF is valid for.
            If set to None, the range (0, 2pi) is used.
        dec_range : 2-element tuple | None
            The range in declination this spatial PDF is valid for.
            If set to None, the range (-pi/2, +pi/2) is used.
        pd_event_data_field_name : str | None
            The probability density values can be pre-calculated by the user.
            This specifies the name of the event data field, where these values
            are stored.
        """
        if ra_range is None:
            ra_range = (0, 2*np.pi)
        if dec_range is None:
            dec_range = (-np.pi/2, np.pi/2)

        super().__init__(
            ra_range=ra_range,
            dec_range=dec_range,
            **kwargs)

        self.pd_event_data_field_name = pd_event_data_field_name

    @property
    def pd_event_data_field_name(self):
        """The event data field name where pre-calculated probability density
        values are stored.
        """
        return self._pd_event_data_field_name

    @pd_event_data_field_name.setter
    def pd_event_data_field_name(self, name):
        name = str_cast(
            name,
            'The pd_event_data_field_name property must be castable to type '
            f'str! Its current type is {classname(name)}!')
        self._pd_event_data_field_name = name

    def get_pd(self, tdm, params_recarray=None, tl=None):
        """Calculates the spatial signal probability density of each event for
        all sources.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF values. The following data fields need to
            be present:

            src_array : numpy record ndarray
                The numpy record ndarray with the following data fields:

                ra : float
                    The right-ascention of the point-like source.
                dec : float
                    The declination of the point-like source.

            ra : float
                The right-ascention in radian of the data event.
            dec : float
                The declination in radian of the data event.
            ang_err: float
                The reconstruction uncertainty in radian of the data event.

            In case the probability density values were pre-calculated,
        params_recarray : None
            Unused interface argument.
        tl : TimeLord instance | None
            The optional TimeLord instance to use for measuring timing
            information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_events,)-shaped numpy ndarray holding the probability density
            for each event. The length of this 1D array depends on the number
            of sources and the events belonging to those sources. In the worst
            case the length is N_sources * N_trial_events.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each fit parameter. By definition this PDF does not depend
            on any fit parameters and hence, this dictionary is empty.
        """
        get_data = tdm.get_data

        # Check if the probability density was pre-calculated.
        if self._pd_event_data_field_name in tdm:
            pd = get_data(self._pd_event_data_field_name)
            return (pd, dict())

        src_array = get_data('src_array')
        ra = get_data('ra')
        dec = get_data('dec')
        sigma = get_data('ang_err')

        src_evt_idxs = tdm.src_evt_idxs
        if src_evt_idxs is None:
            # Make the source position angles two-dimensional so the PDF value
            # can be calculated via numpy broadcasting automatically for several
            # sources.
            src_ra = src_array['ra'][:, np.newaxis]
            src_dec = src_array['dec'][:, np.newaxis]
        else:
            # Pick the event values based on the event selection.
            (src_idxs, evt_idxs) = src_evt_idxs
            src_ra = np.take(src_array['ra'], src_idxs)
            src_dec = np.take(src_array['dec'], src_idxs)

            dec = np.take(dec, evt_idxs)
            ra = np.take(ra, evt_idxs)
            sigma = np.take(sigma, evt_idxs)

        psi = angular_separation(src_ra, src_dec, ra, dec)

        pd = 0.5/(np.pi*sigma**2)*np.exp(-0.5*(psi/sigma)**2)

        # In case the src_evt_idxs was None, pd is a N_sources,N_events array,
        # which needs to be flatten.
        pd = pd.flatten()

        return (pd, dict())


class SignalTimePDF(TimePDF, IsSignalPDF):
    """This class provides a time PDF class for a signal source. It consists of
    a Livetime instance and a TimeProfileModel instance. Together they construct
    the actual signal time PDF, which has detector down-time taking into
    account.
    """

    def __init__(self, livetime, time_profile, **kwargs):
        """Creates a new signal time PDF instance for a given time profile of
        the source.

        Parameters
        ----------
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_profile : instance of TimeFluxProfile
            The time flux profile of the source.
        """
        super().__init__(**kwargs)

        self.livetime = livetime
        self.time_profile = time_profile

        # Define the time axis with the time boundaries of the live-time.
        self.add_axis(PDFAxis(
            name='time',
            vmin=self._livetime.time_window[0],
            vmax=self._livetime.time_window[1]))

        # Get the total integral, I, of the time profile and the sum, S, of the
        # integrals for each detector on-time interval during the time profile,
        # in order to be able to rescale the time profile to unity with
        # overlapping detector off-times removed.
        (self._I, self._S) = self._calculate_time_profile_I_and_S()

    @property
    def livetime(self):
        """The instance of Livetime, which provides the detector live-time
        information.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if(not isinstance(lt, Livetime)):
            raise TypeError(
                'The livetime property must be an instance of Livetime!')
        self._livetime = lt

    @property
    def time_profile(self):
        """The instance of TimeProfileModel providing the (assumed) physical
        time profile of the source.
        """
        return self._time_profile

    @time_profile.setter
    def time_profile(self, tp):
        if not isinstance(tp, TimeFluxProfile):
            raise TypeError(
                'The time_profile property must be an instance of '
                'TimeFluxProfile!')
        self._time_profile = tp

    def __str__(self):
        """Pretty string representation of the signal time PDF.
        """
        s = f'{classname(self)}(\n' +\
            ' '*display.INDENTATION_WIDTH +\
            f'livetime = {str(self._livetime)},\n' +\
            ' '*display.INDENTATION_WIDTH +\
            f'time_profile = {str(self._time_profile)}\n' +\
            ')'
        return s

    def _calculate_time_profile_I_and_S(self):
        """Calculates the total integral, I, of the time profile and the sum, A,
        of the time-profile integrals during the detector on-time intervals.

        Returns
        -------
        I : float
            The total integral of the source time-profile.
        S : float
            The sum of the source time-profile integrals during the detector
            on-time intervals.
        """
        ontime_intervals = self._livetime.get_ontime_intervals_between(
            self._time_profile.t_start, self._time_profile.t_end)
        I = self._time_profile.get_total_integral()
        S = np.sum(self._time_profile.get_integral(
            ontime_intervals[:, 0], ontime_intervals[:, 1]))
        return (I, S)

    def assert_is_valid_for_trial_data(self, tdm):
        """Checks if the time PDF is valid for all the given trial data.
        It checks if the time of all events is within the defined time axis of
        the PDF.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial data.
            The following data fields must exist:

            'time' : float
                The MJD time of the data event.

        Raises
        ------
        ValueError
            If some of the data is outside the time range of the PDF.
        """
        time_axis = self.axes['time']

        time = tdm.get_data('time')

        if np.any((time < time_axis.vmin) |
                  (time > time_axis.vmax)):
            raise ValueError(
                'Some data is outside the time range '
                f'({time_axis.vmin:.3f}, {time_axis.vmax:.3f})!')

    def get_pd(self, tdm, params_recarray, tl=None):
        """Calculates the signal time probability density of each event for the
        given
        set of signal time parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            - 'time' : float
                The MJD time of the event.
        params_recarray : instance of numpy record ndarray
            The numpy record ndarray holding the local parameter values for each
            source.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped 1D numpy ndarray holding the probability
            density value for each trial event and source.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each fit parameter.
        """
        n_sources = len(params_recarray)

        if tdm.src_evt_idxs is None:
            src_idxs = np.repeat(np.arange(n_sources), tdm.n_selected_events)
            evt_idxs = np.tile(np.arange(tdm.n_selected_events), n_sources)
            n_vals = tdm.n_selected_events * n_sources
        else:
            (src_idxs, evt_idxs) = tdm.src_evt_idxs
            n_vals = len(evt_idxs)

        pd = np.zeros((n_vals,), dtype=np.float64)

        events_time = tdm.get_data('time')
        for (src_idx, params_row) in enumerate(params_recarray):
            params = dict(zip(params_recarray.dtype.fields.keys(), params_row))

            # Update the time-profile if its parameter values have changed and
            # recalculate self._I and self._S if an update was actually
            # performed.
            updated = self._time_profile.set_params(params)
            if updated:
                (self._I, self._S) = self._calculate_time_profile_I_and_S()

            src_m = src_idxs == src_idx
            idxs = evt_idxs[src_m]

            times = events_time[idxs]

            # Get a mask of the event times which fall inside a detector on-time
            # interval.
            on = self._livetime.is_on(times)

            # The sum of the on-time integrals of the time profile, A, will be
            # zero if the time profile is entirly during detector off-time.
            if(self._S > 0):
                pd[src_m] = self._time_profile(times[on]) / (self._I * self._S)

        return (pd, dict())


class SignalMultiDimGridPDF(MultiDimGridPDF, IsSignalPDF):
    """This class provides a multi-dimensional signal PDF. The PDF is created
    from pre-calculated PDF data on a grid. The grid data is interpolated using
    a :class:`scipy.interpolate.RegularGridInterpolator` instance.
    """

    def __init__(self, axis_binnings, path_to_pdf_splinetable=None,
                 pdf_grid_data=None, norm_factor_func=None):
        """Creates a new signal PDF instance for a multi-dimensional PDF given
        as PDF values on a grid. The grid data is interpolated with a
        :class:`scipy.interpolate.RegularGridInterpolator` instance. As grid
        points the bin edges of the axis binning definitions are used.

        Parameters
        ----------
        axis_binnings : sequence of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each BinningDefinition instance defines
            the event field name that should be used for querying the PDF.
        path_to_pdf_splinetable : str
            The path to the file containing the spline table.
            The spline table contains a  pre-computed fit to pdf_grid_data.
        pdf_grid_data : n-dimensional numpy ndarray
            The n-dimensional numpy ndarray holding the PDF values at given grid
            points. The grid points must match the bin edges of the given
            BinningDefinition instances of the `axis_binnings` argument.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function
            must be `__call__(pdf, events, fitparams)`, where `pdf` is this PDF
            instance, `events` is a numpy record ndarray holding the events for
            which to calculate the PDF values, and `fitparams` is a dictionary
            with the current fit parameter names and values.
        """
        super(SignalMultiDimGridPDF, self).__init__(
            axis_binnings=axis_binnings,
            path_to_pdf_splinetable=path_to_pdf_splinetable,
            pdf_grid_data=pdf_grid_data,
            norm_factor_func=norm_factor_func)


class SignalMultiDimGridPDFSet(MultiDimGridPDFSet, IsSignalPDF):
    """This class extends the MultiDimGridPDFSet PDF class to be a signal PDF.
    See the documentation of the :class:`skyllh.core.pdf.MultiDimGridPDFSet`
    class for what this PDF provides.
    """

    def __init__(self, param_set, param_grid_set, gridparams_pdfs,
                 interpolmethod=None, **kwargs):
        """Creates a new SignalMultiDimGridPDFSet instance, which holds a set of
        MultiDimGridPDF instances, one for each point of a parameter grid set.

        Parameters
        ----------
        param_set : Parameter instance | sequence of Parameter instances |
                    ParameterSet instance
            The set of parameters defining the model parameters of this PDF.
        param_grid_set : ParameterGrid instance | ParameterGridSet instance
            The set of ParameterGrid instances, which define the grid values of
            the model parameters, the given MultiDimGridPDF instances belong to.
        gridparams_pdfs : sequence of (dict, MultiDimGridPDF) tuples
            The sequence of 2-element tuples which define the mapping of grid
            values to PDF instances.
        interpolmethod : subclass of GridManifoldInterpolationMethod
            The class specifying the interpolation method. This must be a
            subclass of ``GridManifoldInterpolationMethod``.
            If set to None, the default grid manifold interpolation method
            ``Linear1DGridManifoldInterpolationMethod`` will be used.
        """
        super(SignalMultiDimGridPDFSet, self).__init__(
            param_set=param_set,
            param_grid_set=param_grid_set,
            gridparams_pdfs=gridparams_pdfs,
            interpolmethod=interpolmethod,
            pdf_type=SignalMultiDimGridPDF,
            **kwargs)


class SignalMappedMultiDimGridPDFSet(MappedMultiDimGridPDFSet, IsSignalPDF):
    """This class extends the MappedMultiDimGridPDFSet PDF class to be a signal
    PDF. See the documentation of the
    :class:`skyllh.core.pdf.MappedMultiDimGridPDFSet` class for what this PDF
    provides.
    """

    def __init__(self, param_grid_set, gridparams_pdfs,
                 interpolmethod=None, **kwargs):
        """Creates a new SignalMappedMultiDimGridPDFSet instance, which holds a
        set of MultiDimGridPDF instances, one for each point of a parameter grid
        set.

        Parameters
        ----------
        param_grid_set : ParameterGrid instance | ParameterGridSet instance
            The set of ParameterGrid instances, which define the grid values of
            the model parameters, the given MultiDimGridPDF instances belong to.
        gridparams_pdfs : sequence of (dict, MultiDimGridPDF) tuples
            The sequence of 2-element tuples which define the mapping of grid
            values to PDF instances.
        """
        super(SignalMappedMultiDimGridPDFSet, self).__init__(
            param_grid_set=param_grid_set,
            gridparams_pdfs=gridparams_pdfs,
            pdf_type=SignalMultiDimGridPDF,
            **kwargs)


class SignalNDPhotosplinePDF(NDPhotosplinePDF, IsSignalPDF):
    """This class provides a multi-dimensional signal PDF created from a
    n-dimensional photospline fit. The photospline package is used to evaluate
    the PDF fit.
    """

    def __init__(
            self,
            axis_binnings,
            param_set,
            path_to_pdf_splinefit,
            norm_factor_func=None):
        """Creates a new signal PDF instance for a n-dimensional photospline PDF
        fit.

        Parameters
        ----------
        axis_binnings : BinningDefinition | sequence of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each BinningDefinition instance defines
            the event field name that should be used for querying the PDF.
        param_set : Parameter | ParameterSet
            The Parameter instance or ParameterSet instance defining the
            parameters of this PDF. The ParameterSet holds the information
            which parameters are fixed and which are floating (i.e. fitted).
        path_to_pdf_splinefit : str
            The path to the file containing the photospline fit.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function must be
            `__call__(pdf, tdm, params)`, where `pdf` is this PDF
            instance, `tdm` is an instance of TrialDataManager holding the
            event data for which to calculate the PDF values, and `params` is a
            dictionary with the current parameter names and values.
        """
        super(SignalNDPhotosplinePDF, self).__init__(
            axis_binnings=axis_binnings,
            param_set=param_set,
            path_to_pdf_splinefit=path_to_pdf_splinefit,
            norm_factor_func=norm_factor_func
        )
