# -*- coding: utf-8 -*-

"""The ``signalpdf`` module contains possible signal PDF models for the
likelihood function.
"""

import numpy as np

import scipy.stats

from skyllh.core.debugging import (
    get_logger,
    is_tracing_enabled,
)
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Linear1DGridManifoldInterpolationMethod,
)
from skyllh.core.pdf import (
    PDF,
    PDFSet,
    IsSignalPDF,
    MultiDimGridPDF,
    NDPhotosplinePDF,
    SpatialPDF,
    TimePDF,
)
from skyllh.core.py import (
    classname,
    str_cast,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.core.utils.coords import (
    angular_separation,
)


class GaussianPSFPointLikeSourceSignalSpatialPDF(
        SpatialPDF,
        IsSignalPDF):
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
            pmm=None,
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

    def calculate_pd(self, tdm):
        """Calculates the gaussian PSF probability density values for all events
        and sources.

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

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            for each event. The length of this 1D array depends on the number
            of sources and the events belonging to those sources. In the worst
            case the length is N_sources * N_trial_events.
        """
        get_data = tdm.get_data

        src_array = get_data('src_array')
        ra = get_data('ra')
        dec = get_data('dec')
        sigma = get_data('ang_err')

        (src_idxs, evt_idxs) = tdm.src_evt_idxs
        src_ra = np.take(src_array['ra'], src_idxs)
        src_dec = np.take(src_array['dec'], src_idxs)

        dec = np.take(dec, evt_idxs)
        ra = np.take(ra, evt_idxs)
        sigma_sq = np.take(sigma**2, evt_idxs)

        psi = angular_separation(src_ra, src_dec, ra, dec)

        pd = 0.5/(np.pi*sigma_sq) * np.exp(-0.5*(psi**2/sigma_sq))

        return pd

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
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
            The (N_values,)-shaped numpy ndarray holding the probability density
            for each event. The length of this 1D array depends on the number
            of sources and the events belonging to those sources. In the worst
            case the length is N_sources * N_trial_events.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each fit parameter. By definition this PDF does not depend
            on any fit parameters and hence, this dictionary is empty.
        """

        logger = get_logger(f'{__name__}.{classname(self)}.get_pd')

        # Check if the probability density was pre-calculated.
        if self._pd_event_data_field_name in tdm:
            if is_tracing_enabled():
                logger.debug(
                    'Retrieve precalculated probability density values from '
                    f'data field "{self._pd_event_data_field_name}"')
            pd = tdm[self._pd_event_data_field_name]
            return (pd, dict())

        pd = self.calculate_pd(tdm)

        return (pd, dict())


class RayleighPSFPointSourceSignalSpatialPDF(
        SpatialPDF,
        IsSignalPDF):
    r"""This spatial signal PDF model describes the spatial PDF for a point-like
    source following a Rayleigh distribution in the opening angle between the
    source and reconstructed muon direction.
    Mathematically, it's the convolution of a point in the sky, i.e. the source
    location, with the PSF. The result of this convolution has the following
    form

    .. math::

        1/(2\pi \sin\Psi) * \Psi/\sigma^2 \exp(-\Psi^2/(2\sigma^2)),

    where :math:`\sigma` is the spatial uncertainty of the event and
    :math:`\Psi` the distance on the sphere between the source and the data
    event.

    This PDF requires the ``src_array`` source data field, that is numpy
    structured ndarray with the data fields ``ra`` and ``dec`` holding the
    right-ascention and declination of the point-like sources, respectively.
    """
    def __init__(
            self,
            ra_range=None,
            dec_range=None,
            **kwargs):
        r"""Creates a new spatial signal PDF for point-like sources with a
        Rayleigh point-spread-function (PSF).

        Parameters
        ----------
        ra_range : 2-element tuple | None
            The range in right-ascention this spatial PDF is valid for.
            If set to None, the range (0, 2pi) is used.
        dec_range : 2-element tuple | None
            The range in declination this spatial PDF is valid for.
            If set to None, the range (-pi/2, +pi/2) is used.
        """
        if ra_range is None:
            ra_range = (0, 2*np.pi)
        if dec_range is None:
            dec_range = (-np.pi/2, np.pi/2)

        super().__init__(
            pmm=None,
            ra_range=ra_range,
            dec_range=dec_range,
            **kwargs
        )

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the spatial signal probability density of each event for
        all sources.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF values. The following data fields need to
            be present:

            psi : float
                The opening angle in radian between the source direction and the
                reconstructed muon direction.
            ang_err: float
                The reconstruction uncertainty in radian of the data event.

        params_recarray : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.

        Returns
        -------
        pd : (N_values,)-shaped numpy ndarray
            The (N_values,)-shaped 1D numpy ndarray holding the probability
            density value for each event and source in unit 1/rad.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter. By definition this PDF does not
            depend on any global fit parameters and hence, this dictionary is
            empty.
        """
        get_data = tdm.get_data

        (src_idxs, evt_idxs) = tdm.src_evt_idxs

        psi = get_data('psi')
        sigma = get_data('ang_err')
        sigma_sq = np.take(sigma**2, evt_idxs)

        pd = (
            0.5/(np.pi*np.sin(psi)) *
            (psi / sigma_sq) *
            np.exp(-0.5*(psi**2/sigma_sq))
        )

        grads = dict()

        return (pd, grads)


class SignalTimePDF(
        TimePDF,
        IsSignalPDF):
    """This class provides a signal time PDF class. It consists of
    a :class:`~skyllh.core.livetime.Livetime` instance and a
    :class:`~skyllh.physics.flux_model.TimeFluxProfile` instance. Together they
    construct the actual signal time PDF, which has detector down-time taking
    into account.
    """

    def __init__(
            self,
            pmm,
            livetime,
            time_flux_profile,
            **kwargs):
        """Creates a new signal time PDF instance for a given time flux profile
        and detector live time.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper defining the mapping of the
            global parameters to the local source parameters.
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_flux_profile : instance of TimeFluxProfile
            The signal's time flux profile.
        """
        super().__init__(
            pmm=pmm,
            livetime=livetime,
            time_flux_profile=time_flux_profile,
            **kwargs)

    def get_pd(
            self,
            tdm,
            src_params_recarray,
            tl=None):
        """Calculates the signal time probability density of each event for the
        given set of time parameter values for each source.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            ``'time'`` : float
                The MJD time of the event.

        src_params_recarray : instance of numpy structured ndarray
            The numpy structured ndarray holding the local parameter values for
            each source.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped 1D numpy ndarray holding the probability
            density value for each trial event and source.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter.
        """
        (src_idxs, evt_idxs) = tdm.src_evt_idxs
        n_values = len(evt_idxs)

        pd = np.zeros((n_values,), dtype=np.float64)

        events_time = tdm.get_data('time')
        for (src_idx, src_params_row) in enumerate(src_params_recarray):
            params = dict(zip(
                src_params_recarray.dtype.fields.keys(),
                src_params_row))

            # Update the time flux profile if its parameter values have changed
            # and recalculate self._I and self._S if an update was actually
            # performed.
            updated = self._time_flux_profile.set_params(params)
            if updated:
                (self._I, self._S) = self._calculate_time_flux_profile_integrals()

            src_m = src_idxs == src_idx
            idxs = evt_idxs[src_m]

            times = events_time[idxs]

            # Get a mask of the event times which fall inside a detector on-time
            # interval.
            on = self._livetime.is_on(times)

            pd_src = pd[src_m]
            pd_src[on] = (
                self._time_flux_profile(t=times[on]) * self._S / self._I**2
            )
            pd[src_m] = pd_src

        return (pd, dict())


class FixedGaussianSignalTimePDF(
        PDF,
        IsSignalPDF):
    """This class provides a gaussian time PDF, which is fixed to a given
    position in time and has a fixed width. It does not depend on any fit
    parameters.
    """

    def __init__(
            self,
            grl,
            mu,
            sigma,
            **kwargs):
        """Creates a new signal time PDF instance for a given time profile of
        the source.

        Parameters
        ----------
        grl : instance of structured ndarray
            The structured numpy ndarray of length N_runs, holding the detector
            good run list information. The following data fields need to be
            present:

            start : float
                The start time of the run.
            stop : float
                The stop time of the run.

        mu : float
            The mean of the gaussian flare.
        sigma : float
            The sigma of the gaussian flare.
        """
        super().__init__(
            pmm=None,
            **kwargs)

        self.mu = mu
        self.sigma = sigma
        self.grl = grl

    def calc_uptime_norm(self):
        """Calculates the normalization by taking the dataset uptime into
        account. Distributions like scipy.stats.norm are normalized (-inf, inf).
        These must be re-normalized such that the function sums to 1 over the
        finite good run list domain.

        Returns
        -------
        norm : float
            Normalization such that cdf sums to 1 over the good run list domain.
        """
        cdf = scipy.stats.norm(self.mu, self.sigma).cdf

        integral = np.sum(cdf(self.grl['stop']) - cdf(self.grl['start']))

        if np.isclose(integral, 0):
            return 0

        norm = 1 / integral

        return norm

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the signal time probability density of each event and
        source.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            time : float
                The MJD time of the event.

        params_recarray : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.

        Returns
        -------
        pd : (N_values,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability density for each event.
        grads : empty array of float
            Empty, since it does not depend on any fit parameter
        """
        evt_idxs = tdm.src_evt_idxs[1]

        time = np.take(tdm.get_data('time'), evt_idxs)

        uptime_norm = self.calc_uptime_norm()

        pd = scipy.stats.norm.pdf(time, self.mu, self.sigma) * uptime_norm

        grads = dict()

        return (pd, grads)


class SignalMultiDimGridPDF(
        MultiDimGridPDF,
        IsSignalPDF):
    """This class provides a multi-dimensional signal PDF. The PDF is created
    from pre-calculated PDF data on a grid. The grid data is interpolated using
    a :class:`scipy.interpolate.RegularGridInterpolator` instance.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new PDF instance for a multi-dimensional PDF given
        as PDF values on a grid or as PDF values stored in a photospline table.

        See the documentation of the
        :meth:`skyllh.core.pdf.MultiDimGridPDF.__init__` method for the
        documentation of possible arguments.
        """
        super().__init__(
            *args,
            **kwargs)


class SignalMultiDimGridPDFSet(
        PDF,
        PDFSet,
        IsSignalPDF):
    """This class provides a set of MultiDimGridPDF instances that implements
    also the PDF interface.
    """

    def __init__(
            self,
            pmm,
            param_set,
            param_grid_set,
            gridparams_pdfs,
            interpol_method_cls=None,
            **kwargs):
        """Creates a new MultiDimGridPDFSet instance, which holds a set of
        MultiDimGridPDF instances, one for each point of a parameter grid set.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper that defines the mapping of
            the global parameters to local model parameters.
        param_set : instance of Parameter | sequence of instance of Parameter | instance of ParameterSet
            The set of parameters defining the parameters of this PDF.
        param_grid_set : ParameterGrid instance | ParameterGridSet instance
            The set of ParameterGrid instances, which define the grid values of
            the model parameters, the given MultiDimGridPDF instances belong to.
        gridparams_pdfs : sequence of (dict, MultiDimGridPDF) tuples
            The sequence of 2-element tuples which define the mapping of grid
            values to PDF instances.
        interpol_method_cls : subclass of GridManifoldInterpolationMethod
            The class specifying the interpolation method. This must be a
            subclass of ``GridManifoldInterpolationMethod``.
            If set to None, the default grid manifold interpolation method
            ``Linear1DGridManifoldInterpolationMethod`` will be used.
        """
        super().__init__(
            pmm=pmm,
            param_set=param_set,
            param_grid_set=param_grid_set,
            **kwargs)

        if interpol_method_cls is None:
            interpol_method_cls = Linear1DGridManifoldInterpolationMethod
        self.interpol_method_cls = interpol_method_cls

        # Add the given MultiDimGridPDF instances to the PDF set.
        for (gridparams, pdf) in gridparams_pdfs:
            self.add_pdf(pdf, gridparams)

        # Create the interpolation method instance.
        self._interpol_method = self._interpol_method_cls(
            func=self._evaluate_pdfs,
            param_grid_set=self.param_grid_set)

        # Save the parameter names needed for the interpolation for later usage.
        self._interpol_param_names =\
            self.param_grid_set.params_name_list

        self._cache_tdm_trial_data_state_id = None
        self._cache_eventdata = None

    @property
    def interpol_method_cls(self):
        """The class derived from GridManifoldInterpolationMethod
        implementing the interpolation of the PDF grid manifold.
        """
        return self._interpol_method_cls

    @interpol_method_cls.setter
    def interpol_method_cls(self, cls):
        if not issubclass(cls, GridManifoldInterpolationMethod):
            raise TypeError(
                'The interpol_method_cls property must be a sub-class of '
                'GridManifoldInterpolationMethod!')
        self._interpol_method_cls = cls

    def _get_eventdata(self, tdm, tl=None):
        """Creates and caches the event data for this PDFSet. If the
        TrialDataManager's trail data state id changed, the eventdata will be
        recreated.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure task timing.

        Returns
        -------
        eventdata : instance of numpy ndarray
            The (N_values,V)-shaped eventdata ndarray.
        """
        if (self._cache_tdm_trial_data_state_id is None) or\
           (self._cache_tdm_trial_data_state_id != tdm.trial_data_state_id):

            with TaskTimer(tl, 'Create MultiDimGridPDFSet eventdata.'):
                # All PDFs of this PDFSet should have the same axes, so we use
                # the axes from the first PDF in this PDF set.
                pdf = next(iter(self.items()))[1]

                self._cache_tdm_trial_data_state_id = tdm.trial_data_state_id
                self._cache_eventdata =\
                    MultiDimGridPDF.create_eventdata_for_sigpdf(
                        tdm=tdm,
                        axes=pdf.axes)

        return self._cache_eventdata

    def _get_pdf_for_interpol_param_values(
            self,
            interpol_param_values):
        """Retrieves the PDF for the given set of interpolation parameter
        values.

        Parameters
        ----------
        interpol_param_values : instance of numpy ndarray
            The (N_interpol_params,)-shaped numpy ndarray holding the values of
            the interpolation parameters.

        Returns
        -------
        pdf : instance of MultiDimGridPDF
            The requested PDF instance.
        """
        gridparams = dict(
            zip(self._interpol_param_names, interpol_param_values))

        pdf = self.get_pdf(gridparams)

        return pdf

    def _evaluate_pdfs(
            self,
            tdm,
            eventdata,
            gridparams_recarray,
            n_values):
        """Evaluates the PDFs for the given event data. The particular PDF is
        selected based on the grid parameter values for each model.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data.
        eventdata : instance of numpy ndarray
            The (N_values,V)-shaped numpy ndarray holding the event data for
            the PDF evaluation.
        gridparams_recarray : instance of numpy structured ndarray
            The numpy structured ndarray of length N_sources with the
            parameter names and values needed for the interpolation on the grid
            for all sources. If the length of this structured array is
            1, the set of parameters will be used for all sources.
        n_values : int
            The size of the output array.

        Returns
        -------
        pd : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            values for each event.
        """
        logger = get_logger(f'{__name__}.{classname(self)}._evaluate_pdfs')

        # Check for special case when a single set of parameters are provided.
        if len(gridparams_recarray) == 1:
            if is_tracing_enabled():
                logger.debug(
                    'Get PDF for '
                    f'interpol_param_values={gridparams_recarray[0]}.')
            pdf = self._get_pdf_for_interpol_param_values(
                interpol_param_values=gridparams_recarray[0])

            pd = pdf.get_pd_with_eventdata(
                tdm=tdm,
                params_recarray=None,
                eventdata=eventdata)

            return pd

        pd = np.empty(n_values, dtype=np.float64)

        (src_idxs, evt_idxs) = tdm.src_evt_idxs

        v_start = 0
        for (sidx, interpol_param_values) in enumerate(gridparams_recarray):
            pdf = self._get_pdf_for_interpol_param_values(
                interpol_param_values=interpol_param_values)

            # Determine the events that belong to the current source.
            evt_mask = src_idxs == sidx

            n = np.count_nonzero(evt_mask)
            sl = slice(v_start, v_start+n)
            pd[sl] = pdf.get_pd_with_eventdata(
                tdm=tdm,
                params_recarray=None,
                eventdata=eventdata,
                evt_mask=evt_mask)

            v_start += n

        return pd

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Checks if the PDFs of this PDFSet instance are valid for all the
        given trial data events.
        Since all PDFs should have the same axes, only the first PDF will be
        checked.

        This method calls the
        :meth:`~skyllh.core.pdf.PDFSet.assert_is_valid_for_trial_data` method of
        the :class:`~skyllh.core.pdf.PDFSet` class.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If some of the data is outside the axes range of the PDF.
        """
        super().assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def get_pd(
            self,
            tdm,
            params_recarray,
            tl=None):
        """Calculates the probability density for each event, given the given
        parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that will be used to get the data
            from the trial events.
        params_recarray : instance of structured ndarray | None
            The numpy record ndarray holding the parameter name and values for
            each source model.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            value for each source and event.
        grads : dict
            The dictionary holding the PDF gradient value for each event w.r.t.
            each global fit parameter.
            The key of the dictionary is the ID of the global fit parameter.
            The value is the (N_values,)-shaped numpy ndarray holding the
            gradient value for each event.
        """
        logger = get_logger(f'{__name__}.{classname(self)}.get_pd')

        # Create the ndarray for the event data that is needed for the
        # ``MultiDimGridPDF.get_pd_with_eventdata`` method.
        eventdata = self._get_eventdata(
            tdm=tdm,
            tl=tl)

        # Get the interpolated PDF values for the arbitrary parameter values.
        # The (D,N_events)-shaped grads_arr ndarray contains the gradient of the
        # probability density w.r.t. each of the D parameters, which are defined
        # by the param_grid_set. The order of the D gradients is the same as
        # the parameter grids.
        with TaskTimer(
                tl,
                'Call interpolate method to get probability densities for all '
                'events.'):
            if is_tracing_enabled():
                logger.debug(
                    'Call interpol_method with '
                    f'params_recarray={params_recarray} of fields '
                    f'{list(params_recarray.dtype.fields.keys())}.')
            (pd, grads_arr) = self._interpol_method(
                tdm=tdm,
                eventdata=eventdata,
                params_recarray=params_recarray)

        # Construct the gradients dictionary with all the fit parameters, that
        # contribute to the local interpolation parameters.
        grads = dict()

        tdm_n_sources = tdm.n_sources
        for fitparam_id in range(self.pmm.n_global_floating_params):
            grad = np.zeros((tdm.get_n_values(),), dtype=np.float64)

            # Loop through the local interpolation parameters and match them
            # with the global fit parameter fitparam_id.
            fitparam_id_contributes = False
            for (pidx, pname) in enumerate(self._interpol_param_names):
                if pname not in params_recarray.dtype.fields:
                    continue
                p_gpidxs = params_recarray[f'{pname}:gpidx']
                src_mask = p_gpidxs == (fitparam_id + 1)
                n_sources = np.count_nonzero(src_mask)
                if n_sources == 0:
                    continue

                fitparam_id_contributes = True

                if n_sources == tdm_n_sources:
                    # This parameter applies to all sources, hence to all
                    # values, and hence it's the only local parameter
                    # contributing to the global parameter fitparam_id.
                    grad = grads_arr[pidx]
                    break

                # The current parameter does not apply to all sources.
                # Create a values mask that matches a given source mask.
                values_mask = tdm.get_values_mask_for_source_mask(src_mask)
                grad[values_mask] = grads_arr[pidx][values_mask]

            if fitparam_id_contributes:
                grads[fitparam_id] = grad

        return (pd, grads)


class SignalSHGMappedMultiDimGridPDFSet(
        PDF,
        PDFSet,
        IsSignalPDF):
    """This class provides a set of MultiDimGridPDF instances, one for each
    source hypothesis group.
    """

    def __init__(
            self,
            shg_mgr,
            pmm,
            shgidxs_pdfs,
            **kwargs):
        """Creates a new SignalSHGMappedMultiDimGridPDFSet instance, which holds
        a set of MultiDimGridPDF instances, one for each point of a parameter grid set.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups and their sources.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper which defines the mapping of
            global parameters to local source parameters.
        shgidxs_pdfs : sequence of (shg_idx, MultiDimGridPDF) tuples
            The sequence of 2-element tuples which define the mapping of a
            source hypothesis group to a PDF instance.
        """
        super().__init__(
            pmm=pmm,
            param_set=None,
            param_grid_set=None,
            **kwargs)

        if not isinstance(shg_mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr argument must be an instance of '
                'SourceHypoGroupManager! '
                f'Its current type is {classname(shg_mgr)}.')
        self._shg_mgr = shg_mgr

        for (shg_idx, pdf) in shgidxs_pdfs:
            self.add_pdf(
                pdf=pdf,
                gridparams={'shg_idx': shg_idx})

        self._cache_tdm_trial_data_state_id = None
        self._cache_eventdata = None

    @property
    def shg_mgr(self):
        """(read-only) The instance of SourceHypoGroupManager that defines the
        source hypothesis groups and their sources.
        """
        return self._shg_mgr

    def _get_eventdata(self, tdm, tl=None):
        """Creates and caches the event data for this PDFSet. If the
        TrialDataManager's trail data state id changed, the eventdata will be
        recreated.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure task timing.

        Returns
        -------
        eventdata : instance of numpy ndarray
            The (N_values,V)-shaped eventdata ndarray.
        """
        if (self._cache_tdm_trial_data_state_id is None) or\
           (self._cache_tdm_trial_data_state_id != tdm.trial_data_state_id):

            with TaskTimer(tl, 'Create MultiDimGridPDFSet eventdata.'):
                # All PDFs of this PDFSet should have the same axes, so we use
                # the axes from the first PDF in this PDF set.
                pdf = next(iter(self.items()))[1]

                self._cache_tdm_trial_data_state_id = tdm.trial_data_state_id
                self._cache_eventdata =\
                    MultiDimGridPDF.create_eventdata_for_sigpdf(
                        tdm=tdm,
                        axes=pdf.axes)

        return self._cache_eventdata

    def get_pd(
            self,
            tdm,
            params_recarray,
            tl=None):
        """Calculates the probability density for each event, given the given
        parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that will be used to get the data
            from the trial events.
        params_recarray : instance of structured ndarray | None
            The numpy record ndarray holding the parameter name and values for
            each source model.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            value for each event.
        grads : dict
            The dictionary holding the PDF gradient value for each event w.r.t.
            each global fit parameter.
            The key of the dictionary is the ID of the global fit parameter.
            The value is the (N_values,)-shaped numpy ndarray holding the
            gradient value for each event.
            By definition this PDF set does not depend on any fit parameters,
            hence, this dictionary is empty.
        """
        # Create the ndarray for the event data that is needed for the
        # ``MultiDimGridPDF.get_pd_with_eventdata`` method.
        eventdata = self._get_eventdata(
            tdm=tdm,
            tl=tl)

        pd = np.zeros((tdm.get_n_values(),), dtype=np.float64)

        src_idxs = tdm.src_evt_idxs[0]
        src_idxs_arr = np.arange(self._shg_mgr.n_sources)
        for (shg_idx, shg) in enumerate(self._shg_mgr.shg_list):
            # Check if a PDF is defined for this SHG.
            pdf_key = self.make_key({'shg_idx': shg_idx})
            if pdf_key not in self:
                continue

            shg_src_idxs = src_idxs_arr[
                self._shg_mgr.get_src_mask_of_shg(shg_idx)]
            values_mask = np.isin(src_idxs, shg_src_idxs)

            with TaskTimer(tl, f'Get PD values for PDF of SHG {shg_idx}.'):
                pd_shg = self.get_pdf(pdf_key).get_pd_with_eventdata(
                    tdm=tdm,
                    params_recarray=params_recarray,
                    eventdata=eventdata,
                    evt_mask=values_mask)

            pd[values_mask] = pd_shg

        return (pd, dict())


class SignalNDPhotosplinePDF(
        NDPhotosplinePDF,
        IsSignalPDF):
    """This class provides a multi-dimensional signal PDF created from a
    n-dimensional photospline fit. The photospline package is used to evaluate
    the PDF fit.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new signal PDF instance for a n-dimensional photospline PDF
        fit.

        See the documentation of the
        :meth:`skyllh.core.pdf.NDPhotosplinePDF.__init__` method for
        the documentation of arguments.
        """
        super().__init__(
            *args,
            **kwargs)
