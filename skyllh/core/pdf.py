# -*- coding: utf-8 -*-

import abc

import numpy as np

from scipy.interpolate import (
    RegularGridInterpolator,
)

from skyllh.core import (
    tool,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.display import (
    INDENTATION_WIDTH,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.py import (
    NamedObjectCollection,
    bool_cast,
    classname,
    float_cast,
    func_has_n_args,
    issequenceof,
    make_dict_hash,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet,
    ParameterModelMapper,
    ParameterSet,
)
from skyllh.core.timing import (
    TaskTimer,
)


logger = get_logger(__name__)


class PDFAxis(object):
    """This class describes an axis of a PDF. It's main purpose is to define
    the allowed variable space of the PDF. So this information can be used to
    plot a PDF or a PDF ratio.
    """

    def __init__(self, name, vmin, vmax, *args, **kwargs):
        """Creates a new axis for a PDF.

        Parameters
        ----------
        name : str
            The name of the axis.
        vmin : float
            The minimal value of the axis.
        vmax : float
            The maximal value of the axis.
        """
        super().__init__(*args, **kwargs)

        self.name = name
        self.vmin = vmin
        self.vmax = vmax

    @property
    def name(self):
        """The name of the axis.
        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                'The name property must be of type str!')
        self._name = name

    @property
    def vmin(self):
        """The minimal value of the axis.
        """
        return self._vmin

    @vmin.setter
    def vmin(self, v):
        self._vmin = float_cast(
            v,
            'The value for the vmin property must be cast-able to type float!')

    @property
    def vmax(self):
        """The maximal value of the axis.
        """
        return self._vmax

    @vmax.setter
    def vmax(self, v):
        self._vmax = float_cast(
            v,
            'The value for the vmax property must be cast-able to type float!')

    @property
    def range(self):
        """(read-only) The 2-element tuple (vmin,vmax) of the axis.
        """
        return (self._vmin, self._vmax)

    @property
    def length(self):
        """(read-only) The length of the axis as float. It's defined as
        length = vmax - vmin.
        """
        return self._vmax - self._vmin

    def __eq__(self, other):
        """Checks if this PDFAxis object has the same properties than the given
        other PDFAxis object.
        """
        if (self.name == other.name) and\
           np.isclose(self.vmin, other.vmin) and\
           np.isclose(self.vmax, other.vmax):
            return True
        return False

    def __str__(self):
        """Pretty string implementation for the PDFAxis instance.
        """
        s = f'{classname(self)}: {self._name}: ' +\
            f'vmin={self._vmin:g} vmax={self._vmax:g}'
        return s


class PDFAxes(NamedObjectCollection):
    """This class describes the set of PDFAxis objects defining the
    dimensionality of a PDF.
    """

    @staticmethod
    def union(*axeses):
        """Creates a PDFAxes instance that is the union of the given PDFAxes
        instances.

        Parameters
        ----------
        *axeses : PDFAxes instances
            The sequence of PDFAxes instances.

        Returns
        -------
        axes : PDFAxes instance
            The newly created PDFAxes instance that holds the union of the
            PDFAxis instances provided by all the PDFAxes instances.
        """
        if not issequenceof(axeses, PDFAxes):
            raise TypeError(
                'The arguments of the union static function must '
                'be instances of PDFAxes!')
        if not len(axeses) >= 1:
            raise ValueError(
                'At least 1 PDFAxes instance must be provided to '
                'the union static function!')

        axes = PDFAxes(axes=axeses[0])
        for axes_i in axeses[1:]:
            for axis in axes_i:
                if axis.name not in axes:
                    axes += axis

        return axes

    def __init__(self, axes=None, **kwargs):
        """Creates a new PDFAxes instance.

        Parameters
        ----------
        axes : sequence of instance of PDFAxis | None
            The sequence of instance of PDFAxis for this PDFAxes instance.
            If set to ``None``, the PDFAxes instance will be empty.
        """
        super().__init__(
            objs=axes,
            obj_type=PDFAxis,
            **kwargs)

    def __str__(self):
        """Pretty string implementation for the PDFAxes instance.
        """
        return '\n'.join((str(axis) for axis in self))

    def is_same_as(self, axes):
        """Checks if this PDFAxes object has the same axes and range then the
        given PDFAxes object.

        Parameters
        ----------
        axes : instance of PDFAxes | sequence of PDFAxis
            The instance of PDFAxes or the sequence of instance of PDFAxis that
            should be compared to the axes of this PDFAxes instance.

        Returns
        -------
        check : bool
            True, if this PDFAxes and the given PDFAxes have the same axes and
            ranges. False otherwise.
        """
        if len(self) != len(axes):
            return False

        for (self_axis, axes_axis) in zip(self, axes):
            if self_axis != axes_axis:
                return False

        return True


class IsBackgroundPDF(
        object,
):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a background PDF. This is useful for type checking.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsBackgroundPDF class.
        """
        if not isinstance(self, PDF):
            raise TypeError(
                f'The class "{classname(self)}" is not derived from PDF!')

        super().__init__(*args, **kwargs)

    def __mul__(self, other):
        """Creates a BackgroundPDFProduct instance for the multiplication of
        this background PDF and another background PDF.

        Parameters
        ----------
        other : instance of IsBackgroundPDF
            The instance of IsBackgroundPDF, which is the other background PDF.
        """
        if not isinstance(other, IsBackgroundPDF):
            raise TypeError(
                'The other PDF must be an instance of IsBackgroundPDF!')

        return BackgroundPDFProduct(self, other, cfg=self.cfg)


class IsSignalPDF(
        object,
):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a signal PDF.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsSignalPDF class.
        """
        if not isinstance(self, PDF):
            raise TypeError(
                f'The class "{classname(self)}" is not derived from PDF!')

        super().__init__(*args, **kwargs)

    def __mul__(self, other):
        """Creates a SignalPDFProduct instance for the multiplication of this
        signal PDF and another signal PDF.

        Parameters
        ----------
        other : instance of IsSignalPDF
            The instance of IsSignalPDF, which is the other signal PDF.
        """
        if not isinstance(other, IsSignalPDF):
            raise TypeError(
                'The other PDF must be an instance of IsSignalPDF!')

        return SignalPDFProduct(self, other, cfg=self.cfg)


class PDF(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    r"""This is the abstract base class for all probability distribution
    function (PDF) models.
    All PDF model classes must be derived from this class. Mathematically, it
    represents :math:`f(\vec{x}|\vec{p})`, where :math:`\vec{x}` is the
    event data and :math:`\vec{p}` is the set of parameters the PDF is given
    for.
    """

    def __init__(
            self,
            pmm=None,
            param_set=None,
            **kwargs,
    ):
        """Creates a new PDF instance.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper | None
            The instance of ParameterModelMapper defining the global parameters
            and their mapping to local model/source parameters.
            It can be ``None``, if the PDF does not depend on any parameters.
        param_set : instance of Parameter | sequence of instance of Parameter | instance of ParameterSet | None
            If this PDF depends on parameters, this set of parameters
            defines them. If a single parameter instance is given a ParameterSet
            instance will be created holding this single parameter.
            If set to None, this PDF will not depend on any parameters.
        """
        # Make sure that multiple inheritance can be used. This super call will
        # invoke the __init__ method of a possible second inheritance.
        super().__init__(
            **kwargs)

        self.pmm = pmm
        self.param_set = param_set

        self._axes = PDFAxes()

    @property
    def axes(self):
        """(read-only) The PDFAxes object holding the PDFAxis objects for the
        dimensions of the PDF.
        """
        return self._axes

    @property
    def ndim(self):
        """The dimensionality of the PDF. It's defined as the number of PDFAxis
        objects this PDF object has. Note, that the internal dimensionality
        might be smaller than this.
        """
        return len(self._axes)

    @property
    def pmm(self):
        """The instance of ParameterModelMapper that defines the global
        parameters and their mapping to local model/source parameters.
        It can be ``None`` if the PDF does not depend on any parameters.
        """
        return self._pmm

    @pmm.setter
    def pmm(self, mapper):
        if mapper is not None:
            if not isinstance(mapper, ParameterModelMapper):
                raise TypeError(
                    'The pmm property must be an instance of '
                    f'ParameterModelMapper! Its type is "{classname(mapper)}"!')
        self._pmm = mapper

    @property
    def param_set(self):
        """The ParameterSet instance defining the set of parameters this PDF
        depends on. It is ``None``, if this PDF does not depend on any
        parameters.
        """
        return self._param_set

    @param_set.setter
    def param_set(self, param_set):
        if param_set is None:
            param_set = ParameterSet()
        elif not isinstance(param_set, ParameterSet):
            param_set = ParameterSet(param_set)
        self._param_set = param_set

    @property
    def is_signal_pdf(self):
        """(read-only) The flag if this PDF instance represents a signal PDF.
        A PDF is a signal PDF if it derives from the ``IsSignalPDF`` class.
        """
        return isinstance(self, IsSignalPDF)

    @property
    def is_background_pdf(self):
        """(read-only) The flag if this PDF instance represents a background
        PDF. A PDF is a background PDF if it derives from the
        ``IsBackgroundPDF``  class.
        """
        return isinstance(self, IsBackgroundPDF)

    def add_axis(self, axis):
        """Adds the given PDFAxis object to this PDF.
        """
        if not isinstance(axis, PDFAxis):
            raise TypeError(
                'The axis argument must be an instance of PDFAxis!')
        self._axes += axis

    @abc.abstractmethod
    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """This method is supposed to check if this PDF is valid for
        all the given trial data. This means, it needs to check if there
        is a PDF value for each trial data event that will be used in the
        likelihood evaluation. This is just a seatbelt.
        The method must raise a ``ValueError`` if the PDF is not valid for the
        given trial data.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If some of the trial data is outside the PDF's value space.
        """
        pass

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """This method is called when a new trial is initialized. Derived
        classes can use this call hook to pre-compute time-expensive data, which
        do not depend on any fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the new trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure timing information.
        """
        pass

    @abc.abstractmethod
    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """This abstract method is supposed to calculate the probability density
        for the specified events given the specified parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the data events for which
            the probability density should be calculated.
            What data fields are required is defined by the derived PDF class
            and depends on the application.
        params_recarray : numpy record ndarray | None
            The (N_models,)-shaped numpy structured ndarray holding the local
            parameter names and values of the models.
            The models are defined by the ParameterModelMapper instance.
            The parameter values can be different for the different models.
            In case of the signal PDF, the models are the sources.
            The record array must contain two fields for each source parameter,
            one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.
            This can be ``None`` for PDFs that do not depend on any parameters.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            for each event. The length of this 1D array depends on the number
            of sources and the events belonging to those sources. In the worst
            case the length is N_values = N_sources * N_trial_events.
            The assignment of values to sources is given by the
            :py:attr:`~skyllh.core.trialdata.TrialDataManager.src_evt_idxs`
            property.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter. The key of the dictionary is the
            id of the global fit parameter. The value is a (N_values,)-shaped
            numpy ndarray.
        """
        pass


class PDFProduct(
        PDF,
):
    """The PDFProduct class represents the product of two PDF instances, i.e.
    ``pdf1 * pdf2``. It is derived from the PDF class and hence is a PDF itself.
    """

    def __init__(self, pdf1, pdf2, **kwargs):
        """Creates a new PDFProduct instance, which implements the operation
        ``pdf1 * pdf2``.
        The axes of the two PDF instances will be merged.

        Parameters
        ----------
        pdf1 : instance of PDF
            The left-hand-side PDF in the operation ``pdf1 * pdf2``.
        pdf2 : instance of PDF
            The right-hand-side PDF in the operation ``pdf1 * pdf2``.
        """
        self.pdf1 = pdf1
        self.pdf2 = pdf2

        if pdf1.pmm is not pdf2.pmm:
            raise ValueError(
                'The ParameterModelMapper instance of pdf1 is not the same as '
                'for pdf2!')

        # Create the ParameterSet instance that is the union of the ParameterSet
        # instances of the two PDFs.
        param_set = ParameterSet.union(
            self._pdf1.param_set, self._pdf2.param_set)

        super().__init__(
            pmm=pdf1.pmm,
            param_set=param_set,
            **kwargs)

        # The resulting PDFAxes object of this PDF instance is the union of the
        # two PDFAxes instances of the two PDF instances.
        self._axes = PDFAxes.union(pdf1.axes, pdf2.axes)

    @property
    def pdf1(self):
        """The left-hand-side PDF in the operation ``pdf1 * pdf2``.
        It must be an instance of PDF.
        """
        return self._pdf1

    @pdf1.setter
    def pdf1(self, pdf):
        if not isinstance(pdf, PDF):
            raise TypeError(
                'The pdf1 property must be an instance of PDF!')
        self._pdf1 = pdf

    @property
    def pdf2(self):
        """The right-hand-side PDF in the operation ``pdf1 * pdf2``.
        It must be an instance of PDF.
        """
        return self._pdf2

    @pdf2.setter
    def pdf2(self, pdf):
        if not isinstance(pdf, PDF):
            raise TypeError(
                'The pdf2 property must be an instance of PDF!')
        self._pdf2 = pdf

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Calls the :meth:`assert_is_valid_for_trial_data` method of ``pdf1``
        and ``pdf2``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that should be used to get the
            trial data from.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If this PDF does not cover the trial data.
        """
        self._pdf1.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

        self._pdf2.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """Calls the ``initialize_for_new_trial`` method of the two PDF
        instances of this PDF product.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the new trial event data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.
        """
        self._pdf1.initialize_for_new_trial(tdm=tdm, tl=tl, **kwargs)
        self._pdf2.initialize_for_new_trial(tdm=tdm, tl=tl, **kwargs)

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the probability density for the trial events given the
        specified parameters by calling the `get_pd` method of `pdf1`
        and `pdf2` and combining the two property densities by multiplication.
        The gradients will be calculated using the product rule of
        differentiation.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params_recarray : instance of numpy.ndarray | None
            The (N_models,)-shaped structured numpy ndarray holding the
            parameter values of the models. The the documentation of the
            :meth:`~skyllh.core.pdf.PDF.get_pd` method of the
            :class:`~skyllh.core.pdf.PDF` class for further information.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing
            information.

        Returns
        -------
        pd : instance of numpy.ndarray
            The (N_events,)-shaped numpy ndarray holding the probability density
            for each event. In case of a signal PDF product the shape will be
            (N_sources,N_events).
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each fit parameter. The key of the dictionary is the id
            of the global fit parameter. The value is the (N_events,)-shaped
            numpy ndarray. In case of a signal PDF product, the value is a
            (N_sources,N_events)-shaped ndarray.
        """
        pdf1 = self._pdf1
        pdf2 = self._pdf2

        with TaskTimer(
                tl,
                f'Get probability densities from {classname(pdf1)} (pdf1) and '
                f'{classname(pdf2)} (pdf2).'):
            (pd1, grads1) = pdf1.get_pd(
                tdm=tdm,
                params_recarray=params_recarray,
                tl=tl)
            (pd2, grads2) = pdf2.get_pd(
                tdm=tdm,
                params_recarray=params_recarray,
                tl=tl)

        pd = pd1 * pd2

        # Loop over the set of global fit parameter gradients.
        grads = dict()
        for gpid in set(list(grads1.keys()) + list(grads2.keys())):
            # Calculate the gradient w.r.t. the fit parameter of id ``pgid``.

            # There are four possible cases to calculate the gradient for
            # the parameter gpid:
            #     1. Both PDFs depend on this fit parameter, the gradient is
            #        calculated through the product rule of differentiation.
            #     2. Only PDF1 depends on this fit parameter.
            #     3. Only PDF2 depends on this fit parameter.
            #     4. Both PDFs are independent of this fit parameter, the
            #        gradient is 0.
            pdf1_has_fitparam = gpid in grads1
            pdf2_has_fitparam = gpid in grads2
            if pdf1_has_fitparam and pdf2_has_fitparam:
                # Case 1
                grad1 = grads1[gpid]
                grad2 = grads2[gpid]
                grads[gpid] = pd1*grad2 + pd2*grad1
            elif pdf1_has_fitparam:
                # Case 2
                grad1 = grads1[gpid]
                grads[gpid] = pd2*grad1
            elif pdf2_has_fitparam:
                # Case 3
                grad2 = grads2[gpid]
                grads[gpid] = pd1*grad2

        return (pd, grads)


class SignalPDFProduct(
        PDFProduct,
        IsSignalPDF):
    """This class provides a signal PDF that is the product of two signal PDF
    instances.
    """

    def __init__(self, pdf1, pdf2, **kwargs):
        """Creates a new PDF product of two signal PDFs.
        """
        super().__init__(
            pdf1=pdf1,
            pdf2=pdf2,
            **kwargs)


class BackgroundPDFProduct(
        PDFProduct,
        IsBackgroundPDF):
    """This class provides a background PDF that is the product of two
    background PDF instances.
    """

    def __init__(self, pdf1, pdf2, **kwargs):
        """Creates a new PDF product of two background PDFs.
        """
        super().__init__(
            pdf1=pdf1,
            pdf2=pdf2,
            **kwargs)


class SpatialPDF(
        PDF,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for a spatial PDF model. A spatial PDF
    has two axes, right-ascension (ra) and declination (dec).
    """

    def __init__(self, ra_range, dec_range, **kwargs):
        """Constructor of a spatial PDF. It adds the PDF axes "ra" and "dec"
        with the specified ranges of coverage.

        Parameters
        ----------
        ra_range : 2-element tuple
            The tuple specifying the right-ascension range this PDF covers.
        dec_range : 2-element tuple
            The tuple specifying the declination range this PDF covers.
        """
        super().__init__(**kwargs)

        self.add_axis(
            PDFAxis(
                name='ra',
                vmin=ra_range[0],
                vmax=ra_range[1]))
        self.add_axis(
            PDFAxis(
                name='dec',
                vmin=dec_range[0],
                vmax=dec_range[1]))

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Checks if this spatial PDF is valid for all the given experimental
        data.
        It checks if all the data is within the right-ascension and declination
        range.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
            The following data fields must exist:

            - 'ra' : float
                The right-ascension of the data event.
            - 'dec' : float
                The declination of the data event.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If some of the data is outside the right-ascension or declination
            range.
        """
        ra_axis = self.axes['ra']
        dec_axis = self.axes['dec']

        ra = tdm.get_data('ra')
        dec = tdm.get_data('dec')

        # Check if all the data is within the right-ascension range.
        if np.any((ra < ra_axis.vmin) | (ra > ra_axis.vmax)):
            raise ValueError(
                'Some data is outside the right-ascension range '
                f'({ra_axis.vmin:.3f}, {ra_axis.vmax:.3f})!')

        # Check if all the data is within the declination range.
        if np.any((dec < dec_axis.vmin) | (dec > dec_axis.vmax)):
            raise ValueError(
                'Some data is outside the declination range '
                f'({dec_axis.vmin:.3f}, {dec_axis.vmax:.3f})!')


class EnergyPDF(
        PDF,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for an energy PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TimePDF(
        PDF,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for a time PDF. It consists of
    a :class:`~skyllh.core.livetime.Livetime` instance and a
    :class:`~skyllh.core.flux_model.TimeFluxProfile` instance. Together they
    construct the actual time PDF, which has detector down-time taking
    into account.
    """

    def __init__(
            self,
            livetime,
            time_flux_profile,
            **kwargs,
    ):
        """Creates a new time PDF instance for a given time flux profile and
        detector live time.

        Parameters
        ----------
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_profile : instance of TimeFluxProfile
            The signal's time flux profile.
        **kwargs
            Additional keyword arguments are passed to the constructor of the
            base class, :class:`~skyllh.core.pdf.PDF`.
        """
        super().__init__(
            **kwargs)

        self.livetime = livetime
        self.time_flux_profile = time_flux_profile

        # Define the time axis with the time boundaries of the live-time.
        self.add_axis(
            PDFAxis(
                name='time',
                vmin=self._livetime.time_window[0],
                vmax=self._livetime.time_window[1]))

        # Get sum, S, of the integrals for each detector on-time interval during
        # the time flux profile, in order to be able to rescale the time flux
        # profile to unity with overlapping detector off-times removed.
        self._S = self._calculate_sum_of_ontime_time_flux_profile_integrals()

    @property
    def livetime(self):
        """The instance of Livetime, which provides the detector live-time
        information.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if not isinstance(lt, Livetime):
            raise TypeError(
                'The livetime property must be an instance of Livetime!')
        self._livetime = lt

    @property
    def time_flux_profile(self):
        """The instance of TimeFluxProfile providing the physical time flux
        profile.
        """
        return self._time_flux_profile

    @time_flux_profile.setter
    def time_flux_profile(self, profile):
        if not isinstance(profile, TimeFluxProfile):
            raise TypeError(
                'The time_flux_profile property must be an instance of '
                'TimeFluxProfile! '
                f'Its current type is {classname(profile)}!')
        self._time_flux_profile = profile

    def __str__(self):
        """Pretty string representation of the time PDF.
        """
        s = (
            f'{classname(self)}(\n'
            ' '*INDENTATION_WIDTH +
            f'livetime = {str(self._livetime)},\n'
            ' '*INDENTATION_WIDTH +
            f'time_flux_profile = {str(self._time_flux_profile)}\n'
            ')'
        )

        return s

    def _calculate_sum_of_ontime_time_flux_profile_integrals(self):
        """Calculates the sum, S, of the time flux profile integrals during the
        detector on-time intervals.

        Returns
        -------
        S : float
            The sum of the time flux profile integrals during the detector
            on-time intervals.
        """
        uptime_intervals = self._livetime.get_uptime_intervals_between(
            self._time_flux_profile.t_start,
            self._time_flux_profile.t_stop)

        S = np.sum(
            self._time_flux_profile.get_integral(
                uptime_intervals[:, 0],
                uptime_intervals[:, 1]))

        return S

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Checks if the time PDF is valid for all the given trial data.
        It checks if the time of all events is within the defined time axis of
        the PDF.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial data.
            The following data fields must exist:

            ``'time'`` : float
                The time of the data event.

        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

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
                'Some trial data is outside the time range '
                f'[{time_axis.vmin:.3f}, {time_axis.vmax:.3f}]!')


class MultiDimGridPDF(
        PDF,
):
    """This class provides a multi-dimensional PDF. The PDF is created from
    pre-calculated PDF data on a grid. The grid data is either interpolated
    using a :class:`scipy.interpolate.RegularGridInterpolator` instance, or is
    provided as a photospline fit through a photospline table file.
    """

    def __init__(
            self,
            pmm,
            axis_binnings,
            path_to_pdf_splinetable=None,
            pdf_grid_data=None,
            norm_factor_func=None,
            cache_pd_values=False,
            **kwargs,
    ):
        """Creates a new PDF instance for a multi-dimensional PDF given
        as PDF values on a grid or as PDF values stored in a photospline table.

        In case of PDF values on a grid, the grid data is interpolated with a
        :class:`scipy.interpolate.RegularGridInterpolator` instance. As grid
        points the bin edges of the axis binning definitions are used.

        In case of PDF values stored in a photospline table, this table is
        loaded via the ``photospline.SplineTable`` class.

        Note::

            By definition this PDF must not depend on any fit parameters.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper that defines the mapping of
            the global parameters to local model parameters.
        axis_binnings : instance of BinningDefinition | sequence of instance of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each instance of BinningDefinition defines
            the event field name that should be used for querying the PDF.
        path_to_pdf_splinetable : str | None
            The path to the file containing the spline table, which contains
            a pre-computed fit to the grid data.
            If specified, ``pdf_grid_data`` must be ``None``.
        pdf_grid_data : instance of numpy ndarray | None
            The n-dimensional numpy ndarray holding the PDF values at given grid
            points. The grid points must match the bin edges of the given
            BinningDefinition instances of the ``axis_binnings`` argument.
            If specified, ``path_to_pdf_splinetable`` must be ``None``.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function must be

                ``__call__(pdf, tdm, params_recarray, eventdata, evt_mask=None)``,

            where ``pdf`` is this PDF instance, ``tdm`` is an instance of
            TrialDataManager holding the event data for which to calculate the
            PDF values, ``params_recarray`` is a numpy structured ndarray
            holding the local parameter names and values, ``eventdata`` is
            is a (V,N_values)-shaped numpy ndarray holding the event data
            necessary for this PDF, and ``evt_mask`` is an optional
            (N_values,)-shaped numpy ndarray holding the mask for the events,
            i.e. rows in ``eventdata``, which should be considered. If ``None``,
            all events should be considered.
        cache_pd_values : bool
            Flag if the probability density values should be cached.
            The evaluation of the photospline fit might be slow and caching the
            probability density values might increase performance.
        """
        super().__init__(
            pmm=pmm,
            **kwargs)

        # Need either splinetable or grid of pdf values.
        if path_to_pdf_splinetable is None and\
           pdf_grid_data is None:
            raise ValueError(
                'At least one of the following arguments are required: '
                'path_to_pdf_splinetable (str) or '
                'pdf_grid_data (numpy.ndarray)!')
        elif (path_to_pdf_splinetable is not None and
              pdf_grid_data is not None):
            raise ValueError(
                'Only one of the two arguments path_to_pdf_splinetable and '
                'pdf_grid_data can be specified!')

        # If a path to the photospline tables is given, we raise an error if
        # the photospline package is not loaded.
        if path_to_pdf_splinetable is not None:
            if not isinstance(path_to_pdf_splinetable, str):
                raise TypeError(
                    'The path_to_pdf_splinetable argument must be None or an '
                    'instance of str!'
                    'Its current type is '
                    f'{classname(path_to_pdf_splinetable)}.')

            if not tool.is_available('photospline'):
                raise ImportError(
                    'The path_to_pdf_splinetable argument is specified, but '
                    'the "photospline" package is not available!')

            tool.assert_tool_version('photospline', '>=2.2.0')

        if pdf_grid_data is not None:
            if not isinstance(pdf_grid_data, np.ndarray):
                raise TypeError(
                    'The pdf_grid_data argument must be an instance of numpy '
                    f'ndarray. Its current type is {classname(pdf_grid_data)}!')

        self.axis_binning_list = axis_binnings
        self.norm_factor_func = norm_factor_func
        self.cache_pd_values = cache_pd_values

        # Define the PDF axes.
        for axis_binning in self._axis_binning_list:
            self.add_axis(PDFAxis(
                name=axis_binning.name,
                vmin=axis_binning.lower_edge,
                vmax=axis_binning.upper_edge
            ))

        # Create the internal PDF object.
        if path_to_pdf_splinetable is None:
            self._pdf = RegularGridInterpolator(
                tuple([binning.binedges for binning in self._axis_binning_list]),
                pdf_grid_data,
                method='linear',
                bounds_error=False,
                fill_value=0)
        else:
            self._pdf = tool.get('photospline').SplineTable(
                path_to_pdf_splinetable)

        # The basis function indices (centers) is a (V,N_values)-shaped numpy
        # ndarray holding the spline table indices for the eventdata.
        self.basis_function_indices = None

        # Because this PDF does not depend on any fit parameters, the PDF values
        # can be cached as long as the trial data state ID of the trial data
        # manager has not changed.
        self._cache_tdm_trial_data_state_id = None
        self._cache_pd = None

        logger.debug(
            f'Created {classname(self)} instance with axis name list '
            f'{str(self._axes.name_list)}')

    @property
    def axis_binning_list(self):
        """The list of BinningDefinition instances for each PDF axis.
        The name of each BinningDefinition instance defines the event field
        name that should be used for querying the PDF.
        """
        return self._axis_binning_list

    @axis_binning_list.setter
    def axis_binning_list(self, binnings):
        if isinstance(binnings, BinningDefinition):
            binnings = [binnings]
        if not issequenceof(binnings, BinningDefinition):
            raise TypeError(
                'The axis_binning_list property must be an instance of '
                'BinningDefinition or a sequence of BinningDefinition '
                'instances! '
                f'Its current type is {classname(binnings)}.')
        self._axis_binning_list = list(binnings)

    @property
    def basis_function_indices(self):
        """The (V,N_values)-shaped numpy.ndarray of int holding the basis
        function indices of the photospline table for the current trial
        eventdata.
        """
        return self._basis_function_indices

    @basis_function_indices.setter
    def basis_function_indices(self, bfi):
        if bfi is not None:
            if not isinstance(bfi, np.ndarray):
                raise TypeError(
                    'The basis_function_indices property must be None, or an '
                    'instance of numpy.ndarray! '
                    f'It\'s current type is {classname(bfi)}!')
            if bfi.ndim != 2:
                raise ValueError(
                    'The ndarray dimensionality of the basis_function_indices '
                    f'property must be 2! Currently it is {bfi.ndim}!')
        self._basis_function_indices = bfi

    @property
    def norm_factor_func(self):
        """The function that calculates the possible required normalization
        factor. The call signature of this function must be

            ``__call__(pdf, tdm, params_recarray, eventdata, evt_mask=None)``,

        where ``pdf`` is this PDF instance, ``tdm`` is an instance of
        TrialDataManager holding the events for which to calculate the PDF
        values, ``params_recarray`` is a numpy structured ndarray holding the
        local parameter names and values, ``eventdata`` is a (V,N_values)-shaped
        numpy ndarray holding the event data necessary for this PDF, and
        ``evt_mask`` is an optional (N_values,)-shaped numpy ndarray holding the
        mask for the events, i.e. rows in ``eventdata``, which should be
        considered. If ``None``, all events should be considered..
        This property can be set to ``None``. In that case a unity returning
        function is used.
        """
        return self._norm_factor_func

    @norm_factor_func.setter
    def norm_factor_func(self, func):
        if func is None:
            # Define a normalization function that just returns 1 for each
            # event.
            def func(pdf, tdm, params_recarray, eventdata, evt_mask=None):
                if evt_mask is None:
                    n_values = eventdata.shape[1]
                else:
                    n_values = np.count_nonzero(evt_mask)
                return np.ones((n_values,), dtype=np.float64)

        if not callable(func):
            raise TypeError(
                'The norm_factor_func property must be a callable object!')
        if not func_has_n_args(func, 5):
            raise TypeError(
                'The norm_factor_func property must be a function with 5 '
                'arguments!')
        self._norm_factor_func = func

    @property
    def cache_pd_values(self):
        """Flag if the probability density values should be cached.
        """
        return self._cache_pd_values

    @cache_pd_values.setter
    def cache_pd_values(self, b):
        self._cache_pd_values = bool_cast(
            b,
            'The cache_pd_values property must be cast-able to type bool!')

    @property
    def pdf(self):
        """(read-only) The instance of RegularGridInterpolator or instance of
        photospline.SplineTable that represents the internal PDF object.
        """
        return self._pdf

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """Checks if the PDF is valid for all values of the given evaluation
        data. The evaluation data values must be within the ranges of the PDF
        axes.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial data for which
            the PDF should be valid.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If any of the evaluation trial data is out of its axis range.
        """
        for axis in self._axes:
            data = tdm.get_data(axis.name)
            m = (data < axis.vmin) | (data > axis.vmax)
            if np.any(m):
                raise ValueError(
                    f'Some of the trial data for PDF axis "{axis.name}" is out'
                    f'of range ({axis.vmin:g},{axis.vmax:g})! '
                    f'Data values out of range: {data[m]}')

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """This method is called whenever a new trial is initialized.
        """
        # We need to recalculate the the basis function indices for the
        # photospline table.
        self.basis_function_indices = None

    def _initialize_cache(
            self,
            tdm,
    ):
        """Initializes the cache variables.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that hold the trial data events.
        """
        self._cache_tdm_trial_data_state_id = None
        self._cache_pd = None

    def _store_pd_values_to_cache(
            self,
            tdm,
            pd,
            evt_mask=None,
    ):
        """Stores the given pd values into the pd array cache.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that hold the trial data events.
        pd : instance of numpy ndarray
            The (N,)-shaped numpy ndarray holding the pd values to be stored.
        evt_mask : instance of numpy ndarray | None
            The (N_values,)-shaped numpy ndarray defining the elements of the
            (N_values,)-shaped pd cache array where the given pd values should
            get stored. If set to ``None``, the the ``pd`` array must be of
            length N_values.
        """
        self._cache_tdm_trial_data_state_id = tdm.trial_data_state_id

        if self._cache_pd is None:
            self._cache_pd = np.full(
                pd.shape,
                np.nan,
                dtype=np.float64)

        if evt_mask is None:
            self._cache_pd[:] = pd
            return

        self._cache_pd[evt_mask] = pd

    def _get_cached_pd_values(
            self,
            tdm,
            evt_mask=None):
        """Retrieves cached pd values for the given events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that hold the trial data events.
        evt_mask : instance of numpy ndarray | None
            The (N_values,)-shaped numpy ndarray defining the elements of the
            (N_values,)-shaped pd cache array for which pd values should get
            returned.
            If set to ``None`` all N_values values will get retrieved.

        Returns
        -------
        pd : instance of numpy ndarray | None
            Returns ``None``, when no cached values are available.
            Otherwise the (N,)-shaped numpy ndarray holding the pd values where
            evt_mask evaluates to True.
        """
        if self._cache_tdm_trial_data_state_id is None or\
           self._cache_tdm_trial_data_state_id != tdm.trial_data_state_id:
            self._initialize_cache(tdm=tdm)
            return None

        if evt_mask is None:
            pd = self._cache_pd
        else:
            pd = self._cache_pd[evt_mask]
            # If this PDF is evaluated for different sources, i.e. a subset of
            # pd values, those values could still be NaN and still need to be
            # calculated.
            if np.any(np.isnan(pd)):
                return None

        return pd

    def get_pd_with_eventdata(
            self,
            tdm,
            params_recarray,
            eventdata,
            evt_mask=None,
            tl=None,
    ):
        """Calculates the probability density value for the given ``eventdata``.

        This method is useful when PDF values for the same trial data need to
        be evaluated.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params_recarray : instance of numpy structured ndarray | None
            The (N_models,)-shaped numpy structured ndarray holding the local
            parameter names and values of the models.
            By definition, this PDF does not depend on any parameters.
        eventdata : instance of numpy.ndarray
            The (V,N_values)-shaped numpy ndarray holding the V data attributes
            for each of the N_values events needed for the evaluation of the
            PDF.
        evt_mask : instance of numpy ndarray | None
            The (N_values,)-shaped numpy ndarray defining the elements of the
            N_values pd array for which pd values should get calculated.
            This is needed to determine if the requested pd values are already
            cached.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : (N,)-shaped numpy ndarray
            The (N,)-shaped numpy ndarray holding the probability density
            value for each model and event. The length of this array depends on
            the ``evt_mask`` argument. Only values are returned where
            ``evt_mask`` evaluates to ``True``.
            If ``evt_mask`` is set to ``None``, the length is N_values.
        """
        if self._cache_pd_values:
            pd = self._get_cached_pd_values(
                tdm=tdm,
                evt_mask=evt_mask)
            if pd is not None:
                return pd

        # Cached pd values are not available at this point.

        if isinstance(self._pdf, RegularGridInterpolator):
            with TaskTimer(tl, 'Get pd from RegularGridInterpolator.'):
                if evt_mask is None:
                    pd = self._pdf(eventdata.T)
                else:
                    pd = self._pdf(eventdata.T[evt_mask])
        else:
            V = eventdata.shape[0]

            if self.basis_function_indices is None:
                with TaskTimer(tl, 'Get basis function indices from photospline.'):
                    self.basis_function_indices = self._pdf.search_centers(
                        [eventdata[i] for i in range(0, V)]
                    )

            with TaskTimer(tl, 'Get pd from photospline fit.'):
                if evt_mask is None:
                    pd = self._pdf.evaluate(
                        [eventdata[i] for i in range(0, V)],
                        [self.basis_function_indices[i] for i in range(0, V)],
                    )
                else:
                    pd = self._pdf.evaluate(
                        [eventdata[i][evt_mask] for i in range(0, V)],
                        [self.basis_function_indices[i][evt_mask] for i in range(0, V)],
                    )

        with TaskTimer(tl, 'Normalize MultiDimGridPDF with norm factor.'):
            norm = self._norm_factor_func(
                pdf=self,
                tdm=tdm,
                params_recarray=params_recarray,
                eventdata=eventdata,
                evt_mask=evt_mask)

            pd *= norm

        if self._cache_pd_values:
            self._store_pd_values_to_cache(
                tdm=tdm,
                pd=pd,
                evt_mask=evt_mask)

        return pd

    @staticmethod
    def create_eventdata_for_sigpdf(
            tdm,
            axes,
    ):
        """Creates the (V,N_values)-shaped eventdata ndarray necessary for
        evaluating the signal PDF.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data.
        axes : instance of PDFAxes
            The instance of PDFAxes defining the data field names for the PDF.

        Returns
        -------
        eventdata : instance of numpy.ndarray
            The (V,N_values)-shaped numpy ndarray holding the event data for
            evaluating the signal PDF.
        """
        eventdata_fields = []

        (src_idxs, evt_idxs) = tdm.src_evt_idxs
        for axis in axes:
            name = axis.name
            data = tdm.get_data(name)
            if tdm.is_event_data_field(name):
                eventdata_fields.append(np.take(data, evt_idxs))
            elif tdm.is_source_data_field(name):
                eventdata_fields.append(np.take(data, src_idxs))
            elif tdm.is_srcevt_data_field(name):
                eventdata_fields.append(data)
            else:
                TypeError(
                    f'Unable to determine the type of the data field {name}!')

        eventdata = np.array(eventdata_fields)

        return eventdata

    @staticmethod
    def create_eventdata_for_bkgpdf(
            tdm,
            axes,
    ):
        """Creates the (V,N_values)-shaped eventdata ndarray necessary for
        evaluating the background PDF.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data.
        axes : instance of PDFAxes
            The instance of PDFAxes defining the data field names for the PDF.
        """
        eventdata_fields = []

        for axis in axes:
            eventdata_fields.append(tdm.get_data(axis.name))

        eventdata = np.array(eventdata_fields)

        return eventdata

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None,
    ):
        """Calculates the probability density for the given trial events given
        the specified local parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which the PDF values should get calculated.
        params_recarray : instance of numpy structured ndarray | None
            The (N_models,)-shaped numpy structured ndarray holding the local
            parameter names and values of the models.
            By definition, this PDF does not depend on any parameters.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : (N_values,)-shaped numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            value for each source and event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter. Since this PDF does not depend on
            any fit parameter, this is an empty dictionary.
        """
        if self._cache_pd_values:
            pd = self._get_cached_pd_values(
                tdm=tdm)
            if pd is not None:
                return (pd, dict())

        with TaskTimer(tl, 'Get PDF eventdata.'):
            if self.is_signal_pdf:
                eventdata = self.create_eventdata_for_sigpdf(
                    tdm=tdm,
                    axes=self._axes)
            elif self.is_background_pdf:
                eventdata = self.create_eventdata_for_bkgpdf(
                    tdm=tdm,
                    axes=self._axes)
            else:
                raise TypeError(
                    'The PDF is neither a signal nor a background PDF!')

        with TaskTimer(tl, 'Get pd for all selected events.'):
            # The call to get_pd_with_eventdata will cache the pd values.
            pd = self.get_pd_with_eventdata(
                tdm=tdm,
                params_recarray=params_recarray,
                eventdata=eventdata,
                tl=tl)

        return (pd, dict())


class PDFSet(
        HasConfig,
):
    """This class describes a set of PDF objects which are related to each other
    via different values of a set of parameters. A signal PDF usually
    consists of multiple same-kind PDFs for different signal parameters.
    In general background PDFs could have parameters, too.

    This class has the ``params_grid_set`` property holding the set of
    parameter grids. Also it holds a dictionary with the PDFs for the different
    sets of parameter values. PDF instances can be added via the :meth:`add_pdf`
    method and can be retrieved via the :meth:`get_pdf` method.
    """

    def __init__(
            self,
            param_grid_set,
            **kwargs):
        """Constructs a new PDFSet instance.

        Parameters
        ----------
        param_grid_set : instance of ParameterGrid |
                         instance of ParameterGridSet
            The instance of ParameterGridSet with the parameter grids defining
            the discrete parameter values for which the PDFs of this PDF set
            are made for.
        """
        # Call super to support multiple class inheritance.
        super().__init__(
            **kwargs)

        self.param_grid_set = param_grid_set

        self._gridparams_hash_pdf_dict = dict()

    @property
    def param_grid_set(self):
        """The ParameterGridSet instance defining the grid values of
        the different parameters.
        """
        return self._param_grid_set

    @param_grid_set.setter
    def param_grid_set(self, obj):
        if isinstance(obj, ParameterGrid):
            obj = ParameterGridSet([obj])
        if obj is not None:
            if not isinstance(obj, ParameterGridSet):
                raise TypeError(
                    'The params_grid_set property must be an instance of type '
                    'ParameterGridSet!')
        self._param_grid_set = obj

    @property
    def gridparams_list(self):
        """(read-only) The list of dictionaries of all the parameter
        permutations on the grid.
        """
        return self._param_grid_set.parameter_permutation_dict_list

    @property
    def pdf_keys(self):
        """(read-only) The list of stored PDF object keys.
        """
        return list(self._gridparams_hash_pdf_dict.keys())

    @property
    def axes(self):
        """(read-only) The PDFAxes object of one of the PDFs of this PDF set.
        All PDFs of this set are supposed to have the same axes.
        """
        key = next(iter(self._gridparams_hash_pdf_dict.keys()))
        return self._gridparams_hash_pdf_dict[key].axes

    def __contains__(self, key):
        """Checks if the given key exists in this PDFSet instance.

        Parameters
        ----------
        key : dict | int
            If a dictionary is provided, it must be the gridparams dictionary
            containing the grid parameter names and vales.
            If an integer is provided, it must be the hash of the gridparams
            dictionary.
        """
        if isinstance(key, dict):
            key = make_dict_hash(key)

        if not isinstance(key, int):
            raise TypeError(
                'The key argument must be of type dict or int! '
                f'currently its type is {classname(key)}.')

        return key in self._gridparams_hash_pdf_dict

    def __getitem__(self, key):
        """Implements the access operator ``self[gridparams_hash]``.
        """
        return self.get_pdf(key)

    def __iter__(self):
        """Returns an iterator of the PDF dictionary of this PDFSet.
        """
        return iter(self._gridparams_hash_pdf_dict)

    def items(self):
        """Returns an iterator over the (gridparams_hash, PDF) pairs of this
        PDFSet instance.
        """
        return self._gridparams_hash_pdf_dict.items()

    def values(self):
        """Returns an iterator over the PDF instances of the PDFSet instance.
        """
        return self._gridparams_hash_pdf_dict.values()

    def make_key(self, gridparams):
        """Creates the key for the given grid parameter dictionary.

        Parameters
        ----------
        gridparams : dict
            The dictionary holding the grid parameter names and values.

        Returns
        -------
        key : int
            The key for the given grid parameter dictionary.
        """
        return make_dict_hash(gridparams)

    def add_pdf(self, pdf, gridparams):
        """Adds the given PDF object for the given parameters to the internal
        registry. If this PDF set is not empty, the to-be-added PDF must have
        the same axes than the already added PDFs.

        Parameters
        ----------
        pdf : instance of PDF
            The PDF instance, that should be added
        gridparams : dict
            The dictionary with the grid parameter values, which identify
            the PDF object.

        Raises
        ------
        KeyError
            If the given PDF was already added for the given set of parameters.
        TypeError
            If any of the method's arguments has the wrong type.
        ValueError
            If the axes of the given PDFs are not the same as the axes of the
            already added PDFs.
        """
        logger = get_logger(f'{__name__}.{classname(self)}.add_pdf')

        if not isinstance(pdf, PDF):
            raise TypeError(
                'The pdf argument must be an instance of PDF!'
                f'But its type is "{classname(pdf)}!')
        if not isinstance(gridparams, dict):
            raise TypeError(
                'The gridparams argument must be of type dict!'
                f'But its type is "{classname(gridparams)}"!')

        gridparams_hash = make_dict_hash(gridparams)
        if gridparams_hash in self._gridparams_hash_pdf_dict:
            raise KeyError(
                f'The PDF with grid parameters {str(gridparams)} was '
                'already added!')

        # Check that the new PDF has the same axes than the already added PDFs.
        if len(self._gridparams_hash_pdf_dict) > 0:
            some_pdf = self._gridparams_hash_pdf_dict[
                next(iter(self._gridparams_hash_pdf_dict.keys()))]
            if not pdf.axes.is_same_as(some_pdf.axes):
                raise ValueError(
                    'The given PDF does not have the same axes than the '
                    'already added PDFs!\n'
                    f'New axes:\n{str(pdf.axes)}\n'
                    f'Old axes:\n{str(some_pdf.axes)}')

        if self._cfg.is_tracing_enabled:
            logger.debug(f'Adding PDF for gridparams {gridparams}.')

        self._gridparams_hash_pdf_dict[gridparams_hash] = pdf

    def get_pdf(self, gridparams):
        """Retrieves the PDF object for the given set of fit parameters.

        Parameters
        ----------
        gridparams : dict | int
            The dictionary with the grid parameter names and values for which
            the PDF object should get retrieved. If an integer is given, it is
            assumed to be the PDF key.

        Returns
        -------
        pdf : instance if PDF
            The PDF instance for the given parameters.

        Raises
        ------
        KeyError
            If no PDF instance was created for the given set of parameters.
        """
        if isinstance(gridparams, int):
            gridparams_hash = gridparams
        elif isinstance(gridparams, dict):
            gridparams_hash = make_dict_hash(gridparams)
        else:
            raise TypeError(
                'The gridparams argument must be of type dict or int!')

        if gridparams_hash not in self._gridparams_hash_pdf_dict:
            raise KeyError(
                'No PDF was created for the parameter set '
                f'"{str(gridparams)}"!')

        pdf = self._gridparams_hash_pdf_dict[gridparams_hash]

        return pdf

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """This method is called whenever a new trial data is available. It
        calls the :meth:`~skyllh.core.pdf.PDF.initialize_for_new_trial` method
        of each PDF.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the new trial data events.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.
        """
        for pdf in self._gridparams_hash_pdf_dict.values():
            pdf.initialize_for_new_trial(
                tdm=tdm,
                tl=tl,
                **kwargs)

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Checks if the PDFs of this PDFSet instance are valid for all the
        given trial data events.
        Since all PDFs should have the same axes, only the first PDF will be
        checked. It calls the
        :meth:`~skyllh.core.pdf.PDF.assert_is_valid_for_trial_data` method of
        the first :class:`~skyllh.core.pdf.PDF` instance.

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
        key = next(iter(self._gridparams_hash_pdf_dict.keys()))
        pdf = self._gridparams_hash_pdf_dict[key]
        pdf.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def get_pd(
            self,
            gridparams,
            tdm,
            params_recarray=None,
            tl=None):
        """Calls the ``get_pd`` method of the PDF instance that belongs to the
        given grid parameter values ``gridparams``.

        Parameters
        ----------
        gridparams : dict
            The dictionary holding the parameter values, which define PDF
            instance within this PDFSet instance.
            Note, that the parameter values must match a set of parameter grid
            values for which a PDF instance has been created and added to this
            PDFSet instance.
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability density of the events should be calculated.
        params_recarray : instance of ndarray | None
            The numpy record ndarray holding the parameter name and values for
            each source model.

        Returns
        -------
        pd : numpy ndarray
            The 1D numpy ndarray holding the probability density values for each
            event and source.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for further information.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for further information.
        """
        pdf = self.get_pdf(gridparams)

        return pdf.get_pd(
            tdm=tdm,
            params_recarray=params_recarray,
            tl=tl)
