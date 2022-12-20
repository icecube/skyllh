# -*- coding: utf-8 -*-

from skyllh.core.binning import BinningDefinition
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Linear1DGridManifoldInterpolationMethod
)
from skyllh.core.py import (
    ObjectCollection,
    classname,
    func_has_n_args,
    issequenceof,
    typename
)
from skyllh.core.config import CFG
from skyllh.core.debugging import get_logger
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet,
    ParameterSet,
    make_params_hash
)
from skyllh.core.timing import TaskTimer
from skyllh.core.trialdata import TrialDataManager


import abc
import numpy as np
import scipy as scp

from scipy.interpolate import RegularGridInterpolator

# Try to load the photospline tool.
PHOTOSPLINE_LOADED = True
try:
    import photospline
except ImportError:
    PHOTOSPLINE_LOADED = False


logger = get_logger(__name__)


class PDFAxis(object):
    """This class describes an axis of a PDF. It's main purpose is to define
    the allowed variable space of the PDF. So this information can be used to
    plot a PDF or a PDF ratio.
    """

    def __init__(self, name, vmin, vmax):
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
        super(PDFAxis, self).__init__()

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
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

    @property
    def vmin(self):
        """The minimal value of the axis.
        """
        return self._vmin

    @vmin.setter
    def vmin(self, v):
        self._vmin = float(v)

    @property
    def vmax(self):
        """The maximal value of the axis.
        """
        return self._vmax

    @vmax.setter
    def vmax(self, v):
        self._vmax = float(v)

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
        if((self.name == other.name) and
           (self.vmin == other.vmin) and
           (self.vmax == other.vmax)
           ):
            return True
        return False

    def __str__(self):
        """Pretty string implementation for the PDFAxis instance.
        """
        s = '{}: {}: vmin={:g} vmax={:g}'.format(
            classname(self), self._name, self._vmin, self._vmax)
        return s


class PDFAxes(ObjectCollection):
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
        if(not issequenceof(axeses, PDFAxes)):
            raise TypeError('The arguments of the union static function must '
                            'be instances of PDFAxes!')
        if(not len(axeses) >= 1):
            raise ValueError('At least 1 PDFAxes instance must be provided to '
                             'the union static function!')

        axes = PDFAxes(axes=axeses[0])
        for axes_i in axeses[1:]:
            for axis in axes_i:
                if(not axes.has_axis(axis.name)):
                    axes += axis

        return axes

    def __init__(self, axes=None):
        super(PDFAxes, self).__init__(objs=axes, obj_type=PDFAxis)

    def __str__(self):
        """Pretty string implementation for the PDFAxes instance.
        """
        s = ''
        for i in range(len(self)):
            if(i > 0):
                s += '\n'
            s += str(self[i])
        return s

    @property
    def axis_name_list(self):
        """(read-only) The list of the names of all the axes of this PDFAxes
        instance.
        """
        return [ axis.name for axis in self ]

    def get_axis(self, name):
        """Retrieves the PDFAxis object with the given name.

        Parameters
        ----------
        name : str | int
            The name of the axis to retrieve. If an integer is given, it
            specifies the index of the axis.

        Returns
        -------
        axis : PDFAxis
            The PDFAxis object.

        Raises
        ------
        KeyError
            If the axis could not be found.
        """
        if(isinstance(name, int)):
            return self[name]

        for axis in self:
            if(axis.name == name):
                return axis

        raise KeyError(
            'The PDFAxis with name "%s" could not be found!' % (name))

    def has_axis(self, name):
        """Checks if an axis of the given name is present in this PDFAxes
        instance.

        Parameters
        ----------
        name : str
            The name of this axis.

        Returns
        -------
        check : bool
            True, if this Axis object has an axis of the given name,
            False otherwise.
        """
        if(not isinstance(name, str)):
            raise TypeError(
                'The name argument must be an instance of str!')

        for axis in self:
            if(axis.name == name):
                return True
        return False

    def is_same_as(self, axes):
        """Checks if this PDFAxes object has the same axes and range then the
        given PDFAxes object.

        Returns
        -------
        check : bool
            True, if this PDFAxes and the given PDFAxes have the same axes and
            ranges. False otherwise.
        """
        if(len(self) != len(axes)):
            return False
        for i in range(len(self)):
            if(not self[i] == axes[i]):
                return False

        return True


class IsBackgroundPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a background PDF. This is useful for type checking.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsBackgroundPDF class.
        """
        super(IsBackgroundPDF, self).__init__(*args, **kwargs)

    def __mul__(self, other):
        """Creates a BackgroundPDFProduct instance for the multiplication of
        this background PDF and another background PDF.

        Parameters
        ----------
        other : instance of IsBackgroundPDF
            The instance of IsBackgroundPDF, which is the other background PDF.
        """
        if(not isinstance(other, IsBackgroundPDF)):
            raise TypeError('The other PDF must be an instance of '
                            'IsBackgroundPDF!')

        return BackgroundPDFProduct(self, other)


class IsSignalPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a signal PDF.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsSignalPDF class.
        """
        super(IsSignalPDF, self).__init__(*args, **kwargs)

    def __mul__(self, other):
        """Creates a SignalPDFProduct instance for the multiplication of this
        signal PDF and another signal PDF.

        Parameters
        ----------
        other : instance of IsSignalPDF
            The instance of IsSignalPDF, which is the other signal PDF.
        """
        if(not isinstance(other, IsSignalPDF)):
            raise TypeError('The other PDF must be an instance of '
                            'IsSignalPDF!')

        return SignalPDFProduct(self, other)


class PDF(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for all probability distribution
    function (PDF) models.
    All PDF model classes must be derived from this class. Mathematically, it
    represents :math::`f(\vec{x}|\vec{p})`, where :math::`\vec{x}` is the
    event data and :math::`\vec{p}` is the set of parameters the PDF is given
    for.
    """

    def __init__(self, param_set=None, **kwargs):
        """Creates a new PDF instance.

        Parameters
        ----------
        param_set : Parameter instance | sequence of Parameter instances |
                   ParameterSet instance | None
            If this PDF depends on parameters, this set of parameters
            defines them. If a single parameter instance is given a ParameterSet
            instance will be created holding this single parameter.
            If set to None, this PDF will not depend on any parameters.
        """
        # Make sure that multiple inheritance can be used. This super call will
        # invoke the __init__ method of a possible second inheritance.
        super(PDF, self).__init__(**kwargs)

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
    def param_set(self):
        """The ParameterSet instance defining the set of parameters this PDF
        depends on. It is ``None``, if this PDF does not depend on any
        parameters.
        """
        return self._param_set

    @param_set.setter
    def param_set(self, param_set):
        if(param_set is None):
            param_set = ParameterSet()
        elif(not isinstance(param_set, ParameterSet)):
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
        if(not isinstance(axis, PDFAxis)):
            raise TypeError(
                'The axis argument must be an instance of PDFAxis!')
        self._axes += axis

    def assert_is_valid_for_trial_data(self, tdm):
        """This method is supposed to check if this PDF is valid for
        all the given trial data. This means, it needs to check if there
        is a PDF value for each trial data event that will be used in the
        likelihood evaluation. This is just a seatbelt.
        The method must raise a ValueError if the PDF is not valid for the
        given trial data.
        """
        raise NotImplementedError('The derived PDF class "%s" did not '
                                  'implement the "assert_is_valid_for_trial_data" method!' % (
                                      classname(self)))

    @abc.abstractmethod
    def get_prob(self, tdm, params=None, tl=None):
        """This abstract method is supposed to calculate the probability density
        for the specified events given the specified parameter values.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. What data fields are required
            is defined by the derived PDF class and depends on the application.
        params : dict | None
            The dictionary containing the parameter names and values for which
            the probability should get calculated.
            This can be ``Ǹone`` for PDFs that do not depend on any parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability density for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.
        """
        pass


class PDFProduct(PDF, metaclass=abc.ABCMeta):
    """The PDFProduct class represents the product of two PDF instances. It
    is derived from the PDF class and hence is a PDF itself.
    """

    def __init__(self, pdf1, pdf2):
        """Creates a new PDFProduct instance, which implements the operation
        `pdf1 * pdf2`.
        The axes of the two PDF instances will be merged.

        Parameters
        ----------
        pdf1 : instance of PDF
            The left-hand-side PDF in the operation `pdf1 op pdf2`.
        pdf2 : instance of PDF
            The right-hand-side PDF in the operation `pdf1 op pdf2`.
        """
        self.pdf1 = pdf1
        self.pdf2 = pdf2

        # Create the ParameterSet instance that is the union of the ParameterSet
        # instances of the two PDFs.
        param_set = ParameterSet.union(
            self._pdf1.param_set, self._pdf2.param_set)

        super(PDFProduct, self).__init__(
            param_set=param_set)

        # The resulting PDFAxes object of this PDF instance is the union of the
        # two PDFAxes instances of the two PDF instances.
        self._axes = PDFAxes.union(pdf1.axes, pdf2.axes)

    @property
    def pdf1(self):
        """The left-hand-side PDF in the operation `pdf1 op pdf2`. It must be an
        instance of PDF.
        """
        return self._pdf1

    @pdf1.setter
    def pdf1(self, pdf):
        if(not isinstance(pdf, PDF)):
            raise TypeError('The pdf1 property must be an instance of PDF!')
        self._pdf1 = pdf

    @property
    def pdf2(self):
        """The right-hand-side PDF in the operation `pdf1 op pdf2`. It must be
        an instance of PDF.
        """
        return self._pdf2

    @pdf2.setter
    def pdf2(self, pdf):
        if(not isinstance(pdf, PDF)):
            raise TypeError('The pdf2 property must be an instance of PDF!')
        self._pdf2 = pdf

    def assert_is_valid_for_trial_data(self, tdm):
        """Calls the ``assert_is_valid_for_trial_data`` method of ``pdf1`` and
        ``pdf2``.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance that should be used to get the trial
            data from.

        Raises
        ------
        ValueError
            If this PDF does not cover the trial data.
        """
        self._pdf1.assert_is_valid_for_trial_data(tdm)
        self._pdf2.assert_is_valid_for_trial_data(tdm)

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability density for the trial events given the
        specified parameters by calling the `get_prob` method of `pdf1`
        and `pdf2` and combining the two property densities by multiplication.
        The gradients will be calculated using the product rule of
        differentiation.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params : dict | None
            The dictionary containing the parameter names and values for
            which the probability should get calculated.
        tl : TimeLord instance | None
            The optional TimeLord instance to use for measuring timing
            information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability for each trial event.
        grads : (N_fitparams,N_events)-shaped numpy ndarray
            The gradients of the PDF product w.r.t. the fit parameter of this
            PDFProduct instance.

        """
        pdf1 = self._pdf1
        pdf2 = self._pdf2

        with TaskTimer(tl, 'Get signal prob from table.'):
            (prob1, grads1) = pdf1.get_prob(tdm, params, tl=tl)
            (prob2, grads2) = pdf2.get_prob(tdm, params, tl=tl)

        prob = prob1 * prob2

        pdf1_param_set = pdf1.param_set
        pdf2_param_set = pdf2.param_set

        N_events = prob.shape[0]
        fitparam_names = self.param_set.floating_param_name_list
        grads = np.zeros((len(fitparam_names), N_events), dtype=np.float64)
        for (pidx, fitparam_name) in enumerate(fitparam_names):
            # Calculate the gradient w.r.t. fitparam.

            # There are four possible cases to calculate the gradient for
            # the parameter fitparam:
            #     1. Both PDFs depend on this fit parameter, the gradient is
            #        calculated through the product rule of differentiation.
            #     2. Only PDF1 depends on this fit parameter.
            #     3. Only PDF2 depends on this fit parameter.
            #     4. Both PDFs are independ of this fit parameter, the gradient
            #        is 0.
            pdf1_has_fitparam = pdf1_param_set.has_floating_param(
                fitparam_name)
            pdf2_has_fitparam = pdf2_param_set.has_floating_param(
                fitparam_name)
            if(pdf1_has_fitparam and pdf2_has_fitparam):
                # Case 1
                grad1 = grads1[pdf1.param_set.get_floating_pidx(fitparam_name)]
                grad2 = grads2[pdf2.param_set.get_floating_pidx(fitparam_name)]
                grads[pidx] = prob2*grad1 + prob1*grad2
            elif(pdf1_has_fitparam):
                # Case 2
                grad1 = grads1[pdf1.param_set.get_floating_pidx(fitparam_name)]
                grads[pidx] = prob2*grad1
            elif(pdf2_has_fitparam):
                # Case 3
                grad2 = grads2[pdf2.param_set.get_floating_pidx(fitparam_name)]
                grads[pidx] = prob1*grad2

        n_src = len(tdm.get_data('src_array')['ra'])
        if(n_src == 1):
            # Only one source in the signal hypothesis.
            return (prob, grads)
        else:
            # Signal hypothesis contains multiple sources, and the overall weight is obtained by
            # multiplying the detector weight src_w, and the hypothesis weight src_w_W.

            src_w = tdm.get_data('src_array')['src_w'] * tdm.get_data('src_array')['src_w_W']
            src_w_grad = tdm.get_data('src_array')['src_w_grad'] * tdm.get_data('src_array')['src_w_W']

            # Normalize source weights and grads.
            norm_src_w_temp = src_w.sum()
            src_w /= norm_src_w_temp
            src_w_grad /= norm_src_w_temp

            src_ev_idxs = tdm.src_ev_idxs

            if src_ev_idxs is not None:
                (src_idxs, ev_idxs) = src_ev_idxs
                prob = scp.sparse.csr_matrix((prob, (ev_idxs, src_idxs)))
            else:
                prob = prob.reshape((n_src, int(prob.shape[0]/n_src))).T
            prob_res = prob.dot(src_w)

            n_ev = tdm.n_selected_events
            norm_w = src_w.sum()

            grads_tot = np.zeros((len(fitparam_names), n_ev), dtype=np.float64)
            for (pidx, fitparam_name) in enumerate(fitparam_names):
                if src_ev_idxs is not None:
                    grad_i = scp.sparse.csr_matrix((grads[pidx], (ev_idxs, src_idxs)))
                else:
                    grad_i = prob.reshape((n_src, int(grads[pidx].shape[0]/n_src))).T

                if fitparam_name == 'gamma':
                    d_wf = prob.dot(src_w_grad) + grad_i.dot(src_w)
                    grads_tot[pidx] = (d_wf * norm_w - src_w_grad.sum() * prob_res) / norm_w**2
                else:
                    d_wf = grad_i.dot(src_w)
                    grads_tot[pidx] = (d_wf * norm_w) / norm_w**2

            return (prob_res, grads_tot)


class SignalPDFProduct(PDFProduct, IsSignalPDF):
    """This class provides a signal PDF that is the product of two signal PDF
    instances.
    """

    def __init__(self, pdf1, pdf2):
        super(SignalPDFProduct, self).__init__(pdf1, pdf2)


class BackgroundPDFProduct(PDFProduct, IsBackgroundPDF):
    """This class provides a background PDF that is the product of two
    background PDF instances.
    """

    def __init__(self, pdf1, pdf2):
        super(BackgroundPDFProduct, self).__init__(pdf1, pdf2)


class SpatialPDF(PDF, metaclass=abc.ABCMeta):
    """This is the abstract base class for a spatial PDF model. A spatial PDF
    has two axes, right-ascention (ra) and declination (dec).
    """

    def __init__(self, ra_range, dec_range, *args, **kwargs):
        """Constructor of a spatial PDF. It adds the PDF axes "ra" and "dec"
        with the specified ranges of coverage.

        Parameters
        ----------
        ra_range : 2-element tuple
            The tuple specifying the right-ascention range this PDF covers.
        dec_range : 2-element tuple
            The tuple specifying the declination range this PDF covers.
        """
        super(SpatialPDF, self).__init__(*args, **kwargs)

        self.add_axis(PDFAxis(name='ra',
                              vmin=ra_range[0], vmax=ra_range[1]))
        self.add_axis(PDFAxis(name='dec',
                              vmin=dec_range[0], vmax=dec_range[1]))

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this spatial PDF is valid for all the given experimental
        data.
        It checks if all the data is within the right-ascention and declination
        range.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:

            - 'ra' : float
                The right-ascention of the data event.
            - 'dec' : float
                The declination of the data event.

        Raises
        ------
        ValueError
            If some of the data is outside the right-ascention or declination
            range.
        """
        ra_axis = self.get_axis('ra')
        dec_axis = self.get_axis('dec')

        sinDec_binning = self.get_binning('sin_dec')
        exp_sinDec = np.sin(data_exp['dec'])

        # Check if all the data is within the right-ascention range.
        if(np.any((data_exp['ra'] < ra_axis.vmin) |
                  (data_exp['ra'] > ra_axis.vmax))):
            raise ValueError(
                'Some data is outside the right-ascention range (%.3f, %.3f)!' % (ra_axis.vmin, ra_axis.vmax))

        # Check if all the data is within the declination range.
        if(np.any((data_exp['dec'] < dec_axis.vmin) |
                  (data_exp['dec'] > dec_axis.vmax))):
            raise ValueError('Some data is outside the declination range (%.3f, %.3f)!' % (
                dec_axis.vmin, dec_axis.vmax))


class EnergyPDF(PDF, metaclass=abc.ABCMeta):
    """This is the abstract base class for an energy PDF model.
    """

    def __init__(self, *args, **kwargs):
        super(EnergyPDF, self).__init__(*args, **kwargs)


class TimePDF(PDF, metaclass=abc.ABCMeta):
    """This is the abstract base class for a time PDF model.
    """

    def __init__(self, *args, **kwargs):
        super(TimePDF, self).__init__(*args, **kwargs)


class MultiDimGridPDF(PDF):
    """This class provides a multi-dimensional PDF created from pre-calculated
    PDF data on a grid. The grid data is interpolated using a
    :class:`scipy.interpolate.RegularGridInterpolator` instance.
    """

    def __init__(
            self, axis_binnings, path_to_pdf_splinetable=None,
            pdf_grid_data=None, norm_factor_func=None):
        """Creates a new PDF instance for a multi-dimensional PDF given
        as PDF values on a grid. The grid data is interpolated with a
        :class:`scipy.interpolate.RegularGridInterpolator` instance. As grid
        points the bin edges of the axis binning definitions are used.

        Parameters
        ----------
        axis_binnings : BinningDefinition | sequence of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each BinningDefinition instance defines
            the event field name that should be used for querying the PDF.
        path_to_pdf_splinetable : str | None
            The path to the file containing the spline table.
            The spline table contains a pre-computed fit to pdf_grid_data.
        pdf_grid_data : n-dimensional numpy ndarray | None
            The n-dimensional numpy ndarray holding the PDF values at given grid
            points. The grid points must match the bin edges of the given
            BinningDefinition instances of the `axis_binnings` argument.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function must be
            `__call__(pdf, tdm, params)`, where `pdf` is this PDF
            instance, `tdm` is an instance of TrialDataManager holding the
            event data for which to calculate the PDF values, and `params` is a
            dictionary with the current parameter names and values.
        """
        super(MultiDimGridPDF, self).__init__()

        # Need either splinetable or grid of pdf values.
        if((path_to_pdf_splinetable is None) and (pdf_grid_data is None)):
            raise ValueError(
                'At least one of the following arguments are required: '
                'path_to_pdf_splinetable (str) or '
                'pdf_grid_data (numpy.ndarray)!')
        elif((path_to_pdf_splinetable is not None) and
           (pdf_grid_data is not None)):
            raise ValueError(
                'Only one of the two arguments path_to_pdf_splinetable and '
                'pdf_grid_data can be specified!')

        # If a path to the photospline tables is given, we raise an error if
        # the photospline package is not loaded.
        if(path_to_pdf_splinetable is not None):
            if(not isinstance(path_to_pdf_splinetable, str)):
                raise TypeError(
                    'The path_to_pdf_splinetable argument must be None or of '
                    'type str!')

            if(not PHOTOSPLINE_LOADED):
                raise ImportError(
                    'The path_to_pdf_splinetable argument is specified, but '
                    'the "photospline" package is not available!')

        if(pdf_grid_data is not None):
            if(not isinstance(pdf_grid_data, np.ndarray)):
                raise TypeError(
                    'The pdf_grid_data argument must be an instance of numpy '
                    'ndarray. The current type is {}!'.format(
                        type(pdf_grid_data)))

        self.axis_binning_list = axis_binnings
        self.norm_factor_func = norm_factor_func

        # Define the PDF axes.
        for axis_binning in self._axis_binnning_list:
            self.add_axis(PDFAxis(
                name=axis_binning.name,
                vmin=axis_binning.lower_edge,
                vmax=axis_binning.upper_edge
            ))

        # Create the internal PDF object.
        if(path_to_pdf_splinetable is not None):
            self._pdf = photospline.SplineTable(path_to_pdf_splinetable)
        else:
            self._pdf = RegularGridInterpolator(
                tuple([binning.binedges for binning in self._axis_binnning_list]),
                pdf_grid_data,
                method='linear',
                bounds_error=False,
                fill_value=0
            )

        # Because this PDF does not depend on any fit parameters, the PDF values
        # can be cached as long as the trial data state ID of the trial data
        # manager has not changed.
        self._cache_tdm_trial_data_state_id = None
        self._cache_prob = None

        logger.debug(
            'Created %s instance with axis name list %s' % (
                classname(self), str(self._axes.axis_name_list)))

    @property
    def axis_binning_list(self):
        """The list of BinningDefinition instances for each PDF axis.
        The name of each BinningDefinition instance defines the event field
        name that should be used for querying the PDF.
        """
        return self._axis_binnning_list

    @axis_binning_list.setter
    def axis_binning_list(self, binnings):
        if(isinstance(binnings, BinningDefinition)):
            binnings = [binnings]
        if(not issequenceof(binnings, BinningDefinition)):
            raise TypeError(
                'The axis_binning_list property must be an instance of '
                'BinningDefinition or a sequence of BinningDefinition '
                'instances!')
        self._axis_binnning_list = list(binnings)

    @property
    def norm_factor_func(self):
        """The function that calculates the possible required normalization
        factor. The call signature of this function must be
        `__call__(pdf, tdm, fitparams)`, where `pdf` is this PDF
        instance, `tdm` is an instance of TrialDataManager holding the events
        for which to calculate the PDF values, and `fitparams` is a dictionary
        with the current fit parameter names and values. This property can be
        set to `None`. In that case a unity returning function is used.
        """
        return self._norm_factor_func

    @norm_factor_func.setter
    def norm_factor_func(self, func):
        if(func is None):
            # Define a normalization function that just returns 1 for each
            # event.
            def func(pdf, tdm, fitparams, eventdata):
                n_src = len(tdm.get_data('src_array')['ra'])
                if(n_src == 1):
                    n_dim = tdm.n_selected_events
                else:
                    n_dim = eventdata.shape[0]
                return np.ones((n_dim,), dtype=np.float64)

        if(not callable(func)):
            raise TypeError(
                'The norm_factor_func property must be a callable object!')
        if(not func_has_n_args(func, 4)):
            raise TypeError(
                'The norm_factor_func property must be a function with 4 '
                'arguments!')
        self._norm_factor_func = func

    def assert_is_valid_for_trial_data(self, tdm):
        """Checks if the PDF is valid for all values of the given evaluation
        data. The evaluation data values must be within the ranges of the PDF
        axes.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The instance of TrialDataManager that holds the data which is going
            to be evaluated.

        Raises
        ------
        ValueError
            If any of the evaluation data is out of its axis range.
        """
        for axis in self._axes:
            data = tdm.get_data(axis.name)
            if(np.any(data < axis.vmin) or
               np.any(data > axis.vmax)
               ):
                raise ValueError(
                    'Some of the trial data for PDF axis '
                    '"%s" is out of range (%g,%g)!' % (
                        axis.name, axis.vmin, axis.vmax))

    def get_prob_with_eventdata(self, tdm, params, eventdata, tl=None):
        """Calculates the probability for the trial events given the specified
        parameters. This method has the additional argument ``eventdata`` which
        must be a 2d ndarray containing the trial event data in the correct
        order for the evaluation of the RegularGridInterpolator or photospline
        table instance.
        This method is usefull when PDF values for the same trial data need to
        get evaluated.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params : dict | None
            The dictionary containing the parameters the probability should get
            calculated for. By definition, this PDF does not depend on any
            parameters.
        eventdata : 2D (N_events,V)-shaped ndarray
            The 2D numpy ndarray holding the V data attributes for each event
            needed for the evaluation of the PDF.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        """
        do_caching = CFG['caching']['pdf']['MultiDimGridPDF']
        if(do_caching):
            tdm_trial_data_state_id = tdm.trial_data_state_id
            cache_tdm_trial_data_state_id = self._cache_tdm_trial_data_state_id

            if(cache_tdm_trial_data_state_id == tdm_trial_data_state_id):
                return self._cache_prob

        if(isinstance(self._pdf, RegularGridInterpolator)):
            with TaskTimer(tl, 'Get prob from RegularGridInterpolator.'):
                prob = self._pdf(eventdata)
        else:
            with TaskTimer(tl, 'Get prob from photospline fit.'):
                V = eventdata.shape[1]
                prob = self._pdf.evaluate_simple(
                    [eventdata[:, i] for i in range(0, V)])

        with TaskTimer(tl, 'Normalize MultiDimGridPDF with norm factor.'):
            norm = self._norm_factor_func(self, tdm, params, eventdata)
            prob *= norm
        if(do_caching):
            self._cache_tdm_trial_data_state_id = tdm_trial_data_state_id
            self._cache_prob = prob

        return prob

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability for the trial events given the specified
        parameters.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params : dict | None
            The dictionary containing the parameter names and values the
            probability should get calculated for. Since this PDF does not
            depend on any parameters, this could be ``None``.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        grads : None
            Because this PDF does not depend on any parameters, no gradients
            w.r.t. the parameters are returned.
        """
        do_caching = CFG['caching']['pdf']['MultiDimGridPDF']

        if(do_caching):
            tdm_trial_data_state_id = tdm.trial_data_state_id
            cache_tdm_trial_data_state_id = self._cache_tdm_trial_data_state_id

            if((cache_tdm_trial_data_state_id is not None) and
               (cache_tdm_trial_data_state_id == tdm_trial_data_state_id)):
                return (self._cache_prob, None)

        with TaskTimer(tl, 'Get PDF event data.'):
            if(self.is_signal_pdf):
                # Evaluate the relevant quantities for
                # all events and sources (relevant for stacking analyses).
                if tdm.src_ev_idxs is not None:
                    (src_idxs, ev_idxs) = tdm.src_ev_idxs
                    eventdata = np.array(
                        [
                            # Check `psi` axis name.
                            tdm.get_data(axis.name)
                            if ('psi' in axis.name)

                            # Check `src` axis name.
                            else tdm.get_data(axis.name)[src_idxs]
                            if ('src' in axis.name)

                            # Default case.
                            else tdm.get_data(axis.name)[ev_idxs]
                            for axis in self._axes
                        ]
                    ).T
                else:
                    n_src = len(tdm.get_data('src_array')['ra'])
                    l_ev = len(tdm.get_data('ra'))
                    eventdata = np.array(
                        [
                            # Check `psi` axis name.
                            tdm.get_data(axis.name)
                            if ('psi' in axis.name)

                            # Check `src` axis name.
                            else tdm.get_data(axis.name)
                            if (('src' in axis.name) and (n_src == 1))
                            else np.repeat(tdm.get_data(axis.name), l_ev)
                            if (('src' in axis.name) and (n_src != 1))

                            # Default case.
                            else np.tile(tdm.get_data(axis.name), n_src)
                            for axis in self._axes
                        ]
                    ).T
            elif (self.is_background_pdf):
                eventdata = np.array(
                    [tdm.get_data(axis.name) for axis in self._axes]).T
            else:
                raise TypeError('Pdf type is unknown!')

        with TaskTimer(tl, 'Get prob for all selected events.'):
            prob = self.get_prob_with_eventdata(tdm, params, eventdata, tl=tl)
        if(do_caching):
            self._cache_tdm_trial_data_state_id = tdm_trial_data_state_id
            self._cache_prob = prob

        return (prob, None)


class NDPhotosplinePDF(PDF):
    """This class provides a multi-dimensional PDF created from a n-dimensional
    photospline fit. The photospline package is used to evaluate the PDF fit.
    """

    def __init__(
            self,
            axis_binnings,
            param_set,
            path_to_pdf_splinefit,
            norm_factor_func=None):
        """Creates a new PDF instance for a n-dimensional photospline PDF fit.

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
        if(not PHOTOSPLINE_LOADED):
            raise ImportError(
                'The photospline package could not be loaded!')

        super(NDPhotosplinePDF, self).__init__(
            param_set=param_set)

        self._n_fitparams = self._param_set.n_floating_params

        if(isinstance(axis_binnings, BinningDefinition)):
            axis_binnings = [axis_binnings]
        if(not issequenceof(axis_binnings, BinningDefinition)):
            raise TypeError(
                'The axis_binnings argument must be an instance of '
                'BinningDefinition or a sequence of BinningDefinition '
                'instances!')

        if(not isinstance(path_to_pdf_splinefit, str)):
            raise TypeError(
                'The path_to_pdf_splinefit argument must be an instance of '
                'str!')

        self.norm_factor_func = norm_factor_func

        # Define the PDF axes and create a mapping of fit parameter names to
        # axis indices.
        self._fitparam_name_to_axis_idx_map = dict()
        for (axis_idx, axis_binning) in enumerate(axis_binnings):
            axis_name = axis_binning.name

            self.add_axis(PDFAxis(
                name=axis_name,
                vmin=axis_binning.lower_edge,
                vmax=axis_binning.upper_edge
            ))

            if(self._param_set.has_floating_param(axis_name)):
                self._fitparam_name_to_axis_idx_map[axis_name] = axis_idx

        self._pdf = photospline.SplineTable(path_to_pdf_splinefit)

        logger.debug(
            'Created %s instance with axis name list %s' % (
                classname(self), str(self._axes.axis_name_list)))

    @property
    def norm_factor_func(self):
        """The function that calculates the possible required normalization
        factor. The call signature of this function must be
        `__call__(pdf, tdm, fitparams)`, where `pdf` is this PDF
        instance, `tdm` is an instance of TrialDataManager holding the events
        for which to calculate the PDF values, and `fitparams` is a dictionary
        with the current fit parameter names and values. This property can be
        set to `None`. In that case a unity returning function is used.
        """
        return self._norm_factor_func
    @norm_factor_func.setter
    def norm_factor_func(self, func):
        if(func is None):
            # Define a normalization function that just returns 1 for each
            # event.
            def func(pdf, tdm, fitparams):
                n_src = len(tdm.get_data('src_array')['ra'])
                if(n_src == 1):
                    n_dim = tdm.n_selected_events
                else:
                    if tdm.src_ev_idxs is None:
                        n_dim = tdm.n_selected_events * n_src
                    else:
                        n_dim = len(tdm.src_ev_idxs[0])
                return np.ones((n_dim,), dtype=np.float64)

        if(not callable(func)):
            raise TypeError(
                'The norm_factor_func property must be a callable object!')
        if(not func_has_n_args(func, 3)):
            raise TypeError(
                'The norm_factor_func property must be a function with 3 '
                'arguments!')
        self._norm_factor_func = func

    def assert_is_valid_for_trial_data(self, tdm):
        """Checks if the PDF is valid for all values of the given trial data.
        The trial data values must be within the ranges of the PDF axes.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The instance of TrialDataManager that holds the trial data which is
            going to be evaluated.

        Raises
        ------
        ValueError
            If any of the trial data is out of its axis range.
        """
        for axis in self._axes:
            data = tdm.get_data(axis.name)
            if(np.any(data < axis.vmin) or
               np.any(data > axis.vmax)
               ):
                raise ValueError(
                    'Some of the trial data for PDF axis '
                    '"%s" is out of range (%g,%g)!' % (
                        axis.name, axis.vmin, axis.vmax))

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability for the trial events given the specified
        parameters.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        params : dict | None
            The dictionary containing the parameter names and values the
            probability should get calculated for.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.
        """
        with TaskTimer(tl, 'Get PDF event data.'):
            if self.is_signal_pdf:
                if tdm.src_ev_idxs is not None:
                    (src_idxs, ev_idxs) = tdm.src_ev_idxs
                    eventdata = np.empty(
                        (len(ev_idxs), len(self._axes)), dtype=np.float64)
                    for (axis_idx, axis) in enumerate(self._axes):
                        axis_name = axis.name
                        if(axis_name in tdm):
                            if 'src' in axis_name:
                                axis_data = tdm.get_data(axis_name)[src_idxs]
                            elif 'psi' in axis_name:
                                axis_data = tdm.get_data(axis_name)
                            else:
                                axis_data = tdm.get_data(axis_name)[ev_idxs]
                        else:
                            # The requested data field (for the axis) is not
                            # part of the trial data, so it must be a parameter.
                            if(axis_name not in params):
                                raise KeyError(
                                    'The PDF axis "{}" is not part of the '
                                    'trial data and is not a parameter!'.format(
                                        axis_name))

                            axis_data = np.full(
                                (len(ev_idxs),), params[axis_name],
                                dtype=np.float64)
                        eventdata[:, axis_idx] = axis_data

            elif self.is_background_pdf:
                eventdata = np.empty(
                    (tdm.n_selected_events, len(self._axes)), dtype=np.float64)

                for (axis_idx, axis) in enumerate(self._axes):
                    axis_name = axis.name
                    if(axis_name in tdm):
                        axis_data = tdm.get_data(axis_name)
                    else:
                        # The requested data field (for the axis) is not part
                        # of the trial data, so it must be a parameter.
                        if(axis_name not in params):
                            raise KeyError(
                                'The PDF axis "{}" is not part of the trial data '
                                'and is not a parameter!'.format(
                                    axis_name))

                        axis_data = np.full(
                            (tdm.n_selected_events,), params[axis_name],
                            dtype=np.float64)
                eventdata[:, axis_idx] = axis_data
        self__pdf_evaluate_simple = self._pdf.evaluate_simple

        with TaskTimer(tl, 'Get prob from photospline fit.'):
            V = eventdata.shape[1]
            evaluate_simple_data = [eventdata[:, i] for i in range(0, V)]
            prob = self__pdf_evaluate_simple(evaluate_simple_data)

        with TaskTimer(tl, 'Normalize NDPhotosplinePDF with norm factor.'):
            norm = self._norm_factor_func(self, tdm, params)
            prob *= norm

        if(self._n_fitparams == 0):
            # This PDF does not depend on any fit parameters.
            return (prob, None)

        with TaskTimer(tl, 'Get grads from photospline fit.'):
            self__param_set = self._param_set
            grads = np.empty((self._n_fitparams, len(prob)), dtype=np.float64)
            # Loop through the fit parameters of this PDF and calculate their
            # derivative.
            for (fitparam_idx, fitparam_name) in enumerate(
                    self__param_set.floating_param_name_list):
                # Determine the axis index of this fit parameter.
                axis_idx = self._fitparam_name_to_axis_idx_map[fitparam_name]
                mode = 2**axis_idx
                grad = self__pdf_evaluate_simple(
                    evaluate_simple_data, mode)
                grad *= norm
                grads[fitparam_idx, :] = grad

        return (prob, grads)


class PDFSet(object):
    """This class describes a set of PDF objects which are related to each other
    via different values of a set of fit parameters. A signal PDF usually
    consists of multiple same-kind PDFs for different signal fit parameters.
    In general background PDFs could have fit parameters, too.

    This class has the ``fitparams_grid_set`` property holding the set of fit
    parameter grids. Also it holds a dictionary with the PDFs for the different
    sets of fit parameter values. The type of the PDF objects is defined through
    the ``pdf_type`` property. PDF objects of type ``pdf_type`` can be added
    via the ``add_pdf`` method and retrieved via the ``get_pdf`` method.
    """

    def __init__(self, pdf_type, fitparams_grid_set, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this PDFSet class.

        Parameters
        ----------
        pdf_type : type
            The PDF class that can be added to the set.
        fitparams_grid_set : ParameterGridSet | ParameterGrid
            The ParameterGridSet with the fit parameter grids defining the
            descrete fit parameter values for which the PDFs of this PDF set
            are made for.
        """
        # Call super to support multiple class inheritance.
        super(PDFSet, self).__init__(*args, **kwargs)

        if(not issubclass(pdf_type, PDF)):
            raise TypeError('The pdf_type argument must be a subclass of PDF!')
        self._pdf_type = pdf_type
        self.fitparams_grid_set = fitparams_grid_set
        self._gridfitparams_hash_pdf_dict = dict()

    @property
    def pdf_type(self):
        """(read-only) The PDF type which can be added to the PDF set.
        """
        return self._pdf_type

    @property
    def fitparams_grid_set(self):
        """ DEPRECATED (Use param_grid_set instead!)
        The ParameterGridSet object defining the value grids of
        the different fit parameters.
        """
        return self._fitparams_grid_set

    @fitparams_grid_set.setter
    def fitparams_grid_set(self, obj):
        if(isinstance(obj, ParameterGrid)):
            obj = ParameterGridSet([obj])
        # Allow None for the MappedMultiDimGridPDFSet construction.
        # Could create an unexpected behavior in other analyses!
        if(not isinstance(obj, ParameterGridSet) and obj is not None):
            raise TypeError('The fitparams_grid_set property must be an object '
                            'of type ParameterGridSet!')
        self._fitparams_grid_set = obj

    @property
    def param_grid_set(self):
        return self._fitparams_grid_set

    @property
    def gridfitparams_list(self):
        """(read-only) The list of dictionaries of all the fit parameter
        permutations on the grid.
        """
        return self._fitparams_grid_set.parameter_permutation_dict_list

    @property
    def pdf_keys(self):
        """(read-only) The list of stored PDF object keys.
        """
        return list(self._gridfitparams_hash_pdf_dict.keys())

    @property
    def pdf_axes(self):
        """(read-only) The PDFAxes object of one of the PDFs of this PDF set.
        All PDFs of this set are supposed to have the same axes.
        """
        key = next(iter(self._gridfitparams_hash_pdf_dict.keys()))
        return self._gridfitparams_hash_pdf_dict[key].axes

    def items(self):
        """Returns the list of 2-element tuples for the PDF stored in this
        PDFSet object.
        """
        return self._gridfitparams_hash_pdf_dict.items()

    def add_pdf(self, pdf, gridfitparams):
        """Adds the given PDF object for the given parameters to the internal
        registry. If this PDF set is not empty, the to-be-added PDF must have
        the same axes than the already added PDFs.

        Parameters
        ----------
        pdf : pdf_type
            The object derived from ``pdf_type`` that should be added.
        gridfitparams : dict
            The dictionary with the grid fit parameter values, which identify
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
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The pdf argument must be an instance of %s!' % (
                typename(self.pdf_type)))
        if(not isinstance(gridfitparams, dict)):
            raise TypeError('The fitparams argument must be of type dict!')

        gridfitparams_hash = make_params_hash(gridfitparams)
        if(gridfitparams_hash in self._gridfitparams_hash_pdf_dict):
            raise KeyError('The PDF with grid fit parameters %s was already '
                           'added!' % (str(gridfitparams)))

        # Check that the new PDF has the same axes than the already added PDFs.
        if(len(self._gridfitparams_hash_pdf_dict) > 0):
            some_pdf = self._gridfitparams_hash_pdf_dict[
                next(iter(self._gridfitparams_hash_pdf_dict.keys()))]
            if(not pdf.axes.is_same_as(some_pdf.axes)):
                raise ValueError(
                    'The given PDF does not have the same axes than the '
                    'already added PDFs!\n'
                    'New axes:\n{}\n'
                    'Old axes:\n{}'.format(
                        str(pdf.axes), str(some_pdf.axes))
                    )

        self._gridfitparams_hash_pdf_dict[gridfitparams_hash] = pdf

    def get_pdf(self, gridfitparams):
        """Retrieves the PDF object for the given set of fit parameters.

        Parameters
        ----------
        gridfitparams : dict | int
            The dictionary with the grid fit parameters for which the PDF object
            should get retrieved. If an integer is given, it is assumed to be
            the PDF key.

        Returns
        -------
        pdf : pdf_type
            The pdf_type object for the given parameters.

        Raises
        ------
        KeyError
            If no PDF object was created for the given set of parameters.
        """
        if(isinstance(gridfitparams, int)):
            gridfitparams_hash = gridfitparams
        elif(isinstance(gridfitparams, dict)):
            gridfitparams_hash = make_params_hash(gridfitparams)
        else:
            raise TypeError(
                'The gridfitparams argument must be of type dict or int!')

        if(gridfitparams_hash not in self._gridfitparams_hash_pdf_dict):
            raise KeyError(
                'No PDF was created for the parameter set "%s"!' % (str(gridfitparams)))

        pdf = self._gridfitparams_hash_pdf_dict[gridfitparams_hash]
        return pdf


class MultiDimGridPDFSet(PDF, PDFSet):
    def __init__(
            self, param_set, param_grid_set, gridparams_pdfs,
            interpolmethod=None, pdf_type=MultiDimGridPDF,
            **kwargs):
        """Creates a new MultiDimGridPDFSet instance, which holds a set of
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
        pdf_type : type
            The PDF class that can be added to the set.
        """
        super(MultiDimGridPDFSet, self).__init__(
            param_set=param_set,
            pdf_type=pdf_type,
            fitparams_grid_set=param_grid_set,
            **kwargs)

        if(interpolmethod is None):
            interpolmethod = Linear1DGridManifoldInterpolationMethod
        self.interpolmethod = interpolmethod

        # Add the given MultiDimGridPDF instances to the PDF set.
        for (gridparams, pdf) in gridparams_pdfs:
            self.add_pdf(pdf, gridparams)

        # Create the interpolation method instance.
        self._interpolmethod_instance = self._interpolmethod(
            self._get_prob_for_gridparams_with_eventdata_func(), param_grid_set)

    @property
    def interpolmethod(self):
        """The class derived from GridManifoldInterpolationMethod
        implementing the interpolation of the PDF grid manifold.
        """
        return self._interpolmethod

    @interpolmethod.setter
    def interpolmethod(self, cls):
        if(not issubclass(cls, GridManifoldInterpolationMethod)):
            raise TypeError('The interpolmethod property must be a sub-class '
                            'of GridManifoldInterpolationMethod!')
        self._interpolmethod = cls

    def _get_prob_for_gridparams_with_eventdata_func(self):
        """Returns a function with call signature __call__(gridparams, eventdata)
        that will return the probability for each event given by ``eventdata``
        from the PDFs that is registered for the given gridparams parameter
        values.
        """
        def _get_prob_for_gridparams_with_eventdata(tdm, gridparams, eventdata):
            """Gets the probability for each event given by ``eventdata`` from
            the PDFs that is registered for the given gridparams parameter
            values.

            Parameters
            ----------
            tdm : TrialDataManager
                The TrialDataManager instance holding the trial data.
            gridparams : dict
                The dictionary with the grid parameter names and values, that
                reference the registered PDF of interest.
            eventdata : (N_events,V)-shaped numpy ndarray
                The ndarray holding the data for the PDF evaluation.

            Returns
            -------
            prob : (N_events,)-shaped ndarray
                The ndarray holding the probability values for each event.
            """
            pdf = self.get_pdf(gridparams)
            prob = pdf.get_prob_with_eventdata(tdm, gridparams, eventdata)
            return prob

        return _get_prob_for_gridparams_with_eventdata

    def assert_is_valid_for_trial_data(self, tdm):
        """Checks if this PDF set is valid for all the given trial data. Since
        the PDFs have the same axes, we just need to check the first PDFs.
        """
        # Get one of the PDFs.
        pdf = next(iter(self.items()))[1]
        pdf.assert_is_valid_for_trial_data(tdm)

    def get_prob(self, tdm, params, tl=None):
        """Calculates the probability density for each event, given the given
        parameter values.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance that will be used to get the data
            from the trial events.
        params : dict
            The dictionary holding the parameter names and values for which the
            probability should get calculated. Because this PDF is a PDFSet,
            there should be at least one parameter.
        tl : TimeLord instance | None
            The optional TimeLord instance to use for measuring timing
            information.

        Returns
        -------
        prob : (N_events,)-shaped 1D ndarray
            The probability values for each event.
        grads : (N_fitparams,N_events)-shaped 2D ndarray
            The PDF gradients w.r.t. the PDF fit parameters for each event.
        """
        # Create the ndarray for the event data that is needed for the
        # ``MultiDimGridPDF.get_prob_with_eventdata`` method.
        # All PDFs of this PDFSet should have the same axes, so use the axes
        # from any of the PDFs in this PDF set.
        if(isinstance(self, IsSignalPDF)):
            # Evaluate the relevant quantities for
            # all events and sources (relevant for stacking analyses).
            if tdm.src_ev_idxs is not None:
                (src_idxs, ev_idxs) = tdm.src_ev_idxs
                eventdata = np.array(
                    [
                        # Check `psi` axis name.
                        tdm.get_data(axis.name)
                        if ('psi' in axis.name)

                        # Check `src` axis name.
                        else tdm.get_data(axis.name)[src_idxs]
                        if ('src' in axis.name)

                        # Default case.
                        else tdm.get_data(axis.name)[ev_idxs]
                        for axis in self.pdf_axes
                    ]
                ).T
            else:
                n_src = len(tdm.get_data('src_array')['ra'])
                l_ev = len(tdm.get_data('ra'))
                eventdata = np.array(
                    [
                        # Check `psi` axis name.
                        tdm.get_data(axis.name)
                        if ('psi' in axis.name)

                        # Check `src` axis name.
                        else tdm.get_data(axis.name)
                        if (('src' in axis.name) and (n_src == 1))
                        else np.repeat(tdm.get_data(axis.name), l_ev)
                        if (('src' in axis.name) and (n_src != 1))

                        # Default case.
                        else np.tile(tdm.get_data(axis.name), n_src)
                        for axis in self.pdf_axes
                    ]
                ).T

        elif (isinstance(self, IsBackgroundPDF)):
            eventdata = np.array([tdm.get_data(axis.name)
                                  for axis in self.pdf_axes]).T

        # Get the interpolated PDF values for the arbitrary parameter values.
        # The (D,N_events)-shaped grads_ ndarray contains the gradient of the
        # probability density w.r.t. each of the D parameters, which are defined
        # by the param_grid_set. The order of the D gradients is the same as
        # the parameter grids.
        with TaskTimer(tl, 'Get signal PDFs for all events.'):
            (prob, grads_) = self._interpolmethod_instance.get_value_and_gradients(
                tdm, eventdata, params)

        # Handle the special (common) case were there is only one fit parameter
        # and it coincides with the only grid parameter of this PDFSet.
        fitparams = self.param_set.floating_params
        params_grid_set_pnames = self.param_grid_set.parameter_names

        if((len(fitparams) == 1) and (len(params_grid_set_pnames) == 1) and
           (params_grid_set_pnames[0] == fitparams[0].name)):
            return (prob, grads_)

        # Create an array for the gradients, which will only contain the
        # gradients for the fit (floating) parameters.
        grads = np.zeros((len(fitparams), prob.shape[0]), dtype=np.float64)

        # Create a dictionary to map the name of the grid parameter to its
        # index.
        paramgridset_pname_to_pidx = dict(
            [(pname, pidx) for (pidx, pname) in
             enumerate(params_grid_set_pnames)])

        for (pidx, fitparam) in enumerate(fitparams):
            pname = fitparam.name
            # Check if the fit parameter is part of the PDFSet's grid
            # parameters. If so, the gradient is provided by the interpolation
            # method. If not, the gradient is zero for this fit parameter.
            if(pname in paramgridset_pname_to_pidx):
                grads[pidx] = grads_[paramgridset_pname_to_pidx[pname]]

        return (prob, grads)


class MappedMultiDimGridPDFSet(PDF, PDFSet):
    def __init__(
            self, param_grid_set, gridparams_pdfs, src_hypo_group_manager,
            pdf_type=MultiDimGridPDF, **kwargs):
        """Creates a new MappedMultiDimGridPDFSet instance, which holds a set of
        MultiDimGridPDF instances, one for each point of a parameter grid set.

        Parameters
        ----------
        param_grid_set : ParameterGrid instance | ParameterGridSet instance
            The set of ParameterGrid instances, which define the grid values of
            the model parameters, the given MultiDimGridPDF instances belong to.
        gridparams_pdfs : sequence of (dict, MultiDimGridPDF) tuples
            The sequence of 2-element tuples which define the mapping of grid
            values to PDF instances.
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances and flux models.
        pdf_type : type
            The PDF class that can be added to the set.
        """
        super(MappedMultiDimGridPDFSet, self).__init__(
            param_set=None,
            pdf_type=pdf_type,
            fitparams_grid_set=param_grid_set,
            **kwargs)

        self.fluxmodel_to_source_mapping = src_hypo_group_manager.get_fluxmodel_to_source_mapping()

        # Add the given MultiDimGridPDF instances to the PDF set.
        for (gridparams, pdf) in gridparams_pdfs:
            self.add_pdf(pdf, gridparams)

    @property
    def fluxmodel_to_source_mapping(self):
        """The fluxmodel to source indices mapping list used for
        MappedMultiDimGridPDFSet evaluation to get the corresponding KDE PDF.
        """
        return self._fluxmodel_to_source_mapping
    @fluxmodel_to_source_mapping.setter
    def fluxmodel_to_source_mapping(self, mapping_list):
        if(not issequenceof(mapping_list, tuple)):
            raise TypeError(
                'The `fluxmodel_to_source_mapping` property must be a sequence of '
                'tuples.')
        self._fluxmodel_to_source_mapping = mapping_list

    def assert_is_valid_for_trial_data(self, tdm):
        """Checks if this PDF set is valid for all the given trial data. Since
        the PDFs have the same axes, we just need to check the first PDFs.
        """
        # Get one of the PDFs.
        pdf = next(iter(self.items()))[1]
        pdf.assert_is_valid_for_trial_data(tdm)

    def get_prob(self, tdm, params, tl=None):
        """Calculates the probability density for each event, given the given
        parameter values.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance that will be used to get the data
            from the trial events.
        params : dict
            The dictionary holding the parameter names and values for which the
            probability should get calculated. Because this PDF is a PDFSet,
            there should be at least one parameter.
        tl : TimeLord instance | None
            The optional TimeLord instance to use for measuring timing
            information.

        Returns
        -------
        prob : (N_events,)-shaped 1D ndarray
            The probability values for each event.
        grads : (N_fitparams,N_events)-shaped 2D ndarray
            The PDF gradients w.r.t. the PDF fit parameters for each event.
        """
        # Create the ndarray for the event data that is needed for the
        # ``MultiDimGridPDF.get_prob_with_eventdata`` method.
        # All PDFs of this PDFSet should have the same axes, so use the axes
        # from any of the PDFs in this PDF set.
        if(isinstance(self, IsSignalPDF)):
            # Evaluate the relevant quantities for
            # all events and sources (relevant for stacking analyses).
            if tdm.src_ev_idxs is not None:
                (src_idxs, ev_idxs) = tdm.src_ev_idxs
                eventdata = np.array(
                    [
                        # Check `psi` axis name.
                        tdm.get_data(axis.name)
                        if ('psi' in axis.name)

                        # Check `src` axis name.
                        else tdm.get_data(axis.name)[src_idxs]
                        if ('src' in axis.name)

                        # Default case.
                        else tdm.get_data(axis.name)[ev_idxs]
                        for axis in self.pdf_axes
                    ]
                ).T
            else:
                n_src = len(tdm.get_data('src_array')['ra'])
                l_ev = len(tdm.get_data('ra'))
                eventdata = np.array(
                    [
                        # Check `psi` axis name.
                        tdm.get_data(axis.name)
                        if ('psi' in axis.name)

                        # Check `src` axis name.
                        else tdm.get_data(axis.name)
                        if (('src' in axis.name) and (n_src == 1))
                        else np.repeat(tdm.get_data(axis.name), l_ev)
                        if (('src' in axis.name) and (n_src != 1))

                        # Default case.
                        else np.tile(tdm.get_data(axis.name), n_src)
                        for axis in self.pdf_axes
                    ]
                ).T

                # Construct `src_idxs` for masking with `fluxmodel_mask`.
                src_idxs = np.repeat(np.arange(n_src), l_ev)

        elif (isinstance(self, IsBackgroundPDF)):
            eventdata = np.array([tdm.get_data(axis.name)
                                  for axis in self.pdf_axes]).T

        # Get the interpolated PDF values for the arbitrary parameter values.
        # The (D,N_events)-shaped grads ndarray contains the gradient of the
        # probability density w.r.t. each of the D parameters, which are defined
        # by the param_grid_set. The order of the D gradients is the same as
        # the parameter grids.

        # Iterate over fluxmodels in `fluxmodel_to_source_mapping` list.
        prob = np.zeros(eventdata.shape[0])
        grads = np.zeros(eventdata.shape[0])
        for (fluxmodel_hash, src_list) in self.fluxmodel_to_source_mapping:
            # Mask for selecting events corresponding to specific flux.
            fluxmodel_mask = np.isin(src_idxs, src_list)

            # Pass params in case normalization function depends on it.
            # KDE normalization function does not depend on params.
            with TaskTimer(tl, 'Get signal PDFs for specific flux events.'):
                prob_i = self.get_pdf(fluxmodel_hash).get_prob_with_eventdata(
                    tdm, params, eventdata[fluxmodel_mask])

            prob[fluxmodel_mask] = prob_i

        return (prob, grads)
