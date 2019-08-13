# -*- coding: utf-8 -*-

import abc
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from skyllh.core.binning import BinningDefinition
from skyllh.core.py import (
    ObjectCollection,
    func_has_n_args,
    issequenceof,
    range,
    typename
)
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet,
    make_params_hash
)

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


class PDFAxes(ObjectCollection):
    """This class describes the set of PDFAxis objects defining the
    dimensionality of a PDF.
    """
    def __init__(self, axes=None):
        super(PDFAxes, self).__init__(obj_type=PDFAxis, obj_list=axes)

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

        raise KeyError('The PDFAxis with name "%s" could not be found!'%(name))

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
        """Creates a CombinedBackgroundPDF instance for the combination of this
        background PDF and another background PDF.

        Parameters
        ----------
        other : instance of IsBackgroundPDF
            The instance of IsBackgroundPDF, which is the other background PDF.
        """
        if(not isinstance(other, IsBackgroundPDF)):
            raise TypeError('The other PDF must be an instance of '
                'IsBackgroundPDF!')

        return CombinedBackgroundPDF(self, other, op=np.multiply)


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
        """Creates a CombinedSignalPDF instance for the combination of this
        signal PDF and another signal PDF.

        Parameters
        ----------
        other : instance of IsSignalPDF
            The instance of IsSignalPDF, which is the other signal PDF.
        """
        if(not isinstance(other, IsSignalPDF)):
            raise TypeError('The other PDF must be an instance of '
                'IsSignalPDF!')

        return CombinedSignalPDF(self, other, op=np.multiply)


class PDF(object):
    """The abstract base class for all probability distribution functions (PDF)
    models.
    All PDF model classes must be derived from this class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        # Make sure that multiple inheritance can be used. This super call will
        # invoke the __init__ method of a possible second inheritance.
        super(PDF, self).__init__(*args, **kwargs)

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

    def add_axis(self, axis):
        """Adds the given PDFAxis object to this PDF.
        """
        if(not isinstance(axis, PDFAxis)):
            raise TypeError('The axis argument must be an instance of PDFAxis!')
        self._axes += axis

    @abc.abstractmethod
    def assert_is_valid_for_exp_data(self, data_exp):
        """This abstract method is supposed to check if this PDF is valid for
        all the given experimental data. This means, it needs to check if there
        is a PDF value for each data event that will be used in the likelihood
        evaluation. This is just a seatbelt. The method must raise a ValueError
        if the PDF is not valid for the given experimental data.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data to check for.

        Raises
        ------
        ValueError
            If the PDF is not valid for the given experimental data.
        """
        pass

    @abc.abstractmethod
    def get_prob(self, trial_data_manager, fitparams):
        """This abstract method is supposed to calculate the probability for
        the specified events given the specified fit parameters.

        Parameters
        ----------
        trial_data_manager : TrialDataManager instance
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. What data fields are required
            is defined by the derived PDF class and depends on the application.
        fitparams : dict
            The dictionary containing the fit parameters for which the
            probability should get calculated.
            What fit parameters these are is defined by derived PDF class and
            depends on the application.

        Returns
        -------
        prob : (N_events,) or (N_events,N_sources) shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event. If the PDF
            is dependent on the source, it returns a 2D numpy ndarray with the
            probability for each of the N_events events and each of the
            N_sources sources. By definition the 2D case is applicable only for
            signal PDFs.
        """
        pass


class CombinedPDF(PDF):
    """The CombinedPDF class describes a combination of two PDF instances. It
    is derived from the PDF class and hence is a PDF itself. An operator defines
    the type of combination of the two PDF instances.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, pdf1, pdf2, op, op_kwargs):
        """Creates a new CombinedPDF instance, which implements the operation
        `pdf1 op pdf2`. The operation is defined by the operator function `op`.
        The axes of the two PDF instances will be merged.

        Parameters
        ----------
        pdf1 : instance of PDF
            The left-hand-side PDF in the operation `pdf1 op pdf2`.
        pdf2 : instance of PDF
            The right-hand-side PDF in the operation `pdf1 op pdf2`.
        op : callable with two arguments
            The numeric operator function, which combines the PDF values of
            `pdf1` and `pdf2`. The function must have the following call
            signature: `__call__(pdf1_values, pdf2_values)`, where `pdf1_values`
            are the PDF values of `pdf1` and `pdf2_values` the pdf values of
            `pdf2`.
        op_kwargs : dict | None
            Possible additional keyword arguments that should be passed to the
            operator function.
        """
        super(CombinedPDF, self).__init__()

        self.pdf1 = pdf1
        self.pdf2 = pdf2
        self.operator = op
        self.operator_kwargs = op_kwargs

        self._axes = pdf1.axes + pdf2.axes

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

    @property
    def operator(self):
        """The operator function that combines the PDF values of `pdf1` and
        `pdf2`.
        """
        return self._operator
    @operator.setter
    def operator(self, op):
        if(not callable(op)):
            raise TypeError('The operator property must be callable!')
        if(not isinstance(op, np.ufunc)):
            # The func_has_n_args function works only on Python functions.
            if(not func_has_n_args(op, 2)):
                raise TypeError('The operator property must be a function with '
                    '2 arguments!')
        self._operator = op

    @property
    def operator_kwargs(self):
        """Additional keyword arguments that should be passed to the operator
        function.
        """
        return self._operator_kwargs
    @operator_kwargs.setter
    def operator_kwargs(self, kwargs):
        if(kwargs is None):
            kwargs = dict()
        if(not isinstance(kwargs, dict)):
            raise TypeError('The operator_kwargs property must be None or an '
                'instance of dict!')
        self._operator_kwargs = kwargs

    def assert_is_valid_for_exp_data(self, data_exp):
        """Calls the `assert_is_valid_for_exp_data` method of `pdf1` and `pdf2`.
        """
        self._pdf1.assert_is_valid_for_exp_data(data_exp)
        self._pdf2.assert_is_valid_for_exp_data(data_exp)

    def get_prob(self, tdm, fitparams=None):
        """Calculates the probability for the trial events given the
        specified fit parameters by calling the `get_prob` method of `pdf1`
        and `pdf2` and combining the two properties using the defined operator
        function `operator`.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        fitparams : dict | None
            The dictionary containing the fit parameter names and values for
            which the probability should get calculated.

        Returns
        -------
        prob : (N_events,) shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        """
        prob = self._operator(
            self._pdf1.get_prob(tdm, fitparams),
            self._pdf2.get_prob(tdm, fitparams),
            **self._operator_kwargs
        )

        return prob


class CombinedSignalPDF(CombinedPDF, IsSignalPDF):
    """This class provides a combined PDF for two signal PDF instances.
    """
    def __init__(self, pdf1, pdf2, op, op_kwargs=None):
        super(CombinedSignalPDF, self).__init__(pdf1, pdf2, op, op_kwargs)


class CombinedBackgroundPDF(CombinedPDF, IsBackgroundPDF):
    """This class provides a combined PDF for two background PDF instances.
    """
    def __init__(self, pdf1, pdf2, op, op_kwargs=None):
        super(CombinedBackgroundPDF, self).__init__(pdf1, pdf2, op, op_kwargs)


class SpatialPDF(PDF):
    """This is the abstract base class for a spatial PDF model. A spatial PDF
    has two axes, right-ascention (ra) and declination (dec).
    """
    __metaclass__ = abc.ABCMeta

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
            raise ValueError('Some data is outside the right-ascention range (%.3f, %.3f)!'%(ra_axis.vmin, ra_axis.vmax))

        # Check if all the data is within the declination range.
        if(np.any((data_exp['dec'] < dec_axis.vmin) |
                  (data_exp['dec'] > dec_axis.vmax))):
            raise ValueError('Some data is outside the declination range (%.3f, %.3f)!'%(dec_axis.vmin, dec_axis.vmax))


class EnergyPDF(PDF):
    """This is the abstract base class for an energy PDF model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(EnergyPDF, self).__init__(*args, **kwargs)


class TimePDF(PDF):
    """This is the abstract base class for a time PDF model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(TimePDF, self).__init__(*args, **kwargs)


class MultiDimGridPDF(PDF):
    """This class provides a multi-dimensional PDF created from pre-calculated
    PDF data on a grid. The grid data is interpolated using a
    :class:`scipy.interpolate.RegularGridInterpolator` instance.
    """
    def __init__(self, axis_binnings, pdf_grid_data, norm_factor_func=None):
        """Creates a new PDF instance for a multi-dimensional PDF given
        as PDF values on a grid. The grid data is interpolated with a
        :class:`scipy.interpolate.RegularGridInterpolator` instance. As grid
        points the bin edges of the axis binning definitions are used.

        Parameters
        ----------
        axis_binnings : sequence of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each BinningDefinition instance defines
            the event field name that should be used for querying the PDF.
        pdf_grid_data : n-dimensional numpy ndarray
            The n-dimensional numpy ndarray holding the PDF values at given grid
            points. The grid points must match the bin edges of the given
            BinningDefinition instances of the `axis_binnings` argument.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function must be
            `__call__(pdf, tdm, fitparams)`, where `pdf` is this PDF
            instance, `tdm` is an instance of TrialDataManager holding the
            events for which to calculate the PDF values, and `fitparams` is a
            dictionary with the current fit parameter names and values.
        """
        super(MultiDimGridPDF, self).__init__()

        self.axis_binning_list = axis_binnings
        self.norm_factor_func = norm_factor_func

        # Define the PDF axes.
        for axis_binning in self._axis_binnning_list:
            self.add_axis(PDFAxis(
                name=axis_binning.name,
                vmin=axis_binning.lower_edge,
                vmax=axis_binning.upper_edge
            ))

        self._pdf = RegularGridInterpolator(
            tuple([ binning.binedges for binning in self._axis_binnning_list ]),
            pdf_grid_data,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

    @property
    def axis_binning_list(self):
        """The list of BinningDefinition instances for each PDF axis.
        The name of each BinningDefinition instance defines the event field
        name that should be used for querying the PDF.
        """
    @axis_binning_list.setter
    def axis_binning_list(self, binnings):
        if(isinstance(binnings, BinningDefinition)):
            binnings = [binnings]
        if(not issequenceof(binnings, BinningDefinition)):
            raise TypeError('The axis_binning_list property must be an '
                'instance of BinningDefinition or a sequence of '
                'BinningDefinition instances!')
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
            func = lambda pdf, tdm, fitparams: np.ones((tdm.n_events,), dtype=np.float)
        if(not callable(func)):
            raise TypeError('The norm_factor_func property must be a callable '
                'object!')
        if(not func_has_n_args(func, 3)):
            raise TypeError('The norm_factor_func property must be a function '
                'with 3 arguments!')
        self._norm_factor_func = func

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if the PDF is valid for all values of the given experimental
        data. This method is deprecated!
        """
        pass

    def assert_is_valid_for_trial_data(self, trial_data_manager):
        """Checks if the PDF is valid for all values of the given evaluation
        data. The evaluation data values must be within the ranges of the PDF
        axes.

        Parameters
        ----------
        trial_data_manager : TrialDataManager instance
            The instance of TrialDataManager that holds the data which is going
            to be evaluated.

        Raises
        ------
        ValueError
            If any of the evaluation data is out of its axis range.
        """
        for axis in self._axes:
            data = trial_data_manager.get_data(axis.name)
            if(np.any(data < axis.vmin) or
               np.any(data > axis.vmax)
            ):
                raise ValueError('Some of the trial data for PDF axis '
                    '"%s" is out of range (%g,%g)!'%(
                    axis.name, axis.vmin, axis.vmax))

    def get_prob(self, tdm, fitparams=None):
        """Calculates the probability for the trial events given the
        specified fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF values should get calculated.
        fitparams : dict | None
            The dictionary containing the fit parameters for which the
            probability should get calculated.

        Returns
        -------
        prob : (N_events,) shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        """
        x = np.array([ tdm.get_data(axis.name) for axis in self._axes ]).T
        prob = self._pdf(x)

        norm = self._norm_factor_func(self, tdm, fitparams)
        prob *= norm

        return prob


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
        """The ParameterGridSet object defining the value grids of the different
        fit parameters.
        """
        return self._fitparams_grid_set
    @fitparams_grid_set.setter
    def fitparams_grid_set(self, obj):
        if(isinstance(obj, ParameterGrid)):
            obj = ParameterGridSet([obj])
        if(not isinstance(obj, ParameterGridSet)):
            raise TypeError('The fitparams_grid_set property must be an object of type ParameterGridSet!')
        self._fitparams_grid_set = obj

    @property
    def gridfitparams_list(self):
        """(read-only) The list of dictionaries of all the fit parameter
        permutations on the grid.
        """
        return self.fitparams_grid_set.parameter_permutation_dict_list

    @property
    def pdf_keys(self):
        """(read-only) The list of stored PDF object keys.
        """
        return self._gridfitparams_hash_pdf_dict.keys()

    def items(self):
        """Returns the list of 2-element tuples for the PDF stored in this
        PDFSet object.
        """
        return self._gridfitparams_hash_pdf_dict.items()

    def add_pdf(self, pdf, gridfitparams):
        """Adds the given PDF object for the given parameters to the internal
        registry.

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
        """
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The pdf argument must be an instance of %s!'%(typename(self.pdf_type)))
        if(not isinstance(gridfitparams, dict)):
            raise TypeError('The fitparams argument must be of type dict!')

        gridfitparams_hash = make_params_hash(gridfitparams)
        if(gridfitparams_hash in self._gridfitparams_hash_pdf_dict):
            raise KeyError('The PDF with grid fit parameters %s was already added!'%(str(gridfitparams)))
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
            raise TypeError('The gridfitparams argument must be of type dict or int!')

        if(gridfitparams_hash not in self._gridfitparams_hash_pdf_dict):
            raise KeyError('No PDF was created for the parameter set "%s"!'%(str(gridfitparams)))

        pdf = self._gridfitparams_hash_pdf_dict[gridfitparams_hash]
        return pdf
