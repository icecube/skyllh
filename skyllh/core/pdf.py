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


class PDFDataField(object):
    """This class defines a data field and its calculation that is used by a
    PDF class instance. The calculation is defined through an external function
    with the call signature: `__call__(pdf, events, fitparams)`, where `pdf` is
    the PDF class instance, and `events` is the numpy record ndarray holding the
    event data. The return type of this function must be a numpy ndarray.
    """
    def __init__(self, name, func):
        """Creates a new instance of a PDFDataField instance.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the new data field. The call signature
            must be `__call__(pdf, events, fitparams)`, where `pdf` is the PDF
            class instance, and `events` is the numpy record ndarray holding the
            event data. The return type of this function must be a numpy
            ndarray.
        """
        super(PDFDataField, self).__init__()

        self.name = name
        self.func = func

        # Define the member variable that holds the numpy ndarray with the data
        # field values.
        self._values = None

    @property
    def name(self):
        """The name of the data field.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be an instance of str!')
        self._name = name

    @property
    def func(self):
        """The function that calculates the data field values.
        """
        return self._func
    @func.setter
    def func(self, f):
        if(not callable(f)):
            raise TypeError('The func property must be a callable object!')
        if(not func_has_n_args(f, 3)):
            raise TypeError('The func property must be a function with 3 '
                'arguments!')
        self._func = f

    @property
    def values(self):
        """(read-only) The calculated data values of the data field.
        """
        return self._values

    def calculate(self, pdf, events, fitparams):
        """Calculates the data field values utilizing the defined external
        function.

        Parameters
        ----------
        pdf : PDF instance
            The PDF instance for which to calculate the PDF data field values.
        events : numpy record ndarray
            The numpy record ndarray holding the event data.
        fitparams : dict
            The dictionary holding the current fit parameter names and values.
        """
        self._values = self._func(pdf, events, fitparams)


class PDFDataFields(object):
    """This class provides the functionality of additional event data fields for
    PDF classes, that are required by the PDF class and have to be calculated
    based on properties of the PDF class instance.
    """
    def __init__(self):
        super(PDFDataFields, self).__init__()

        self._precalc_data_fields = []
        self._precalc_data_field_reg = dict()

        self._dynamic_data_fields = []
        self._dynamic_data_field_reg = dict()

    def add(self, name, func, precalc=False):
        """Adds a new PDF data field.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the new data field. The call signature
            must be `__call__(pdf, events, fitparams)`, where `pdf` is the PDF
            class instance, `events` is the numpy record ndarray holding the
            event data, and fitparams is the dictionary holding the current fit
            parameter values. The return type of this function must be a numpy
            ndarray.
        precalc : bool
            Flag if this data field can be pre-calculated whenever a new trial
            is being initialized. Otherwise the data field gets calculated for
            every PDF value evaluation, which would be required in cases where
            the calculation of the data field depends on fit parameter values.
        """
        if(not isinstance(precalc, bool)):
            raise TypeError('The precalc argument must be an instance of bool!')

        if((name in self._precalc_data_field_reg) or
           (name in self._dynamic_data_field_reg)):
            raise KeyError('The PDF data field "%s" is already defined!'%(name))

        data_field = PDFDataField(name, func)

        if(precalc is True):
            self._precalc_data_fields.append(data_field)
            self._precalc_data_field_reg[name] = data_field
        else:
            self._dynamic_data_fields.append(data_field)
            self._dynamic_data_field_reg[name] = data_field

    def __contains__(self, name):
        """Checks if the given data field is contained in this data field
        collection.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        check : bool
            True if the data field is contained in this data field collection,
            False otherwise.
        """
        if(name in self._precalc_data_field_reg):
            return True
        if(name in self._dynamic_data_field_reg):
            return True

        return False

    def __getitem__(self, name):
        """Retrieves the values of a given data field.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        values : numpy ndarray
            The numpy ndarray holding the calculated field values.

        Raises
        ------
        KeyError
            If the given data field is not defined.
        """
        if(name in self._precalc_data_field_reg):
            return self._precalc_data_field_reg[name].values
        if(name in self._dynamic_data_field_reg):
            return self._dynamic_data_field_reg[name].values

        raise KeyError('The PDF data field "%s" is not defined!'%(name))

    def get_field_values(self, name):
        """Retrieves the values of a given data field. This is equivalent to
        __getitem__(name).

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        values : numpy ndarray
            The numpy ndarray holding the calculated field values.

        Raises
        ------
        KeyError
            If the given data field is not defined.
        """
        return self.__getitem__(name)

    def calc_precalc_data_fields(self, pdf, events):
        """Calculates the data values of the data fields that can be
        pre-calculated.

        Parameters
        ----------
        pdf : PDF instance
            The PDF instance for which the data fields should get calculated.
        events : numpy record ndarray
            The numpy record ndarray holding the event data.
        """
        fitparams = dict()
        for data_field in self._precalc_data_fields:
            data_field.calculate(pdf, events, fitparams)

    def calc_dynamic_data_fields(self, pdf, events, fitparams):
        """Calculates the data values of the dynamic data fields.

        Parameters
        ----------
        pdf : PDF instance
            The PDF instance for which the data fields should get calculated.
        events : numpy record ndarray
            The numpy record ndarray holding the event data.
        fitparams : dict | None
            The dictionary holding the current fit parameter names and values.
        """
        for data_field in self._dynamic_data_fields:
            data_field.calculate(pdf, events, fitparams)


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
        self._data_fields = PDFDataFields()

    @property
    def axes(self):
        """(read-only) The PDFAxes object holding the PDFAxis objects for the
        dimensions of the PDF.
        """
        return self._axes

    @property
    def data_fields(self):
        """The PDFDataFields instance providing the functionality to define
        additional data fields that depend on the properties of the PDF class
        instance.
        """
        return self._data_fields

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

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """This method must be reimplemented by the derived class if the
        derived PDF class relies on the source hypothesis group manager.
        """
        pass

    def initialize_for_new_trial(self, events):
        """This method must be reimplemented by the derived class if the
        derived PDF class requires notification whenever a new data trial with
        the given events is being initialized. This base method calculates PDF
        data fields that can be pre-calculated. Hence, this method needs to be
        called by the derived implementation of this method.
        """
        self._data_fields.calc_precalc_data_fields(self, events)

    def calc_dynamic_data_fields(self, events, fitparams):
        """Calculates the dynamic data fields, that depend on current fit
        parameter values.
        """
        self._data_fields.calc_dynamic_data_fields(self, events, fitparams)

    def get_data(self, name, eval_data):
        """Gets the data for the given data field name. The data is stored
        either in the eval_data numpy record array or in one of the additional
        PDF data fields. Data from the evaluation data is prefered.

        Parameters
        ----------
        name : str
            The name of the data field for which to retrieve the data.
        eval_data : numpy record ndarray
            The data that is going to be evaluated.

        Returns
        -------
        data : numpy ndarray
            The data of the requested data field.

        Raises
        ------
        KeyError
            If the given PDF data field is not defined.
        """
        if(name in eval_data.dtype.names):
            return eval_data[name]
        return self._data_fields[name]

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
    def get_prob(self, events, fitparams):
        """This abstract method is supposed to calculate the probability for
        the specified events given the specified fit parameters.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
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
    `scipy.interpolate.RegularGridInterpolator` instance.
    """
    def __init__(self, axis_binnings, pdf_grid_data, norm_factor_func=None):
        """Creates a new PDF instance for a multi-dimensional PDF given
        as PDF values on a grid. The grid data is interpolated with a
        `scipy.interpolate.RegularGridInterpolator` instance. As grid points
        the bin edges of the axis binning definitions are used.

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
            The call signature of this function
            must be `__call__(pdf, events, fitparams)`, where `pdf` is this PDF
            instance, `events` is a numpy record ndarray holding the events for
            which to calculate the PDF values, and `fitparams` is a dictionary
            with the current fit parameter names and values.
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
        `__call__(pdf, events, fitparams)`, where `pdf` is this PDF instance,
        `events` is a numpy record ndarray holding the events for which to
        calculate the PDF values, and `fitparams` is a dictionary with the
        current fit parameter names and values.
        This property can be set to `None`. In that case a unity returning
        function is used.
        """
        return self._norm_factor_func
    @norm_factor_func.setter
    def norm_factor_func(self, func):
        if(func is None):
            # Define a normalization function that just returns 1 for each
            # event.
            func = lambda pdf, events, fitparams: np.ones_like(events)
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

    def assert_is_valid_for_eval_data(self, eval_data):
        """Checks if the PDF is valid for all values of the given evaluation
        data. The evaluation data values must be within the ranges of the PDF
        axes.

        Parameters
        ----------
        eval_data : numpy record ndarray
            The data that is going to be evaluated.

        Raises
        ------
        ValueError
            If any of the evaluation data is out of its axis range.
        """
        for axis in self._axes:
            data = self.get_data(axis.name, eval_data)
            if(np.any(data < axis.vmin) or
               np.any(data > axis.vmax)
            ):
                raise ValueError('Some of the evaluation data for PDF axis '
                    '"%s" is out of range (%g,%g)!'%(
                    axis.name, axis.vmin, axis.vmax))

    def get_prob(self, events, fitparams=None):
        """Calculates the probability for the specified events given the
        specified fit parameters.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
            probability should be calculated.
        fitparams : dict | None
            The dictionary containing the fit parameters for which the
            probability should get calculated.

        Returns
        -------
        prob : (N_events,) shaped numpy ndarray
            The 1D numpy ndarray with the probability for each event.
        """
        # Calculate possible dynamic data fields.
        self.calc_dynamic_data_fields(events, fitparams)

        x = np.array([ self.get_data(axis.name, events) for axis in self._axes ]).T
        prob = self._pdf(x)

        norm = self._norm_factor_func(self, events, fitparams)
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

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Calls the ``change_source_hypo_group_manager`` method of all the PDF
        instances added to this PDF set.
        """
        for (key, pdf) in self._gridfitparams_hash_pdf_dict.items():
            pdf.change_source_hypo_group_manager(src_hypo_group_manager)


class IsBackgroundPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a background PDF. This is useful for type checking.
    """
    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsBackgroundPDF class.
        """
        super(IsBackgroundPDF, self).__init__(*args, **kwargs)


class IsSignalPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a signal PDF.
    """
    def __init__(self, *args, **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsSignalPDF class.
        """
        super(IsSignalPDF, self).__init__(*args, **kwargs)
