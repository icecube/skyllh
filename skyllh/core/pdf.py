# -*- coding: utf-8 -*-

import abc

from skyllh.core.py import typename, ObjectCollection, range
from skyllh.core.parameters import ParameterGrid, ParameterGridSet, make_params_hash

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

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """This method must be reimplemented by the derived class if the
        derived PDF class relies on the source hypothesis group manager.
        """
        pass

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
        for (key, pdf) in self._gridfitparams_hash_pdf_dict.iteritems():
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
