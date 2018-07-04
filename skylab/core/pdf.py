# -*- coding: utf-8 -*-

import abc

from skylab.core.parameters import ParameterGridSet

class PDF(object):
    """The abstract base class for all probability distribution functions (PDF)
    models.
    All PDF model classes must be derived from this class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # Make sure that multiple inheritance can be used. This super call will
        # invoke the __init__ method of a possible second inheritance.
        super(PDF, self).__init__()

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

        Errors
        ------
        ValueError
            If the PDF is not valid for the given experimental data.
        """
        pass

    @abc.abstractmethod
    def get_prob(self, events, params, *args, **kwargs):
        """This abstract method is supposed to calculate the probability for
        the specified events given the specified parameters.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
            probability should be calculated for. What data fields are required
            is defined by the derived PDF class and depends on the application.
        params : dict
            The dictionary containing the parameters needed for the calculation.
            What parameters these are is defined by derived PDF class and
            depends on the application.
        """
        pass

class SpatialPDF(PDF):
    """This is the abstract base class for a spatial PDF model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(SpatialPDF, self).__init__()

class EnergyPDF(PDF):
    """This is the abstract base class for an energy PDF model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(EnergyPDF, self).__init__()

class TimePDF(PDF):
    """This is the abstract base class for a time PDF model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TimePDF, self).__init__()

class IsBackgroundPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a background PDF. This is useful for type checking.
    """
    def __init__(self):
        super(IsBackgroundPDF, self).__init__()

class IsSignalPDF(object):
    """This is a classifier class that can be used by other classes to indicate
    that the class describes a signal PDF. This is useful for type checking.
    """
    def __init__(self):
        super(IsSignalPDF, self).__init__()

        self.signal_parameter_grid_set = ParameterGridSet()

    @property
    def signal_parameter_grid_set(self):
        """The ParameterGridSet object defining the value grids of the different
        signal parameters.
        """
        return self._signal_parameter_grid_set
    @signal_parameter_grid_set.setter
    def signal_parameter_grid_set(self, obj):
        if(not isinstance(obj, ParameterGridSet)):
            raise TypeError('The signal_parameter_grid_set property must be an object of type ParameterGridSet!')
        self._signal_parameter_grid_set = obj
