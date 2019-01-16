# -*- coding: utf-8 -*-

import logging
import numpy as np

class BinningDefinition(object):
    """The BinningDefinition class provides a structure to hold histogram
    binning definitions for an analyis.
    """
    def __init__(self, name, binedges):
        """Creates a new binning definition object.

        Parameters
        ----------
        name : str
            The name of the binning definition.
        binedges : sequence
            The sequence of the bin edges, which should be used for the binning.
        """
        self.logger = logging.getLogger(__name__)

        self.name = name
        self.binedges = binedges

    def __eq__(self, other):
        """Checks if object ``other`` is equal to this BinningDefinition object.
        """
        if(not isinstance(other, BinningDefinition)):
            raise TypeError('The other object in the equal comparison must be an instance of BinningDefinition!')
        if(self.name != other.name):
            return False
        if(np.any(self.binedges != other.binedges)):
            return False
        return True

    @property
    def name(self):
        """The name of the binning setting. This must be an unique name
        for all the different binning settings used within a season.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError("The name must be of type 'str'!")
        self._name = name

    @property
    def binedges(self):
        """The numpy.ndarray holding the bin edges.
        """
        return self._binedges
    @binedges.setter
    def binedges(self, arr):
        arr = np.atleast_1d(arr)
        self._binedges = np.array(arr, dtype=np.float64)

    @property
    def nbins(self):
        """The number of bins, based on the number of bin edges (minus 1).
        """
        return self._binedges.size - 1

    @property
    def bincenters(self):
        """The center values of the bins.
        """
        return 0.5*(self._binedges[:-1] + self._binedges[1:])

    @property
    def lower_edge(self):
        """The lowest bin edge of the binning.
        """
        return self._binedges[0]

    @property
    def upper_edge(self):
        """The upper most edge of the binning.
        """
        return self._binedges[-1]

    @property
    def range(self):
        """The tuple (lower_edge, upper_edge) of the binning.
        """
        return (self.lower_edge, self.upper_edge)

    def any_data_out_of_binning_range(self, data):
        """Checks if any of the given data is outside of the binning range.

        Parameters
        ----------
        data : 1d ndarray
            The array with the data values to check.

        Returns
        -------
        outofrange : bool
            True if any data value is outside the binning range.
            False otherwise.
        """
        outofrange = np.any((data < self.lower_edge) |
                            (data > self.upper_edge))
        if outofrange:
            self.logger.debug('Out of range data: %s', data[np.logical_or(data < self.lower_edge, data > self.upper_edge)])
        return outofrange


class UsesBinning(object):
    """This is a classifier class that can be used to define, that a class uses
    binning information for one or more dimensions.

    This class defines the property ``binnings``, which is a list of
    BinningDefinition objects.

    This class provides the method ``has_same_binning_as(obj)`` to determine if
    a given object (that also uses binning) has the same binning.
    """
    def __init__(self, *args, **kwargs):
        # Make sure that multiple inheritance can be used.
        super(UsesBinning, self).__init__(*args, **kwargs)

        # Define the list of binning definition objects and a name->list_index
        # mapping for faster access.
        self._binnings = []
        self._binning_name2idx = {}

    @property
    def binnings(self):
        """(read-only) The list of BinningDefinition objects, one for each
        dimension.
        """
        return self._binnings

    @property
    def binning_ndim(self):
        """(read-only) The number of dimensions that uses binning.
        """
        return len(self._binnings)

    def has_same_binning_as(self, obj):
        """Checks if this object has the same binning as the given object.

        Parameters
        ----------
        obj : class instance derived from UsesBinning
            The object that should be checked for same binning.

        Returns
        -------
        check : bool
            True if ``obj`` uses the same binning, False otherwise.
        """
        if(not isinstance(obj, UsesBinning)):
            raise TypeError('The obj argument must be an instance of UsesBinning!')

        for (self_binning, obj_binning) in zip(self.binnings, obj.binnings):
            if(not (self_binning == obj_binning)):
                return False
        return True

    def add_binning(self, binning, name=None):
        """Adds the given binning definition to the list of binnings.

        Parameters
        ----------
        binning : BinningDefinition
            The binning definition to add.
        name : str | (default) None
            The name of the binning. If not None and it's different to the
            name of the given binning definition, a copy of the
            BinningDefinition object is made and the new name is set.
        """
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The binning argument must be an instance of BinningDefinition!')

        # Create a copy of the BinningDefinition object if the name differs.
        if(name is not None):
            if(not isinstance(name, str)):
                raise TypeError('The name argument must be of type str!')
            if(name != binning.name):
                binning = BinningDefinition(name, binning.binedges)

        self._binnings.append(binning)
        self._binning_name2idx[binning.name] = len(self._binnings)-1

    def get_binning(self, name):
        """Retrieves the binning definition of the given name.

        Parameters
        ----------
        name : str | int
            The name of the binning definition. A string specifies the name and
            an integer the dimension index.

        Returns
        -------
        binning : BinningDefinition
            The binning definition of the given name.
        """
        if(isinstance(name, str)):
            if(name not in self._binning_name2idx):
                raise KeyError('The binning definition "%s" is not defined!'%(name))
            binning = self._binnings[self._binning_name2idx[name]]
        elif(isinstance(name, int)):
            binning = self._binnings[name]
        else:
            raise TypeError('The name argument must be of type str or int!')

        return binning
