# -*- coding: utf-8 -*-

"""The analyis module of skylab contains analyis related utility classes.
"""

import numpy as np

class BinningDefinition(object):
    """The BinningDefinition class provides a structure to hold histogram
    binning definitions for an analyis.
    """
    def __init__(self, key, binedges):
        """Creates a new binning definition object.

        Parameters
        ----------
        key : str
            The key (name) of the binning definition.
        binedges : sequence
            The sequence of the bin edges, which should be used for the binning.
        """
        self.key = key
        self.binedges = binedges

    @property
    def key(self):
        """The key (name) of the binning setting. This must be an unique name
        for all the different binning settings used within a season.
        """
        return self._key
    @key.setter
    def key(self, key):
        if(not isinstance(key, str)):
            raise TypeError("The key must be of type 'str'!")
        self._key = key

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

class UsesBinning(object):
    """This is a classifier class that can be used to define, that a class uses
    binning information for one or more dimensions.

    This class defines the property ``binnings``, which is a list of
    BinningDefinition objects.

    This class provides the method ``has_same_binning(obj)`` to determine if
    a given object (that also uses binning) has the same binning.
    """
    def __init__(self):
        print('Entering UsesBinning.__init__')
        # Make sure that multiple inheritance can be used.
        super(UsesBinning, self).__init__()

        # Define the list of binning definition objects and a name->list_index
        # mapping for faster access.
        self._binnings = []
        self._binning_key2idx = {}
        print('Leaving UsesBinning.__init__')

    @property
    def binnings(self):
        """(read-only) The list of BinningDefinition objects, one for each
        dimension.
        """
        return self._binnings

    @property
    def binning_ndim(self):
        """(read-only)
        """
        return len(self._binnings)

    def add_binning(self, binning):
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The binning argument must be an instance of BinningDefinition!')
        self._binnings.append(binning)
        self._binning_key2idx[binning.key] = len(self._binnings)-1

    def get_binning(self, key):
        """Retrieves the binning definition of the given key.

        Parameters
        ----------
        key : str | int
            The key of the binning definition. A string specifies the name and
            an integer the dimension index.

        Returns
        -------
        binning : BinningDefinition
            The binning definition of the given key.
        """
        if(isinstance(key, str)):
            if(key not in self._binning_key2idx):
                raise KeyError('The binning definition "%s" is not defined!'%(key))
            binning = self._binnings[self._binning_key2idx[key]]
        elif(isinstance(key, int)):
            binning = self._binnings[key]
        else:
            raise TypeError('The key argument must be of type str or int!')

        return binning
