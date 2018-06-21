# -*- coding: utf-8 -*-

"""The analyis module of skylab contains analyis related utility classes.
"""

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
