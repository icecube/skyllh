# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import solve

from skyllh.core.py import classname


def rebin(
        bincontent: np.array,
        old_binedges: np.array,
        new_binedges: np.array,
        negatives=False):
    """Rebins the binned counts to the new desired grid. This function
    uses a method of moments approach. Currently it uses a three moments
    appraoch. At the edges of the array it uses a two moments approach.

    Parameters
    ----------
    bincontent: (n,)-shaped 1D numpy ndarray
        The binned content which should be rebinned.
    old_binedges: (n+1,)-shaped 1D numpy ndarray
        The old grid's bin edges. The shape needs to be the same as
        `bincontent`.
    new_binedges: (m+1)-shaped 1D numpy ndarray
        The new bin edges to use.
    binning_scheme: str
        The binning scheme to use. Choices are "log" (logarithmic)
        or "lin" (linear). This decides how to calculate the midpoints
        of each bin.
    negatives: bool
        Switch to keep or remove negative values in the final binning.

    Returns
    -------
    new_bincontent: 1D numpy ndarray
        The new binned counts for the new binning.

    Raises
    ------
    ValueError:
        Unknown binning scheme.

    Authors
    -------
    - Dr. Stephan Meighen-Berger
    - Dr. Martin Wolf
    """
    old_bincenters = 0.5*(old_binedges[1:] + old_binedges[:-1])

    # Checking if shapes align.
    if bincontent.shape != old_bincenters.shape:
        ValueError('The arguments bincontent and old_binedges do not match!'
            'bincontent must be (n,)-shaped and old_binedges must be (n+1,)-'
            'shaped!')

    # Setting up the new binning.
    new_bincenters = 0.5*(new_binedges[1:] + new_binedges[:-1])


    new_widths = np.diff(new_binedges)
    new_nbins = len(new_widths)

    # Create output array with zeros.
    new_bincontent = np.zeros(new_bincenters.shape)

    # Looping over the old bin contents and distributing
    for (idx, bin_val) in enumerate(bincontent):
        # Ignore empty bins.
        if bin_val == 0.:
            continue

        old_bincenter = old_bincenters[idx]

        new_point = (np.abs(new_binedges - old_bincenter)).argmin()

        if new_point == 0:
            # It the first bin. Use 2-moments method.
            start_idx = new_point
            end_idx = new_point + 1

            mat = np.vstack(
                (
                    new_widths[start_idx:end_idx+1],
                    new_widths[start_idx:end_idx+1]
                    * new_bincenters[start_idx:end_idx+1]
                )
            )

            b = bin_val * np.array([
                1.,
                old_bincenter
            ])
        elif new_point == new_nbins-1:
            # It the last bin. Use 2-moments method.
            start_idx = new_point - 1
            end_idx = new_point

            mat = np.vstack(
                (
                    new_widths[start_idx:end_idx+1],
                    new_widths[start_idx:end_idx+1]
                    * new_bincenters[start_idx:end_idx+1]
                )
            )

            b = bin_val * np.array([
                1.,
                old_bincenter
            ])
        else:
            # Setting up the equation for 3 moments (mat*x = b)
            # x is the values we want
            start_idx = new_point - 1
            end_idx = new_point + 1

            mat = np.vstack(
                (
                    new_widths[start_idx:end_idx+1],
                    new_widths[start_idx:end_idx+1]
                    * new_bincenters[start_idx:end_idx+1],
                    new_widths[start_idx:end_idx+1]
                    * new_bincenters[start_idx:end_idx+1]**2
                )
            )

            b = bin_val * np.array([
                1.,
                old_bincenter,
                old_bincenter**2
            ])

        # Solving and adding to the new bin content.
        new_bincontent[start_idx:end_idx+1] += solve(mat, b)

        if not negatives:
            new_bincontent[new_bincontent < 0.] = 0.

    new_bincontent = new_bincontent / (
        np.sum(new_bincontent) / np.sum(bincontent))

    return new_bincontent


def get_bincenters_from_binedges(edges):
    """Calculates the bin center values from the given bin edge values.

    Parameters
    ----------
    edges : 1D numpy ndarray
        The (n+1,)-shaped 1D ndarray holding the bin edge values.

    Returns
    -------
    bincenters : 1D numpy ndarray
        The (n,)-shaped 1D ndarray holding the bin center values.
    """
    return 0.5*(edges[:-1] + edges[1:])

def get_binedges_from_bincenters(centers):
    """Calculates the bin edges from the given bin center values. The bin center
    values must be evenly spaced.

    Parameters
    ----------
    centers : 1D numpy ndarray
        The (n,)-shaped 1D ndarray holding the bin center values.

    Returns
    -------
    edges : 1D numpy ndarray
        The (n+1,)-shaped 1D ndarray holding the bin edge values.
    """
    d = np.diff(centers)
    if not np.all(np.isclose(np.diff(d), 0)):
        raise ValueError('The bin center values are not evenly spaced!')
    d = d[0]
    print(d)

    edges = np.zeros((len(centers)+1,), dtype=np.double)
    edges[:-1] = centers - d/2
    edges[-1] = centers[-1] + d/2

    return edges

def get_bin_indices_from_lower_and_upper_binedges(le, ue, values):
    """Returns the bin indices for the given lower and upper bin edges the given
    values fall into.

    Parameters
    ----------
    le : 1D numpy ndarray
        The lower bin edges.
    ue : 1D numpy ndarray
        The upper bin edges.
    values : 1D numpy ndarray
        The values for which to get the bin indices.

    Returns
    -------
    idxs : 1D numpy ndarray
        The bin indices of the given values.
    """
    if np.any(values < le[0]):
        invalid_values = values[values < le[0]]
        raise ValueError(
            '{} values ({}) are smaller than the lowest bin edge ({})!'.format(
                len(invalid_values), str(invalid_values), le[0]))
    if np.any(values > ue[-1]):
        invalid_values = values[values > ue[-1]]
        raise ValueError(
            '{} values ({}) are larger than the largest bin edge ({})!'.format(
                len(invalid_values), str(invalid_values), ue[-1]))

    m = (
        (values[:,np.newaxis] >= le[np.newaxis,:]) &
        (values[:,np.newaxis] <= ue[np.newaxis,:])
    )
    idxs = np.nonzero(m)[1]

    return idxs


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
        self.name = name
        self.binedges = binedges

    def __str__(self):
        """Pretty string representation.
        """
        s = '%s: %s\n'%(classname(self), self._name)
        s += str(self._binedges)
        return s

    def __eq__(self, other):
        """Checks if object ``other`` is equal to this BinningDefinition object.
        """
        if(not isinstance(other, BinningDefinition)):
            raise TypeError('The other object in the equal comparison must be '
                'an instance of BinningDefinition!')
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
    def binwidths(self):
        """(read-only) The widths of the bins.
        """
        return np.diff(self._binedges)

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
        return outofrange

    def get_binwidth_from_value(self, value):
        """Returns the width of the bin the given value falls into.
        """
        idx = np.digitize(value, self._binedges) - 1

        bin_width = self.binwidths[idx]

        return bin_width

    def get_subset(self, lower_edge, upper_edge):
        """Creates a new BinningDefinition instance which contains only a subset
        of the bins of this BinningDefinition instance. The range of the subset
        is given by a lower and upper edge value.

        Parameters
        ----------
        lower_edge : float
            The lower edge value of the subset.
        upper_edge : float
            The upper edge value of the subset.

        Returns
        -------
        new_binning : BinningDefinition instance
            The new BinningDefinition instance holding the binning subset.
        """

        idxs = np.indices((len(self._binedges),))[0]
        m = (self._binedges >= lower_edge) & (self._binedges <= upper_edge)

        idx_lower = np.min(idxs[m])
        # Include the lower edge of the bin the lower_edge value falls into.
        if(self._binedges[idx_lower] > lower_edge):
            idx_lower -= 1

        idx_upper = np.max(idxs[m])
        # Include the upper edge of the bin the upper_edge value falls into.
        if(self._binedges[idx_upper] < upper_edge):
            idx_upper += 1

        new_binedges = self._binedges[idx_lower:idx_upper+1]

        return BinningDefinition(self._name, new_binedges)


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
            raise TypeError('The obj argument must be an instance of '
                'UsesBinning!')

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
            raise TypeError('The binning argument must be an instance of '
                'BinningDefinition!')

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
                raise KeyError('The binning definition "%s" is not defined!'%(
                    name))
            binning = self._binnings[self._binning_name2idx[name]]
        elif(isinstance(name, int)):
            binning = self._binnings[name]
        else:
            raise TypeError('The name argument must be of type str or int!')

        return binning
