# -*- coding: utf-8 -*-

"""The seasons module provides functionality to handle a likelihood function for
multiple seasons, where the detector response is different for each season.
"""

import numpy as np

from skylab.core import storage

class BinningDefinition(object):
    """The BinningDefinition class provides a structure to hold histogram
    binning definitions.
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

class Season(object):
    def __init__(self, name, exp_pathfilenames, mc_pathfilenames, livetime):
        """Creates a new Season instance of a given name. A season holds the
        file locations to the experimental and monte-carlo data files. These
        files must contain numpy record arrays.

        Parameters
        ----------
        name : str
            The name of the season.
        exp_pathfilenames : str | list of str
            The path and filenames of the data files holding the experimental
            data events in the format of numpy record arrays (.npy files).
        mc_pathfilenames : str | list of str
            The path and filenames of the data files holding the monte-carlo
            data events in the format of numpy record arrays (.npy files).
        livetime : float
            the live-time of the season, specified in days.
        """
        self.name = name
        self.exp_pathfilename_list = exp_pathfilenames
        self.mc_pathfilename_list = mc_pathfilenames
        self.livetime = livetime

        # _data_exp and _data_mc will hold the actual data after the data has
        # been loaded.
        self._data_exp = None
        self._data_mc = None
        self._data_preparation_functions = list()
        self._binning_definitions = dict()

    @property
    def name(self):
        """The name of the season. This must be an unique identifier among
        all the different seasons.
        """
        return self._name
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def exp_pathfilename_list(self):
        """The list of path and file names of the numpy record array data files
        that stores the experimental data for this season.
        """
        return self._exp_pathfilename_list
    @exp_pathfilename_list.setter
    def exp_pathfilename_list(self, pathfilenames):
        if(isinstance(pathfilenames, str)):
            pathfilenames = [pathfilenames]
        if(not isinstance(pathfilenames, list)):
            raise TypeError('The exp_pathfilename_list property must be of type list!')
        self._exp_pathfilename_list = pathfilenames

    @property
    def mc_pathfilename_list(self):
        """The list of path and file names of the numpy record array data files
        that stores the monte-carlo data for this season.
        """
        return self._mc_pathfilename_list
    @mc_pathfilename_list.setter
    def mc_pathfilename_list(self, pathfilenames):
        if(isinstance(pathfilenames, str)):
            pathfilenames = [pathfilenames]
        if(not isinstance(pathfilenames, list)):
            raise TypeError('The mc_pathfilename_list property must be of type list!')
        self._mc_pathfilename_list = pathfilenames

    @property
    def livetime(self):
        """The live-time of the season in days.
        """
        return self._lifetime
    @livetime.setter
    def livetime(self, livetime):
        if(not isinstance(livetime, float)):
            raise TypeError('The lifetime of the season must be of type float!')
        self._lifetime = livetime

    @property
    def data_exp(self):
        """The numpy record array holding the experimental data of this season.

        Error Exceptions
        ----------------
        RuntimeError
            If the data has not been loaded via the ``load_data`` method.
        """
        if(self._data_exp is None):
            raise RuntimeError('The data has not been loaded yet!')
        return self._data_exp

    @property
    def data_mc(self):
        """The numpy record array holding the monte-carlo data of this season.

        Error Exceptions
        ----------------
        RuntimeError
            If the data has not been loaded via the ``load_data`` method.
        """
        if(self._data_mc is None):
            raise RuntimeError('The data has not been loaded yet!')
        return self._data_mc

    @property
    def data_preparation_functions(self):
        """The list of callback functions that will be called to prepare the
        data (experimental and monte-carlo).
        """
        return self._data_preparation_functions

    def load_data(self):
        """Loads the data of the season. Afterwards, the data can be accessed
        through the properties ``data_exp`` and ``data_mc``.

        Note: This does not call the ``prepare_data`` method! It only loads
              the data as the method names says.
        """
        fileloader_exp = storage.create_FileLoader(self.exp_pathfilename_list)
        fileloader_mc  = storage.create_FileLoader(self.mc_pathfilename_list)
        self._data_exp = fileloader_exp.load_data()
        self._data_mc  = fileloader_mc.load_data()

    def add_data_preparation(self, func):
        """Adds the given data preparation function to the season.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(exp, mc) that will prepare
            the data after it was loaded. The arguments 'exp' and 'mc' are
            numpy record arrays holding the experimental and monte-carlo data,
            respectively. The return value must be a two-element tuple of the
            form (exp, mc) with the modified experimental and monto-carlo data.

        """
        if(not callable(func)):
            raise TypeError('The argument "func" must be a callable object with call signature __call__(exp, mc)!')
        self._data_preparation_functions.append(func)

    def add_binning_definition(self, binning):
        """Adds a binning setting to this season instance.

        Parameters
        ----------
        binning : BinningDefinition
            The BinningDefinition object holding the binning.
        """
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The "binning" argument must be of type BinningDefinition!')
        if(binning.key in self._binning_definitions):
            raise KeyError('The binning definition "%s" is already defined for season "%s"!'%(binning.key, self._name))

        self._binning_definitions[binning.key] = binning

    def get_binning_definition(self, key):
        """Gets the BinningDefinition object for the given binning key.
        """
        if(key not in self._binning_definitions):
            raise KeyError('The given binning key "%s" has not been added yet!'%(key))
        return self._binning_definitions[key]

    def define_binning(self, key, binedges):
        """Defines a binning for ``key``, and adds it as binning setting.

        Parameters
        ----------
        key : str
            The name of the binning setting.
        binedges : sequence
            The sequence of the bin edges, which should be used for this binning
            definition.

        Returns
        -------
        binning : BinningDefinition
            The BinningDefinition object which was created and added to this
            season.
        """
        binning = BinningDefinition(key, binedges)
        self.add_binning_definition(binning)
        return binning

    def prepare_data(self):
        """Prepares the data by calling the assigned data preparation callback
        functions for this season.
        """
        for func in self._data_preparation_functions:
            (self._data_exp, self._data_mc) = func(self._data_exp, self._data_mc)

class MultiSeason(object):
    """The MultiSeason class is a collection of multiple Season objects.
    """
    def __init__(self, seasons=None):
        """Creates a new MultiSeason instance.
        """
        self._season_list = list()

    def add_season(self, season):
        """Adds a new season to the list of seasons.

        Parameters
        ----------
        season : Season
            The Season object that should be added.

        Return
        ------
            Returns this MultiSeason instance to concatenate several add_season
            method calls.
        """
        if(not isinstance(season, Season)):
            raise TypeError('The season argument must be of type Season!')
        self._season_list.append(season)
        return self
