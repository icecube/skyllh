# -*- coding: utf-8 -*-

import os
import numpy as np
from copy import deepcopy

from skylab.core import display
from skylab.core.py import issequence, issequenceof
from skylab.core import storage

class BinningDefinition(object):
    """The BinningDefinition class provides a structure to hold histogram
    binning definitions for a dataset.
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

class Dataset(object):
    """The Dataset class describes a set of self-consistent experimental and
    simulated detector data. Usually this is for a certain time period, i.e.
    season.

    Independet dataset of the same kind, e.g. event selection, can be joined
    through a DatasetCollection object.
    """
    def __init__(self, name, exp_pathfilenames, mc_pathfilenames, livetime, version, verqualifiers=None):
        """Creates a new dataset object that describes a self-consistent set of
        data.

        Parameters
        ----------
        name : str
            The name of the dataset.
        livetime : float
            The integrated live-time in days of the dataset.
        version: int
            The version number of the dataset. Higher version numbers indicate
            newer datasets.
        verqualifiers: dict | None
            If specified, this dictionary specifies version qualifiers. These
            can be interpreted as subversions of the dataset. The format of the
            dictionary must be 'qualifier (str): version (int)'.
        """
        self.name = name
        self.exp_pathfilename_list = exp_pathfilenames
        self.mc_pathfilename_list = mc_pathfilenames
        self.livetime = livetime
        self.version = version
        self.verqualifiers = verqualifiers

        self.description = ''

        # _data_exp and _data_mc will hold the actual data after the data has
        # been loaded.
        self._data_exp = None
        self._data_mc = None
        self._data_preparation_functions = list()
        self._binning_definitions = dict()

    @property
    def name(self):
        """The name of the dataset. This must be an unique identifier among
        all the different datasets.
        """
        return self._name
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def description(self):
        """The (longer) description of the dataset.
        """
        return self._description
    @description.setter
    def description(self, description):
        if(not isinstance(description, str)):
            raise TypeError('The description of the dataset must be of type str!')
        self._description = description

    @property
    def exp_pathfilename_list(self):
        """The list of fully qualified file names of the data files that store
        the experimental data for this dataset.
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
        """The list of fully qualified file names of the data files that store
        the monte-carlo data for this dataset.
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
        """The integrated live-time in days of the dataset.
        """
        return self._lifetime
    @livetime.setter
    def livetime(self, livetime):
        if(not isinstance(livetime, float)):
            raise TypeError('The lifetime of the dataset must be of type float!')
        self._lifetime = livetime

    @property
    def version(self):
        """The main version (int) of the dataset.
        """
        return self._version
    @version.setter
    def version(self, version):
        if(not isinstance(version, int)):
            raise TypeError('The version of the dataset must be of type int!')
        self._version = version

    @property
    def verqualifiers(self):
        """The dictionary holding the version qualifiers, i.e. sub-version
        qualifiers. If set to None, an empty dictionary will be used.
        The dictionary must have the type form of str:int.
        """
        return self._verqualifiers
    @verqualifiers.setter
    def verqualifiers(self, verqualifiers):
        if(verqualifiers is None):
            verqualifiers = dict()
        if(not isinstance(verqualifiers, dict)):
            raise TypeError('The version qualifiers must be of type dict!')
        # Check if the dictionary has format str:int.
        for (q,v) in verqualifiers.iteritems():
            if(not isinstance(q, str)):
                raise TypeError('The version qualifier "%s" must be of type str!'%(q))
            if(not isinstance(v, int)):
                raise TypeError('The version for the qualifier "%s" must be of type int!'%(q))
        self._verqualifiers = verqualifiers

    @property
    def version_str(self):
        """The version string of the dataset. This combines all the version
        information about the dataset.
        """
        s = '%03d'%(self._version)
        for (q,v) in self._verqualifiers.iteritems():
            s += q+'%02d'%(v)
        return s

    @property
    def data_exp(self):
        """The numpy record array holding the experimental data of the dataset.

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
        """The numpy record array holding the monte-carlo data of the dataset.

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

    @property
    def is_data_loaded(self):
        """Information (boolean) if the data was already loaded.
        """
        if(isinstance(self._data_exp, np.ndarray) and
           isinstance(self._data_mc, np.ndarray)
          ):
            return True
        return False

    def __gt__(self, ds):
        """Implementation to support the operation ``b = self > ds``, where
        ``self`` is this Dataset object and ``ds`` an other Dataset object.
        The comparison is done based on the version information of both
        datasets. Larger version numbers for equal version qualifiers indicate
        newer (greater) datasets.

        The two datasets must be of the same kind, i.e. have the same name, in
        order to make the version comparison senseful.

        Returns
        -------
        bool
            True, if this dataset is newer than the reference dataset.
            False, if this dataset is as new or older than the reference
            dataset.
        """
        # Datasets of different names cannot be compared usefully.
        if(self._name != ds._name):
            return False

        # Larger main version numbers indicate newer datasets.
        if(self._version > ds._version):
            return True

        # Look for version qualifiers that make this dataset older than the
        # reference dataset.
        qs1 = self._verqualifiers.keys()
        qs2 = ds._verqualifiers.keys()

        # If a qualifier of self is also specified for ds, the version number
        # of the self qualifier must be larger than the version number of the ds
        # qualifier, in order to consider self as newer dataset.
        # If a qualifier is present in self but not in ds, self is considered
        # newer.
        for q in qs1:
            if(q in qs2 and qs1[q] <= qs2[q]):
                return False
        # If there is a qualifier in ds but not in self, self is considered
        # older.
        for q in qs2:
            if(q not in qs1):
                return False

        return True

    def __str__(self):
        """Implementation of the pretty string representation of the Dataset
        object.
        """
        s = 'Dataset "%s": v%s\n'%(self.name, self.version_str)
        pad = ' '*4
        s += '%s { livetime = %.3f days}\n'%(pad, self.livetime)
        if(self.description != ''):
            s += '%s Description:\n'%(pad) + self.description + '\n'
        s += '%s Experimental data:\n'%(pad)
        for pathfilename in self.exp_pathfilename_list:
            s += '%s%s\n'%(pad*2, pathfilename)
        s += '%s MC data:\n'%(pad)
        for pathfilename in self.mc_pathfilename_list:
            s += '%s%s\n'%(pad*2, pathfilename)

        return s

    def load_data(self):
        """Loads the data of the dataset. Afterwards, the data can be accessed
        through the properties ``data_exp`` and ``data_mc``.

        Note: This does not call the ``prepare_data`` method! It only loads
              the data as the method names says.
        """
        fileloader_exp = storage.create_FileLoader(self.exp_pathfilename_list)
        fileloader_mc  = storage.create_FileLoader(self.mc_pathfilename_list)
        self._data_exp = fileloader_exp.load_data()
        self._data_mc  = fileloader_mc.load_data()

    def add_data_preparation(self, func):
        """Adds the given data preparation function to the dataset.

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

    def prepare_data(self):
        """Prepares the data by calling the assigned data preparation callback
        functions of this dataset.
        """
        for func in self._data_preparation_functions:
            (self._data_exp, self._data_mc) = func(self._data_exp, self._data_mc)

    def load_and_prepare_data(self):
        """Loads and prepares the experimental and monte-carlo data of this
        dataset by calling its ``load_data`` and ``prepare_data`` methods.
        """
        self.load_data()
        self.prepare_data()

    def add_binning_definition(self, binning):
        """Adds a binning setting to this dataset.

        Parameters
        ----------
        binning : BinningDefinition
            The BinningDefinition object holding the binning information.
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
        """Defines a binning for ``key``, and adds it as binning definition.

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


class DatasetCollection(object):
    """The DatasetCollection class describes a collection of different datasets.

    New datasets can be added via the add-assign operator (+=), which calls
    the ``add_datasets`` method.
    """
    def __init__(self, name, description=''):
        """Creates a new DatasetCollection instance.

        Parameters
        ----------
        name : str
            The name of the collection.
        description : str
            The (longer) description of the dataset collection.
        """
        self.name = name
        self.description = description

        self._datasets = dict()

    @property
    def name(self):
        """The name (str) of the dataset collection.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name of the dataset collection must be of type str!')
        self._name = name

    @property
    def description(self):
        """The (longer) description of the dataset collection.
        """
        return self._description
    @description.setter
    def description(self, description):
        if(not isinstance(description, str)):
            raise TypeError('The description of the dataset collection must be of type str!')
        self._description = description

    @property
    def dataset_names(self):
        """The list of names of the assigned datasets.
        """
        return sorted(self._datasets.keys())

    def __iadd__(self, ds):
        """Implementation of the ``self += dataset`` operation to add a
        Dataset object to this dataset collection.
        """
        if(not isinstance(ds, Dataset)):
            raise TypeError('The dataset object must be a subclass of Dataset!')

        self.add_datasets(ds)

        return self

    def __str__(self):
        """Implementation of the pretty string representation of the
        DatasetCollection instance. It shows the available datasets.
        """
        lines  = 'DatasetCollection "%s"\n'%(self.name)
        lines += "-"*display.PAGE_WIDTH + "\n"
        lines += "Description:\n" + self.description + "\n"
        lines += "Available datasets:\n"

        for name in self.dataset_names:
            lines += '\n'
            lines += display.add_leading_text_line_padding(2, str(self._datasets[name]))

        return lines

    def add_datasets(self, datasets):
        """Adds the given Dataset object(s) to this dataset collection.

        Parameters
        ----------
        datasets : Dataset | sequence of Dataset
            The Dataset object or the sequence of Dataset objects that should be
            added to the dataset collection.

        Returns
        -------
        self : DatasetCollection
            This DatasetCollection object in order to be able to chain several
            add_dataset calls.
        """
        if(not issequence(datasets)):
            datasets = [datasets]

        for dataset in datasets:
            if(not isinstance(dataset, Dataset)):
                raise TypeError('The dataset object must be a sub-class of Dataset!')

            if(dataset.name in self._datasets):
                raise KeyError('Dataset "%s" already exists!'%(dataset.name))

            self._datasets[dataset.name] = dataset

        return self

    def remove_dataset(self, name):
        """Removes the given dataset from the collection.

        Parameters
        ----------
        name : str
            The name of the dataset that should get removed.
        """
        if(name not in self._datasets):
            raise KeyError('Dataset "%s" is not part of the dataset collection "%s", nothing to remove!'%(name, self.name))

        self._datasets.pop(name)

    def get_dataset(self, name):
        """Retrieves a Dataset object from this dataset collection.

        Parameters
        ----------
        name : str
            The name of the dataset.

        Returns
        -------
        dataset : Dataset
            The Dataset object holding all the information about the dataset.
        """
        if(name not in self._datasets):
            raise KeyError('The dataset "%s" is not part of the dataset collection "%s"!'%(name, self.name))
        return self._datasets[name]

    def define_binning(self, key, binedges):
        """Defines a binning definition and adds it to all the datasets of this
        dataset collection.

        Parameters
        ----------
        key : str
            The name of the binning definition.
        binedges : sequence
            The sequence of the bin edges, that should be used for the binning.
        """
        for (name, dataset) in self._datasets.iteritems():
            dataset.define_binning(key, binedges)

    def add_data_preparation(self, func):
        """Adds the data preparation function to all the datasets of this
        dataset collection.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(exp, mc) that will prepare
            the data after it was loaded. The arguments 'exp' and 'mc' are
            numpy record arrays holding the experimental and monte-carlo data,
            respectively. The return value must be a two-element tuple of the
            form (exp, mc) with the modified experimental and monto-carlo data.
        """
        for (name, dataset) in self._datasets.iteritems():
            dataset.add_data_preparation(func)
