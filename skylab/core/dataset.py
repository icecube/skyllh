# -*- coding: utf-8 -*-

import os
import numpy as np
from copy import deepcopy

from skylab.core import display
from skylab.core.py import issequenceof
from skylab.core.seasons import Season

class Dataset(object):
    """The Dataset class describes an entire dataset, i.e. event selection, that
    can contain several seasons. For each season it stores specific
    configurations, like sine declination binning for background PDFs.
    """
    def __init__(self, name, version, verqualifiers=None, copy=None):
        """Defines a dataset for usage in skylab. A dataset might consist of
        several seasons. Each season has experimental and monte-carlo data
        assigned.

        Parameters:
        -----------
        name : str
            The name of the dataset.
        version: int
            The version number of the dataset. Higher version numbers indicate
            newer datasets.
        verqualifiers: dict | None
            If specified, this dictionary specifies version qualifiers. These
            can be seen as subversions. The format of the dictionary must be
            'qualifier (str): version (int)'.
        copy: Dataset | None
            If specified, a given existing Dataset gets copied and will have
            the new specified name and version.
        """
        self.name = name
        self.version = version
        self.verqualifiers = verqualifiers

        if(copy is not None):
            if(isinstance(copy, Dataset)):
                self._seasons = deepcopy(copy._seasons)
            else:
                raise TypeError('The "copy" argument must be of type "Dataset"!')
            return

        self._seasons = dict()

    @property
    def name(self):
        """The name (str) of the dataset.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name of the dataset must be of type str!')
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
        s = 'v%03d'%(self._version)
        for (q,v) in self._verqualifiers.iteritems():
            s += q+'%02d'%(v)
        return s

    @property
    def season_names(self):
        """The list of the season names assigned to the dataset.
        """
        return sorted(self._seasons.keys())

    def __str__(self):
        """Implementation of the pretty string representation of the Dataset
        object. It shows the available seasons.
        """

        lines  = 'Dataset "%s"\n'%(self.name)
        lines += "-"*display.PAGE_WIDTH + "\n"
        lines += "Description:\n" + self.description + "\n"
        lines += "Available Seasons:\n\n"

        for (name, season) in self._seasons.iteritems():
            pad = ' '*4
            lines += ' "%s" { livetime = %.3f days }\n'%(name, season.livetime)
            lines += '%s Experimental data:\n'%(pad)
            for pathfilename in season.exp_pathfilename_list:
                lines += '%s%s\n'%(pad*2, pathfilename)
            lines += '%s MC data:\n'%(pad)
            for pathfilename in season.mc_pathfilename_list:
                lines += '%s%s\n'%(pad*2, pathfilename)

        return lines

    def __gt__(self, ds):
        """Implementation to support the operation ``b = self > ds``, where
        ``self`` is this Dataset object and ``ds`` an other Dataset object.
        The comparison is done based on the version information of both
        datasets. Larger version numbers for equal version qualifiers indicate
        newer (greater) datasets.

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

    def add_season(self, name, exp_pathfilenames, mc_pathfilenames, livetime):
        """Adds a new Season object to the dataset and returns it.

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

        Returns
        -------
        season : Season
            The Season object that was added to the internal season dictionary.
        """
        if(name in self.season_names):
            raise KeyError('Season "%s" already exists!'%(name))

        season = Season(
            name = name,
            exp_pathfilenames = exp_pathfilenames,
            mc_pathfilenames = mc_pathfilenames,
            livetime = livetime
        )

        self._seasons[name] = season

        return season

    def remove_season(self, name):
        """Removes the given season from the dataset.

        Parameters
        ----------
        name : str
            The season name.
        """
        if(name not in self.season_names()):
            raise KeyError('Season "%s" is not defined, nothing to remove!'%(name))

        self._seasons.pop(name)

    def get_season(self, name):
        """Retrieves the Season object for the given season.

        Parameters
        ----------
        name : str
            The name of the season.

        Returns
        -------
        season : Season
            The Season object holding all the information about the season.
        """
        if(name not in self._seasons):
            raise KeyError('The season "%s" is not defined!'%(name))
        return self._seasons[name]

    def define_binning(self, key, binedges):
        """Defines a binning definition and adds it to all the seasons of this
        dataset.

        Parameters
        ----------
        key : str
            The name of the binning definition.
        binedges : sequence
            The sequence of the bin edges, that should be used for the binning.
        """
        for (name, season) in self._seasons.iteritems():
            season.define_binning(key, binedges)

    def add_data_preparation(self, func):
        """Adds the data preparation function to all the seasons of this
        dataset.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(exp, mc) that will prepare
            the data after it was loaded. The arguments 'exp' and 'mc' are
            numpy record arrays holding the experimental and monte-carlo data,
            respectively. The return value must be a two-element tuple of the
            form (exp, mc) with the modified experimental and monto-carlo data.
        """
        for (name, season) in self._seasons.iteritems():
            season.add_data_preparation(func)

    def load_data(self, seasons=None):
        """Loads the experimental and monte-carlo data of the given seasons
        from disk into memory.

        Parameters
        ----------
        seasons : str | list of str | None
            The name, or the list of names of the seasons which data should be
            loaded. If None, the data of all seasons of the dataset will be
            loaded.
        """
        if(seasons is None):
            seasons = self.season_names
        if(isinstance(seasons, str)):
            seasons = [seasons]
        if(not issequenceof(seasons, str)):
            raise TypeError('The seasons argument must be a sequence of str!')

        for name in seasons:
            self.get_season(name).load_data()

    def prepare_data(self, seasons=None):
        """Calls the 'prepare_data' method of the given seasons.

        Parameters
        ----------
        seasons : str | list of str | None
            The name, or the list of names of the seasons which prepare_data
            method should be called.
            If None, the prepare_data method of all the seasons of the dataset
            will be called.
        """
        if(seasons is None):
            seasons = self.season_names
        if(isinstance(seasons, str)):
            seasons = [seasons]
        if(not issequenceof(seasons, str)):
            raise TypeError('The seasons argument must be a sequence of str!')

        for name in seasons:
            self.get_season(name).prepare_data()

