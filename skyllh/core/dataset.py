# -*- coding: utf-8 -*-

import os
import os.path
import numpy as np
from numpy.lib import recfunctions as np_rfn
from copy import deepcopy

from skyllh.core.binning import BinningDefinition
from skyllh.core.livetime import Livetime
from skyllh.core.py import (
    float_cast,
    issequence,
    issequenceof,
    list_of_cast,
    str_cast
)
from skyllh.core import display
from skyllh.core.storage import (
    DataFieldRecordArray,
    create_FileLoader
)
from skyllh.core.timing import TaskTimer


class Dataset(object):
    """The Dataset class describes a set of self-consistent experimental and
    simulated detector data. Usually this is for a certain time period, i.e.
    season.

    Independet dataset of the same kind, e.g. event selection, can be joined
    through a DatasetCollection object.
    """
    _EXP_FIELD_NAMES = ('ra', 'dec', 'ang_err', 'time', 'log_energy')
    _MC_FIELD_NAMES = ('true_ra', 'true_dec', 'true_energy', 'mcweight')

    @staticmethod
    def add_required_exp_field_names(cls, fieldnames):
        """Static method to add required experimental data field names to the
        list of already required field names for experimental data.
        This method is useful for derived dataset classes.

        Parameters
        ----------
        cls : class object
            The class object, for which the new set of required exp field names
            should apply.
        fieldnames : str | list of str
            The field name or the list of field names to add.
        """
        if(not issequence(fieldnames)):
            fieldnames = [fieldnames]
        if(not issequenceof(fieldnames, str)):
            raise TypeError('The fieldnames argument must be a sequence of str objects!')

        cls._EXP_FIELD_NAMES = tuple(list(Dataset._EXP_FIELD_NAMES) + fieldnames)

    @staticmethod
    def add_required_mc_field_names(cls, fieldnames):
        """Static method to add required monte-carlo field names to the list of
        already required field names for the monte-carlo.
        This method is useful for derived dataset classes.

        Parameters
        ----------
        cls : class object
            The class object, for which the new set of required mc field names
            should apply.
        fieldnames : str | list of str
            The field name or the list of field names to add.
        """
        if(not issequence(fieldnames)):
            fieldnames = [fieldnames]
        if(not issequenceof(fieldnames, str)):
            raise TypeError('The fieldnames argument must be a sequence of str objects!')

        cls._MC_FIELD_NAMES = tuple(list(Dataset._MC_FIELD_NAMES) + fieldnames)

    @staticmethod
    def get_combined_exp_pathfilenames(datasets):
        """Creates the combined list of exp pathfilenames of all the given
        datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        exp_pathfilenames : list
            The combined list of exp pathfilenames.
        """
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The datasets argument must be a sequence of Dataset instances!')

        exp_pathfilenames = []
        for ds in datasets:
            exp_pathfilenames += ds.exp_pathfilename_list

        return exp_pathfilenames

    @staticmethod
    def get_combined_mc_pathfilenames(datasets):
        """Creates the combined list of mc pathfilenames of all the given
        datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        mc_pathfilenames : list
            The combined list of mc pathfilenames.
        """
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The datasets argument must be a sequence of Dataset instances!')

        mc_pathfilenames = []
        for ds in datasets:
            mc_pathfilenames += ds.mc_pathfilename_list

        return mc_pathfilenames

    @staticmethod
    def get_combined_livetime(datasets):
        """Sums the live-time of all the given datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        livetime : float
            The sum of all the individual live-times.
        """
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The datasets argument must be a sequence of Dataset instances!')

        livetime = np.sum([ ds.livetime for ds in datasets ])

        return livetime

    def __init__(self, name, exp_pathfilenames, mc_pathfilenames, livetime, version, verqualifiers=None):
        """Creates a new dataset object that describes a self-consistent set of
        data.

        Parameters
        ----------
        name : str
            The name of the dataset.
        exp_pathfilenames : str | sequence of str
            The file name(s), including paths, of the experimental data file(s).
        mc_pathfilenames : str | sequence of str
            The file name(s), including paths, of the monte-carlo data file(s).
        livetime : float | None
            The integrated live-time in days of the dataset. It can be None for
            cases where the live-time is retrieved directly from the data files
            uppon data loading.
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

        self.exp_field_name_renaming_dict = dict()
        self.mc_field_name_renaming_dict = dict()

        self._data_preparation_functions = list()
        self._binning_definitions = dict()
        self._aux_data_definitions = dict()

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
            raise TypeError('The description of the dataset must be of '
                'type str!')
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
        if(not issequenceof(pathfilenames, str)):
            raise TypeError('The exp_pathfilename_list property must be of '
                'type str or a sequence of str!')
        self._exp_pathfilename_list = list(pathfilenames)

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
        if(not issequenceof(pathfilenames, str)):
            raise TypeError('The mc_pathfilename_list property must be of '
                'type str or a sequence of str!')
        self._mc_pathfilename_list = list(pathfilenames)

    @property
    def livetime(self):
        """The integrated live-time in days of the dataset. This can be None in
        cases where the livetime is retrieved directly from the data files.
        """
        return self._lifetime
    @livetime.setter
    def livetime(self, lt):
        if(lt is not None):
            lt = float_cast(lt,
                'The lifetime property of the dataset must be castable to '
                'type float!')
        self._lifetime = lt

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
        for (q,v) in verqualifiers.items():
            if(not isinstance(q, str)):
                raise TypeError('The version qualifier "%s" must be of type str!'%(q))
            if(not isinstance(v, int)):
                raise TypeError('The version for the qualifier "%s" must be of type int!'%(q))
        # We need to take a deep copy in order to make sure that two datasets
        # don't share the same version qualifier dictionary.
        self._verqualifiers = deepcopy(verqualifiers)

    @property
    def exp_field_name_renaming_dict(self):
        """The dictionary specifying the field names of the experimental data
        which need to get renamed just after loading the data. The dictionary
        values are the new names.
        """
        return self._exp_field_name_renaming_dict
    @exp_field_name_renaming_dict.setter
    def exp_field_name_renaming_dict(self, d):
        if(not isinstance(d, dict)):
            raise TypeError('The exp_field_name_renaming_dict property must '
                'be an instance of dict!')
        self._exp_field_name_renaming_dict = d

    @property
    def mc_field_name_renaming_dict(self):
        """The dictionary specifying the field names of the monte-carlo data
        which need to get renamed just after loading the data. The dictionary
        values are the new names.
        """
        return self._mc_field_name_renaming_dict
    @mc_field_name_renaming_dict.setter
    def mc_field_name_renaming_dict(self, d):
        if(not isinstance(d, dict)):
            raise TypeError('The mc_field_name_renaming_dict property must '
                'be an instance of dict!')
        self._mc_field_name_renaming_dict = d

    @property
    def exp_field_names(self):
        """(read-only) The tuple of numpy record ndarray field names for the
        experimental data.
        """
        return self.__class__._EXP_FIELD_NAMES

    @property
    def mc_field_names(self):
        """(read-only) The tuple of numpy record ndarray field names for the
        monto-carlo data.
        """
        return self.__class__._MC_FIELD_NAMES

    @property
    def version_str(self):
        """The version string of the dataset. This combines all the version
        information about the dataset.
        """
        s = '%03d'%(self._version)
        for (q,v) in self._verqualifiers.items():
            s += q+'%02d'%(v)
        return s

    @property
    def data_preparation_functions(self):
        """The list of callback functions that will be called to prepare the
        data (experimental and monte-carlo).
        """
        return self._data_preparation_functions

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

    def update_version_qualifiers(self, verqualifiers):
        """Updates the version qualifiers of the dataset. The update can only
        be done by increasing the version qualifier integer or by adding new
        version qualifiers.

        Parameters
        ----------
        verqualifiers : dict
            The dictionary with the new version qualifiers.

        Raises
        ------
        ValueError
            If the integer number of an existing version qualifier is not larger
            than the old one.
        """
        for q in verqualifiers:
            # If the qualifier already exist, it must have a larger integer
            # number.
            if((q in self._verqualifiers) and
               (verqualifiers[q] <= self._verqualifiers[q])):
                raise ValueError('The integer number (%d) of the version qualifier "%s" is not larger than the old integer number (%d)'%(verqualifiers[q], q, self._verqualifiers[q]))
            self._verqualifiers[q] = verqualifiers[q]

    def load_data(self, livetime=None, tl=None):
        """Loads the data, which is described by the dataset.

        Note: This does not call the ``prepare_data`` method! It only loads
              the data as the method names says.

        Parameters
        ----------
        livetime : float | None
            If not None, uses this livetime (in days) for the DatasetData
            instance, otherwise uses the Dataset livetime property value for
            the DatasetData instance.
        tl : TimeLord instance | None
            The TimeLord instance to use to time the data loading procedure.

        Returns
        -------
        data : DatasetData
            A DatasetData instance holding the experimental and monte-carlo
            data.
        """
        fileloader_exp = create_FileLoader(self._exp_pathfilename_list)
        fileloader_mc  = create_FileLoader(self._mc_pathfilename_list)

        with TaskTimer(tl, 'Loading exp data from disk.'):
            data_exp = DataFieldRecordArray(fileloader_exp.load_data())
            data_exp.rename_fields(self._exp_field_name_renaming_dict)

        with TaskTimer(tl, 'Loading mc data from disk.'):
            data_mc = DataFieldRecordArray(fileloader_mc.load_data())
            data_mc.rename_fields(self._mc_field_name_renaming_dict)

        if(livetime is None):
            livetime = self.livetime
        if(livetime is None):
            raise ValueError('No livetime was provided for dataset '
                '"%s"!'%(self.name))

        # Load all the auxiliary data for this dataset.
        data_aux = dict()
        for (aux_name, aux_pathfilename_list) in self._aux_data_definitions.items():
            with TaskTimer(tl, 'Loaded aux data "%s" from disk.'%(aux_name)):
                fileloader_aux = create_FileLoader(aux_pathfilename_list)
                data_aux[aux_name] = fileloader_aux.load_data()

        data = DatasetData(data_exp, data_mc, data_aux, livetime)

        return data

    def add_data_preparation(self, func):
        """Adds the given data preparation function to the dataset.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(data) that will prepare
            the data after it was loaded. The argument 'data' is a DatasetData
            instance holding the experimental and monte-carlo data. The function
            must alter the properties of the DatasetData instance.

        """
        if(not callable(func)):
            raise TypeError('The argument "func" must be a callable object with call signature __call__(data)!')
        self._data_preparation_functions.append(func)

    def remove_data_preparation(self, index=-1):
        """Removes data preparation function from the dataset.

        Parameters
        ----------
        index : int, optional
            Index of which data preparation function to remove. Default value
            is the last added function.
        """
        del self._data_preparation_functions[index]

    def prepare_data(self, data, tl=None):
        """Prepares the data by calling the data preparation callback functions
        of this dataset.

        Parameters
        ----------
        data : DatasetData instance
            The DatasetData instance holding the data.
        tl : TimeLord instance | None
            The TimeLord instance that should be used to time the data
            preparation.
        """
        for data_prep_func in self._data_preparation_functions:
            task = 'Preparing data of dataset "'+self.name+'" by '\
                '"'+data_prep_func.__name__+'".'
            with TaskTimer(tl, task):
                data_prep_func(data)

    def load_and_prepare_data(
            self, livetime=None, keep_fields=None, compress=False, tl=None):
        """Loads and prepares the experimental and monte-carlo data of this
        dataset by calling its ``load_data`` and ``prepare_data`` methods.
        After loading the data it drops all unnecessary data fields if they are
        not listed in ``keep_fields``.
        In the end it asserts the data format of the experimental and
        monte-carlo data.

        Parameters
        ----------
        livetime : float | None
            The user-defined livetime in days of the data set. If not set to
            None, livetime information from the data set will get ignored and
            this value of the livetime will be used.
        keep_fields : sequence of str | None
            The list of additional data fields that should get kept.
            By default only the required data fields are kept.
        compress : bool
            Flag if the float64 data fields of the data should get converted,
            i.e. compressed, into float32 data fields, in order to save main
            memory.
            The only field, which will not get converted is the 'mcweight'
            field, in order to ensure reliable calculations.
            Default is False.
        tl : TimeLord instance | None
            The TimeLord instance that should be used to time the data loading
            and preparation.

        Returns
        -------
        data : DatasetData
            The DatasetData instance holding the experimental and monte-carlo
            data.
        """
        if(keep_fields is None):
            keep_fields = tuple()
        if(not issequenceof(keep_fields, str)):
            raise TypeError('The keep_fields argument must be None, or a '
                'sequence of str!')
        keep_fields = tuple(keep_fields)

        data = self.load_data(livetime=livetime, tl=tl)
        self.prepare_data(data, tl=tl)

        # Drop unrequired data fields.
        with TaskTimer(tl, 'Cleaning exp data.'):
            data.exp.tidy_up(keep_fields=(type(self)._EXP_FIELD_NAMES +
                                          keep_fields))
        with TaskTimer(tl, 'Cleaning MC data.'):
            data.mc.tidy_up(keep_fields=(type(self)._EXP_FIELD_NAMES +
                                         type(self)._MC_FIELD_NAMES +
                                         keep_fields))

        # Convert float64 fields into float32 fields if requested.
        if(compress):
            dtype_convertions = { np.dtype(np.float64): np.dtype(np.float32) }
            with TaskTimer(tl, 'Compressing exp data.'):
                data.exp.convert_dtypes(dtype_convertions)

            with TaskTimer(tl, 'Compressing MC data.'):
                data.mc.convert_dtypes(dtype_convertions,
                    except_fields=('mcweight',))

        with TaskTimer(tl, 'Asserting data format.'):
            assert_data_format(self, data)

        return data

    def add_binning_definition(self, binning):
        """Adds a binning setting to this dataset.

        Parameters
        ----------
        binning : BinningDefinition
            The BinningDefinition object holding the binning information.
        """
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The "binning" argument must be of type BinningDefinition!')
        if(binning.name in self._binning_definitions):
            raise KeyError('The binning definition "%s" is already defined for season "%s"!'%(binning.name, self._name))

        self._binning_definitions[binning.name] = binning

    def get_binning_definition(self, name):
        """Gets the BinningDefinition object for the given binning name.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        Returns
        -------
        binning_definition : BinningDefinition instance
            The requested BinningDefinition instance.
        """
        if(name not in self._binning_definitions):
            raise KeyError('The given binning name "%s" has not been added to the dataset yet!'%(name))
        return self._binning_definitions[name]

    def has_binning_definition(self, name):
        """Checks if the dataset has a defined binning definition with the given
        name.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        Returns
        -------
        check : bool
            True if the binning definition exists, False otherwise.
        """
        if(name in self._binning_definitions):
            return True
        return False

    def define_binning(self, name, binedges):
        """Defines a binning for ``name``, and adds it as binning definition.

        Parameters
        ----------
        name : str
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
        binning = BinningDefinition(name, binedges)
        self.add_binning_definition(binning)
        return binning

    def add_aux_data_definition(self, name, pathfilenames):
        """Adds the given data files as auxiliary data definition to the
        dataset.

        Parameters
        ----------
        name : str
            The name of the auxiliary data. The name is used as identifier for
            the data within SkyLLH.
        pathfilenames : str | sequence of str
            The file name(s) (including paths) of the data file(s).
        """
        name = str_cast(name,
            'The name argument must be castable to type str!')
        pathfilenames = list_of_cast(str, pathfilenames,
            'The pathfilenames argument must be of type str or a sequence '
            'of str!')

        if(name in self._aux_data_definitions):
            raise KeyError('The auxiliary data definition "%s" is already '
                'defined for dataset "%s"!'%(name, self.name))

        self._aux_data_definitions[name] = pathfilenames


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
                raise TypeError('The dataset object must be a sub-class of '
                    'Dataset!')

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
            raise KeyError('Dataset "%s" is not part of the dataset '
                'collection "%s", nothing to remove!'%(name, self.name))

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
            raise KeyError('The dataset "%s" is not part of the dataset '
                'collection "%s"!'%(name, self.name))
        return self._datasets[name]

    def set_exp_field_name_renaming_dict(self, d):
        """Sets the dictionary with the data field names of the experimental
        data that needs to be renamed just after loading the data. The
        dictionary will be set to all added data sets.

        Parameters
        ----------
        d : dict
            The dictionary with the old field names as keys and the new field
            names as values.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.exp_field_name_renaming_dict = d

    def set_mc_field_name_renaming_dict(self, d):
        """Sets the dictionary with the data field names of the monte-carlo
        data that needs to be renamed just after loading the data. The
        dictionary will be set to all added data sets.

        Parameters
        ----------
        d : dict
            The dictionary with the old field names as keys and the new field
            names as values.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.mc_field_name_renaming_dict = d

    def define_binning(self, name, binedges):
        """Defines a binning definition and adds it to all the datasets of this
        dataset collection.

        Parameters
        ----------
        name : str
            The name of the binning definition.
        binedges : sequence
            The sequence of the bin edges, that should be used for the binning.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.define_binning(name, binedges)

    def add_data_preparation(self, func):
        """Adds the data preparation function to all the datasets of this
        dataset collection.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(data) that will prepare
            the data after it was loaded. The argument 'data' is the DatasetData
            instance holding the experimental and monte-carlo data.
            This function must alter the properties of the DatasetData instance.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.add_data_preparation(func)

    def remove_data_preparation(self, index=-1):
        """Removes data preparation function from all the datasets of this
        dataset collection.

        Parameters
        ----------
        index : int, optional
            Index of which data preparation function to remove. Default value
            is the last added function.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.remove_data_preparation(index=index)

    def update_version_qualifiers(self, verqualifiers):
        """Updates the version qualifiers of all datasets of this dataset
        collection.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.update_version_qualifiers(verqualifiers)


class DatasetData(object):
    """This class provides the container for the actual experimental and
    monto-carlo data. It also holds a reference to the Dataset instance, which
    holds the data's meta information.
    """
    def __init__(self, data_exp, data_mc, data_aux, livetime):
        """Creates a new DatasetData instance.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the experimental data.
        data_mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monto-carlo data.
        data_aux : dict
            The dictionary holding the auxiliary data for the dataset. The key
            of the dictionary identifies the auxiliary data. The value is the
            data in any type and format.
        livetime : float
            The integrated livetime in days of the data.
        """
        super(DatasetData, self).__init__()

        self.exp = data_exp
        self.mc = data_mc
        self.aux = data_aux
        self.livetime = livetime

    @property
    def exp(self):
        """The DataFieldRecordArray instance holding the experimental data.
        """
        return self._exp
    @exp.setter
    def exp(self, data):
        if(not isinstance(data, DataFieldRecordArray)):
            raise TypeError('The exp property must be an instance of '
                'DataFieldRecordArray!')
        self._exp = data

    @property
    def mc(self):
        """The DataFieldRecordArray instance holding the monte-carlo data.
        """
        return self._mc
    @mc.setter
    def mc(self, data):
        if(not isinstance(data, DataFieldRecordArray)):
            raise TypeError('The mc property must be an instance of '
                'DataFieldRecordArray!')
        self._mc = data

    @property
    def aux(self):
        """The dictionary holding the auxiliary data of the dataset. The key
        of the dictionary identifies the auxiliary data. The value is the data
        in any type and format.
        """
        return self._aux
    @aux.setter
    def aux(self, d):
        if(not isinstance(d, dict)):
            raise TypeError('The aux property must be an instance of dict!')
        self._aux = d

    @property
    def livetime(self):
        """The integrated livetime in days of the data.
        """
        return self._livetime
    @livetime.setter
    def livetime(self, lt):
        lt = float_cast(lt,
            'The livetime property must be castable to type float!')
        self._livetime = lt

    @property
    def exp_field_names(self):
        """(read-only) The list of field names present in the experimental data.
        """
        return self._exp.field_name_list

    @property
    def mc_field_names(self):
        """(read-only) The list of field names present in the monte-carlo data.
        """
        return self._mc.field_name_list


def assert_data_format(dataset, data):
    """Checks the format of the experimental and monte-carlo data.

    Raises
    ------
    KeyError
        If a required data field is missing.
    """
    def _get_missing_keys(keys, required_keys):
        missing_keys = []
        for reqkey in required_keys:
            if(reqkey not in keys):
                missing_keys.append(reqkey)
        return missing_keys

    dataset_cls = type(dataset)

    # Check experimental data keys.
    missing_exp_keys = _get_missing_keys(data.exp.field_name_list,
        dataset_cls._EXP_FIELD_NAMES)
    if(len(missing_exp_keys) != 0):
        raise KeyError('The following data fields are missing for the experimental data of dataset "%s": '%(dataset.name)+', '.join(missing_exp_keys))

    # Check monte-carlo data keys.
    missing_mc_keys = _get_missing_keys(data.mc.field_name_list,
        dataset_cls._EXP_FIELD_NAMES + dataset_cls._MC_FIELD_NAMES)
    if(len(missing_mc_keys) != 0):
        raise KeyError('The following data fields are missing for the monte-carlo data of dataset "%s": '%(dataset.name)+', '.join(missing_mc_keys))


def remove_events(data_exp, mjds):
    """Utility function to remove events having the specified MJD time stamps.

    Parameters
    ----------
    data_exp : numpy record ndarray
        The numpy record ndarray holding the experimental data events.
    mjds : float | array of floats
        The MJD time stamps of the events, that should get removed from the
        experimental data array.

    Returns
    -------
    data_exp : numpy record ndarray
        The array holding the experimental data events with the specified events
        removed.
    """
    mjds = np.atleast_1d(mjds)

    for time in mjds:
        mask = data_exp['time'] == time
        if(np.sum(mask) > 1):
            raise LookupError('The MJD time stamp %f is not unique!'%(time))
        data_exp = data_exp[~mask]

    return data_exp

def generate_data_file_path(
    default_base_path, default_sub_path,
    version, verqualifiers,
    base_path=None, sub_path=None
):
    """Generates the path to the data files based on the given base path and
    sub path. If base_path is None, default_base_path is used. If sub_path is
    None, default_sub_path is used.

    The default_sub_path and sub_path can contain the following wildcards:
        - '%(version)d'
        - '%(<verqualifiers_key>)d'

    Parameters
    ----------
    default_base_path : str
        The default base path if base_path is None.
    default_sub_path : str
        The default sub path if sub_path is None.
    version : int
        The version of the data sample.
    verqualifiers : dict
        The dictionary holding the version qualifiers of the data sample.
    base_path : str | None
        The user-specified base path.
    sub_path : str | None
        The user-specified sub path.

    Returns
    -------
    path : str
        The generated data file path.
    """
    if(base_path is None):
        base_path = default_base_path

    if(sub_path is None):
        sub_path = default_sub_path

    subdict = dict( [('version', version)] + list(verqualifiers.items()) )
    sub_path = sub_path%subdict

    path = os.path.join(base_path, sub_path)

    return path

def get_data_subset(data, livetime, t_start, t_end):
    """Gets DatasetData and Livetime objects with data subsets between the given
    time range from t_start to t_end.

    Parameters
    ----------
    data : DatasetData
        The DatasetData object.
    livetime : Livetime
        The Livetime object.
    t_start : float
        The MJD start time of the time range to consider.
    t_end : float
        The MJD end time of the time range to consider.

    Returns
    -------
    dataset_data_subset : DatasetData
        DatasetData object with subset of the data between the given time range
        from t_start to t_end.
    livetime_subset : Livetime
        Livetime object with subset of the data between the given time range
        from t_start to t_end.
    """
    if(not isinstance(data, DatasetData)):
        raise TypeError('The "data" argument must be of type DatasetData!')
    if(not isinstance(livetime, Livetime)):
        raise TypeError('The "livetime" argument must be of type Livetime!')

    exp_slice = np.logical_and(data.exp['time'] >= t_start,
                               data.exp['time'] < t_end)
    mc_slice = np.logical_and(data.mc['time'] >= t_start,
                              data.mc['time'] < t_end)

    data_exp = data.exp[exp_slice]
    data_mc = data.mc[mc_slice]

    uptime_mjd_intervals_arr = livetime.get_ontime_intervals_between(t_start, t_end)
    livetime_subset = Livetime(uptime_mjd_intervals_arr)

    dataset_data_subset = DatasetData(data_exp, data_mc, data.aux, livetime_subset.livetime)

    return (dataset_data_subset, livetime_subset)
