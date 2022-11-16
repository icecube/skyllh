# -*- coding: utf-8 -*-

import os
import os.path
import numpy as np
from copy import deepcopy

from skyllh.core.binning import BinningDefinition
from skyllh.core.config import CFG
from skyllh.core.livetime import Livetime
from skyllh.core.progressbar import ProgressBar
from skyllh.core.py import (
    float_cast,
    issequence,
    issequenceof,
    list_of_cast,
    str_cast
)
from skyllh.core import display
from skyllh.core.display import ANSIColors
from skyllh.core.storage import (
    DataFieldRecordArray,
    create_FileLoader
)
from skyllh.core.timing import TaskTimer


class Dataset(object):
    """The Dataset class describes a set of self-consistent experimental and
    simulated detector data. Usually this is for a certain time period, i.e.
    a season.

    Independet data sets of the same kind, e.g. event selection, can be joined
    through a DatasetCollection object.
    """
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

    def __init__(
            self, name, exp_pathfilenames, mc_pathfilenames, livetime,
            default_sub_path_fmt, version, verqualifiers=None,
            base_path=None, sub_path_fmt=None):
        """Creates a new dataset object that describes a self-consistent set of
        data.

        Parameters
        ----------
        name : str
            The name of the dataset.
        exp_pathfilenames : str | sequence of str | None
            The file name(s), including paths, of the experimental data file(s).
            This can be None, if a MC-only study is performed.
        mc_pathfilenames : str | sequence of str
            The file name(s), including paths, of the monte-carlo data file(s).
        livetime : float | None
            The integrated live-time in days of the dataset. It can be None for
            cases where the live-time is retrieved directly from the data files
            uppon data loading.
        default_sub_path_fmt : str
            The default format of the sub path of the data set.
            This must be a string that can be formatted via the ``format``
            method of the ``str`` class.
        version : int
            The version number of the dataset. Higher version numbers indicate
            newer datasets.
        verqualifiers : dict | None
            If specified, this dictionary specifies version qualifiers. These
            can be interpreted as subversions of the dataset. The format of the
            dictionary must be 'qualifier (str): version (int)'.
        base_path : str | None
            The user-defined base path of the data set.
            Usually, this is the path of the location of the data directory.
            If set to ``None`` the configured repository base path
            ``CFG['repository']['base_path']`` is used.
        sub_path_fmt : str | None
            The user-defined format of the sub path of the data set.
            If set to ``None``, the ``default_sub_path_fmt`` will be used.
        """
        self.name = name
        self.exp_pathfilename_list = exp_pathfilenames
        self.mc_pathfilename_list = mc_pathfilenames
        self.livetime = livetime
        self.default_sub_path_fmt = default_sub_path_fmt
        self.version = version
        self.verqualifiers = verqualifiers
        self.base_path = base_path
        self.sub_path_fmt = sub_path_fmt

        self.description = ''

        self._loading_extra_exp_field_name_list = list()
        self._loading_extra_mc_field_name_list = list()

        self._exp_field_name_renaming_dict = dict()
        self._mc_field_name_renaming_dict = dict()

        self._data_preparation_functions = list()
        self._binning_definitions = dict()
        self._aux_data_definitions = dict()
        self._aux_data = dict()

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
        """The list of file names of the data files that store the experimental
        data for this dataset.
        If a file name is given with a relative path, it will be relative to the
        root_dir property of this Dataset instance.
        """
        return self._exp_pathfilename_list
    @exp_pathfilename_list.setter
    def exp_pathfilename_list(self, pathfilenames):
        if(pathfilenames is None):
            pathfilenames = []
        if(isinstance(pathfilenames, str)):
            pathfilenames = [pathfilenames]
        if(not issequenceof(pathfilenames, str)):
            raise TypeError('The exp_pathfilename_list property must be of '
                'type str or a sequence of str!')
        self._exp_pathfilename_list = list(pathfilenames)

    @property
    def exp_abs_pathfilename_list(self):
        """(read-only) The list of absolute path file names of the experimental
        data files.
        """
        return self.get_abs_pathfilename_list(self._exp_pathfilename_list)

    @property
    def mc_pathfilename_list(self):
        """The list of file names of the data files that store the monte-carlo
        data for this dataset.
        If a file name is given with a relative path, it will be relative to the
        root_dir property of this Dataset instance.
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
    def mc_abs_pathfilename_list(self):
        """(read-only) The list of absolute path file names of the monte-carlo
        data files.
        """
        return self.get_abs_pathfilename_list(self._mc_pathfilename_list)

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
    def base_path(self):
        """The base path of the data set. This can be ``None``.
        """
        return self._base_path
    @base_path.setter
    def base_path(self, path):
        if(path is not None):
            path = str_cast(path, 'The base_path property must be castable to '
                'type str!')
            if(not os.path.isabs(path)):
                raise ValueError('The base_path property must be an absolute '
                    'path!')
        self._base_path = path

    @property
    def default_sub_path_fmt(self):
        """The default format of the sub path of the data set. This must be a
        string that can be formatted via the ``format`` method of the ``str``
        class.
        """
        return self._default_sub_path_fmt
    @default_sub_path_fmt.setter
    def default_sub_path_fmt(self, fmt):
        fmt = str_cast(fmt, 'The default_sub_path_fmt property must be '
            'castable to type str!')
        self._default_sub_path_fmt = fmt

    @property
    def sub_path_fmt(self):
        """The format of the sub path of the data set. This must be a string
        that can be formatted via the ``format`` method of the ``str`` class.
        If set to ``None``, this property will return the
        ``default_sub_path_fmt`` property.
        """
        if(self._sub_path_fmt is None):
            return self._default_sub_path_fmt
        return self._sub_path_fmt
    @sub_path_fmt.setter
    def sub_path_fmt(self, fmt):
        if(fmt is not None):
            fmt = str_cast(fmt, 'The sub_path_fmt property must be None, or '
                'castable to type str!')
        self._sub_path_fmt = fmt

    @property
    def root_dir(self):
        """(read-only) The root directory to use when data files are specified
        with relative paths. It is constructed from the ``base_path`` and the
        ``sub_path_fmt`` properties via the ``generate_data_file_root_dir``
        function.
        """
        return generate_data_file_root_dir(
            default_base_path=CFG['repository']['base_path'],
            default_sub_path_fmt=self._default_sub_path_fmt,
            version=self._version,
            verqualifiers=self._verqualifiers,
            base_path=self._base_path,
            sub_path_fmt=self._sub_path_fmt)

    @property
    def loading_extra_exp_field_name_list(self):
        """The list of extra field names that should get loaded when loading
        experimental data. These should only be field names that are required
        during the data preparation of this specific data set.
        """
        return self._loading_extra_exp_field_name_list
    @loading_extra_exp_field_name_list.setter
    def loading_extra_exp_field_name_list(self, fieldnames):
        if(isinstance(fieldnames, str)):
            fieldnames = [ fieldnames ]
        elif(not issequenceof(fieldnames, str)):
            raise TypeError('The loading_extra_exp_field_name_list property '
                'must be an instance of str or a sequence of str type '
                'instances!')
        self._loading_extra_exp_field_name_list = list(fieldnames)

    @property
    def loading_extra_mc_field_name_list(self):
        """The list of extra field names that should get loaded when loading
        monte-carlo data. These should only be field names that are required
        during the data preparation of this specific data set.
        """
        return self._loading_extra_mc_field_name_list
    @loading_extra_mc_field_name_list.setter
    def loading_extra_mc_field_name_list(self, fieldnames):
        if(isinstance(fieldnames, str)):
            fieldnames = [ fieldnames ]
        elif(not issequenceof(fieldnames, str)):
            raise TypeError('The loading_extra_mc_field_name_list property '
                'must be an instance of str or a sequence of str type '
                'instances!')
        self._loading_extra_mc_field_name_list = list(fieldnames)

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
    def exists(self):
        """(read-only) Flag if all the data files of this data set exists. It is
        ``True`` if all data files exist and ``False`` otherwise.
        """
        for pathfilename in (self.exp_abs_pathfilename_list +
                             self.mc_abs_pathfilename_list):
            if(not os.path.exists(pathfilename)):
                return False
        return True

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

    def _gen_datafile_pathfilename_entry(self, pathfilename):
        """Generates a string containing the given pathfilename and if it exists
        (FOUND) or not (NOT FOUND).

        Parameters
        ----------
        pathfilename : str
            The fully qualified path and filename of the file.

        Returns
        -------
        s : str
            The generated string.
        """
        if(os.path.exists(pathfilename)):
            s = '['+ANSIColors.OKGREEN+'FOUND'+ANSIColors.ENDC+']'
        else:
            s = '['+ANSIColors.FAIL+'NOT FOUND'+ANSIColors.ENDC+']'
        s += ' ' + pathfilename
        return s

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

        s1 = ''

        if(self.livetime is None):
            s1 += '{ livetime = UNDEFINED }'
        else:
            s1 += '{ livetime = %.3f days }'%(self.livetime)
        s1 += '\n'

        if(self.description != ''):
            s1 += 'Description:\n' + self.description + '\n'

        s1 += 'Experimental data:\n'
        s2 = ''
        for (idx, pathfilename) in enumerate(self.exp_abs_pathfilename_list):
            if(idx > 0):
                s2 += '\n'
            s2 += self._gen_datafile_pathfilename_entry(pathfilename)
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)
        s1 += '\n'

        s1 += 'MC data:\n'
        s2 = ''
        for (idx, pathfilename) in enumerate(self.mc_abs_pathfilename_list):
            if(idx > 0):
                s2 += '\n'
            s2 += self._gen_datafile_pathfilename_entry(pathfilename)
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)
        s1 += '\n'

        if(len(self._aux_data_definitions) > 0):
            s1 += 'Auxiliary data:\n'
            s2 = ''
            for (idx,(name, pathfilename_list)) in enumerate(
                self._aux_data_definitions.items()):
                if(idx > 0):
                    s2 += '\n'

                s2 += name+':'
                s3 = ''
                pathfilename_list = self.get_abs_pathfilename_list(
                    pathfilename_list)
                for pathfilename in pathfilename_list:
                    s3 += '\n' + self._gen_datafile_pathfilename_entry(pathfilename)
                s2 += display.add_leading_text_line_padding(
                    display.INDENTATION_WIDTH, s3)
            s1 += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, s2)

        s += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)

        return s

    def get_abs_pathfilename_list(self, pathfilename_list):
        """Returns a list where each entry of the given pathfilename_list is
        an absolute path. Relative paths will be prefixed with the root_dir
        property of this Dataset instance.

        Parameters
        ----------
        pathfilename_list : sequence of str
            The sequence of file names, either with relative, or absolute paths.

        Returns
        -------
        abs_pathfilename_list : list of str
            The list of file names with absolute paths.
        """
        root_dir = self.root_dir

        abs_pathfilename_list = []
        for pathfilename in pathfilename_list:
            if(os.path.isabs(pathfilename)):
                abs_pathfilename_list.append(
                    pathfilename)
            else:
                abs_pathfilename_list.append(
                    os.path.join(root_dir, pathfilename))

        return abs_pathfilename_list

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
        got_new_verqualifiers = False
        verqualifiers_keys = verqualifiers.keys()
        self_verqualifiers_keys = self._verqualifiers.keys()
        if(len(verqualifiers_keys) > len(self_verqualifiers_keys)):
            # New version qualifiers must be a subset of the old version
            # qualifiers.
            for q in self_verqualifiers_keys:
                if(not q in verqualifiers_keys):
                    raise ValueError('The version qualifier {} has been '
                        'dropped!'.format(q))
            got_new_verqualifiers = True

        existing_verqualifiers_incremented = False
        for q in verqualifiers:
            if((q in self._verqualifiers) and
               (verqualifiers[q] > self._verqualifiers[q])):
                existing_verqualifiers_incremented = True
            self._verqualifiers[q] = verqualifiers[q]

        if(not (got_new_verqualifiers or existing_verqualifiers_incremented)):
            raise ValueError('Version qualifier values did not increment and '
                'no new version qualifiers were added!')

    def load_data(
            self, keep_fields=None, livetime=None, dtc_dict=None,
            dtc_except_fields=None, efficiency_mode=None, tl=None):
        """Loads the data, which is described by the dataset.

        Note: This does not call the ``prepare_data`` method! It only loads
              the data as the method names says.

        Parameters
        ----------
        keep_fields : list of str | None
            The list of user-defined data fields that should get loaded and kept
            in addition to the analysis required data fields.
        livetime : float | None
            If not None, uses this livetime (in days) for the DatasetData
            instance, otherwise uses the Dataset livetime property value for
            the DatasetData instance.
        dtc_dict : dict | None
            This dictionary defines how data fields of specific
            data types should get converted into other data types.
            This can be used to use less memory. If set to None, no data
            convertion is performed.
        dtc_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        efficiency_mode : str | None
            The efficiency mode the data should get loaded with. Possible values
            are:

                - 'memory':
                    The data will be load in a memory efficient way. This will
                    require more time, because all data records of a file will
                    be loaded sequentially.
                - 'time'
                    The data will be loaded in a time efficient way. This will
                    require more memory, because each data file gets loaded in
                    memory at once.

            The default value is ``'time'``. If set to ``None``, the default
            value will be used.
        tl : TimeLord instance | None
            The TimeLord instance to use to time the data loading procedure.

        Returns
        -------
        data : DatasetData
            A DatasetData instance holding the experimental and monte-carlo
            data.
        """
        def _conv_new2orig_field_names(new_field_names, orig2new_renaming_dict):
            """Converts the given ``new_field_names`` into their original name
            given the original-to-new field name renaming dictionary.
            """
            if(new_field_names is None):
                return None

            new2orig_renaming_dict = dict()
            for (k,v) in orig2new_renaming_dict.items():
                new2orig_renaming_dict[v] = k

            orig_field_names = [
                new2orig_renaming_dict.get(new_field_name, new_field_name)
                    for new_field_name in new_field_names
            ]

            return orig_field_names

        if(keep_fields is None):
            keep_fields = []

        # Load the experimental data if there is any.
        if(len(self._exp_pathfilename_list) > 0):
            fileloader_exp = create_FileLoader(self.exp_abs_pathfilename_list)
            with TaskTimer(tl, 'Loading exp data from disk.'):
                # Create the list of field names that should get kept.
                keep_fields = list(set(
                    _conv_new2orig_field_names(
                        CFG['dataset']['analysis_required_exp_field_names'] +
                        self._loading_extra_exp_field_name_list +
                        keep_fields,
                        self._exp_field_name_renaming_dict
                    )
                ))

                data_exp = fileloader_exp.load_data(
                    keep_fields=keep_fields,
                    dtype_convertions=dtc_dict,
                    dtype_convertion_except_fields=_conv_new2orig_field_names(
                        dtc_except_fields,
                        self._exp_field_name_renaming_dict),
                    efficiency_mode=efficiency_mode)
                data_exp.rename_fields(self._exp_field_name_renaming_dict)
        else:
            data_exp = None

        # Load the monte-carlo data.
        with TaskTimer(tl, 'Loading mc data from disk.'):
            fileloader_mc = create_FileLoader(self.mc_abs_pathfilename_list)
            keep_fields = list(set(
                _conv_new2orig_field_names(
                    CFG['dataset']['analysis_required_exp_field_names'] +
                    self._loading_extra_exp_field_name_list +
                    keep_fields,
                    self._exp_field_name_renaming_dict) +
                _conv_new2orig_field_names(
                    CFG['dataset']['analysis_required_mc_field_names'] +
                    self._loading_extra_mc_field_name_list +
                    keep_fields,
                    self._mc_field_name_renaming_dict)
            ))
            data_mc = fileloader_mc.load_data(
                keep_fields=keep_fields,
                dtype_convertions=dtc_dict,
                dtype_convertion_except_fields=_conv_new2orig_field_names(
                    dtc_except_fields,
                    self._mc_field_name_renaming_dict),
                efficiency_mode=efficiency_mode)
            data_mc.rename_fields(self._mc_field_name_renaming_dict)

        if(livetime is None):
            livetime = self.livetime
        if(livetime is None):
            raise ValueError('No livetime was provided for dataset '
                '"%s"!'%(self.name))

        data = DatasetData(data_exp, data_mc, livetime)

        return data

    def load_aux_data(self, name, tl=None):
        """Loads the auxiliary data for the given auxiliary data definition.

        Parameters
        ----------
        name : str
            The name of the auxiliary data.
        tl : TimeLord instance | None
            The TimeLord instance to use to time the data loading procedure.

        Returns
        -------
        data : unspecified
            The loaded auxiliary data.
        """
        name = str_cast(name,
            'The name argument must be castable to type str!')

        # Check if the data was defined in memory.
        if(name in self._aux_data):
            with TaskTimer(tl, 'Loaded aux data "%s" from memory.'%(name)):
                data = self._aux_data[name]
            return data

        if(name not in self._aux_data_definitions):
            raise KeyError('The auxiliary data named "%s" does not exist!'%(
                name))

        aux_pathfilename_list = self._aux_data_definitions[name]
        with TaskTimer(tl, 'Loaded aux data "%s" from disk.'%(name)):
            fileloader_aux = create_FileLoader(self.get_abs_pathfilename_list(
                aux_pathfilename_list))
            data = fileloader_aux.load_data()

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

    def remove_data_preparation(self, key=-1):
        """Removes a data preparation function from the dataset.

        Parameters
        ----------
        key : str, int, optional
            The name or the index of the data preparation function that should
            be removed. Default value is ``-1``, i.e. the last added function.

        Raises
        ------
        TypeError
            If the type of the key argument is invalid.
        IndexError
            If the given key is out of range.
        KeyError
            If the data preparation function cannot be found.
        """
        if(isinstance(key, int)):
            n = len(self._data_preparation_functions)
            if((key < -n) or (key >= n)):
                raise IndexError('The given index (%d) for the data '
                    'preparation function is out of range (%d,%d)!'%(
                        key, -n, n-1))
            del self._data_preparation_functions[key]
            return
        elif(isinstance(key, str)):
            for (i,func) in enumerate(self._data_preparation_functions):
                if(func.__name__ == key):
                    del self._data_preparation_functions[i]
                    return
            raise KeyError('The data preparation function "%s" was not found '
                'in the dataset "%s"!'%(key, self._name))

        TypeError('The key argument must be an instance of int or str!')

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
            self, livetime=None, keep_fields=None, compress=False,
            efficiency_mode=None, tl=None):
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
        efficiency_mode : str | None
            The efficiency mode the data should get loaded with. Possible values
            are:

                - 'memory':
                    The data will be load in a memory efficient way. This will
                    require more time, because all data records of a file will
                    be loaded sequentially.
                - 'time'
                    The data will be loaded in a time efficient way. This will
                    require more memory, because each data file gets loaded in
                    memory at once.

            The default value is ``'time'``. If set to ``None``, the default
            value will be used.
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
            keep_fields = list()
        elif(not issequenceof(keep_fields, str)):
            raise TypeError('The keep_fields argument must be None, or a '
                'sequence of str!')
        keep_fields = list(keep_fields)

        dtc_dict = None
        dtc_except_fields = None
        if(compress):
            dtc_dict = { np.dtype(np.float64): np.dtype(np.float32) }
            dtc_except_fields = [ 'mcweight' ]

        data = self.load_data(
            keep_fields=keep_fields,
            livetime=livetime,
            dtc_dict=dtc_dict,
            dtc_except_fields=dtc_except_fields,
            efficiency_mode=efficiency_mode,
            tl=tl)

        self.prepare_data(data, tl=tl)

        # Drop unrequired data fields.
        if(data.exp is not None):
            with TaskTimer(tl, 'Cleaning exp data.'):
                keep_fields_exp = (
                    CFG['dataset']['analysis_required_exp_field_names'] +
                    keep_fields
                )
                data.exp.tidy_up(keep_fields=keep_fields_exp)
        with TaskTimer(tl, 'Cleaning MC data.'):
            keep_fields_mc = (
                CFG['dataset']['analysis_required_exp_field_names'] +
                CFG['dataset']['analysis_required_mc_field_names'] +
                keep_fields
            )
            data.mc.tidy_up(keep_fields=keep_fields_mc)

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
            raise TypeError('The "binning" argument must be of type '
                'BinningDefinition!')
        if(binning.name in self._binning_definitions):
            raise KeyError('The binning definition "%s" is already defined for '
                'dataset "%s"!'%(binning.name, self._name))

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
            raise KeyError('The given binning name "%s" has not been added to '
                'the dataset yet!'%(name))
        return self._binning_definitions[name]

    def remove_binning_definition(self, name):
        """Removes the BinningDefinition object from the dataset.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        """
        if(name not in self._binning_definitions):
            raise KeyError(
                f'The given binning name "{name}" does not exist in the '
                f'dataset "{self.name}", nothing to remove!'
            )

        self._binning_definitions.pop(name)

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

    def replace_binning_definition(self, binning):
        """Replaces an already defined binning definition of this dataset by
        the given binning definition.

        Parameters
        ----------
        binning : BinningDefinition instance
            The instance of BinningDefinition that will replace the data set's
            BinningDefinition instance of the same name.
        """
        if(not isinstance(binning, BinningDefinition)):
            raise TypeError('The "binning" argument must be of type '
                'BinningDefinition!')
        if(binning.name not in self._binning_definitions):
            raise KeyError('The given binning definition "%s" has not been '
                'added to the dataset yet!'%(binning.name))

        self._binning_definitions[binning.name] = binning

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

    def get_aux_data_definition(self, name):
        """Returns the auxiliary data definition from the dataset.

        Parameters
        ----------
        name : str
            The name of the auxiliary data.

        Raises
        ------
        KeyError
            If auxiliary data with the given name does not exist.

        Returns
        -------
        aux_data_definition : list of str
            The locations (pathfilenames) of the files defined in the auxiliary data
                    as auxiliary data definition.
        """

        if(not name in self._aux_data_definitions):
            raise KeyError('The auxiliary data definition "{}" does not '
                'exist in dataset "{}"!'.format(name, self.name))

        return self._aux_data_definitions[name]

    def remove_aux_data_definition(self, name):
        """Removes the auxiliary data definition from the dataset.

        Parameters
        ----------
        name : str
            The name of the dataset that should get removed.
        """
        if(name not in self._aux_data_definitions):
            raise KeyError(
                f'The auxiliary data definition "{name}" does not exist in '
                f'dataset "{self.name}", nothing to remove!'
            )

        self._aux_data_definitions.pop(name)

    def add_aux_data(self, name, data):
        """Adds the given data as auxiliary data to this data set.

        Parameters
        ----------
        name : str
            The name under which the auxiliary data will be stored.
        data : unspecified
            The data that should get stored. This can be of any data type.

        Raises
        ------
        KeyError
            If auxiliary data is already stored under the given name.
        """
        name = str_cast(name,
            'The name argument must be castable to type str!')

        if(name in self._aux_data):
            raise KeyError('The auxiliary data "%s" is already defined for '
                'dataset "%s"!'%(name, self.name))

        self._aux_data[name] = data

    def get_aux_data(self, name):
        """Retrieves the auxiliary data that is stored in this data set under
        the given name.

        Parameters
        ----------
        name : str
            The name under which the auxiliary data is stored.

        Returns
        -------
        data : unspecified
            The retrieved auxiliary data.

        Raises
        ------
        KeyError
            If no auxiliary data is stored with the given name.
        """
        name = str_cast(name,
            'The name argument must be castable to type str!')

        if(name not in self._aux_data):
            raise KeyError('The auxiliary data "%s" is not defined for '
                'dataset "%s"!'%(name, self.name))

        return self._aux_data[name]

    def remove_aux_data(self, name):
        """Removes the auxiliary data that is stored in this data set under
        the given name.

        Parameters
        ----------
        name : str
            The name of the dataset that should get removed.
        """
        if(name not in self._aux_data):
            raise KeyError(
                f'The auxiliary data "{name}" is not defined for dataset '
                f'"{self.name}", nothing to remove!'
            )

        self._aux_data.pop(name)


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

    @property
    def version(self):
        """(read-only) The version number of the datasets collected by this
        dataset collection.
        """
        ds_name = list(self._datasets.keys())[0]
        return self._datasets[ds_name].version

    @property
    def verqualifiers(self):
        """(read-only) The dictionary holding the version qualifiers of the
        datasets collected by this dataset collection.
        """
        ds_name = list(self._datasets.keys())[0]
        return self._datasets[ds_name].verqualifiers

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
        dataset : Dataset instance
            The Dataset object holding all the information about the dataset.

        Raises
        ------
        KeyError
            If the data set of the given name is not present in this data set
            collection.
        """
        if(name not in self._datasets):
            raise KeyError('The dataset "%s" is not part of the dataset '
                'collection "%s"!'%(name, self.name))
        return self._datasets[name]

    def get_datasets(self, names):
        """Retrieves a list of Dataset objects from this dataset collection.

        Parameters
        ----------
        names : str | sequence of str
            The name or sequence of names of the datasets to retrieve.

        Returns
        -------
        datasets : list of Dataset instances
            The list of Dataset instances for the given list of data set names.

        Raises
        ------
        KeyError
            If one of the requested data sets is not present in this data set
            collection.
        """
        if(not issequence(names)):
            names = [names]
        if(not issequenceof(names, str)):
            raise TypeError('The names argument must be an instance of str or '
                'a sequence of str instances!')

        datasets = []
        for name in names:
            datasets.append(self.get_dataset(name))

        return datasets

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

    def set_dataset_prop(self, name, value):
        """Sets the given property to the given name for all data sets of this
        data set collection.

        Parameters
        ----------
        name : str
            The name of the property.
        value : object
            The value to set for the given property.

        Raises
        ------
        KeyError
            If the given property does not exist in the data sets.
        """
        for (dsname, dataset) in self._datasets.items():
            if(not hasattr(dataset, name)):
                raise KeyError('The data set "%s" does not have a property '
                    'named "%s"!'%(dsname, name))
            setattr(dataset, name, value)

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

    def remove_data_preparation(self, key=-1):
        """Removes data preparation function from all the datasets of this
        dataset collection.

        Parameters
        ----------
        index : int, optional
            Index of which data preparation function to remove. Default value
            is the last added function.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.remove_data_preparation(key=key)

    def update_version_qualifiers(self, verqualifiers):
        """Updates the version qualifiers of all datasets of this dataset
        collection.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.update_version_qualifiers(verqualifiers)

    def load_data(self, livetime=None, tl=None, ppbar=None):
        """Loads the data of all data sets of this data set collection.

        Parameters
        ----------
        livetime : float | dict of str => float | None
            If not None, uses this livetime (in days) as livetime for (all) the
            DatasetData instances, otherwise uses the live time from the Dataset
            instance. If a dictionary of data set names and floats is given, it
            defines the livetime for the individual data sets.
        tl : TimeLord instance | None
            The TimeLord instance that should be used to time the data load
            operation.
        ppbar : instance of ProgressBar | None
            The optional parent progress bar.

        Returns
        -------
        data_dict : dictionary str => instance of DatasetData
            The dictionary with the DatasetData instance holding the data of
            an individual data set as value and the data set's name as key.
        """
        if(not isinstance(livetime, dict)):
            livetime_dict = dict()
            for (dsname, dataset) in self._datasets.items():
                livetime_dict[dsname] = livetime
            livetime = livetime_dict

        if(len(livetime) != len(self._datasets)):
            raise ValueError('The livetime argument must be None, a single '
                'float, or a dictionary with %d str:float entries! Currently '
                'the dictionary has %d entries.'%(
                    len(self._datasets), len(livetime)))

        pbar = ProgressBar(len(self._datasets), parent=ppbar).start()
        data_dict = dict()
        for (dsname, dataset) in self._datasets.items():
            data_dict[dsname] = dataset.load_data(
                livetime=livetime[dsname], tl=tl)
            pbar.increment()
        pbar.finish()

        return data_dict


class DatasetData(object):
    """This class provides the container for the actual experimental and
    monto-carlo data. It also holds a reference to the Dataset instance, which
    holds the data's meta information.
    """
    def __init__(self, data_exp, data_mc, livetime):
        """Creates a new DatasetData instance.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray | None
            The instance of DataFieldRecordArray holding the experimental data.
            This can be None for a MC-only study.
        data_mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monto-carlo data.
        livetime : float
            The integrated livetime in days of the data.
        """
        super(DatasetData, self).__init__()

        self.exp = data_exp
        self.mc = data_mc
        self.livetime = livetime

    @property
    def exp(self):
        """The DataFieldRecordArray instance holding the experimental data.
        This is None, if there is no experimental data available.
        """
        return self._exp
    @exp.setter
    def exp(self, data):
        if(data is not None):
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
        This is an empty list if there is no experimental data available.
        """
        if(self._exp is None):
            return []
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

    if(data.exp is not None):
        # Check experimental data keys.
        missing_exp_keys = _get_missing_keys(
            data.exp.field_name_list,
            CFG['dataset']['analysis_required_exp_field_names'])
        if(len(missing_exp_keys) != 0):
            raise KeyError('The following data fields are missing for the '
                'experimental data of dataset "%s": '%(dataset.name)+
                ', '.join(missing_exp_keys))

    # Check monte-carlo data keys.
    missing_mc_keys = _get_missing_keys(
        data.mc.field_name_list,
        CFG['dataset']['analysis_required_exp_field_names'] +
        CFG['dataset']['analysis_required_mc_field_names'])
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

def generate_data_file_root_dir(
    default_base_path, default_sub_path_fmt,
    version, verqualifiers,
    base_path=None, sub_path_fmt=None
):
    """Generates the root directory of the data files based on the given base
    path and sub path format. If base_path is None, default_base_path is used.
    If sub_path_fmt is None, default_sub_path_fmt is used.

    The default_sub_path_fmt and sub_path_fmt arguments can contain the
    following wildcards:
        - '{version:d}'
        - '{<verqualifiers_key>:d}'

    Parameters
    ----------
    default_base_path : str
        The default base path if base_path is None.
    default_sub_path_fmt : str
        The default sub path format if sub_path_fmt is None.
    version : int
        The version of the data sample.
    verqualifiers : dict
        The dictionary holding the version qualifiers of the data sample.
    base_path : str | None
        The user-specified base path.
    sub_path_fmt : str | None
        The user-specified sub path format.

    Returns
    -------
    root_dir : str
        The generated root directory of the data files.
    """
    if(base_path is None):
        if(default_base_path is None):
            raise ValueError('The default_base_path argument must not be None, '
                'when the base_path argument is set to None!')
        base_path = default_base_path

    if(sub_path_fmt is None):
        sub_path_fmt = default_sub_path_fmt

    fmtdict = dict( [('version', version)] + list(verqualifiers.items()) )
    sub_path = sub_path_fmt.format(**fmtdict)

    root_dir = os.path.join(base_path, sub_path)

    return root_dir

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

    dataset_data_subset = DatasetData(data_exp, data_mc, livetime_subset.livetime)

    return (dataset_data_subset, livetime_subset)
