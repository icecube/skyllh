# -*- coding: utf-8 -*-

import os.path

import numpy as np

from skyllh.core import (
    display,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.py import (
    issequenceof,
    module_classname,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
    create_FileLoader,
)
from skyllh.core.timing import (
    TaskTimer,
)


class I3Dataset(
        Dataset,
):
    """The I3Dataset class is an IceCube specific Dataset class that adds
    IceCube specific properties to the Dataset class. These additional
    properties are:

        * good-run-list (GRL)

    """
    @staticmethod
    def get_combined_grl_pathfilenames(datasets):
        """Creates the combined list of grl pathfilenames of all the given
        datasets.

        Parameters
        ----------
        datasets : sequence of I3Dataset
            The sequence of I3Dataset instances.

        Returns
        -------
        grl_pathfilenames : list
            The combined list of grl pathfilenames.
        """
        if not issequenceof(datasets, I3Dataset):
            raise TypeError(
                'The datasets argument must be a sequence of I3Dataset '
                'instances!')

        grl_pathfilenames = []
        for ds in datasets:
            grl_pathfilenames += ds.grl_pathfilename_list

        return grl_pathfilenames

    def __init__(
            self,
            livetime=None,
            grl_pathfilenames=None,
            **kwargs,
    ):
        """Creates a new IceCube specific dataset, that also can hold a list
        of GRL data files.

        Parameters
        ----------
        livetime : float | None
            The live-time of the dataset in days. It can be ``None``, if
            good-run-list data files are provided.
        grl_pathfilenames : str | sequence of str
            The sequence of pathfilenames pointing to the good-run-list (GRL)
            data files.
        """
        super().__init__(
            livetime=livetime,
            **kwargs)

        self._logger = get_logger(module_classname(self))

        self.grl_pathfilename_list = grl_pathfilenames

        self.grl_field_name_renaming_dict = dict()

    @property
    def grl_pathfilename_list(self):
        """The list of file names of the good-run-list (GRL) data files for this
        dataset. If a file name is given with a relative path, it will be
        relative to the ``root_dir`` property of this Dataset instance.
        """
        return self._grl_pathfilename_list

    @grl_pathfilename_list.setter
    def grl_pathfilename_list(self, pathfilenames):
        if pathfilenames is None:
            pathfilenames = []
        if isinstance(pathfilenames, str):
            pathfilenames = [pathfilenames]
        if not issequenceof(pathfilenames, str):
            raise TypeError(
                'The grl_pathfilename_list property must be a sequence of str!')
        self._grl_pathfilename_list = list(pathfilenames)

    @property
    def grl_abs_pathfilename_list(self):
        """(read-only) The list of absolute path file names of the good-run-list
        data files.
        """
        return self.get_abs_pathfilename_list(self._grl_pathfilename_list)

    @property
    def grl_field_name_renaming_dict(self):
        """The dictionary specifying the field names of the good-run-list data
        which need to get renamed just after loading the data. The dictionary
        keys are the old names and their values are the new names.
        """
        return self._grl_field_name_renaming_dict

    @grl_field_name_renaming_dict.setter
    def grl_field_name_renaming_dict(self, d):
        if not isinstance(d, dict):
            raise TypeError(
                'The grl_field_name_renaming_dict property must be an '
                'instance of dict!')
        self._grl_field_name_renaming_dict = d

    @property
    def exists(self):
        """(read-only) Flag if all the data files of this data set exists. It is
        ``True`` if all data files exist and ``False`` otherwise.
        """
        if not super().exists:
            return False

        for pathfilename in self.grl_abs_pathfilename_list:
            if not os.path.exists(pathfilename):
                return False

        return True

    def __str__(self):
        """Implementation of the pretty string representation of the I3Dataset
        object.
        """
        s = super().__str__()
        s += '\n'

        s1 = ''
        s1 += 'GRL data:\n'
        s2 = ''
        if len(self._grl_pathfilename_list) > 0:
            for (idx, pathfilename) in enumerate(self.grl_abs_pathfilename_list):
                if idx > 0:
                    s2 += '\n'
                s2 += self._gen_datafile_pathfilename_entry(pathfilename)
        else:
            s2 += 'None'
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)

        s += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)

        return s

    def create_file_list(
            self,
    ):
        """Creates the list of files of this dataset.
        The file paths are relative to the dataset's root directory.

        Returns
        -------
        file_list : list of str
            The list of files of this dataset.
        """
        file_list = (
            super().create_file_list() +
            self._grl_pathfilename_list
        )

        return file_list

    def load_grl(
            self,
            efficiency_mode=None,
            tl=None,
    ):
        """Loads the good-run-list and returns a DataFieldRecordArray instance
        which should contain the following data fields:

            run : int
                The run number.
            start : float
                The MJD start time of the run.
            stop : float
                The MJD stop time of the run.
            livetime : float
                The livetime in days of the run.
            events : int
                The number of experimental events in the run.

        Parameters
        ----------
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
        grl_data : instance of DataFieldRecordArray
            The DataFieldRecordArray instance holding the good-run-list
            information of the dataset.
        """
        with TaskTimer(tl, 'Loading grl data from disk.'):
            fileloader_grl = create_FileLoader(
                self.grl_abs_pathfilename_list)
            grl_data = fileloader_grl.load_data(
                efficiency_mode=efficiency_mode)
            grl_data.rename_fields(self._grl_field_name_renaming_dict)

        with TaskTimer(tl, 'Sort grl data according to start time'):
            grl_data.sort_by_field(name='start')

        return grl_data

    def load_data(
            self,
            livetime=None,
            keep_fields=None,
            dtc_dict=None,
            dtc_except_fields=None,
            efficiency_mode=None,
            tl=None,
    ):
        """Loads the data, which is described by the dataset. If a good-run-list
        (GRL) is provided for this dataset, only experimental data will be
        selected which matches the GRL.

        Parameters
        ----------
        livetime : instance of Livetime | float | None
            If not None, uses this livetime (if float livetime in days) as
            livetime for the DatasetData instance, otherwise uses the live time
            from the Dataset instance or, if available, the livetime from the
            good-run-list (GRL).
        keep_fields : list of str | None
            The list of user-defined data fields that should get loaded and kept
            in addition to the analysis required data fields.
        dtc_dict : dict | None
            This dictionary defines how data fields of specific data types (key)
            should get converted into other data types (value).
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
            The TimeLord instance that should be used to time the data load
            operation.

        Returns
        -------
        data : instance of DatasetData
            A DatasetData instance holding the experimental and monte-carlo
            data of this data set.
        """
        # Load the dataset files first. This will ensure the dataset is
        # downloaded if necessary.
        data_ = super().load_data(
            livetime=livetime,
            keep_fields=keep_fields,
            dtc_dict=dtc_dict,
            dtc_except_fields=dtc_except_fields,
            efficiency_mode=efficiency_mode,
            tl=tl)

        # Load the good-run-list (GRL) data if it is provided for this dataset,
        # and calculate the livetime based on the GRL.
        data_grl = None
        if len(self._grl_pathfilename_list) > 0:
            data_grl = self.load_grl(
                efficiency_mode=efficiency_mode,
                tl=tl)

        # Load all the defined data.
        data = I3DatasetData(
            data=data_,
            data_grl=data_grl)

        return data

    def prepare_data(  # noqa: C901
            self,
            data,
            tl=None,
    ):
        """Prepares the data for IceCube by pre-calculating the following
        experimental data fields:

        - sin_dec: float
            The sin value of the declination coordinate.

        and monte-carlo data fields:

        - sin_true_dec: float
            The sin value of the true declination coordinate.

        Parameters
        ----------
        data : DatasetData instance
            The DatasetData instance holding the data as numpy record ndarray.
        tl : TimeLord instance | None
            The TimeLord instance that should be used to time the data
            preparation.
        """
        # Set the livetime of the dataset from the GRL data when no livetime
        # was specified previously.
        if data.livetime is None and data.grl is not None:
            if 'start' not in data.grl:
                raise KeyError(
                    f'The GRL data for dataset "{self.name}" has no data '
                    'field named "start"!')
            if 'stop' not in data.grl:
                raise KeyError(
                    f'The GRL data for dataset "{self.name}" has no data '
                    'field named "stop"!')
            data.livetime = np.sum(data.grl['stop'] - data.grl['start'])

        # Execute all the data preparation functions for this dataset.
        super().prepare_data(
            data=data,
            tl=tl)

        if data.exp is not None:
            # Append sin(dec) data field to the experimental data.
            task = 'Appending IceCube-specific data fields to exp data.'
            with TaskTimer(tl, task):
                if 'sin_dec' not in data.exp.field_name_list:
                    data.exp.append_field(
                        'sin_dec', np.sin(data.exp['dec']))

        if data.mc is not None:
            # Append sin(dec) and sin(true_dec) to the MC data.
            task = 'Appending IceCube-specific data fields to MC data.'
            with TaskTimer(tl, task):
                if 'sin_dec' not in data.mc.field_name_list:
                    data.mc.append_field(
                        'sin_dec', np.sin(data.mc['dec']))
                if 'sin_true_dec' not in data.mc.field_name_list:
                    data.mc.append_field(
                        'sin_true_dec', np.sin(data.mc['true_dec']))

        # Select only the experimental data which fits the good-run-list for
        # this dataset.
        if (data.grl is not None) and (data.exp is not None):
            # Select based on run information.
            if ('run' in data.grl) and ('run' in data.exp):
                task = (
                    'Select only the experimental data that matches the run '
                    f'information in the GRL for dataset "{self.name}".')
                with TaskTimer(tl, task):
                    runs = np.unique(data.grl['run'])
                    mask = np.isin(data.exp['run'], runs)

                    if np.any(~mask):
                        n_cut_runs = np.count_nonzero(~mask)
                        self._logger.info(
                            f'Cutting {n_cut_runs} runs from dataset '
                            f'{self.name} due to GRL run information.')
                        data.exp = data.exp[mask]

            # Select based on detector on-time information.
            if ('start' in data.grl) and\
               ('stop' in data.grl) and\
               ('time' in data.exp):
                task = (
                    'Select only the experimental data that matches the '
                    'detector\'s on-time information in the GRL for dataset '
                    f'"{self.name}".')
                with TaskTimer(tl, task):
                    mask = np.zeros((len(data.exp),), dtype=np.bool_)
                    for (start, stop) in zip(data.grl['start'],
                                             data.grl['stop']):
                        mask |= (
                            (data.exp['time'] >= start) &
                            (data.exp['time'] <= stop)
                        )

                    if np.any(~mask):
                        n_cut_evts = np.count_nonzero(~mask)
                        self._logger.info(
                            f'Cutting {n_cut_evts} events from dataset '
                            f'{self.name} due to GRL on-time window '
                            'information.')
                        data.exp = data.exp[mask]


class I3DatasetData(
        DatasetData,
):
    """The class provides the container for the loaded experimental and
    monto-carlo data of a data set. It's the IceCube specific class that also
    holds the good-run-list (GRL) data.
    """
    def __init__(
            self,
            data,
            data_grl,
    ):
        """Constructs a new I3DatasetData instance.

        Parameters
        ----------
        data : instance of DatasetData
            The instance of DatasetData holding the experimental and monte-carlo
            data.
        data_grl : instance of DataFieldRecordArray | None
            The instance of DataFieldRecordArray holding the good-run-list data
            of the dataset. This can be None, if no GRL data is available.
        """
        super().__init__(
            data_exp=data._exp,
            data_mc=data._mc,
            livetime=data._livetime)

        self.grl = data_grl

    @property
    def grl(self):
        """The DataFieldRecordArray instance holding the good-run-list (GRL)
        data of the IceCube data set. It is None, if there is no GRL data
        available for this IceCube data set.
        """
        return self._grl

    @grl.setter
    def grl(self, data):
        if data is not None:
            if not isinstance(data, DataFieldRecordArray):
                raise TypeError(
                    'The grl property must be an instance of '
                    'DataFieldRecordArray!')
        self._grl = data
