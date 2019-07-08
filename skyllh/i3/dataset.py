# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core import display
from skyllh.core.py import issequenceof
from skyllh.core.dataset import (
    Dataset,
    DatasetData
)
from skyllh.core.storage import (
    DataFieldRecordArray,
    create_FileLoader
)
from skyllh.core.timing import TaskTimer


class I3Dataset(Dataset):
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
        if(not issequenceof(datasets, I3Dataset)):
            raise TypeError('The datasets argument must be a sequence of I3Dataset instances!')

        grl_pathfilenames = []
        for ds in datasets:
            grl_pathfilenames += ds.grl_pathfilename_list

        return grl_pathfilenames

    def __init__(self, grl_pathfilenames=None, *args, **kwargs):
        """Creates a new IceCube specific dataset, that also can hold a list
        of GRL data files.

        Parameters
        ----------
        grl_pathfilenames : str | sequence of str

        """
        super(I3Dataset, self).__init__(*args, **kwargs)

        self.grl_pathfilename_list = grl_pathfilenames

    @property
    def grl_pathfilename_list(self):
        """The list of fully qualified file names of the good-run-list
        data files for this dataset.
        """
        return self._grl_pathfilename_list
    @grl_pathfilename_list.setter
    def grl_pathfilename_list(self, pathfilenames):
        if(pathfilenames is None):
            pathfilenames = []
        if(isinstance(pathfilenames, str)):
            pathfilenames = [pathfilenames]
        if(not issequenceof(pathfilenames, str)):
            raise TypeError('The grl_pathfilename_list property must be a sequence of str!')
        self._grl_pathfilename_list = list(pathfilenames)

    @property
    def grl_data(self):
        """(read-only) The numpy record ndarray holding the good-run-list (GRL)
        data of the data set. It is None, if there is no GRL data available for
        this data set.
        """
        return self._grl_data

    def __str__(self):
        """Implementation of the pretty string representation of the I3Dataset
        object.
        """
        s = super(I3Dataset, self).__str__()
        s += '\n'

        s1 = ''
        s1 += 'GRL data:\n'
        if(len(self.grl_pathfilename_list) > 0):
            s2 = '\n'.join(self.grl_pathfilename_list)
        else:
            s2 += 'None'
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)

        s += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)

        return s

    def load_grl(self, tl=None):
        """Loads the good-run-list and returns a structured numpy ndarray with
        the following data fields:

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
                self._grl_pathfilename_list)
            grl_data = DataFieldRecordArray(fileloader_grl.load_data())

        return grl_data

    def load_data(self, livetime=None, tl=None):
        """Loads the data, which is described by the dataset. If a good-run-list
        (GRL) is provided for this dataset, only experimental data will be
        selected which matches the GRL.

        Parameters
        ----------
        livetime : float | None
            If not None, uses this livetime (in days) as livetime for the
            DatasetData instance, otherwise uses the live time from the Dataset
            instance or, if available, the livetime from the good-run-list
            (GRL).
        tl : TimeLord instance | None
            The TimeLord instance that should be used to time the data load
            operation.

        Returns
        -------
        data : instance of DatasetData
            A DatasetData instance holding the experimental and monte-carlo
            data of this data set.
        """
        # Load the good-run-list (GRL) data if it is provided for this dataset,
        # and calculate the livetime based on the GRL.
        data_grl = None
        lt = self.livetime
        if(len(self._grl_pathfilename_list) > 0):
            data_grl = self.load_grl(tl=tl)
            lt = np.sum(data_grl['livetime'])

        # Override the livetime if there is a user defined livetime.
        if(livetime is not None):
            lt = livetime

        # Load all the defined data.
        data = I3DatasetData(
            super(I3Dataset, self).load_data(livetime=lt, tl=tl),
            data_grl)

        # Select only the experimental data which fits the good-run-list for
        # this dataset.
        if(data_grl is not None):
            task = 'Selected only the experimental data that matches the GRL '\
                'for dataset "%s".'%(self.name)
            with TaskTimer(tl, task):
                runs = np.unique(data_grl['run'])
                mask = np.isin(data.exp['run'], runs)
                data.exp = data.exp[mask]

        return data

    def prepare_data(self, data, tl=None):
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
        # Execute all the data preparation functions for this dataset.
        super(I3Dataset, self).prepare_data(data, tl=tl)

        task = 'Appending IceCube-specific data fields to exp data.'
        with TaskTimer(tl, task):
            data.exp.append_field('sin_dec', np.sin(data.exp['dec']))

        # Append sin(dec) and sin(true_dec) to the MC data.
        task = 'Appending IceCube-specific data fields to MC data.'
        with TaskTimer(tl, task):
            data.mc.append_field('sin_dec', np.sin(data.mc['dec']))
            data.mc.append_field('sin_true_dec', np.sin(data.mc['true_dec']))


I3Dataset.add_required_exp_field_names(I3Dataset, ['azi', 'zen', 'sin_dec'])
I3Dataset.add_required_mc_field_names(I3Dataset, ['sin_true_dec'])


class I3DatasetData(DatasetData):
    """The class provides the container for the loaded experimental and
    monto-carlo data of a data set. It's the IceCube specific class that also
    holds the good-run-list (GRL) data.
    """
    def __init__(self, data, data_grl):
        super(I3DatasetData, self).__init__(
            data._exp, data._mc, data._aux, data._livetime)

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
        if(data is not None):
            if(not isinstance(data, DataFieldRecordArray)):
                raise TypeError('The grl property must be an instance of '
                    'DataFieldRecordArray!')
        self._grl = data
