# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.livetime import Livetime
from skyllh.core import storage
from skyllh.i3.dataset import I3Dataset

class I3Livetime(Livetime):
    """The I3Livetime class provides the functionality to load a Livetime object
    from a good-run-list data file.
    """
    @staticmethod
    def from_GRL_files(pathfilenames):
        """Loads an I3Livetime instance from the given good-run-list (GRL) data
        file. The data file needs to contain the following data fields:

            - start : float
                The MJD of the run start.
            - stop : float
                The MJD of the run stop.

        Parameters
        ----------
        pathfilenames : str | list of str
            The list of fully qualified file names of the GRL data files.

        Returns
        -------
        livetime : I3Livetime instance
            The created I3Livetime instance for the provided GRL data.
        """
        grl_data = storage.create_FileLoader(pathfilenames).load_data()

        uptime_mjd_intervals_arr = np.hstack((
            grl_data['start'].reshape((len(grl_data),1)),
            grl_data['stop'].reshape((len(grl_data),1))
        ))

        return I3Livetime(uptime_mjd_intervals_arr)

    @staticmethod
    def from_I3Dataset(ds):
        """Loads an I3Livetime instance from a given I3Dataset instance, which
        must have a good-run-list (GRL) files defined.

        Parameters
        ----------
        ds : I3Dataset instance
            The instance of I3Dataset which defined the good-run-list (GRL)
            files for the dataset.

        Returns
        -------
        livetime : I3Livetime instance
            The created I3Livetime instance for the GRL data from the provided
            dataset.
        """
        if(not isinstance(ds, I3Dataset)):
            raise TypeError('The ds argument must be an instance of I3Dataset!')
        if(len(ds.grl_pathfilename_list) == 0):
            raise ValueError('No GRL files have been defined for the given dataset!')
        return I3Livetime.from_GRL_files(ds.grl_pathfilename_list)

    def __init__(self, uptime_mjd_intervals_arr):
        super(I3Livetime, self).__init__(uptime_mjd_intervals_arr)
