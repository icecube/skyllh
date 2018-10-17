# -*- coding: utf-8 -*-

import numpy as np

from skylab.core.livetime import Livetime
from skylab.core import storage

class I3Livetime(Livetime):
    """The I3Livetime class provides the functionality to load a Livetime object
    from a good-run-list data file.
    """
    @staticmethod
    def from_GRL_files(pathfilenames):
        """Loads an I3Livetime instance from the given good-run-list (GRL) data
        file. The data file needs to contain the following data fields:

            start : float
                The MJD of the run start.
            stop : float
                The MJD of the run stop.
            good_i3 : bool
                The flag if the run had a good in-ice detector.

        Parameters
        ----------
        pathfilenames : str | list of str
            The list of fully qualified file names of the GRL data files.
        """
        grl_data = storage.create_FileLoader(pathfilenames).load_data()

        uptime_mjd_intervals_arr = np.hstack((
            grl_data['start'].reshape((grl_data.shape[0],1)),
            grl_data['stop'].reshape((grl_data.shape[0],1))
        ))

        # Remove bad runs.
        uptime_mjd_intervals_arr = np.compress(
            grl_data['good_i3'], uptime_mjd_intervals_arr, axis=0)

        return I3Livetime(uptime_mjd_intervals_arr)

    def __init__(self, uptime_mjd_intervals_arr):
        super(I3Livetime, self).__init__(uptime_mjd_intervals_arr)
