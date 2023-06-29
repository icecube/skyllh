# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.storage import (
    create_FileLoader,
)
from skyllh.i3.dataset import (
    I3Dataset,
)


class I3Livetime(
        Livetime):
    """The I3Livetime class provides the functionality to load a Livetime object
    from a good-run-list data file.
    """

    @classmethod
    def from_grl_data(cls, grl_data):
        """Creates an I3LiveTime instance from the given good-run-list (GRL)
        data.

        Parameters
        ----------
        grl_data : instance of numpy structured ndarray.
            The numpy structured ndarray of length N_runs holding the start end
            end times of the good runs. The following fields need to exist:

            start : float
                The MJD of the run start.
            end : float
                The MJD of the run stop.

        Returns
        -------
        livetime : instance of I3Livetime
            The created instance of I3Livetime for the provided GRL data.
        """
        uptime_mjd_intervals_arr = np.hstack((
            grl_data['start'].reshape((len(grl_data), 1)),
            grl_data['stop'].reshape((len(grl_data), 1))
        ))

        livetime = cls(
            uptime_mjd_intervals_arr=uptime_mjd_intervals_arr)

        return livetime

    @staticmethod
    def from_grl_files(
            pathfilenames):
        """Loads an I3Livetime instance from the given good-run-list (GRL) data
        file. The data file needs to contain the following data fields:

            start : float
                The MJD of the run start.
            stop : float
                The MJD of the run stop.

        Parameters
        ----------
        pathfilenames : str | list of str
            The list of fully qualified file names of the GRL data files.

        Returns
        -------
        livetime : instance of I3Livetime
            The created instance of I3Livetime for the provided GRL data.
        """
        grl_data = create_FileLoader(pathfilenames).load_data()

        uptime_mjd_intervals_arr = np.hstack((
            grl_data['start'].reshape((len(grl_data), 1)),
            grl_data['stop'].reshape((len(grl_data), 1))
        ))

        livetime = I3Livetime(
            uptime_mjd_intervals_arr=uptime_mjd_intervals_arr)

        return livetime

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
        livetime : instance of I3Livetime
            The created instance of I3Livetime for the GRL data from the
            provided dataset.
        """
        if not isinstance(ds, I3Dataset):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')
        if len(ds.grl_pathfilename_list) == 0:
            raise ValueError(
                'No GRL files have been defined for the given dataset!')

        livetime = I3Livetime.from_grl_files(
            pathfilenames=ds.grl_pathfilename_list)

        return livetime

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new instance of I3Livetime.
        """
        super().__init__(
            *args,
            **kwargs)
