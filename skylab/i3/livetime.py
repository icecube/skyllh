import numpy as np

from skylab.core.livetime import Livetime
from skylab.core import storage

class I3Livetime(Livetime):
    """The I3LiveTime class provides the functionality to load a Livetime object
    from a good-run-list data file.
    """
    def __init__(self):
        super(I3Livetime, self).__init__()

    def load_from_GRL_files(self, pathfilenames):
        """Loads the live time from the given good-run-list (GRL) data file.
        The data file needs to contain the following data fields:

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

        data = np.hstack((
            grl_data['start'].reshape((grl_data.shape[0],1)),
            grl_data['stop'].reshape((grl_data.shape[0],1))
        ))

        # Remove bad runs.
        data = np.compress(grl_data['good_i3'], data, axis=0)

        self._uptime_mjd_intervals = data

