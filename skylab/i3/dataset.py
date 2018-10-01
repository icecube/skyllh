# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib import recfunctions as np_rfn

from skylab.core.dataset import Dataset

class I3Dataset(Dataset):
    """The I3Dataset class is an IceCube specific Dataset class that adds
    IceCube specific properties to the Dataset class. These additional
    properties are:

        * good-run-list (GRL)

    """
    def __init__(self, grl_pathfilenames=None, *args, **kwargs):
        """Creates a new IceCube specific dataset, that also can hold a list
        of GRL data files.
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
        if(not isinstance(pathfilenames, list)):
            raise TypeError('The grl_pathfilename_list property must be of type list!')
        self._grl_pathfilename_list = pathfilenames

    def __str__(self):
        """Implementation of the pretty string representation of the I3Dataset
        object.
        """
        pad = ' '*4
        s = super(I3Dataset, self).__str__()
        if(len(self.grl_pathfilename_list) > 0):
            s += '%s GRL data:\n'%(pad)
            for pathfilename in self.grl_pathfilename_list:
                s += '%s%s\n'%(pad*2, pathfilename)
        return s

    def prepare_data(self, data):
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
        """
        super(I3Dataset, self).prepare_data(data)

        data.exp = np_rfn.append_fields(data.exp,
            'sin_dec', np.sin(data.exp['dec']), dtypes=np.float, usemask=False)
        data.mc = np_rfn.append_fields(data.mc,
            ('sin_dec', 'sin_true_dec'),
            (np.sin(data.mc['dec']), np.sin(data.mc['true_dec'])),
            dtypes=(np.float,np.float), usemask=False)

I3Dataset.add_required_exp_field_names(I3Dataset, ['sin_dec'])
I3Dataset.add_required_mc_field_names(I3Dataset, ['sin_true_dec'])
