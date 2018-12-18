# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib import recfunctions as np_rfn

from skyllh.core.py import issequenceof
from skyllh.core.dataset import Dataset
from skyllh.core.stopwatch import get_stopwatch_lap_taker

class I3Dataset(Dataset):
    """The I3Dataset class is an IceCube specific Dataset class that adds
    IceCube specific properties to the Dataset class. These additional
    properties are:

        * good-run-list (GRL)

    """
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

    def __str__(self):
        """Implementation of the pretty string representation of the I3Dataset
        object.
        """
        pad = ' '*4
        s = super(I3Dataset, self).__str__()
        s += '%s GRL data:\n'%(pad)
        if(len(self.grl_pathfilename_list) > 0):
            for pathfilename in self.grl_pathfilename_list:
                s += '%s%s\n'%(pad*2, pathfilename)
        else:
            s += '%s None'%(pad*2)
        return s

    def prepare_data(self, data, sw=None):
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
        sw : Stopwatch instance | None
            The Stopwatch instance that should be used to time the data
            preparation.
        """
        super(I3Dataset, self).prepare_data(data, sw=sw)

        sw_take_lap = get_stopwatch_lap_taker(sw)

        data.exp = np_rfn.append_fields(data.exp,
            'sin_dec', np.sin(data.exp['dec']), dtypes=np.float, usemask=False)
        sw_take_lap('Appended IceCube-specific data fields to exp data.')

        # Append sin(dec) and sin(true_dec) to the MC data.
        # Note: We do this in two separate calls because it is 5-times faster
        #       than having it done with a single call!
        data.mc = np_rfn.append_fields(data.mc,
            'sin_dec',
            np.sin(data.mc['dec']),
            dtypes=np.float, usemask=False)
        data.mc = np_rfn.append_fields(data.mc,
            'sin_true_dec',
            np.sin(data.mc['true_dec']),
            dtypes=np.float, usemask=False)
        sw_take_lap('Appended IceCube-specific data fields to MC data.')

I3Dataset.add_required_exp_field_names(I3Dataset, ['azi', 'zen', 'sin_dec'])
I3Dataset.add_required_mc_field_names(I3Dataset, ['sin_true_dec'])
