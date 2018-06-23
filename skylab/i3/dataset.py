# -*- coding: utf-8 -*-

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
