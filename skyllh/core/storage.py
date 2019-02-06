# -*- coding: utf-8 -*-

import abc
import pickle
import numpy as np
import os.path

from skyllh.core.py import issequence, issequenceof, range

# Define a file loader registry that holds the FileLoader classes for different
# file formats.
_FILE_LOADER_REG = dict()

def register_FileLoader(formats, fileloader_cls):
    """Registers the given file formats (file extensions) to the given
    FileLoader class.

    Parameters
    ----------
    formats : str | list of str
        The list of file name extensions that should be mapped to the FileLoader
        class.
    fileloader_cls : FileLoader
        The subclass of FileLoader that should be used for the given file
        formats.
    """
    if(isinstance(formats, str)):
        formats = [ formats ]
    if(not issequence(formats)):
        raise TypeError('The "formats" argument must be a sequence!')
    if(not issubclass(fileloader_cls, FileLoader)):
        raise TypeError('The "fileloader_cls" argument must be a subclass of FileLoader!')

    for fmt in formats:
        if(fmt in _FILE_LOADER_REG.keys()):
            raise KeyError('The format "%s" is already registered!'%(fmt))
        _FILE_LOADER_REG[fmt] = fileloader_cls

def create_FileLoader(pathfilenames, **kwargs):
    """Creates the appropriate FileLoader object for the given file names.
    It looks up the FileLoader class from the FileLoader registry for the
    file name extension of the first file name in the given list.

    Parameters
    ----------
    pathfilenames : str | sequence of str
        The sequence of fully qualified file names of the files that should be
        loaded.

    Additional Parameters
    ---------------------
    Additional parameters will be passed to the constructor method of the
    chosen FileLoader class.

    Returns
    -------
    fileloader : FileLoader
        The appropiate FileLoader instance for the given type of data files.
    """
    if(isinstance(pathfilenames, str)):
        pathfilenames = [pathfilenames]
    if(not issequenceof(pathfilenames, str)):
        raise TypeError('The pathfilenames argument must be a sequence of str!')

    # Sort the file names extensions with shorter extensions before longer ones
    # to support a format that is sub-string of another format.
    formats = sorted(_FILE_LOADER_REG.keys())
    for fmt in formats:
        l = len(fmt)
        if(pathfilenames[0][-l:].lower() == fmt.lower()):
            cls = _FILE_LOADER_REG[fmt]
            return cls(pathfilenames, **kwargs)

    raise RuntimeError('No FileLoader class is suitable to load the data file "%s"!'%(pathfilenames[0]))

def assert_file_exists(pathfilename):
    """Checks if the given file exists and raises a RuntimeError if it does
    not exist.
    """
    if(not os.path.isfile(pathfilename)):
        raise RuntimeError('The data file "%s" does not exist!'%(pathfilename))


class FileLoader:
    __metaclass__ = abc.ABCMeta

    def __init__(self, pathfilenames):
        """Initializes a new FileLoader instance.

        Parameters
        ----------
        pathfilenames : sequence
            The sequence of fully qualified file names of the data files that
            need to be loaded. The data arrays of several files will be
            concatenated to a single data array uppon loading the data.
        """
        self.pathfilename_list = pathfilenames

    @property
    def pathfilename_list(self):
        """The list of fully qualified file names of the data files.
        """
        return self._pathfilename_list
    @pathfilename_list.setter
    def pathfilename_list(self, pathfilenames):
        if(isinstance(pathfilenames, str)):
            pathfilenames = [ pathfilenames ]
        if(not issequence(pathfilenames)):
            raise TypeError('The pathfilename_list property must be a sequence type!')
        self._pathfilename_list = list(pathfilenames)

    @abc.abstractmethod
    def load_data(self):
        pass


class NPYFileLoader(FileLoader):
    """The NPYFileLoader class provides the data loading functionality for
    numpy data files containing numpy arrays. It uses the ``numpy.load``
    function for loading the data and the numpy.append function to concatenate
    several data files.
    """
    def __init__(self, pathfilenames):
        super(NPYFileLoader, self).__init__(pathfilenames)

    def load_data(self):
        """Loads the data from the files specified through their fully qualified
        file names.

        Returns
        -------
        data : numpy.recarray
            The numpy record array holding the loaded data.

        Raises
        ------
        RuntimeError if a file does not exist.
        """
        pathfilename = self.pathfilename_list[0]
        assert_file_exists(pathfilename)
        data = np.load(pathfilename)
        for i in range(1, len(self.pathfilename_list)):
            pathfilename = self.pathfilename_list[i]
            assert_file_exists(pathfilename)
            data = np.append(data, np.load(pathfilename))

        return data


class PKLFileLoader(FileLoader):
    """The PKLFileLoader class provides the data loading functionality for
    pickled Python data files containing Python data structures. It uses the
    `pickle.load` function for loading the data from the file.
    """
    def __init__(self, pathfilenames):
        super(PKLFileLoader, self).__init__(pathfilenames)

    def load_data(self):
        """Loads the data from the files specified through their fully qualified
        file names.

        Returns
        -------
        data : Python object | list of Python objects
            The de-pickled Python object. If more than one file was specified,
            this is a list of Python objects, i.e. one object for each file.
            The file <-> object mapping order is preserved.

        Raises
        ------
        RuntimeError if a file does not exist.
        """
        data = []
        for pathfilename in self.pathfilename_list:
            assert_file_exists(pathfilename)
            with open(pathfilename, 'rb') as ifile:
                data.append(pickle.load(ifile))

        if(len(data) == 1):
            data = data[0]

        return data


register_FileLoader(['.npy'], NPYFileLoader)
register_FileLoader(['.pkl'], PKLFileLoader)
