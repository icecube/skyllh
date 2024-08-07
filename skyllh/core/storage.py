# -*- coding: utf-8 -*-

import abc
import copy
import pickle
import os.path
import sys

import numpy as np

from skyllh.core import (
    display as dsp,
    tool,
)
from skyllh.core.py import (
    classname,
    get_byte_size_prefix,
    getsizeof,
    issequence,
    issequenceof,
)


# Define a file loader registry that holds the FileLoader classes for different
# file formats.
_FILE_LOADER_REG = dict()


def register_FileLoader(
        formats,
        fileloader_cls):
    """Registers the given file formats (file extensions) to the given
    FileLoader class.

    Parameters
    ----------
    formats : str | list of str
        The list of file name extensions that should be mapped to the FileLoader
        class.
    fileloader_cls : instance of FileLoader
        The subclass of FileLoader that should be used for the given file
        formats.
    """
    if isinstance(formats, str):
        formats = [formats]
    if not issequence(formats):
        raise TypeError(
            'The "formats" argument must be a sequence!')
    if not issubclass(fileloader_cls, FileLoader):
        raise TypeError(
            'The "fileloader_cls" argument must be a subclass of FileLoader!')

    for fmt in formats:
        if fmt in _FILE_LOADER_REG.keys():
            raise KeyError(
                f'The format "{fmt}" is already registered!')
        _FILE_LOADER_REG[fmt] = fileloader_cls


def create_FileLoader(
        pathfilenames,
        **kwargs):
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
        The appropriate FileLoader instance for the given type of data files.
    """
    if isinstance(pathfilenames, str):
        pathfilenames = [pathfilenames]
    if not issequenceof(pathfilenames, str):
        raise TypeError(
            'The pathfilenames argument must be a sequence of str!')

    # Sort the file names extensions with shorter extensions before longer ones
    # to support a format that is sub-string of another format.
    formats = sorted(_FILE_LOADER_REG.keys())
    for fmt in formats:
        fmt_len = len(fmt)
        if pathfilenames[0][-fmt_len:].lower() == fmt.lower():
            cls = _FILE_LOADER_REG[fmt]
            return cls(pathfilenames, **kwargs)

    raise RuntimeError(
        'No FileLoader class is suitable to load the data file '
        f'"{pathfilenames[0]}"!')


def assert_file_exists(
        pathfilename):
    """Checks if the given file exists and raises a RuntimeError if it does
    not exist.
    """
    if not os.path.isfile(pathfilename):
        raise RuntimeError(
            f'The data file "{pathfilename}" does not exist!')


class FileLoader(
        object,
        metaclass=abc.ABCMeta):
    """Abstract base class for a FileLoader class.
    """
    def __init__(
            self,
            pathfilenames,
            **kwargs):
        """Creates a new FileLoader instance.

        Parameters
        ----------
        pathfilenames : str | sequence of str
            The sequence of fully qualified file names of the data files that
            need to be loaded.
        """
        super().__init__(
            **kwargs)

        self.pathfilename_list = pathfilenames

    @property
    def pathfilename_list(self):
        """The list of fully qualified file names of the data files.
        """
        return self._pathfilename_list

    @pathfilename_list.setter
    def pathfilename_list(self, pathfilenames):
        if isinstance(pathfilenames, str):
            pathfilenames = [pathfilenames]
        if not issequence(pathfilenames):
            raise TypeError(
                'The pathfilename_list property must be of type str or a '
                'sequence of type str!')
        self._pathfilename_list = list(pathfilenames)

    @abc.abstractmethod
    def load_data(self, **kwargs):
        """This method is supposed to load the data from the file.
        """
        pass


class NPYFileLoader(
        FileLoader):
    """The NPYFileLoader class provides the data loading functionality for
    numpy data files containing numpy arrays. It uses the ``numpy.load``
    function for loading the data and the numpy.append function to concatenate
    several data files.
    """
    def __init__(
            self,
            pathfilenames,
            **kwargs):
        """Creates a new NPYFileLoader instance.

        Parameters
        ----------
        pathfilenames : str | sequence of str
            The sequence of fully qualified file names of the data files that
            need to be loaded.
        """
        super().__init__(
            pathfilenames=pathfilenames,
            **kwargs)

    def _load_file_memory_efficiently(
            self,
            pathfilename,
            keep_fields,
            dtype_conversions,
            dtype_conversion_except_fields):
        """Loads a single file in a memory efficient way.

        Parameters
        ----------
        pathfilename : str
            The fully qualified file name of the to-be-loaded file.
        keep_fields : list of str | None
            The list of field names which should be kept.

        Returns
        -------
        data : DataFieldRecordArray instance
            An instance of DataFieldRecordArray holding the data.
        """
        assert_file_exists(pathfilename)

        # Create a memory map into the data file. This loads the data only when
        # accessing the data.
        mmap_ndarray = np.load(pathfilename, mmap_mode='r')
        field_names = mmap_ndarray.dtype.names
        fname_to_fidx = dict([
            (fname, idx)
            for (idx, fname) in enumerate(field_names)
        ])
        dt_fields = mmap_ndarray.dtype.fields
        n_rows = mmap_ndarray.shape[0]

        data = dict()

        # Create empty arrays for each column of length n_rows.
        for fname in field_names:
            # Ignore fields that should not get kept.
            if (keep_fields is not None) and (fname not in keep_fields):
                continue

            # Get the original data type of the field.
            dt = dt_fields[fname][0]
            # Convert the data type if requested.
            if (fname not in dtype_conversion_except_fields) and\
               (dt in dtype_conversions):
                dt = dtype_conversions[dt]

            data[fname] = np.empty((n_rows,), dtype=dt)

        # Loop through the rows of the recarray.
        bs = 4096
        for ridx in range(0, n_rows):
            row = mmap_ndarray[ridx]
            for fname in data.keys():
                fidx = fname_to_fidx[fname]
                data[fname][ridx] = row[fidx]

            # Reopen the data file after each given blocksize.
            if ridx % bs == 0:
                del mmap_ndarray
                mmap_ndarray = np.load(pathfilename, mmap_mode='r')

        # Close the memory map file.
        del mmap_ndarray

        # Create a DataFieldRecordArray out of the dictionary.
        data = DataFieldRecordArray(data, copy=False)

        return data

    def _load_file_time_efficiently(
            self,
            pathfilename,
            keep_fields,
            dtype_conversions,
            dtype_conversion_except_fields):
        """Loads a single file in a time efficient way. This will load the data
        column-wise.
        """
        assert_file_exists(pathfilename)

        # Create a memory map into the data file. This loads the data only when
        # accessing the data.
        mmap_ndarray = np.load(pathfilename, mmap_mode='r')

        # Create a DataFieldRecordArray out of the memory mapped file. We need
        # to copy the data, otherwise we get read-only numpy arrays.
        data = DataFieldRecordArray(
            mmap_ndarray,
            keep_fields=keep_fields,
            dtype_conversions=dtype_conversions,
            dtype_conversion_except_fields=dtype_conversion_except_fields,
            copy=True)

        # Close the memory map file.
        del mmap_ndarray

        return data

    def load_data(  # noqa: C901
            self,
            keep_fields=None,
            dtype_conversions=None,
            dtype_conversion_except_fields=None,
            efficiency_mode=None):
        """Loads the data from the files specified through their fully qualified
        file names.

        Parameters
        ----------
        keep_fields : str | sequence of str | None
            Load the data into memory only for these data fields. If set to
            ``None``, all in-file-present data fields are loaded into memory.
        dtype_conversions : dict | None
            If not None, this dictionary defines how data fields of specific
            data types get converted into the specified data types.
            This can be used to use less memory.
        dtype_conversion_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        efficiency_mode : str | None
            The efficiency mode the data should get loaded with. Possible values
            are:

                ``'memory'``:
                    The data will be load in a memory efficient way. This will
                    require more time, because all data records of a file will
                    be loaded sequentially.
                ``'time'``
                    The data will be loaded in a time efficient way. This will
                    require more memory, because each data file gets loaded in
                    memory at once.

            The default value is ``'time'``. If set to ``None``, the default
            value will be used.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray holding the loaded data.

        Raises
        ------
        RuntimeError if a file does not exist.
        """
        if keep_fields is not None:
            if isinstance(keep_fields, str):
                keep_fields = [keep_fields]
            elif not issequenceof(keep_fields, str):
                raise TypeError(
                    'The keep_fields argument must be None, an instance of '
                    'type str, or a sequence of instances of type str!')

        if dtype_conversions is None:
            dtype_conversions = dict()
        elif not isinstance(dtype_conversions, dict):
            raise TypeError(
                'The dtype_conversions argument must be None, or an instance '
                'of dict!')

        if dtype_conversion_except_fields is None:
            dtype_conversion_except_fields = []
        elif isinstance(dtype_conversion_except_fields, str):
            dtype_conversion_except_fields = [dtype_conversion_except_fields]
        elif not issequenceof(dtype_conversion_except_fields, str):
            raise TypeError(
                'The dtype_conversion_except_fields argument must be a '
                'sequence of str instances.')

        efficiency_mode2func = {
            'memory': self._load_file_memory_efficiently,
            'time': self._load_file_time_efficiently
        }
        if efficiency_mode is None:
            efficiency_mode = 'time'
        if not isinstance(efficiency_mode, str):
            raise TypeError(
                'The efficiency_mode argument must be an instance of type str!')
        if efficiency_mode not in efficiency_mode2func:
            raise ValueError(
                'The efficiency_mode argument value must be one of '
                f'{", ".join(efficiency_mode2func.keys())}!')
        load_file_func = efficiency_mode2func[efficiency_mode]

        # Load the first data file.
        data = load_file_func(
            self._pathfilename_list[0],
            keep_fields=keep_fields,
            dtype_conversions=dtype_conversions,
            dtype_conversion_except_fields=dtype_conversion_except_fields
        )

        # Load possible subsequent data files by appending to the first data.
        for i in range(1, len(self._pathfilename_list)):
            data.append(load_file_func(
                self._pathfilename_list[i],
                keep_fields=keep_fields,
                dtype_conversions=dtype_conversions,
                dtype_conversion_except_fields=dtype_conversion_except_fields
            ))

        return data


class ParquetFileLoader(
        FileLoader
):
    """The ParquetFileLoader class provides the data loading functionality for
    parquet files. It uses the ``pyarrow`` package.
    """
    @tool.requires('pyarrow', 'pyarrow.parquet')
    def __init__(
            self,
            pathfilenames,
            **kwargs
    ):
        """Creates a new file loader instance for parquet data files.

        Parameters
        ----------
        pathfilenames : str | sequence of str
            The sequence of fully qualified file names of the data files that
            need to be loaded.
        """
        super().__init__(
            pathfilenames=pathfilenames,
            **kwargs)

        self.pa = tool.get('pyarrow')
        self.pq = tool.get('pyarrow.parquet')

    def load_data(
            self,
            keep_fields=None,
            dtype_conversions=None,
            dtype_conversion_except_fields=None,
            copy=False,
            **kwargs,
    ):
        """Loads the data from the files specified through their fully qualified
        file names.

        Parameters
        ----------
        keep_fields : str | sequence of str | None
            Load the data into memory only for these data fields. If set to
            ``None``, all in-file-present data fields are loaded into memory.
        dtype_conversions : dict | None
            If not ``None``, this dictionary defines how data fields of specific
            data types get converted into the specified data types.
            This can be used to use less memory.
        dtype_conversion_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        copy : bool
            If set to ``True``, the column data from the pyarrow.Table instance
            will be copied into the DataFieldRecordArray. This should not be
            necessary.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray holding the loaded data.
        """
        assert_file_exists(self.pathfilename_list[0])
        table = self.pq.read_table(self.pathfilename_list[0], columns=keep_fields)
        for pathfilename in self.pathfilename_list[1:]:
            assert_file_exists(pathfilename)
            next_table = self.pq.read_table(pathfilename, columns=keep_fields)
            table = self.pa.concat_tables([table, next_table])

        data = DataFieldRecordArray(
            data=table,
            data_table_accessor=ParquetDataTableAccessor(),
            keep_fields=keep_fields,
            dtype_conversions=dtype_conversions,
            dtype_conversion_except_fields=dtype_conversion_except_fields,
            copy=copy)

        return data


class PKLFileLoader(
        FileLoader):
    """The PKLFileLoader class provides the data loading functionality for
    pickled Python data files containing Python data structures. It uses the
    `pickle.load` function for loading the data from the file.
    """
    def __init__(
            self,
            pathfilenames,
            pkl_encoding=None,
            **kwargs):
        """Creates a new file loader instance for a pickled data file.

        Parameters
        ----------
        pathfilenames : str | sequence of str
            The sequence of fully qualified file names of the data files that
            need to be loaded.
        pkl_encoding : str | None
            The encoding of the pickled data files. If None, the default
            encodings 'ASCII' and 'latin1' will be tried to load the data.
        """
        super().__init__(
            pathfilenames=pathfilenames,
            **kwargs)

        self.pkl_encoding = pkl_encoding

    @property
    def pkl_encoding(self):
        """The encoding of the pickled data files. Can be None.
        If None, the default encodings 'ASCII' and 'latin1' will be tried to
        load the data.
        """
        return self._pkl_encoding

    @pkl_encoding.setter
    def pkl_encoding(self, encoding):
        if encoding is not None:
            if not isinstance(encoding, str):
                raise TypeError(
                    'The pkl_encoding property must be None or of type str!')
        self._pkl_encoding = encoding

    def load_data(
            self,
            **kwargs):
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
        # Define the possible encodings of the pickled files.
        encodings = ['ASCII', 'latin1']
        if self._pkl_encoding is not None:
            encodings = [self._pkl_encoding] + encodings

        data = []
        for pathfilename in self.pathfilename_list:
            assert_file_exists(pathfilename)
            with open(pathfilename, 'rb') as ifile:
                enc_idx = 0
                load_ok = False
                obj = None
                while (not load_ok) and (enc_idx < len(encodings)):
                    try:
                        encoding = encodings[enc_idx]
                        obj = pickle.load(ifile, encoding=encoding)
                    except UnicodeDecodeError:
                        enc_idx += 1
                        # Move the file pointer back to the beginning of the
                        # file.
                        ifile.seek(0)
                    else:
                        load_ok = True
                if obj is None:
                    raise RuntimeError(
                        f'The file "{pathfilename}" could not get unpickled! '
                        'No correct encoding available!')
                data.append(obj)

        if len(data) == 1:
            data = data[0]

        return data


class TextFileLoader(
        FileLoader):
    """The TextFileLoader class provides the data loading functionality for
    data text files where values are stored in a comma, or whitespace, separated
    format. It uses the numpy.loadtxt function to load the data. It reads the
    first line of the text file for a table header.
    """
    def __init__(
            self,
            pathfilenames,
            header_comment='#',
            header_separator=None,
            **kwargs):
        """Creates a new file loader instance for a text data file.

        Parameters
        ----------
        pathfilenames : str | sequence of str
            The sequence of fully qualified file names of the data files that
            need to be loaded.
        header_comment : str
            The character that defines a comment line in the text file.
        header_separator : str | None
            The separator of the header field names. If None, it assumes
            whitespaces.
        """
        super().__init__(
            pathfilenames=pathfilenames,
            **kwargs)

        self.header_comment = header_comment
        self.header_separator = header_separator

    @property
    def header_comment(self):
        """The character that defines a comment line in the text file.
        """
        return self._header_comment

    @header_comment.setter
    def header_comment(self, s):
        if not isinstance(s, str):
            raise TypeError(
                'The header_comment property must be of type str!')
        self._header_comment = s

    @property
    def header_separator(self):
        """The separator of the header field names. If None, it assumes
        whitespaces.
        """
        return self._header_separator

    @header_separator.setter
    def header_separator(self, s):
        if s is not None:
            if not isinstance(s, str):
                raise TypeError(
                    'The header_separator property must be None or of type '
                    'str!')
        self._header_separator = s

    def _extract_column_names(self, line):
        """Tries to extract the column names of the data table based on the
        given line.

        Parameters
        ----------
        line : str
            The text line containing the column names.

        Returns
        -------
        names : list of str | None
            The column names.
            It returns None, if the column names cannot be extracted.
        """
        # Remove possible new-line character and leading white-spaces.
        line = line.strip()
        # Check if the line is a comment line.
        if line[0:len(self._header_comment)] != self._header_comment:
            return None
        # Remove the leading comment character(s).
        line = line.strip(self._header_comment)
        # Remove possible leading whitespace characters.
        line = line.strip()
        # Split the line into the column names.
        names = line.split(self._header_separator)
        # Remove possible whitespaces of column names.
        names = [n.strip() for n in names]

        if len(names) == 0:
            return None

        return names

    def _load_file(
            self,
            pathfilename,
            keep_fields,
            dtype_conversions,
            dtype_conversion_except_fields):
        """Loads the given file.

        Parameters
        ----------
        pathfilename : str
            The fully qualified file name of the data file that
            need to be loaded.
        keep_fields : str | sequence of str | None
            Load the data into memory only for these data fields. If set to
            ``None``, all in-file-present data fields are loaded into memory.
        dtype_conversions : dict | None
            If not None, this dictionary defines how data fields of specific
            data types get converted into the specified data types.
            This can be used to use less memory.
        dtype_conversion_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.

        Returns
        -------
        data : DataFieldRecordArray instance
            The DataFieldRecordArray instance holding the loaded data.
        """
        assert_file_exists(pathfilename)

        with open(pathfilename, 'r') as ifile:
            line = ifile.readline()
            column_names = self._extract_column_names(line)
            if column_names is None:
                raise ValueError(
                    f'The data text file "{pathfilename}" does not contain a '
                    'readable table header as first line!')
            usecols = None
            dtype = [(n, np.float64) for n in column_names]
            if keep_fields is not None:
                # Select only the given columns.
                usecols = []
                dtype = []
                for (idx, name) in enumerate(column_names):
                    if name in keep_fields:
                        usecols.append(idx)
                        dtype.append((name, np.float64))
                usecols = tuple(usecols)
            if len(dtype) == 0:
                raise ValueError(
                    'No data columns were selected to be loaded!')

            data_ndarray = np.loadtxt(
                ifile,
                dtype=dtype,
                comments=self._header_comment,
                usecols=usecols)

        data = DataFieldRecordArray(
            data_ndarray,
            keep_fields=keep_fields,
            dtype_conversions=dtype_conversions,
            dtype_conversion_except_fields=dtype_conversion_except_fields,
            copy=False)

        return data

    def load_data(
            self,
            keep_fields=None,
            dtype_conversions=None,
            dtype_conversion_except_fields=None,
            **kwargs):
        """Loads the data from the data files specified through their fully
        qualified file names.

        Parameters
        ----------
        keep_fields : str | sequence of str | None
            Load the data into memory only for these data fields. If set to
            ``None``, all in-file-present data fields are loaded into memory.
        dtype_conversions : dict | None
            If not None, this dictionary defines how data fields of specific
            data types get converted into the specified data types.
            This can be used to use less memory.
        dtype_conversion_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray holding the loaded data.

        Raises
        ------
        RuntimeError
            If a file does not exist.
        ValueError
            If the table header cannot be read.
        """
        if keep_fields is not None:
            if isinstance(keep_fields, str):
                keep_fields = [keep_fields]
            elif not issequenceof(keep_fields, str):
                raise TypeError(
                    'The keep_fields argument must be None, an instance of '
                    'type str, or a sequence of instances of type str!')

        if dtype_conversions is None:
            dtype_conversions = dict()
        elif not isinstance(dtype_conversions, dict):
            raise TypeError(
                'The dtype_conversions argument must be None, or an instance '
                'of dict!')

        if dtype_conversion_except_fields is None:
            dtype_conversion_except_fields = []
        elif isinstance(dtype_conversion_except_fields, str):
            dtype_conversion_except_fields = [dtype_conversion_except_fields]
        elif not issequenceof(dtype_conversion_except_fields, str):
            raise TypeError(
                'The dtype_conversion_except_fields argument must be a '
                'sequence of str instances.')

        # Load the first data file.
        data = self._load_file(
            self._pathfilename_list[0],
            keep_fields=keep_fields,
            dtype_conversions=dtype_conversions,
            dtype_conversion_except_fields=dtype_conversion_except_fields
        )

        # Load possible subsequent data files by appending to the first data.
        for i in range(1, len(self._pathfilename_list)):
            data.append(self._load_file(
                self._pathfilename_list[i],
                keep_fields=keep_fields,
                dtype_conversions=dtype_conversions,
                dtype_conversion_except_fields=dtype_conversion_except_fields
            ))

        return data


class DataTableAccessor(
        object,
        metaclass=abc.ABCMeta,
):
    """This class provides an interface wrapper to access the data table of a
    particular format in a unified way.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def get_column(self, data, name):
        """This method is supposed to return a numpy.ndarray holding the data of
        the column with name ``name``.

        Parameters
        ----------
        data : any
            The data table.
        name : str
            The name of the column.

        Returns
        -------
        arr : instance of numpy.ndarray
            The column data as numpy ndarray.
        """
        pass

    @abc.abstractmethod
    def get_field_names(self, data):
        """This method is supposed to return a list of field names.
        """
        pass

    @abc.abstractmethod
    def get_field_name_to_dtype_dict(self, data):
        """This method is supposed to return a dictionary with field name and
        numpy dtype instance for each field.
        """
        pass

    @abc.abstractmethod
    def get_length(self, data):
        """This method is supposed to return the length of the data table.
        """
        pass


class NDArrayDataTableAccessor(
        DataTableAccessor,
):
    """This class provides an interface wrapper to access the data table stored
    as a structured numpy ndarray.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column(self, data, name):
        """Gets the column data from the structured ndarray.

        Parameters
        ----------
        data : instance of numpy.ndarray
            The structured numpy ndarray holding the table data.
        name : str
            The name of the column.
        """
        return data[name]

    def get_field_names(self, data):
        return data.dtype.names

    def get_field_name_to_dtype_dict(self, data):
        """Returns the dictionary with field name and numpy dtype instance for
        each field.
        """
        fname_to_dtype_dict = dict([
            (k, v[0]) for (k, v) in data.dtype.fields.items()
        ])
        return fname_to_dtype_dict

    def get_length(self, data):
        """Returns the length of the data table.
        """
        length = data.shape[0]
        return length


class DictDataTableAccessor(
        DataTableAccessor,
):
    """This class provides an interface wrapper to access the data table stored
    as a Python dictionary.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column(self, data, name):
        """Gets the column data from the dictionary.

        Parameters
        ----------
        data : dict
            The dictionary holding the table data.
        name : str
            The name of the column.
        """
        return data[name]

    def get_field_names(self, data):
        return list(data.keys())

    def get_field_name_to_dtype_dict(self, data):
        """Returns the dictionary with field name and numpy dtype instance for
        each field.
        """
        fname_to_dtype_dict = dict([
            (fname, data[fname].dtype) for fname in data.keys()
        ])
        return fname_to_dtype_dict

    def get_length(self, data):
        """Returns the length of the data table.
        """
        length = 0
        if len(data) > 0:
            length = data[next(iter(data))].shape[0]
        return length


class ParquetDataTableAccessor(
        DataTableAccessor,
):
    """This class provides an interface wrapper to access the data table stored
    as a Parquet table.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column(self, data, name):
        """Gets the column data from the Parquet table.

        Parameters
        ----------
        data : instance of pyarrow.Table
            The instance of pyarrow.Table holding the table data.
        name : str
            The name of the column.
        """
        return data[name].to_numpy()

    def get_field_names(self, data):
        return data.column_names

    def get_field_name_to_dtype_dict(self, data):
        """Returns the dictionary with field name and numpy dtype instance for
        each field.
        """
        fname_to_dtype_dict = dict([
            (fname, data.field(fname).type.to_pandas_dtype())
            for fname in data.column_names
        ])
        return fname_to_dtype_dict

    def get_length(self, data):
        """Returns the length of the data table.
        """
        return len(data)


class DataFieldRecordArrayDataTableAccessor(
        DataTableAccessor,
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column(self, data, name):
        """Gets the column data from the Parquet table.

        Parameters
        ----------
        data : instance of pyarrow.Table
            The instance of pyarrow.Table holding the table data.
        name : str
            The name of the column.
        """
        return data[name]

    def get_field_names(self, data):
        return data.field_name_list

    def get_field_name_to_dtype_dict(self, data):
        """Returns the dictionary with field name and numpy dtype instance for
        each field.
        """
        fname_to_dtype_dict = dict([
            (fname, data[fname].dtype)
            for fname in data.field_name_list
        ])
        return fname_to_dtype_dict

    def get_length(self, data):
        """Returns the length of the data table.
        """
        return len(data)


class DataFieldRecordArray(
        object):
    """The DataFieldRecordArray class provides a data container similar to a numpy
    record ndarray. But the data fields are stored as individual numpy ndarray
    objects. Hence, access of single data fields is much faster compared to
    access on the record ndarray.
    """
    def __init__(  # noqa: C901
            self,
            data,
            data_table_accessor=None,
            keep_fields=None,
            dtype_conversions=None,
            dtype_conversion_except_fields=None,
            copy=True,
    ):
        """Creates a DataFieldRecordArray from the given data.

        Parameters
        ----------
        data : any | None
            The tabulated data in any format. The only requirement is that
            there is a DataTableAccessor instance available for the given data
            format. Supported data types are:

                numpy.ndarray
                    A structured numpy ndarray.
                dict
                    A Python dictionary with field names as keys and
                    one-dimensional numpy.ndarrays as values.
                pyarrow.Table
                    An instance of pyarrow.Table.
                DataFieldRecordArray
                    An instance of DataFieldRecordArray.

            If set to `None`, the DataFieldRecordArray instance is initialized
            with no data and the length of the array is set to 0.
        data_table_accessor : instance of DataTableAccessor | None
            The instance of DataTableAccessor which provides column access to
            ``data``. If set to ``None``, an appropriate ``DataTableAccessor``
            instance will be selected based on the type of ``data``.
        keep_fields : str | sequence of str | None
            If not None (default), this specifies the data fields that should
            get kept from the given data. Otherwise all data fields get kept.
        dtype_conversions : dict | None
            If not None, this dictionary defines how data fields of specific
            data types get converted into the specified data types.
            This can be used to use less memory.
        dtype_conversion_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        copy : bool
            Flag if the input data should get copied. Default is True. If a
            DataFieldRecordArray instance is provided, this option is set to
            ``True`` automatically.
        """
        self._data_fields = dict()
        self._len = None

        if data is None:
            data = dict()

        if keep_fields is not None:
            if isinstance(keep_fields, str):
                keep_fields = [keep_fields]
            elif not issequenceof(keep_fields, str):
                raise TypeError(
                    'The keep_fields argument must be None, an instance of '
                    'type str, or a sequence of instances of type str!')

        if dtype_conversions is None:
            dtype_conversions = dict()
        elif not isinstance(dtype_conversions, dict):
            raise TypeError(
                'The dtype_conversions argument must be None, or an instance '
                'of dict!')

        if dtype_conversion_except_fields is None:
            dtype_conversion_except_fields = []
        elif isinstance(dtype_conversion_except_fields, str):
            dtype_conversion_except_fields = [dtype_conversion_except_fields]
        elif not issequenceof(dtype_conversion_except_fields, str):
            raise TypeError(
                'The dtype_conversion_except_fields argument must be a '
                'sequence of str instances.')

        # Select an appropriate data table accessor for the type of data.
        if data_table_accessor is None:
            if isinstance(data, np.ndarray):
                data_table_accessor = NDArrayDataTableAccessor()
            elif isinstance(data, dict):
                data_table_accessor = DictDataTableAccessor()
            elif (tool.is_available('pyarrow') and
                  isinstance(data, tool.get('pyarrow').Table)):
                data_table_accessor = ParquetDataTableAccessor()
            elif isinstance(data, DataFieldRecordArray):
                data_table_accessor = DataFieldRecordArrayDataTableAccessor()
            else:
                raise TypeError(
                    'No TableDataAccessor instance has been specified for the '
                    f'data of type {type(data)}!')

        field_names = data_table_accessor.get_field_names(data)
        fname2dtype = data_table_accessor.get_field_name_to_dtype_dict(data)
        length = data_table_accessor.get_length(data)

        for fname in field_names:
            # Ignore fields that should not get kept.
            if (keep_fields is not None) and (fname not in keep_fields):
                continue

            copy_field = copy
            dt = fname2dtype[fname]
            if (fname not in dtype_conversion_except_fields) and\
               (dt in dtype_conversions):
                dt = dtype_conversions[dt]
                # If a data type conversion is needed, the data of the field
                # needs to get copied.
                copy_field = True

            if copy_field is True:
                # Create a ndarray with the final data type and then assign the
                # values from the data, which technically is a copy.
                field_arr = np.empty((length,), dtype=dt)
                np.copyto(field_arr, data_table_accessor.get_column(data, fname))
            else:
                field_arr = data_table_accessor.get_column(data, fname)

            if self._len is None:
                self._len = len(field_arr)
            elif len(field_arr) != self._len:
                raise ValueError(
                    'All field arrays must have the same length. '
                    f'Field "{fname}" has length {len(field_arr)}, but must be '
                    f'{self._len}!')

            self._data_fields[fname] = field_arr

        if self._len is None:
            # The DataFieldRecordArray is initialized with no fields, i.e. also
            # also no data.
            self._len = 0

        self._field_name_list = list(self._data_fields.keys())
        self._indices = None

    def __contains__(
            self,
            name):
        """Checks if the given field exists in this DataFieldRecordArray
        instance.

        Parameters
        ----------
        name : str
            The name of the field.

        Returns
        -------
        check : bool
            True, if the given field exists in this DataFieldRecordArray
            instance, False otherwise.
        """
        return (name in self._data_fields)

    def __getitem__(
            self,
            name):
        """Implements data field value access.

        Parameters
        ----------
        name : str | numpy ndarray of int or bool
            The name of the data field. If a numpy ndarray is given, it must
            contain the indices for which to retrieve a data selection of the
            entire DataFieldRecordArray. A numpy ndarray of bools can be given
            as well to define a mask.

        Raises
        ------
        KeyError
            If the given data field does not exist.

        Returns
        -------
        data : numpy ndarray | instance of DataFieldRecordArray
            The requested field data or a DataFieldRecordArray holding the
            requested selection of the entire data.
        """
        if isinstance(name, np.ndarray):
            return self.get_selection(name)

        if name not in self._data_fields:
            raise KeyError(
                f'The data field "{name}" is not present in the '
                'DataFieldRecordArray instance.')

        return self._data_fields[name]

    def __setitem__(
            self,
            name,
            arr):
        """Implements data field value assignment. If values are assigned to a
        data field that does not exist yet, it  will be added via the
        ``append_field`` method.

        Parameters
        ----------
        name : str | numpy ndarray of int or bool
            The name of the data field, or a numpy ndarray holding the indices
            or mask of a selection of this DataFieldRecordArray.
        arr : numpy ndarray | instance of DataFieldRecordArray
            The numpy ndarray holding the field values. It must be of the same
            length as this DataFieldRecordArray. If `name` is a numpy ndarray,
            `arr` must be a DataFieldRecordArray.

        Raises
        ------
        ValueError
            If the given data array is not of the same length as this
            DataFieldRecordArray instance.
        """
        if isinstance(name, np.ndarray):
            self.set_selection(name, arr)
            return

        # Check if a new field is supposed to be added.
        if name not in self:
            self.append_field(name, arr)
            return

        # We set a particular already existing data field.
        if len(arr) != self._len:
            raise ValueError(
                f'The length of the to-be-set data ({len(arr)}) must match '
                f'the length ({self._len}) of the DataFieldRecordArray '
                'instance!')

        if not isinstance(arr, np.ndarray):
            raise TypeError(
                'When setting a field directly, the data must be provided as a '
                'numpy ndarray!')

        self._data_fields[name] = arr

    def __len__(self):
        return self._len

    def __sizeof__(self):
        """Calculates the size in bytes of this DataFieldRecordArray instance
        in memory.

        Returns
        -------
        memsize : int
            The memory size in bytes that this DataFieldRecordArray instance
            has.
        """
        memsize = getsizeof([
            self._data_fields,
            self._len,
            self._field_name_list,
            self._indices
        ])
        return memsize

    def __str__(self):
        """Creates a pretty informative string representation of this
        DataFieldRecordArray instance.
        """
        (size, prefix) = get_byte_size_prefix(sys.getsizeof(self))

        max_field_name_len = np.max(
            [len(fname) for fname in self._field_name_list])

        # Generates a pretty string representation of the given field name.
        def _pretty_str_field(name):
            field = self._data_fields[name]
            s = (f'{name.ljust(max_field_name_len)}: '
                 '{'
                 f'dtype: {str(field.dtype)}, '
                 f'vmin: {np.min(field):.3e}, '
                 f'vmax: {np.max(field)}'
                 '}')
            return s

        indent_str = ' '*dsp.INDENTATION_WIDTH
        s = (f'{classname(self)}: {len(self._field_name_list)} fields, '
             f'{len(self)} entries, {np.round(size, 0):.0f} {prefix}bytes ')
        if len(self._field_name_list) > 0:
            s += f'\n{indent_str}fields = '
            s += '{'
            for fname in self._field_name_list:
                s += f'\n{indent_str*2}{_pretty_str_field(fname)}'
            s += f'\n{indent_str}'
            s += '}'

        return s

    @property
    def field_name_list(self):
        """(read-only) The list of the field names of this DataFieldRecordArray.
        """
        return self._field_name_list

    @property
    def indices(self):
        """(read-only) The numpy ndarray holding the indices of this
        DataFieldRecordArray.
        """
        if self._indices is None:
            self._indices = np.arange(self._len)
        return self._indices

    def append(self, arr):
        """Appends the given DataFieldRecordArray to this DataFieldRecordArray
        instance.

        Parameters
        ----------
        arr : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that should get appended to
            this DataFieldRecordArray. It must contain the same data fields.
            Additional data fields are ignored.
        """
        if not isinstance(arr, DataFieldRecordArray):
            raise TypeError(
                'The arr argument must be an instance of DataFieldRecordArray!')

        for fname in self._field_name_list:
            self._data_fields[fname] = np.append(
                self._data_fields[fname], arr[fname])

        self._len += len(arr)
        self._indices = None

    def append_field(self, name, data):
        """Appends a field and its data to this DataFieldRecordArray instance.

        Parameters
        ----------
        name : str
            The name of the new data field.
        data : numpy ndarray
            The numpy ndarray holding the data. The length of the ndarray must
            match the current length of this DataFieldRecordArray instance.

        Raises
        ------
        KeyError
            If the given data field name already exists in this
            DataFieldRecordArray instance.
        ValueError
            If the length of the data array does not equal to the length of the
            data of this DataFieldRecordArray instance.
        TypeError
            If the arguments are of the wrong type.
        """
        if not isinstance(name, str):
            raise TypeError(
                'The name argument must be an instance of str!')
        if not isinstance(data, np.ndarray):
            raise TypeError(
                'The data argument must be an instance of ndarray!')
        if name in self._data_fields:
            raise KeyError(
                f'The data field "{name}" already exists in this '
                f'{classname(self)} instance!')
        if len(data) != self._len:
            raise ValueError(
                f'The length of the given data is {len(data)}, but must be '
                f'{self._len}!')

        self._data_fields[name] = data
        self._field_name_list.append(name)

    def as_numpy_record_array(self):
        """Creates a numpy record ndarray instance holding the data of this
        DataFieldRecordArray instance.

        Returns
        -------
        arr : instance of numpy record ndarray
            The numpy recarray ndarray holding the data of this
            DataFieldRecordArray instance.
        """
        dt = np.dtype([
            (name, self._data_fields[name].dtype)
            for name in self.field_name_list
        ])

        arr = np.empty((len(self),), dtype=dt)
        for name in self.field_name_list:
            arr[name] = self[name]

        return arr

    def copy(
            self,
            keep_fields=None):
        """Creates a new DataFieldRecordArray that is a copy of this
        DataFieldRecordArray instance.

        Parameters
        ----------
        keep_fields : str | sequence of str | None
            If not None (default), this specifies the data fields that should
            get kept from this DataFieldRecordArray. Otherwise all data fields
            get kept.
        """
        return DataFieldRecordArray(self, keep_fields=keep_fields)

    def remove_field(self, name):
        """Removes the given field from this array.

        Parameters
        ----------
        name : str
            The name of the data field that is to be removed.
        """
        self._data_fields.pop(name)
        self._field_name_list.remove(name)

    def get_field_dtype(self, name):
        """Returns the numpy dtype object of the given data field.
        """
        return self._data_fields[name].dtype

    def set_field_dtype(self, name, dt):
        """Sets the data type of the given field.

        Parameters
        ----------
        name : str
            The name of the data field.
        dt : numpy.dtype
            The dtype instance defining the new data type.
        """
        if name not in self:
            raise KeyError(
                f'The data field "{name}" does not exist in this '
                f'{classname(self)} instance!')
        if not isinstance(dt, np.dtype):
            raise TypeError(
                'The dt argument must be an instance of type numpy.dtype!')

        self._data_fields[name] = self._data_fields[name].astype(dt, copy=False)

    def convert_dtypes(
            self,
            conversions,
            except_fields=None):
        """Converts the data type of the data fields of this
        DataFieldRecordArray. This method can be used to compress the data.

        Parameters
        ----------
        conversions : dict of `old_dtype` -> `new_dtype`
            The dictionary with the old dtype as key and the new dtype as value.
        except_fields : sequence of str | None
            The sequence of field names, which should not get converted.
        """
        if not isinstance(conversions, dict):
            raise TypeError(
                'The conversions argument must be an instance of dict!')

        if except_fields is None:
            except_fields = []
        if not issequenceof(except_fields, str):
            raise TypeError(
                'The except_fields argument must be a sequence of str!')

        _data_fields = self._data_fields
        for fname in self._field_name_list:
            if fname in except_fields:
                continue
            old_dtype = _data_fields[fname].dtype
            if old_dtype in conversions:
                new_dtype = conversions[old_dtype]
                _data_fields[fname] = _data_fields[fname].astype(new_dtype)

    def get_selection(
            self,
            indices):
        """Creates an DataFieldRecordArray that contains a selection of the data
        of this DataFieldRecordArray instance.

        Parameters
        ----------
        indices : (N,)-shaped numpy ndarray of int or bool
            The numpy ndarray holding the indices for which to select the data.

        Returns
        -------
        data_field_array : instance of DataFieldRecordArray
            The DataFieldRecordArray that contains the selection of the
            original DataFieldRecordArray. The selection data is a copy of the
            original data.
        """
        data = dict()
        for fname in self._field_name_list:
            # Get the data selection from the original data. This creates a
            # copy.
            data[fname] = self._data_fields[fname][indices]
        return DataFieldRecordArray(data, copy=False)

    def set_selection(
            self,
            indices,
            arr):
        """Sets a selection of the data of this DataFieldRecordArray instance
        to the data given in arr.

        Parameters
        ----------
        indices : (N,)-shaped numpy ndarray of int or bool
            The numpy ndarray holding the indices or mask for which to set the
            data.
        arr : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selection data.
            It must have the same fields defined as this DataFieldRecordArray
            instance.
        """
        if not isinstance(arr, DataFieldRecordArray):
            raise TypeError(
                'The arr argument must be an instance of DataFieldRecordArray!')

        for fname in self._field_name_list:
            self._data_fields[fname][indices] = arr[fname]

    def rename_fields(
            self,
            conversions,
            must_exist=False):
        """Renames the given fields of this array.

        Parameters
        ----------
        conversions : dict of `old_name` -> `new_name`
            The dictionary holding the old and new names of the data fields.
        must_exist : bool
            Flag if the given fields must exist. If set to ``True`` and a field
            does not exist, a KeyError is raised.

        Raises
        ------
        KeyError
            If ``must_exist`` is set to ``True`` and a given field does not
            exist.
        """
        for (old_fname, new_fname) in conversions.items():
            if old_fname in self.field_name_list:
                self._data_fields[new_fname] = self._data_fields.pop(old_fname)
            elif must_exist is True:
                raise KeyError(
                    f'The required field "{old_fname}" does not exist!')

        self._field_name_list = list(self._data_fields.keys())

    def tidy_up(
            self,
            keep_fields):
        """Removes all fields that are not specified through the keep_fields
        argument.

        Parameters
        ----------
        keep_fields : str | sequence of str
            The field name(s), that should not be removed.

        Raises
        ------
        TypeError
            If keep_fields is not an instance of str or a sequence of str
            instances.
        """
        if isinstance(keep_fields, str):
            keep_fields = [keep_fields]
        if not issequenceof(keep_fields, str):
            raise TypeError(
                'The keep_fields argument must be a sequence of str!')

        # We need to make a copy of the field_name_list because that list will
        # get changed by the `remove_field` method.
        field_name_list = copy.copy(self._field_name_list)
        for fname in field_name_list:
            if fname not in keep_fields:
                self.remove_field(fname)

    def sort_by_field(
            self,
            name):
        """Sorts the data along the given field name in ascending order.

        Parameters
        ----------
        name : str
            The name of the field along the events should get sorted.

        Returns
        -------
        sorted_idxs : (n_events,)-shaped numpy ndarray
            The numpy ndarray holding the indices of the sorted array.

        Raises
        ------
        KeyError
            If the given data field does not exist.
        """
        if name not in self._data_fields:
            raise KeyError(
                f'The data field "{name}" does not exist in this '
                f'{classname(self)} instance!')

        sorted_idxs = np.argsort(self._data_fields[name])

        for fname in self._field_name_list:
            self._data_fields[fname] = self._data_fields[fname][sorted_idxs]

        return sorted_idxs


register_FileLoader(['.npy'], NPYFileLoader)
register_FileLoader(['.parquet'], ParquetFileLoader)
register_FileLoader(['.pkl'], PKLFileLoader)
register_FileLoader(['.csv'], TextFileLoader)
