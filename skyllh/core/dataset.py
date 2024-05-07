# -*- coding: utf-8 -*-

import abc
import os
import os.path
import shutil
import stat

import numpy as np

from copy import (
    deepcopy,
)

from skyllh.core import (
    display,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.datafields import (
    DataFields,
    DataFieldStages as DFS,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.display import (
    ANSIColors,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
    float_cast,
    get_class_of_func,
    issequence,
    issequenceof,
    list_of_cast,
    module_class_method_name,
    str_cast,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
    create_FileLoader,
)
from skyllh.core.timing import (
    TaskTimer,
)


class DatasetOrigin(
    object,
):
    """The DatasetOrigin class provides information about the origin of a
    dataset, so the files of a dataset can be downloaded from the origin.
    """
    def __init__(
            self,
            base_path,
            sub_path,
            transfer_func,
            filename=None,
            host=None,
            port=None,
            username=None,
            password=None,
            post_transfer_func=None,
            **kwargs,
    ):
        """Creates a new instance to define the origin of a dataset.

        Parameters
        ----------
        base_path : str
            The dataset's base directory at the origin.
        sub_path : str
            The dataset's sub directory at the origin.
        transfer_func : callable
            The callable object that should be used to transfer the dataset.
            This function requires the following call signature::

                __call__(origin, file_list, dst_base_path, user=None, password=None)

            where ``origin`` is an instance of DatasetOrigin, ``file_list`` is
            a list of str specifying the files to transfer, ``dst_base_path`` is
            an instance of str specifying the destination base path on the local
            machine, ``user`` is the user name required to connect to the remote
            host, and ``password`` is the password for the user name required to
            connect to the remote host.
        filename : str | None
            If the origin is not a directory but a file, this specifies the
            filename.
        host : str | None
            The name or IP of the remote host.
        port : int | None
            The port number to use when connecting to the remote host.
        username : str | None
            The user name required to connect to the remote host.
        password : str | None
            The password for the user name required to connect to the remote
            host.
        post_transfer_func : callable | None
            The callable object that should be called after the dataset has been
            transferred by the ``transfer_func``function. It can be used to
            extract an archive file.
            This function requires the following call signature::

                __call__(ds, dst_path)

            where ``ds`` is an instance of ``Dataset``, and ``dst_path`` is the
            destination path.
        """
        super().__init__(**kwargs)

        self.base_path = base_path
        self.sub_path = sub_path
        self.transfer_func = transfer_func
        self.filename = filename
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.post_transfer_func = post_transfer_func

    @property
    def base_path(self):
        """The dataset's base directory at the origin.
        """
        return self._base_path

    @base_path.setter
    def base_path(self, obj):
        if not isinstance(obj, str):
            raise TypeError(
                'The base_path property must be an instance of str! '
                f'Its current type is {classname(obj)}!')
        self._base_path = obj

    @property
    def sub_path(self):
        """The dataset's sub directory at the origin.
        """
        return self._sub_path

    @sub_path.setter
    def sub_path(self, obj):
        if not isinstance(obj, str):
            raise TypeError(
                'The sub_path property must be an instance of str! '
                f'Its current type is {classname(obj)}!')
        self._sub_path = obj

    @property
    def root_dir(self):
        """(read-only) The dataset's root directory at the origin, which is
        the combination of ``base_path`` and ``sub_path``.
        """
        return os.path.join(self._base_path, self._sub_path)

    @property
    def transfer_func(self):
        """The callable object that should be used to transfer the dataset.
        """
        return self._transfer_func

    @transfer_func.setter
    def transfer_func(self, obj):
        if not callable(obj):
            raise TypeError(
                'The property transfer_func must be a callable object! '
                f'Its current type is {classname(obj)}!')
        self._transfer_func = obj

    @property
    def filename(self):
        """The file name if the origin is a file instead of a directory.
        """
        return self._filename

    @filename.setter
    def filename(self, obj):
        if obj is not None:
            if not isinstance(obj, str):
                raise TypeError(
                    'The property filename must be None, or an instance of '
                    'str! '
                    f'Its current type is {classname(obj)}!')
        self._filename = obj

    @property
    def is_directory(self):
        """(read-only) Flag if the origin refers to a directory (``True``) or a
        file (``False``).
        """
        return (self._filename is None)

    @property
    def host(self):
        """The name or IP of the remote host.
        """
        return self._host

    @host.setter
    def host(self, obj):
        if obj is not None:
            if not isinstance(obj, str):
                raise TypeError(
                    'The property host must be None, or an instance of str! '
                    f'Its current type is {classname(obj)}!')
        self._host = obj

    @property
    def port(self):
        """The port number to use when connecting to the remote host.
        """
        return self._port

    @port.setter
    def port(self, obj):
        if obj is not None:
            if not isinstance(obj, int):
                raise TypeError(
                    'The property port must be None, or an instance of int! '
                    f'Its current type is {classname(obj)}!')
        self._port = obj

    @property
    def username(self):
        """The user name required to connect to the remote host.
        """
        return self._username

    @username.setter
    def username(self, obj):
        if obj is not None:
            if not isinstance(obj, str):
                raise TypeError(
                    'The property username must be None, or an instance of '
                    'str! '
                    f'Its current type is {classname(obj)}!')
        self._username = obj

    @property
    def password(self):
        """The password for the user name required to connect to the remote
        host.
        """
        return self._password

    @password.setter
    def password(self, obj):
        if obj is not None:
            if not isinstance(obj, str):
                raise TypeError(
                    'The property password must be None, or an instance of '
                    'str! '
                    f'Its current type is {classname(obj)}!')
        self._password = obj

    @property
    def post_transfer_func(self):
        """The callable object that should be called after the dataset has been
        transferred by the ``transfer_func`` callable.
        """
        return self._post_transfer_func

    @post_transfer_func.setter
    def post_transfer_func(self, obj):
        if obj is not None:
            if not callable(obj):
                raise TypeError(
                    'The property post_transfer_func must be a callable '
                    'object! '
                    f'Its current type is {classname(obj)}!')
        self._post_transfer_func = obj

    def __str__(self):
        """Pretty string representation of this class.
        """
        transfer_cls = get_class_of_func(self.transfer_func)

        s = f'{classname(self)} '+'{\n'
        s1 = f'base_path = "{self.base_path}"\n'
        s1 += f'sub_path = "{self.sub_path}"\n'
        if self._filename is not None:
            s1 += f'filename = {self._filename}\n'
        s1 += f'user@host:port = {self.username}@{self.host}:{self.port}\n'
        s1 += 'password = '
        if self.password is not None:
            s1 += 'set\n'
        else:
            s1 += 'not set\n'
        s1 += f'transfer class = {classname(transfer_cls)}\n'
        s += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)
        s += '}'

        return s

    def is_locally_available(self):
        """Checks if the dataset origin is available locally by checking if the
        given path exists on the local host.

        Returns
        -------
        check : bool
            ``True`` if the path specified in this dataset origin is an absolute
            path and exists on the local host, ``False`` otherwise.
        """
        root_dir = self.root_dir

        if (
            self.is_directory and
            os.path.abspath(root_dir) == root_dir and
            os.path.exists(root_dir) and
            os.path.isdir(root_dir)
        ):
            return True

        return False


class TemporaryTextFile(
    object,
):
    """This class provides a temporary text file with a given content while
    being within a with statement. Exiting the with statement will remove the
    temporary text file.

    Example:

    .. code::

        with TemporaryTextFile('myfile.txt', 'My file content'):
            # Do something that requires the text file ``myfile.txt``.
        # At this point the text file is removed again.

    """

    def __init__(
            self,
            pathfilename,
            text,
            mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
    ):
        self.pathfilename = pathfilename
        self.text = text
        self.mode = mode

    def __enter__(self):
        with open(self.pathfilename, 'w') as fd:
            fd.write(self.text)
        os.chmod(self.pathfilename, self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.pathfilename)


class SystemCommandError(
    Exception,
):
    """This custom exception will be raised when a system command failed.
    """
    pass


class DatasetTransferError(
    Exception,
):
    """This custom exception defines an error that should be raised when the
    actual transfer of the dataset files failed.
    """
    pass


class DatasetTransfer(
    object,
    metaclass=abc.ABCMeta,
):
    """Base class for a dataset transfer mechanism.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def execute_system_command(
            cmd,
            logger,
            success_rcode=0,
    ):
        """Executes the given system command via a ``os.system`` call.

        Parameters
        ----------
        cmd : str
            The system command to execute.
        logger : instance of logging.Logger
            The logger to use for debug messages.
        success_rcode : int | None
            The return code that indicates success of the system command.
            If set to ``None``, no return code checking is performed.

        Raises
        ------
        SystemCommandError
            If the system command did not return ``success_rcode``.
        """
        logger.debug(f'Running command "{cmd}"')
        rcode = os.system(cmd)
        if (success_rcode is not None) and (rcode != success_rcode):
            raise SystemCommandError(
                f'The system command "{cmd}" failed with return code {rcode}!')

    @staticmethod
    def ensure_dst_path(
            dst_path,
    ):
        """Ensures the existence of the given destination path.

        Parameters
        ----------
        dst_path : str
            The destination path.
        """
        if not os.path.isdir(dst_path):
            # Throws if dst_path exists as a file.
            os.makedirs(dst_path)

    @abc.abstractmethod
    def transfer(
            self,
            origin,
            file_list,
            dst_base_path,
            username=None,
            password=None,
    ):
        """This method is supposed to transfer the dataset origin path to the
        given destination path.

        Parameters
        ----------
        origin : instance of DatasetOrigin
            The instance of DatasetOrigin defining the origin of the dataset.
        file_list : list of str
            The list of files, relative to the origin base path, which should be
            transferred.
        dst_base_path : str
            The destination base path into which the dataset files will be
            transferred.
        username : str | None
            The user name required to connect to the remote host.
        password : str | None
            The password for the user name required to connect to the remote
            host.

        Raises
        ------
        DatasetTransferError
            If the actual transfer of the dataset files failed.
        """
        pass

    @staticmethod
    def post_transfer_unzip(
            ds,
            dst_path,
    ):
        """This is a post-transfer function. It will unzip the transferred file
        into the dst_path if the origin path was a zip file.
        """
        if ds.origin.filename is None:
            return

        if not ds.origin.filename.lower().endswith('.zip'):
            return

        cls = get_class_of_func(DatasetTransfer.post_transfer_unzip)
        logger = get_logger(f'{classname(cls)}.post_transfer_unzip')

        # Unzip the dataset file.
        zip_file = os.path.join(dst_path, ds.origin.filename)
        cmd = f'unzip "{zip_file}" -d "{dst_path}"'
        DatasetTransfer.execute_system_command(cmd, logger)

        # Remove the zip file.
        try:
            os.remove(zip_file)
        except Exception as exc:
            logger.warn(str(exc))


class RSYNCDatasetTransfer(
    DatasetTransfer,
):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs)

    def transfer(  # noqa: C901
            self,
            origin,
            file_list,
            dst_base_path,
            username=None,
            password=None,
    ):
        """Transfers the given dataset to the given destination path using the
        ``rsync`` program.

        Parameters
        ----------
        ds : instance of Dataset
            The instance of Dataset containing the origin property specifying
            the origin of the dataset.
        file_list : list of str
            The list of files, relative to the origin base path, which should be
            transferred.
        dst_base_path : str
            The destination base path into which the dataset files will be
            transferred.
        username : str | None
            The user name required to connect to the remote host.
        password : str | None
            The password for the user name required to connect to the remote
            host.
        """
        cls = get_class_of_func(self.transfer)
        logger = get_logger(f'{classname(cls)}.transfer')

        host = origin.host

        # Make sure the origin and destination base paths end with a directory
        # separator.
        origin_base_path = origin.base_path
        if origin_base_path[-len(os.path.sep):] != os.path.sep:
            origin_base_path += os.path.sep
        if dst_base_path[-len(os.path.sep):] != os.path.sep:
            dst_base_path += os.path.sep

        file_list_pathfilename = os.path.join(
            os.getcwd(),
            f'.{id(origin)}.rsync_file_list.txt')

        # Create file list file content.
        # Skip files which already exists.
        file_list_filecontent = ''
        for file in file_list:
            dst_pathfilename = os.path.join(dst_base_path, file)
            if not os.path.exists(dst_pathfilename):
                file_list_filecontent += f'{file}\n'

        if username is None:
            # No user name is defined.
            with TemporaryTextFile(
                pathfilename=file_list_pathfilename,
                text=file_list_filecontent,
            ):
                cmd = (
                    f'rsync '
                    '-avrRL '
                    '--progress '
                    f'--files-from="{file_list_pathfilename}" '
                    f'{host}:"{origin_base_path}" "{dst_base_path}"'
                )
                try:
                    DatasetTransfer.execute_system_command(cmd, logger)
                except SystemCommandError as err:
                    raise DatasetTransferError(str(err))

        elif password is not None:
            # User and password is defined.
            pwdfile = os.path.join(
                os.getcwd(),
                f'.{id(origin)}.rsync_passwd.txt')

            with TemporaryTextFile(
                pathfilename=pwdfile,
                text=password,
                mode=stat.S_IRUSR,
            ):
                with TemporaryTextFile(
                    pathfilename=file_list_pathfilename,
                    text=file_list_filecontent,
                ):
                    cmd = (
                        f'rsync '
                        '-avrRL '
                        '--progress '
                        f'--password-file "{pwdfile}" '
                        f'--files-from="{file_list_pathfilename}" '
                        f'{username}@{host}:"{origin_base_path}" "{dst_base_path}"'
                    )
                    try:
                        DatasetTransfer.execute_system_command(cmd, logger)
                    except SystemCommandError as err:
                        raise DatasetTransferError(str(err))
        else:
            # Only the user name is defined.
            with TemporaryTextFile(
                pathfilename=file_list_pathfilename,
                text=file_list_filecontent,
            ):
                cmd = (
                    f'rsync '
                    '-avrRL '
                    '--progress '
                    f'--files-from="{file_list_pathfilename}" '
                    f'{username}@{host}:"{origin_base_path}" "{dst_base_path}"'
                )
                try:
                    DatasetTransfer.execute_system_command(cmd, logger)
                except SystemCommandError as err:
                    raise DatasetTransferError(str(err))


class WGETDatasetTransfer(
    DatasetTransfer,
):
    def __init__(self, protocol, **kwargs):
        super().__init__(
            **kwargs)

        self.protocol = protocol

    @property
    def protocol(self):
        """The protocol to use for the transfer.
        """
        return self._protocol

    @protocol.setter
    def protocol(self, obj):
        if not isinstance(obj, str):
            raise TypeError(
                'The property protocol must be an instance of str! '
                f'Its current type is {classname(obj)}!')
        self._protocol = obj

    def transfer(
            self,
            origin,
            file_list,
            dst_base_path,
            username=None,
            password=None,
    ):
        """Transfers the given dataset to the given destination path using the
        ``wget`` program.

        Parameters
        ----------
        origin : instance of DatasetOrigin
            The instance of DatasetOrigin defining the origin of the dataset.
        file_list : list of str
            The list of files relative to the origin's base path, which
            should be transferred.
        dst_base_path : str
            The destination base path into which the dataset will be
            transferred.
        username : str | None
            The user name required to connect to the remote host.
        password : str | None
            The password for the user name required to connect to the remote
            host.
        """
        cls = get_class_of_func(self.transfer)
        logger = get_logger(f'{classname(cls)}.transfer')

        host = origin.host
        port = origin.port

        for file in file_list:
            dst_pathfilename = os.path.join(dst_base_path, file)
            if os.path.exists(dst_pathfilename):
                logger.debug(
                    f'File "{dst_pathfilename}" already exists. Skipping.')
                continue

            path = os.path.join(origin.base_path, file)

            dst_sub_path = os.path.dirname(file)
            if dst_sub_path == '':
                dst_path = dst_base_path
            else:
                dst_path = os.path.join(dst_base_path, dst_sub_path)
            DatasetTransfer.ensure_dst_path(dst_path)

            cmd = 'wget '
            if username is None:
                # No user name is specified.
                pass
            elif password is not None:
                # A user name and password is specified.
                cmd += (
                    f'--user="{username}" '
                    f'--password="{password}" '
                )
            else:
                # Only a user name is specified.
                cmd += (
                    f'--user={username} '
                )
            cmd += f'{self.protocol}://{host}'
            if port is not None:
                cmd += f':{port}'
            if path[0:1] != '/':
                cmd += '/'
            cmd += f'{path} -P {dst_path}'
            try:
                DatasetTransfer.execute_system_command(cmd, logger)
            except SystemCommandError as err:
                raise DatasetTransferError(str(err))


class Dataset(
        HasConfig,
):
    """The Dataset class describes a set of self-consistent experimental and
    simulated detector data. Usually this is for a certain time period, i.e.
    a season.

    Independent data sets of the same kind, e.g. event selection, can be joined
    through a DatasetCollection object.
    """
    @staticmethod
    def get_combined_exp_pathfilenames(
            datasets):
        """Creates the combined list of exp pathfilenames of all the given
        datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        exp_pathfilenames : list
            The combined list of exp pathfilenames.
        """
        if not issequenceof(datasets, Dataset):
            raise TypeError(
                'The datasets argument must be a sequence of Dataset '
                'instances!')

        exp_pathfilenames = []
        for ds in datasets:
            exp_pathfilenames += ds.exp_pathfilename_list

        return exp_pathfilenames

    @staticmethod
    def get_combined_mc_pathfilenames(
            datasets):
        """Creates the combined list of mc pathfilenames of all the given
        datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        mc_pathfilenames : list
            The combined list of mc pathfilenames.
        """
        if not issequenceof(datasets, Dataset):
            raise TypeError(
                'The datasets argument must be a sequence of Dataset '
                'instances!')

        mc_pathfilenames = []
        for ds in datasets:
            mc_pathfilenames += ds.mc_pathfilename_list

        return mc_pathfilenames

    @staticmethod
    def get_combined_livetime(
            datasets):
        """Sums the live-time of all the given datasets.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances.

        Returns
        -------
        livetime : float
            The sum of all the individual live-times.
        """
        if not issequenceof(datasets, Dataset):
            raise TypeError(
                'The datasets argument must be a sequence of Dataset '
                'instances!')

        livetime = np.sum([
            ds.livetime
            for ds in datasets
        ])

        return livetime

    def __init__(
            self,
            name,
            exp_pathfilenames,
            mc_pathfilenames,
            livetime,
            default_sub_path_fmt,
            version,
            verqualifiers=None,
            base_path=None,
            sub_path_fmt=None,
            origin=None,
            **kwargs,
    ):
        """Creates a new dataset object that describes a self-consistent set of
        data.

        Parameters
        ----------
        name : str
            The name of the dataset.
        exp_pathfilenames : str | sequence of str | None
            The file name(s), including paths, of the experimental data file(s).
            This can be None, if a MC-only study is performed.
        mc_pathfilenames : str | sequence of str | None
            The file name(s), including paths, of the monte-carlo data file(s).
            This can be None, if a MC-less analysis is performed.
        livetime : float | None
            The integrated live-time in days of the dataset. It can be None for
            cases where the live-time is retrieved directly from the data files
            upon data loading.
        default_sub_path_fmt : str
            The default format of the sub path of the data set.
            This must be a string that can be formatted via the ``format``
            method of the ``str`` class.
        version : int
            The version number of the dataset. Higher version numbers indicate
            newer datasets.
        verqualifiers : dict | None
            If specified, this dictionary specifies version qualifiers. These
            can be interpreted as subversions of the dataset. The format of the
            dictionary must be 'qualifier (str): version (int)'.
        base_path : str | None
            The user-defined base path of the data set.
            Usually, this is the path of the location of the data directory.
            If set to ``None`` the configured repository base path
            ``Config['repository']['base_path']`` is used.
        sub_path_fmt : str | None
            The user-defined format of the sub path of the data set.
            If set to ``None``, the ``default_sub_path_fmt`` will be used.
        origin : instance of DatasetOrigin | None
            The instance of DatasetOrigin defining the origin of the dataset,
            so the dataset can be transferred automatically to the user's
            device.
        """
        super().__init__(**kwargs)

        self.name = name
        self.exp_pathfilename_list = exp_pathfilenames
        self.mc_pathfilename_list = mc_pathfilenames
        self.livetime = livetime
        self.default_sub_path_fmt = default_sub_path_fmt
        self.version = version
        self.verqualifiers = verqualifiers
        self.base_path = base_path
        self.sub_path_fmt = sub_path_fmt
        self.origin = origin

        self.description = ''

        self._datafields = dict()

        self._exp_field_name_renaming_dict = dict()
        self._mc_field_name_renaming_dict = dict()

        self._data_preparation_functions = list()
        self._binning_definitions = dict()
        self._aux_data_definitions = dict()
        self._aux_data = dict()

    @property
    def name(self):
        """The name of the dataset. This must be an unique identifier among
        all the different datasets.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def description(self):
        """The (longer) description of the dataset.
        """
        return self._description

    @description.setter
    def description(self, description):
        if not isinstance(description, str):
            raise TypeError(
                'The description of the dataset must be of type str!')
        self._description = description

    @property
    def datafields(self):
        """The dictionary holding the names and stages of required data fields
        specific for this dataset.
        """
        return self._datafields

    @datafields.setter
    def datafields(self, fields):
        if not isinstance(fields, dict):
            raise TypeError(
                'The datafields property must be a dictionary! '
                f'Its current type is "{classname(fields)}"!')
        self._datafields = fields

    @property
    def exp_pathfilename_list(self):
        """The list of file names of the data files that store the experimental
        data for this dataset.
        If a file name is given with a relative path, it will be relative to the
        root_dir property of this Dataset instance.
        """
        return self._exp_pathfilename_list

    @exp_pathfilename_list.setter
    def exp_pathfilename_list(self, pathfilenames):
        if pathfilenames is None:
            pathfilenames = []
        if isinstance(pathfilenames, str):
            pathfilenames = [pathfilenames]
        if not issequenceof(pathfilenames, str):
            raise TypeError(
                'The exp_pathfilename_list property must be of type str or a '
                'sequence of str!')
        self._exp_pathfilename_list = list(pathfilenames)

    @property
    def exp_abs_pathfilename_list(self):
        """(read-only) The list of absolute path file names of the experimental
        data files.
        """
        return self.get_abs_pathfilename_list(self._exp_pathfilename_list)

    @property
    def mc_pathfilename_list(self):
        """The list of file names of the data files that store the monte-carlo
        data for this dataset.
        If a file name is given with a relative path, it will be relative to the
        root_dir property of this Dataset instance.
        """
        return self._mc_pathfilename_list

    @mc_pathfilename_list.setter
    def mc_pathfilename_list(self, pathfilenames):
        if pathfilenames is None:
            pathfilenames = []
        if isinstance(pathfilenames, str):
            pathfilenames = [pathfilenames]
        if not issequenceof(pathfilenames, str):
            raise TypeError(
                'The mc_pathfilename_list property must be of type str or a '
                'sequence of str!')
        self._mc_pathfilename_list = list(pathfilenames)

    @property
    def mc_abs_pathfilename_list(self):
        """(read-only) The list of absolute path file names of the monte-carlo
        data files.
        """
        return self.get_abs_pathfilename_list(self._mc_pathfilename_list)

    @property
    def livetime(self):
        """The integrated live-time in days of the dataset. This can be None in
        cases where the livetime is retrieved directly from the data files.
        """
        return self._lifetime

    @livetime.setter
    def livetime(self, lt):
        if lt is not None:
            lt = float_cast(
                lt,
                'The lifetime property of the dataset must be cast-able to '
                'type float!')
        self._lifetime = lt

    @property
    def version(self):
        """The main version (int) of the dataset.
        """
        return self._version

    @version.setter
    def version(self, version):
        if not isinstance(version, int):
            raise TypeError(
                'The version of the dataset must be of type int!')
        self._version = version

    @property
    def verqualifiers(self):
        """The dictionary holding the version qualifiers, i.e. sub-version
        qualifiers. If set to None, an empty dictionary will be used.
        The dictionary must have the type form of str:int.
        """
        return self._verqualifiers

    @verqualifiers.setter
    def verqualifiers(self, verqualifiers):
        if verqualifiers is None:
            verqualifiers = dict()
        if not isinstance(verqualifiers, dict):
            raise TypeError('The version qualifiers must be of type dict!')
        # Check if the dictionary has format str:int.
        for (q, v) in verqualifiers.items():
            if not isinstance(q, str):
                raise TypeError(
                    f'The version qualifier "{q}" must be of type str!')
            if not isinstance(v, int):
                raise TypeError(
                    f'The version for the qualifier "{q}" must be of type int!')
        # We need to take a deep copy in order to make sure that two datasets
        # don't share the same version qualifier dictionary.
        self._verqualifiers = deepcopy(verqualifiers)

    @property
    def base_path(self):
        """The base path of the data set. This can be ``None``.
        """
        return self._base_path

    @base_path.setter
    def base_path(self, path):
        if path is not None:
            path = str_cast(
                path,
                'The base_path property must be cast-able to type str!')
            if not os.path.isabs(path):
                raise ValueError(
                    'The base_path property must be an absolute path!')
        self._base_path = path

    @property
    def default_sub_path_fmt(self):
        """The default format of the sub path of the data set. This must be a
        string that can be formatted via the ``format`` method of the ``str``
        class.
        """
        return self._default_sub_path_fmt

    @default_sub_path_fmt.setter
    def default_sub_path_fmt(self, fmt):
        fmt = str_cast(
            fmt,
            'The default_sub_path_fmt property must be cast-able to type str!')
        self._default_sub_path_fmt = fmt

    @property
    def sub_path_fmt(self):
        """The format of the sub path of the data set. This must be a string
        that can be formatted via the ``format`` method of the ``str`` class.
        If set to ``None``, this property will return the
        ``default_sub_path_fmt`` property.
        """
        if self._sub_path_fmt is None:
            return self._default_sub_path_fmt
        return self._sub_path_fmt

    @sub_path_fmt.setter
    def sub_path_fmt(self, fmt):
        if fmt is not None:
            fmt = str_cast(
                fmt,
                'The sub_path_fmt property must be None, or cast-able to type '
                'str!')
        self._sub_path_fmt = fmt

    @property
    def origin(self):
        """The instance of DatasetOrigin defining the origin of the dataset.
        This can be ``None`` if the dataset has no origin defined.
        """
        return self._origin

    @origin.setter
    def origin(self, obj):
        if obj is not None:
            if not isinstance(obj, DatasetOrigin):
                raise TypeError(
                    'The origin property must be None, or an instance of '
                    'DatasetOrigin! '
                    f'Its current type is {classname(obj)}!')
        self._origin = obj

    @property
    def root_dir(self):
        """(read-only) The root directory to use when data files are specified
        with relative paths. It is constructed from the ``base_path`` and the
        ``sub_path_fmt`` properties via the ``generate_data_file_root_dir``
        function.
        """
        return generate_data_file_root_dir(
            default_base_path=self._cfg['repository']['base_path'],
            default_sub_path_fmt=self._default_sub_path_fmt,
            version=self._version,
            verqualifiers=self._verqualifiers,
            base_path=self._base_path,
            sub_path_fmt=self._sub_path_fmt)

    @property
    def exp_field_name_renaming_dict(self):
        """The dictionary specifying the field names of the experimental data
        which need to get renamed just after loading the data. The dictionary
        values are the new names.
        """
        return self._exp_field_name_renaming_dict

    @exp_field_name_renaming_dict.setter
    def exp_field_name_renaming_dict(self, d):
        if not isinstance(d, dict):
            raise TypeError(
                'The exp_field_name_renaming_dict property must be an instance '
                'of dict!')
        self._exp_field_name_renaming_dict = d

    @property
    def mc_field_name_renaming_dict(self):
        """The dictionary specifying the field names of the monte-carlo data
        which need to get renamed just after loading the data. The dictionary
        values are the new names.
        """
        return self._mc_field_name_renaming_dict

    @mc_field_name_renaming_dict.setter
    def mc_field_name_renaming_dict(self, d):
        if not isinstance(d, dict):
            raise TypeError(
                'The mc_field_name_renaming_dict property must be an instance '
                'of dict!')
        self._mc_field_name_renaming_dict = d

    @property
    def exists(self):
        """(read-only) Flag if all the data files of this dataset exists. It is
        ``True`` if all data files exist and ``False`` otherwise.
        """
        file_list = self.create_file_list()
        abs_file_list = self.get_abs_pathfilename_list(file_list)
        for abs_file in abs_file_list:
            if not os.path.exists(abs_file):
                return False
        return True

    @property
    def version_str(self):
        """The version string of the dataset. This combines all the version
        information about the dataset.
        """
        s = f'{self._version:03d}'
        for (q, v) in self._verqualifiers.items():
            s += f'{q}{v:02d}'
        return s

    @property
    def data_preparation_functions(self):
        """The list of callback functions that will be called to prepare the
        data (experimental and monte-carlo).
        """
        return self._data_preparation_functions

    def _gen_datafile_pathfilename_entry(self, pathfilename):
        """Generates a string containing the given pathfilename and if it exists
        (FOUND) or not (NOT FOUND).

        Parameters
        ----------
        pathfilename : str
            The fully qualified path and filename of the file.

        Returns
        -------
        s : str
            The generated string.
        """
        if os.path.exists(pathfilename):
            s = '['+ANSIColors.OKGREEN+'FOUND'+ANSIColors.ENDC+']'
        else:
            s = '['+ANSIColors.FAIL+'NOT FOUND'+ANSIColors.ENDC+']'
        s += ' ' + pathfilename
        return s

    def __gt__(self, ds):
        """Implementation to support the operation ``b = self > ds``, where
        ``self`` is this Dataset object and ``ds`` an other Dataset object.
        The comparison is done based on the version information of both
        datasets. Larger version numbers for equal version qualifiers indicate
        newer (greater) datasets.

        The two datasets must be of the same kind, i.e. have the same name, in
        order to make the version comparison senseful.

        Returns
        -------
        bool
            True, if this dataset is newer than the reference dataset.
            False, if this dataset is as new or older than the reference
            dataset.
        """
        # Datasets of different names cannot be compared usefully.
        if self._name != ds._name:
            return False

        # Larger main version numbers indicate newer datasets.
        if self._version > ds._version:
            return True

        # Look for version qualifiers that make this dataset older than the
        # reference dataset.
        qs1 = self._verqualifiers.keys()
        qs2 = ds._verqualifiers.keys()

        # If a qualifier of self is also specified for ds, the version number
        # of the self qualifier must be larger than the version number of the ds
        # qualifier, in order to consider self as newer dataset.
        # If a qualifier is present in self but not in ds, self is considered
        # newer.
        for q in qs1:
            if q in qs2 and qs1[q] <= qs2[q]:
                return False
        # If there is a qualifier in ds but not in self, self is considered
        # older.
        for q in qs2:
            if q not in qs1:
                return False

        return True

    def __str__(self):  # noqa: C901
        """Implementation of the pretty string representation of the Dataset
        object.
        """
        s = f'Dataset "{self.name}": v{self.version_str}\n'

        s1 = ''

        if self.livetime is None:
            s1 += '{ livetime = UNDEFINED }'
        else:
            s1 += '{ 'f'livetime = {self.livetime:.3f} days'' }'
        s1 += '\n'

        if self.description != '':
            s1 += 'Description:\n' + self.description + '\n'

        s1 += 'Experimental data:\n'
        s2 = ''
        for (idx, pathfilename) in enumerate(self.exp_abs_pathfilename_list):
            if idx > 0:
                s2 += '\n'
            s2 += self._gen_datafile_pathfilename_entry(pathfilename)
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)
        s1 += '\n'

        s1 += 'MC data:\n'
        s2 = ''
        for (idx, pathfilename) in enumerate(self.mc_abs_pathfilename_list):
            if idx > 0:
                s2 += '\n'
            s2 += self._gen_datafile_pathfilename_entry(pathfilename)
        s1 += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s2)
        s1 += '\n'

        if len(self._aux_data_definitions) > 0:
            s1 += 'Auxiliary data:\n'
            s2 = ''
            for (idx, (name, pathfilename_list)) in enumerate(
                    self._aux_data_definitions.items()):
                if idx > 0:
                    s2 += '\n'

                s2 += name+':'
                s3 = ''
                pathfilename_list = self.get_abs_pathfilename_list(
                    pathfilename_list)
                for pathfilename in pathfilename_list:
                    s3 += '\n' + self._gen_datafile_pathfilename_entry(pathfilename)
                s2 += display.add_leading_text_line_padding(
                    display.INDENTATION_WIDTH, s3)
            s1 += display.add_leading_text_line_padding(
                display.INDENTATION_WIDTH, s2)

        s += display.add_leading_text_line_padding(
            display.INDENTATION_WIDTH, s1)

        return s

    def remove_data(self):
        """Removes the data of this dataset by removing the dataset's root
        directory and everything in it. If the root directory is a symbolic
        link, only this link will be removed.

        Raises
        ------
        RuntimeError
            If the dataset's root directory is neither a symlink nor a
            directory.
        """
        root_dir = self.root_dir

        if os.path.islink(root_dir):
            os.remove(root_dir)
            return

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)
            return

        raise RuntimeError(
            f'The root directory "{root_dir}" of dataset {self.name} is '
            'neither a symlink nor a directory!')

    def get_abs_pathfilename_list(
            self,
            pathfilename_list,
    ):
        """Returns a list where each entry of the given pathfilename_list is
        an absolute path. Relative paths will be prefixed with the root_dir
        property of this Dataset instance.

        Parameters
        ----------
        pathfilename_list : sequence of str
            The sequence of file names, either with relative, or absolute paths.

        Returns
        -------
        abs_pathfilename_list : list of str
            The list of file names with absolute paths.
        """
        root_dir = self.root_dir

        abs_pathfilename_list = []
        for pathfilename in pathfilename_list:
            if os.path.isabs(pathfilename):
                abs_pathfilename_list.append(
                    pathfilename)
            else:
                abs_pathfilename_list.append(
                    os.path.join(root_dir, pathfilename))

        return abs_pathfilename_list

    def get_missing_files(
            self,
    ):
        """Determines which files of the dataset are missing and returns the
        list of files.

        Returns
        -------
        missing_files : list of str
            The list of files that are missing. The files are relative to the
            dataset's root directory.
        """
        file_list = self.create_file_list()
        abs_file_list = self.get_abs_pathfilename_list(file_list)

        missing_files = [
            file
            for (file, abs_file) in zip(file_list, abs_file_list)
            if not os.path.exists(abs_file)
        ]

        return missing_files

    def update_version_qualifiers(
            self,
            verqualifiers,
    ):
        """Updates the version qualifiers of the dataset. The update can only
        be done by increasing the version qualifier integer or by adding new
        version qualifiers.

        Parameters
        ----------
        verqualifiers : dict
            The dictionary with the new version qualifiers.

        Raises
        ------
        ValueError
            If the integer number of an existing version qualifier is not larger
            than the old one.
        """
        got_new_verqualifiers = False
        verqualifiers_keys = verqualifiers.keys()
        self_verqualifiers_keys = self._verqualifiers.keys()
        if len(verqualifiers_keys) > len(self_verqualifiers_keys):
            # New version qualifiers must be a subset of the old version
            # qualifiers.
            for q in self_verqualifiers_keys:
                if q not in verqualifiers_keys:
                    raise ValueError(
                        f'The version qualifier {q} has been dropped!')
            got_new_verqualifiers = True

        existing_verqualifiers_incremented = False
        for q in verqualifiers:
            if (q in self._verqualifiers) and\
               (verqualifiers[q] > self._verqualifiers[q]):
                existing_verqualifiers_incremented = True
            self._verqualifiers[q] = verqualifiers[q]

        if not (got_new_verqualifiers or existing_verqualifiers_incremented):
            raise ValueError(
                'Version qualifier values did not increment and no new version '
                'qualifiers were added!')

    def create_file_list(
            self,
    ):
        """Creates the list of files that are linked to this dataset.
        The file paths are relative to the dataset's root directory.

        Returns
        -------
        file_list : list of str
            The list of files of this dataset.
        """
        file_list = (
            self._exp_pathfilename_list +
            self._mc_pathfilename_list
        )

        for aux_pathfilename_list in self._aux_data_definitions.values():
            file_list += aux_pathfilename_list

        return file_list

    def make_data_available(  # noqa: C901
            self,
            username=None,
            password=None,
    ):
        """Makes the data of the dataset available.
        If the root directory of the dataset does not exist locally, the dataset
        is transferred from its origin to the local host. If the origin is
        already available locally, only a symlink is created to the origin path.

        Parameters
        ----------
        username : str | None
            The user name required to connect to the remote host of the origin.
            If set to ``None``, the
        password : str | None
            The password of the user name required to connect to the remote host
            of the origin.

        Returns
        -------
        success : bool
            ``True`` if the data was made available successfully, ``False``
            otherwise.
        """
        logger = get_logger(
            module_class_method_name(self, 'make_data_available')
        )

        if len(self.get_missing_files()) == 0:
            logger.debug(
                f'All files of dataset "{self.name}" already exist. '
                'Nothing to download.')
            return True

        if self.origin is None:
            logger.warn(
                f'No origin defined for dataset "{self.name}"! '
                'Cannot download dataset!')
            return False

        # Check if the dataset origin is locally available. In that case we
        # just create a symlink.
        if self.origin.is_locally_available():
            root_dir = self.root_dir

            # Check if the symlink to the root directory already exists.
            if os.path.isdir(root_dir):
                return True

            # Make sure all directories leading to the symlink exist.
            dirname = os.path.dirname(root_dir)
            if dirname != '':
                os.makedirs(dirname, exist_ok=True)

            cmd = f'ln -s "{self.origin.root_dir}" "{root_dir}"'
            DatasetTransfer.execute_system_command(cmd, logger)
            return True

        if self._cfg['repository']['download_from_origin'] is False:
            logger.warn(
                f'The data of dataset "{self.name}" is locally not available '
                'and the download from the origin is disabled through the '
                'configuration!')
            return False

        if username is None:
            username = self.origin.username
        if password is None:
            password = self.origin.password

        base_path = generate_base_path(
            default_base_path=self._cfg['repository']['base_path'],
            base_path=self._base_path)

        logger.debug(
            f'Downloading dataset "{self.name}" from origin into base path '
            f'"{base_path}". username="{username}".')

        # Check if the origin is a directory. If not we just transfer that one
        # file.
        if self.origin.is_directory:
            file_list = [
                os.path.join(self.origin.sub_path, pathfilename)
                for pathfilename in self.create_file_list()
            ]
        else:
            file_list = [
                os.path.join(self.origin.sub_path, self.origin.filename)
            ]

        self.origin.transfer_func(
            origin=self.origin,
            file_list=file_list,
            dst_base_path=base_path,
            username=username,
            password=password)

        if self.origin.post_transfer_func is not None:
            self.origin.post_transfer_func(
                ds=self,
                dst_path=base_path)

        return True

    def load_data(
            self,
            livetime=None,
            keep_fields=None,
            dtc_dict=None,
            dtc_except_fields=None,
            efficiency_mode=None,
            tl=None,
    ):
        """Loads the data, which is described by the dataset.

        .. note:

            This does not call the ``prepare_data`` method! It only loads the
            data as the method names says.

        Parameters
        ----------
        livetime : instance of Livetime | float | None
            If not None, uses this livetime (if float, livetime in days) for the
            DatasetData instance, otherwise uses the Dataset livetime property
            value for the DatasetData instance.
        keep_fields : list of str | None
            The list of user-defined data fields that should get loaded and kept
            in addition to the analysis required data fields.
        dtc_dict : dict | None
            This dictionary defines how data fields of specific data types (key)
            should get converted into other data types (value).
            This can be used to use less memory. If set to None, no data
            conversion is performed.
        dtc_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        efficiency_mode : str | None
            The efficiency mode the data should get loaded with. Possible values
            are:

                ``'memory'``
                    The data will be load in a memory efficient way. This will
                    require more time, because all data records of a file will
                    be loaded sequentially.
                ``'time'``
                    The data will be loaded in a time efficient way. This will
                    require more memory, because each data file gets loaded in
                    memory at once.

            The default value is ``'time'``. If set to ``None``, the default
            value will be used.
        tl : instance of TimeLord | None
            The TimeLord instance to use to time the data loading procedure.

        Returns
        -------
        data : instance of DatasetData
            A instance of DatasetData holding the experimental and monte-carlo
            data.
        """
        def _conv_new2orig_field_names(
                new_field_names,
                orig2new_renaming_dict,
        ):
            """Converts the given ``new_field_names`` into their original name
            given the original-to-new field name renaming dictionary.
            """
            if new_field_names is None:
                return None

            new2orig_renaming_dict = {
                v: k
                for (k, v) in orig2new_renaming_dict.items()
            }

            orig_field_names = [
                new2orig_renaming_dict.get(new_field_name, new_field_name)
                for new_field_name in new_field_names
            ]

            return orig_field_names

        if self._cfg['repository']['download_from_origin'] is True:
            self.make_data_available()

        if keep_fields is None:
            keep_fields = []

        datafields = {**self._cfg['datafields'], **self._datafields}

        # Load the experimental data if there is any.
        if len(self._exp_pathfilename_list) > 0:
            with TaskTimer(tl, 'Loading exp data from disk.'):
                fileloader_exp = create_FileLoader(
                    self.exp_abs_pathfilename_list)
                # Create the list of field names that should get kept.
                keep_fields_exp = list(set(
                    _conv_new2orig_field_names(
                        DataFields.get_joint_names(
                            datafields=datafields,
                            stages=(
                                DFS.DATAPREPARATION_EXP |
                                DFS.ANALYSIS_EXP
                            )
                        ) +
                        keep_fields,
                        self._exp_field_name_renaming_dict
                    )
                ))

                data_exp = fileloader_exp.load_data(
                    keep_fields=keep_fields_exp,
                    dtype_conversions=dtc_dict,
                    dtype_conversion_except_fields=_conv_new2orig_field_names(
                        dtc_except_fields,
                        self._exp_field_name_renaming_dict),
                    efficiency_mode=efficiency_mode)
                data_exp.rename_fields(self._exp_field_name_renaming_dict)
        else:
            data_exp = None

        # Load the monte-carlo data if there is any.
        if len(self._mc_pathfilename_list) > 0:
            with TaskTimer(tl, 'Loading mc data from disk.'):
                fileloader_mc = create_FileLoader(
                    self.mc_abs_pathfilename_list)
                # Determine `keep_fields_mc` for the generic case, where MC
                # field names are an union of exp and mc field names.
                # But the renaming dictionary can differ for exp and MC fields.
                keep_fields_mc = list(set(
                    _conv_new2orig_field_names(
                        DataFields.get_joint_names(
                            datafields=datafields,
                            stages=(
                                DFS.DATAPREPARATION_EXP |
                                DFS.ANALYSIS_EXP
                            )
                        ) +
                        keep_fields,
                        self._exp_field_name_renaming_dict
                    ) +
                    _conv_new2orig_field_names(
                        DataFields.get_joint_names(
                            datafields=datafields,
                            stages=(
                                DFS.DATAPREPARATION_EXP |
                                DFS.ANALYSIS_EXP |
                                DFS.DATAPREPARATION_MC |
                                DFS.ANALYSIS_MC
                            )
                        ) +
                        keep_fields,
                        self._mc_field_name_renaming_dict
                    )
                ))
                data_mc = fileloader_mc.load_data(
                    keep_fields=keep_fields_mc,
                    dtype_conversions=dtc_dict,
                    dtype_conversion_except_fields=_conv_new2orig_field_names(
                        dtc_except_fields,
                        self._mc_field_name_renaming_dict),
                    efficiency_mode=efficiency_mode)
                data_mc.rename_fields(self._mc_field_name_renaming_dict)
        else:
            data_mc = None

        if livetime is None:
            livetime = self.livetime

        data = DatasetData(
            data_exp=data_exp,
            data_mc=data_mc,
            livetime=livetime)

        return data

    def load_aux_data(
            self,
            name,
            tl=None,
    ):
        """Loads the auxiliary data for the given auxiliary data definition.

        Parameters
        ----------
        name : str
            The name of the auxiliary data.
        tl : instance of TimeLord | None
            The TimeLord instance to use to time the data loading procedure.

        Returns
        -------
        data : unspecified
            The loaded auxiliary data.
        """
        name = str_cast(
            name,
            'The name argument must be cast-able to type str!')

        # Check if the data was defined in memory.
        if name in self._aux_data:
            with TaskTimer(tl, f'Loaded aux data "{name}" from memory.'):
                data = self._aux_data[name]
            return data

        if name not in self._aux_data_definitions:
            raise KeyError(
                f'The auxiliary data named "{name}" does not exist!')

        aux_pathfilename_list = self._aux_data_definitions[name]
        with TaskTimer(tl, f'Loaded aux data "{name}" from disk.'):
            fileloader_aux = create_FileLoader(self.get_abs_pathfilename_list(
                aux_pathfilename_list))
            data = fileloader_aux.load_data()

        return data

    def add_data_preparation(
            self,
            func,
    ):
        """Adds the given data preparation function to the dataset.

        Parameters
        ----------
        func : callable
            The object with call signature __call__(data) that will prepare
            the data after it was loaded. The argument 'data' is a DatasetData
            instance holding the experimental and monte-carlo data. The function
            must alter the properties of the DatasetData instance.

        """
        if not callable(func):
            raise TypeError(
                'The argument "func" must be a callable object with call '
                'signature __call__(data)!')
        self._data_preparation_functions.append(func)

    def remove_data_preparation(
            self,
            key=-1,
    ):
        """Removes a data preparation function from the dataset.

        Parameters
        ----------
        key : str, int, optional
            The name or the index of the data preparation function that should
            be removed. Default value is ``-1``, i.e. the last added function.

        Raises
        ------
        TypeError
            If the type of the key argument is invalid.
        IndexError
            If the given key is out of range.
        KeyError
            If the data preparation function cannot be found.
        """
        if isinstance(key, int):
            n = len(self._data_preparation_functions)
            if (key < -n) or (key >= n):
                raise IndexError(
                    f'The given index ({key}) for the data preparation '
                    f'function is out of range ({-n},{n-1})!')
            del self._data_preparation_functions[key]
            return
        elif isinstance(key, str):
            for (i, func) in enumerate(self._data_preparation_functions):
                if func.__name__ == key:
                    del self._data_preparation_functions[i]
                    return
            raise KeyError(
                f'The data preparation function "{key}" was not found in the '
                f'dataset "{self._name}"!')

        TypeError(
            'The key argument must be an instance of int or str!')

    def prepare_data(
            self,
            data,
            tl=None,
    ):
        """Prepares the data by calling the data preparation callback functions
        of this dataset.

        Parameters
        ----------
        data : instance of DatasetData
            The instance of DatasetData holding the data.
        tl : instance of TimeLord | None
            The instance TimeLord that should be used to time the data
            preparation.
        """
        for data_prep_func in self._data_preparation_functions:
            with TaskTimer(
                    tl,
                    f'Preparing data of dataset "{self.name}" by '
                    f'"{data_prep_func.__name__}".'):
                data_prep_func(data)

    def load_and_prepare_data(
            self,
            livetime=None,
            keep_fields=None,
            dtc_dict=None,
            dtc_except_fields=None,
            efficiency_mode=None,
            tl=None,
    ):
        """Loads and prepares the experimental and monte-carlo data of this
        dataset by calling its ``load_data`` and ``prepare_data`` methods.
        After loading the data it drops all unnecessary data fields if they are
        not listed in ``keep_fields``.
        In the end it asserts the data format of the experimental and
        monte-carlo data.

        Parameters
        ----------
        livetime : float | None
            The user-defined livetime in days of the data set. If not set to
            None, livetime information from the data set will get ignored and
            this value of the livetime will be used.
        keep_fields : sequence of str | None
            The list of additional data fields that should get kept.
            By default only the required data fields are kept.
        dtc_dict : dict | None
            This dictionary defines how data fields of specific data types (key)
            should get converted into other data types (value).
            This can be used to use less memory. If set to None, no data
            conversion is performed.
        dtc_except_fields : str | sequence of str | None
            The sequence of field names whose data type should not get
            converted.
        efficiency_mode : str | None
            The efficiency mode the data should get loaded with. Possible values
            are:

                ``'memory'``
                    The data will be load in a memory efficient way. This will
                    require more time, because all data records of a file will
                    be loaded sequentially.
                ``'time'``
                    The data will be loaded in a time efficient way. This will
                    require more memory, because each data file gets loaded in
                    memory at once.

            The default value is ``'time'``. If set to ``None``, the default
            value will be used.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time the data
            loading and preparation.

        Returns
        -------
        data : instance of DatasetData
            The instance of DatasetData holding the experimental and monte-carlo
            data.
        """
        if keep_fields is None:
            keep_fields = list()
        elif not issequenceof(keep_fields, str):
            raise TypeError(
                'The keep_fields argument must be None, or a sequence of str!')
        keep_fields = list(keep_fields)

        data = self.load_data(
            keep_fields=keep_fields,
            livetime=livetime,
            dtc_dict=dtc_dict,
            dtc_except_fields=dtc_except_fields,
            efficiency_mode=efficiency_mode,
            tl=tl)

        self.prepare_data(data, tl=tl)

        # Drop non-required data fields.
        if data.exp is not None:
            with TaskTimer(tl, 'Cleaning exp data.'):
                keep_fields_exp = (
                    DataFields.get_joint_names(
                        datafields=self._cfg['datafields'],
                        stages=(
                            DFS.ANALYSIS_EXP
                        )
                    ) +
                    keep_fields
                )
                data.exp.tidy_up(keep_fields=keep_fields_exp)

        if data.mc is not None:
            with TaskTimer(tl, 'Cleaning MC data.'):
                keep_fields_mc = (
                    DataFields.get_joint_names(
                        datafields=self._cfg['datafields'],
                        stages=(
                            DFS.ANALYSIS_EXP |
                            DFS.ANALYSIS_MC
                        )
                    ) +
                    keep_fields
                )
                data.mc.tidy_up(keep_fields=keep_fields_mc)

        with TaskTimer(tl, 'Asserting data format.'):
            assert_data_format(self, data)

        return data

    def add_binning_definition(
            self,
            binning,
    ):
        """Adds a binning setting to this dataset.

        Parameters
        ----------
        binning : BinningDefinition
            The BinningDefinition object holding the binning information.
        """
        if not isinstance(binning, BinningDefinition):
            raise TypeError(
                'The "binning" argument must be of type BinningDefinition!')
        if binning.name in self._binning_definitions:
            raise KeyError(
                f'The binning definition "{binning.name}" is already defined '
                f'for dataset "{self._name}"!')

        self._binning_definitions[binning.name] = binning

    def get_binning_definition(
            self,
            name,
    ):
        """Gets the BinningDefinition object for the given binning name.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        Returns
        -------
        binning_definition : instance of BinningDefinition
            The requested instance of BinningDefinition.
        """
        if name not in self._binning_definitions:
            raise KeyError(
                f'The given binning name "{name}" has not been added to the '
                'dataset yet!')
        return self._binning_definitions[name]

    def remove_binning_definition(
            self,
            name,
    ):
        """Removes the BinningDefinition object from the dataset.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        """
        if name not in self._binning_definitions:
            raise KeyError(
                f'The given binning name "{name}" does not exist in the '
                f'dataset "{self.name}", nothing to remove!')

        self._binning_definitions.pop(name)

    def has_binning_definition(
            self,
            name,
    ):
        """Checks if the dataset has a defined binning definition with the given
        name.

        Parameters
        ----------
        name : str
            The name of the binning definition.

        Returns
        -------
        check : bool
            True if the binning definition exists, False otherwise.
        """
        if name in self._binning_definitions:
            return True
        return False

    def define_binning(
            self,
            name,
            binedges,
    ):
        """Defines a binning for ``name``, and adds it as binning definition.

        Parameters
        ----------
        name : str
            The name of the binning setting.
        binedges : sequence
            The sequence of the bin edges, which should be used for this binning
            definition.

        Returns
        -------
        binning : instance of BinningDefinition
            The instance of BinningDefinition which was created and added to
            this dataset.
        """
        binning = BinningDefinition(name, binedges)
        self.add_binning_definition(binning)
        return binning

    def replace_binning_definition(self, binning):
        """Replaces an already defined binning definition of this dataset by
        the given binning definition.

        Parameters
        ----------
        binning : instance of BinningDefinition
            The instance of BinningDefinition that will replace the dataset's
            BinningDefinition instance of the same name.
        """
        if not isinstance(binning, BinningDefinition):
            raise TypeError(
                'The "binning" argument must be of type BinningDefinition!')
        if binning.name not in self._binning_definitions:
            raise KeyError(
                f'The given binning definition "{binning.name}" has not been '
                'added to the dataset yet!')

        self._binning_definitions[binning.name] = binning

    def add_aux_data_definition(
            self,
            name,
            pathfilenames,
    ):
        """Adds the given data files as auxiliary data definition to the
        dataset.

        Parameters
        ----------
        name : str
            The name of the auxiliary data definition. The name is used as
            identifier for the data within SkyLLH.
        pathfilenames : str | sequence of str
            The file name(s) (including paths) of the data file(s).
        """
        name = str_cast(
            name,
            'The name argument must be cast-able to type str! '
            f'Its current type is {classname(name)}.')

        pathfilenames = list_of_cast(
            str,
            pathfilenames,
            'The pathfilenames argument must be of type str or a sequence '
            f'of str! Its current type is {classname(pathfilenames)}.')

        if name in self._aux_data_definitions:
            raise KeyError(
                f'The auxiliary data definition "{name}" is already defined '
                f'for dataset "{self.name}"!')

        self._aux_data_definitions[name] = pathfilenames

    def get_aux_data_definition(
            self,
            name,
    ):
        """Returns the auxiliary data definition from the dataset.

        Parameters
        ----------
        name : str
            The name of the auxiliary data definition.

        Raises
        ------
        KeyError
            If auxiliary data with the given name does not exist.

        Returns
        -------
        aux_data_definition : list of str
            The locations (pathfilenames) of the files defined in the auxiliary
            data as auxiliary data definition.
        """
        if name not in self._aux_data_definitions:
            raise KeyError(
                f'The auxiliary data definition "{name}" does not exist in '
                f'dataset "{self.name}"!')

        return self._aux_data_definitions[name]

    def set_aux_data_definition(
            self,
            name,
            pathfilenames,
    ):
        """Sets the files of the auxiliary data definition, which has the given
        name.

        Parameters
        ----------
        name : str
            The name of the auxiliary data definition.
        pathfilenames : str | sequence of str
            The file name(s) (including paths) of the data file(s).
        """
        name = str_cast(
            name,
            'The name argument must be cast-able to type str! '
            f'Its current type is {classname(name)}.')

        pathfilenames = list_of_cast(
            str,
            pathfilenames,
            'The pathfilenames argument must be of type str or a sequence '
            f'of str! Its current type is {classname(pathfilenames)}.')

        if name not in self._aux_data_definitions:
            raise KeyError(
                f'The auxiliary data definition "{name}" is not defined '
                f'for dataset "{self.name}"! Use add_aux_data_definition '
                'instead!')

        self._aux_data_definitions[name] = pathfilenames

    def remove_aux_data_definition(
            self,
            name,
    ):
        """Removes the auxiliary data definition from the dataset.

        Parameters
        ----------
        name : str
            The name of the data definition that should get removed.
        """
        if name not in self._aux_data_definitions:
            raise KeyError(
                f'The auxiliary data definition "{name}" does not exist in '
                f'dataset "{self.name}", nothing to remove!')

        self._aux_data_definitions.pop(name)

    def remove_aux_data_definitions(
            self,
            names,
    ):
        """Removes the auxiliary data definition from the dataset.

        Parameters
        ----------
        names : sequence of str
            The names of the data definitions that should get removed.
        """
        for name in names:
            self.remove_aux_data_definition(
                name=name)

    def add_aux_data(
            self,
            name,
            data,
    ):
        """Adds the given data as auxiliary data to this data set.

        Parameters
        ----------
        name : str
            The name under which the auxiliary data will be stored.
        data : unspecified
            The data that should get stored. This can be of any data type.

        Raises
        ------
        KeyError
            If auxiliary data is already stored under the given name.
        """
        name = str_cast(
            name,
            'The name argument must be cast-able to type str!')

        if name in self._aux_data:
            raise KeyError(
                f'The auxiliary data "{name}" is already defined for dataset '
                f'"{self.name}"!')

        self._aux_data[name] = data

    def get_aux_data(
            self,
            name,
            default=None,
    ):
        """Retrieves the auxiliary data that is stored in this data set under
        the given name.

        Parameters
        ----------
        name : str
            The name under which the auxiliary data is stored.
        default : any | None
            If not ``None``, it specifies the returned default value when the
            auxiliary data does not exists.

        Returns
        -------
        data : unspecified
            The retrieved auxiliary data.

        Raises
        ------
        KeyError
            If no auxiliary data is stored with the given name and no default
            value was specified.
        """
        name = str_cast(
            name,
            'The name argument must be cast-able to type str!')

        if name not in self._aux_data:
            if default is not None:
                return default
            raise KeyError(
                f'The auxiliary data "{name}" is not defined for dataset '
                f'"{self.name}"!')

        return self._aux_data[name]

    def remove_aux_data(
            self,
            name,
    ):
        """Removes the auxiliary data that is stored in this data set under
        the given name.

        Parameters
        ----------
        name : str
            The name of the dataset that should get removed.
        """
        if name not in self._aux_data:
            raise KeyError(
                f'The auxiliary data "{name}" is not defined for dataset '
                f'"{self.name}", nothing to remove!')

        self._aux_data.pop(name)


class DatasetCollection(
        object):
    """The DatasetCollection class describes a collection of different datasets.

    New datasets can be added via the add-assign operator (+=), which calls
    the ``add_datasets`` method.
    """
    def __init__(
            self,
            name,
            description=''):
        """Creates a new DatasetCollection instance.

        Parameters
        ----------
        name : str
            The name of the collection.
        description : str
            The (longer) description of the dataset collection.
        """
        self.name = name
        self.description = description

        self._datasets = dict()

    @property
    def name(self):
        """The name (str) of the dataset collection.
        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                'The name of the dataset collection must be of type str!')
        self._name = name

    @property
    def description(self):
        """The (longer) description of the dataset collection.
        """
        return self._description

    @description.setter
    def description(self, description):
        if not isinstance(description, str):
            raise TypeError(
                'The description of the dataset collection must be of type '
                'str!')
        self._description = description

    @property
    def dataset_names(self):
        """The list of names of the assigned datasets.
        """
        return sorted(self._datasets.keys())

    @property
    def version(self):
        """(read-only) The version number of the datasets collected by this
        dataset collection.
        """
        ds_name = list(self._datasets.keys())[0]
        return self._datasets[ds_name].version

    @property
    def verqualifiers(self):
        """(read-only) The dictionary holding the version qualifiers of the
        datasets collected by this dataset collection.
        """
        ds_name = list(self._datasets.keys())[0]
        return self._datasets[ds_name].verqualifiers

    def __getitem__(
            self,
            key,
    ):
        """Implementation of the access operator ``[key]``.

        Parameters
        ----------
        key : str | sequence of str
            The name or names of the dataset(s) that should get retrieved from
            this dataset collection.

        Returns
        -------
        datasets : instance of Dataset | list of instance of Dataset
            The dataset instance or the list of dataset instances corresponding
            to the given key.
        """
        if not issequence(key):
            return self.get_dataset(key)

        if not issequenceof(key, str):
            raise TypeError(
                'The key for the access operator must be an instance of str or '
                'a sequence of str instances!')

        datasets = [
            self.get_dataset(name)
            for name in key
        ]

        return datasets

    def __iadd__(self, ds):
        """Implementation of the ``self += dataset`` and
        ``self += (dataset1, dataset2, ...)`` operations to add one or several
        Dataset objects to this dataset collection.
        """
        self.add_datasets(ds)

        return self

    def __str__(self):
        """Implementation of the pretty string representation of the
        DatasetCollection instance. It shows the available datasets.
        """
        lines = f'DatasetCollection "{self.name}"\n'
        lines += "-"*display.PAGE_WIDTH + "\n"
        lines += "Description:\n" + self.description + "\n"
        lines += "Available datasets:\n"

        for name in self.dataset_names:
            lines += '\n'
            lines += display.add_leading_text_line_padding(
                2, str(self._datasets[name]))

        return lines

    def add_aux_data(
            self,
            name,
            data,
    ):
        """Adds the given data as auxiliary data to all datasets of this
        dataset collection.

        Parameters
        ----------
        name : str
            The name under which the auxiliary data will be stored.
        data : unspecified
            The data that should get stored. This can be of any data type.

        Raises
        ------
        ValueError
            If no datasets have been added to this dataset collection yet.
        KeyError
            If auxiliary data is already stored under the given name.
        """
        if len(self._datasets) == 0:
            raise ValueError(
                f'The dataset collection "{self.name}" has no datasets added '
                'yet!')

        for dataset in self._datasets.values():
            dataset.add_aux_data(
                name=name,
                data=data)

    def add_datasets(
            self,
            datasets,
    ):
        """Adds the given Dataset object(s) to this dataset collection.

        Parameters
        ----------
        datasets : instance of Dataset | sequence of instance of Dataset
            The instance of Dataset or the sequence of instance of Dataset that
            should be added to the dataset collection.

        Returns
        -------
        self : instance of DatasetCollection
            This instance of DatasetCollection in order to be able to chain
            several ``add_datasets`` calls.
        """
        if not issequence(datasets):
            datasets = [datasets]

        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    'The dataset object must be a sub-class of Dataset!')

            if dataset.name in self._datasets:
                raise KeyError(
                    f'Dataset "{dataset.name}" already exists!')

            self._datasets[dataset.name] = dataset

        return self

    def remove_dataset(
            self,
            name,
    ):
        """Removes the given dataset from the collection.

        Parameters
        ----------
        name : str
            The name of the dataset that should get removed.
        """
        if name not in self._datasets:
            raise KeyError(
                f'Dataset "{name}" is not part of the dataset collection '
                f'"{self.name}", nothing to remove!')

        self._datasets.pop(name)

    def get_dataset(
            self,
            name,
    ):
        """Retrieves a Dataset object from this dataset collection.

        Parameters
        ----------
        name : str
            The name of the dataset.

        Returns
        -------
        dataset : Dataset instance
            The Dataset object holding all the information about the dataset.

        Raises
        ------
        KeyError
            If the data set of the given name is not present in this data set
            collection.
        """
        if name not in self._datasets:
            ds_names = '", "'.join(self.dataset_names)
            ds_names = '"'+ds_names+'"'
            raise KeyError(
                f'The dataset "{name}" is not part of the dataset collection '
                f'"{self.name}"! Possible dataset names are: {ds_names}!')

        return self._datasets[name]

    def get_datasets(
            self,
            names,
    ):
        """Retrieves a list of Dataset objects from this dataset collection.

        Parameters
        ----------
        names : str | sequence of str
            The name or sequence of names of the datasets to retrieve.

        Returns
        -------
        datasets : list of Dataset instances
            The list of Dataset instances for the given list of data set names.

        Raises
        ------
        KeyError
            If one of the requested data sets is not present in this data set
            collection.
        """
        if not issequence(names):
            names = [names]
        if not issequenceof(names, str):
            raise TypeError(
                'The names argument must be an instance of str or a sequence '
                'of str instances!')

        datasets = []
        for name in names:
            datasets.append(self.get_dataset(name))

        return datasets

    def set_exp_field_name_renaming_dict(
            self,
            d,
    ):
        """Sets the dictionary with the data field names of the experimental
        data that needs to be renamed just after loading the data. The
        dictionary will be set to all added data sets.

        Parameters
        ----------
        d : dict
            The dictionary with the old field names as keys and the new field
            names as values.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.exp_field_name_renaming_dict = d

    def set_mc_field_name_renaming_dict(
            self,
            d,
    ):
        """Sets the dictionary with the data field names of the monte-carlo
        data that needs to be renamed just after loading the data. The
        dictionary will be set to all added data sets.

        Parameters
        ----------
        d : dict
            The dictionary with the old field names as keys and the new field
            names as values.
        """
        for (dsname, dataset) in self._datasets.items():
            dataset.mc_field_name_renaming_dict = d

    def set_dataset_prop(
            self,
            name,
            value,
    ):
        """Sets the given property to the given name for all data sets of this
        data set collection.

        Parameters
        ----------
        name : str
            The name of the property.
        value : object
            The value to set for the given property.

        Raises
        ------
        KeyError
            If the given property does not exist in the data sets.
        """
        for (dsname, dataset) in self._datasets.items():
            if not hasattr(dataset, name):
                raise KeyError(
                    f'The dataset "{dsname}" does not have a property named '
                    f'"{name}"!')
            setattr(dataset, name, value)

    def define_binning(
            self,
            name,
            binedges,
    ):
        """Defines a binning definition and adds it to all the datasets of this
        dataset collection.

        Parameters
        ----------
        name : str
            The name of the binning definition.
        binedges : sequence
            The sequence of the bin edges, that should be used for the binning.
        """
        for dataset in self._datasets.values():
            dataset.define_binning(name, binedges)

    def add_data_preparation(
            self,
            func,
    ):
        """Adds the data preparation function to all the datasets of this
        dataset collection.

        Parameters
        ----------
        func : callable
            The object with call signature ``__call__(data)`` that will prepare
            the data after it was loaded. The argument 'data' is the DatasetData
            instance holding the experimental and monte-carlo data.
            This function must alter the properties of the DatasetData instance.
        """
        for dataset in self._datasets.values():
            dataset.add_data_preparation(func)

    def remove_data_preparation(
            self,
            key=-1,
    ):
        """Removes data preparation function from all the datasets of this
        dataset collection.

        Parameters
        ----------
        key : str, int, optional
            The name or the index of the data preparation function that should
            be removed. Default value is ``-1``, i.e. the last added function.

        Raises
        ------
        TypeError
            If the type of the key argument is invalid.
        IndexError
            If the given key is out of range.
        KeyError
            If the data preparation function cannot be found.
        """
        for dataset in self._datasets.values():
            dataset.remove_data_preparation(key=key)

    def update_version_qualifiers(
            self,
            verqualifiers,
    ):
        """Updates the version qualifiers of all datasets of this dataset
        collection.
        """
        for dataset in self._datasets.values():
            dataset.update_version_qualifiers(verqualifiers)

    def load_data(
            self,
            livetime=None,
            tl=None,
            ppbar=None,
            **kwargs,
    ):
        """Loads the data of all data sets of this data set collection.

        Parameters
        ----------
        livetime : float | dict of str => float | None
            If not None, uses this livetime (in days) as livetime for (all) the
            DatasetData instances, otherwise uses the live time from the Dataset
            instance. If a dictionary of data set names and floats is given, it
            defines the livetime for the individual data sets.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time the data load
            operation.
        ppbar : instance of ProgressBar | None
            The optional parent progress bar.
        **kwargs
            Additional keyword arguments are passed to the
            :meth:`~skyllh.core.dataset.Dataset.load_data` method of the
            individual datasets.

        Returns
        -------
        data_dict : dictionary str => instance of DatasetData
            The dictionary with the DatasetData instance holding the data of
            an individual data set as value and the data set's name as key.
        """
        if not isinstance(livetime, dict):
            livetime_dict = dict()
            for (dsname, dataset) in self._datasets.items():
                livetime_dict[dsname] = livetime
            livetime = livetime_dict

        if len(livetime) != len(self._datasets):
            raise ValueError(
                'The livetime argument must be None, a single float, or a '
                f'dictionary with {len(self._datasets)} str:float entries! '
                f'Currently the dictionary has {len(livetime)} entries.')

        pbar = ProgressBar(len(self._datasets), parent=ppbar).start()
        data_dict = dict()
        for (dsname, dataset) in self._datasets.items():
            data_dict[dsname] = dataset.load_data(
                livetime=livetime[dsname],
                tl=tl,
                **kwargs)
            pbar.increment()
        pbar.finish()

        return data_dict


class DatasetData(
        object):
    """This class provides the container for the actual experimental and
    monte-carlo data.
    """
    def __init__(
            self,
            data_exp,
            data_mc,
            livetime,
            **kwargs,
    ):
        """Creates a new DatasetData instance.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray | None
            The instance of DataFieldRecordArray holding the experimental data.
            This can be None for a MC-only study.
        data_mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monte-carlo data.
        livetime : float
            The integrated livetime in days of the data.
        """
        super().__init__(**kwargs)

        self.exp = data_exp
        self.mc = data_mc
        self.livetime = livetime

    @property
    def exp(self):
        """The DataFieldRecordArray instance holding the experimental data.
        This is None, if there is no experimental data available.
        """
        return self._exp

    @exp.setter
    def exp(self, data):
        if data is not None:
            if not isinstance(data, DataFieldRecordArray):
                raise TypeError(
                    'The exp property must be an instance of '
                    'DataFieldRecordArray!')
        self._exp = data

    @property
    def mc(self):
        """The DataFieldRecordArray instance holding the monte-carlo data.
        This is None, if there is no monte-carlo data available.
        """
        return self._mc

    @mc.setter
    def mc(self, data):
        if data is not None:
            if not isinstance(data, DataFieldRecordArray):
                raise TypeError(
                    'The mc property must be an instance of '
                    'DataFieldRecordArray!')
        self._mc = data

    @property
    def livetime(self):
        """The integrated livetime in days of the data.
        This is None, if there is no live-time provided.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if lt is not None:
            lt = float_cast(
                lt,
                'The livetime property must be cast-able to type float!')
        self._livetime = lt

    @property
    def exp_field_names(self):
        """(read-only) The list of field names present in the experimental data.
        This is an empty list if there is no experimental data available.
        """
        if self._exp is None:
            return []
        return self._exp.field_name_list

    @property
    def mc_field_names(self):
        """(read-only) The list of field names present in the monte-carlo data.
        """
        return self._mc.field_name_list


def assert_data_format(
        dataset,
        data,
):
    """Checks the format of the experimental and monte-carlo data.

    Parameters
    ----------
    dataset : instance of Dataset
        The instance of Dataset describing the dataset and holding the local
        configuration.
    data : instance of DatasetData
        The instance of DatasetData holding the actual experimental and
        simulation data of the data set.

    Raises
    ------
    KeyError
        If a required data field is missing.
    """
    cfg = dataset.cfg

    def _get_missing_keys(keys, required_keys):
        missing_keys = []
        for reqkey in required_keys:
            if reqkey not in keys:
                missing_keys.append(reqkey)
        return missing_keys

    if data.exp is not None:
        missing_exp_keys = _get_missing_keys(
            data.exp.field_name_list,
            DataFields.get_joint_names(
                datafields=cfg['datafields'],
                stages=(
                    DFS.ANALYSIS_EXP
                )
            )
        )
        if len(missing_exp_keys) != 0:
            raise KeyError(
                'The following data fields are missing for the experimental '
                f'data of dataset "{dataset.name}": '
                ', '.join(missing_exp_keys))

    if data.mc is not None:
        missing_mc_keys = _get_missing_keys(
            data.mc.field_name_list,
            DataFields.get_joint_names(
                datafields=cfg['datafields'],
                stages=(
                    DFS.ANALYSIS_EXP |
                    DFS.ANALYSIS_MC
                )
            )
        )
        if len(missing_mc_keys) != 0:
            raise KeyError(
                'The following data fields are missing for the monte-carlo '
                f'data of dataset "{dataset.name}": '
                ', '.join(missing_mc_keys))

    if data.livetime is None:
        raise ValueError(
            f'No livetime was specified for dataset "{dataset.name}"!')


def remove_events(
        data_exp,
        mjds,
):
    """Utility function to remove events having the specified MJD time stamps.

    Parameters
    ----------
    data_exp : instance of DataFieldRecordArray
        The instance of DataFieldRecordArray holding the experimental data
        events.
    mjds : float | array of floats
        The MJD time stamps of the events, that should get removed from the
        experimental data array.

    Returns
    -------
    data_exp : instance of DataFieldRecordArray
        The instance of DataFieldRecordArray holding the experimental data
        events with the specified events removed.
    """
    mjds = np.atleast_1d(mjds)

    mask = np.zeros((len(data_exp)), dtype=np.bool_)
    for time in mjds:
        m = data_exp['time'] == time
        if np.count_nonzero(m) > 1:
            raise LookupError(
                f'The MJD time stamp {time} is not unique!')
        mask |= m
    data_exp = data_exp[~mask]

    return data_exp


def generate_base_path(
        default_base_path,
        base_path=None,
):
    """Generates the base path. If base_path is None, default_base_path is used.

    Parameters
    ----------
    default_base_path : str
        The default base path if base_path is None.
    base_path : str | None
        The user-specified base path.

    Returns
    -------
    base_path : str
        The generated base path.
    """
    if base_path is None:
        if default_base_path is None:
            raise ValueError(
                'The default_base_path argument must not be None, when the '
                'base_path argument is set to None!')
        base_path = default_base_path

    return base_path


def generate_sub_path(
        sub_path_fmt,
        version,
        verqualifiers,
):
    """Generates the sub path of the dataset based on the given sub path format.

    Parameters
    ----------
    sub_path_fmt : str
        The format string of the sub path.
    version : int
        The version of the dataset.
    verqualifiers : dict
        The dictionary holding the version qualifiers of the dataset.

    Returns
    -------
    sub_path : str
        The generated sub path.
    """
    fmt_dict = dict(
        [('version', version)] + list(verqualifiers.items())
    )
    sub_path = sub_path_fmt.format(**fmt_dict)

    return sub_path


def generate_data_file_root_dir(
        default_base_path,
        default_sub_path_fmt,
        version,
        verqualifiers,
        base_path=None,
        sub_path_fmt=None,
):
    """Generates the root directory of the data files based on the given base
    path and sub path format. If base_path is None, default_base_path is used.
    If sub_path_fmt is None, default_sub_path_fmt is used.

    The ``default_sub_path_fmt`` and ``sub_path_fmt`` arguments can contain the
    following wildcards:

        ``{version:d}``
            The version integer number of the dataset.
        ``{<verqualifiers_key>:d}``
            The integer number of the specific version qualifier
            ``'verqualifiers_key'``.

    Parameters
    ----------
    default_base_path : str
        The default base path if base_path is None.
    default_sub_path_fmt : str
        The default sub path format if sub_path_fmt is None.
    version : int
        The version of the data sample.
    verqualifiers : dict
        The dictionary holding the version qualifiers of the data sample.
    base_path : str | None
        The user-specified base path.
    sub_path_fmt : str | None
        The user-specified sub path format.

    Returns
    -------
    root_dir : str
        The generated root directory of the data files. This will have no
        trailing directory separator.
    """
    base_path = generate_base_path(
        default_base_path=default_base_path,
        base_path=base_path)

    if sub_path_fmt is None:
        sub_path_fmt = default_sub_path_fmt

    sub_path = generate_sub_path(
        sub_path_fmt=sub_path_fmt,
        version=version,
        verqualifiers=verqualifiers)

    root_dir = os.path.join(base_path, sub_path)

    len_sep = len(os.path.sep)
    if root_dir[-len_sep:] == os.path.sep:
        root_dir = root_dir[:-len_sep]

    return root_dir


def get_data_subset(
        data,
        livetime,
        t_start,
        t_stop,
):
    """Gets instance of DatasetData and instance of Livetime with data subsets
    between the given time range from ``t_start`` to ``t_stop``.

    Parameters
    ----------
    data : DatasetData
        The DatasetData object.
    livetime : Livetime
        The Livetime object.
    t_start : float
        The MJD start time of the time range to consider.
    t_end : float
        The MJD end time of the time range to consider.

    Returns
    -------
    data_subset : instance of DatasetData
        The instance of DatasetData with subset of the data between the given
        time range from ``t_start`` to ``t_stop``.
    livetime_subset : instance of Livetime
        The instance of Livetime for a subset of the data between the given
        time range from ``t_start`` to ``t_stop``.
    """
    if not isinstance(data, DatasetData):
        raise TypeError(
            'The "data" argument must be of type DatasetData!')
    if not isinstance(livetime, Livetime):
        raise TypeError(
            'The "livetime" argument must be of type Livetime!')

    exp_slice = np.logical_and(
        data.exp['time'] >= t_start,
        data.exp['time'] < t_stop)
    mc_slice = np.logical_and(
        data.mc['time'] >= t_start,
        data.mc['time'] < t_stop)

    data_exp = data.exp[exp_slice]
    data_mc = data.mc[mc_slice]

    uptime_mjd_intervals_arr = livetime.get_uptime_intervals_between(
        t_start, t_stop)
    livetime_subset = Livetime(uptime_mjd_intervals_arr)

    data_subset = DatasetData(
        data_exp=data_exp,
        data_mc=data_mc,
        livetime=livetime_subset.livetime)

    return (data_subset, livetime_subset)
