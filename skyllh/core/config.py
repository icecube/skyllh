# -*- coding: utf-8 -*-

"""This file contains the global configuration dictionary, together with some
convenience utility functions to set different configuration settings.
"""

import copy
import os.path
import sys

from astropy import (
    units,
)

from typing import (
    Any,
    Dict,
)

from skyllh.core import (
    tool,
)
from skyllh.core.datafields import (
    DataFieldStages as DFS,
)
from skyllh.core.py import (
    classname,
)


_BASECONFIG = {
    'multiproc': {
        # The number of CPUs to use for functions that allow multi-processing.
        # If this setting is set to an int value in the range [1, N] this
        # setting will be used if a function's local ncpu setting is not
        # specified.
        'ncpu': None,
    },
    'debugging': {
        # The default log format.
        'log_format': (
            '%(asctime)s %(processName)s %(name)s %(levelname)s: '
            '%(message)s'),
        # Flag if detailed debug log messages, i.e. trace log messages, should
        # get generated. This is good for debugging but bad for performance.
        'enable_tracing': False,
    },
    'project': {
        # The project's working directory.
        'working_directory': '.',
    },
    'repository': {
        # A base path of repository datasets.
        'base_path': None,
        'download_from_origin': True,
    },
    'units': {
        # Definition of the internal units to use. These must match with the
        # units of the monte-carlo data files.
        'internal': {
            'angle': units.radian,
            'energy': units.GeV,
            'length': units.cm,
            'time': units.s,
        },
        'defaults': {
            # Definition of default units used for fluxes.
            'fluxes': {
                'angle': units.radian,
                'energy': units.GeV,
                'length': units.cm,
                'time': units.s,
            }
        }
    },
    'datafields': {
        'run': DFS.ANALYSIS_EXP,
        'ra': DFS.ANALYSIS_EXP,
        'dec': DFS.ANALYSIS_EXP,
        'ang_err': DFS.ANALYSIS_EXP,
        'time': DFS.ANALYSIS_EXP,
        'log_energy': DFS.ANALYSIS_EXP,
        'true_ra': DFS.ANALYSIS_MC,
        'true_dec': DFS.ANALYSIS_MC,
        'true_energy': DFS.ANALYSIS_MC,
        'mcweight': DFS.ANALYSIS_MC,
    },
    # Flag if specific calculations in the core module can be cached.
    'caching': {
        'pdf': {
            'MultiDimGridPDF': False,
        }
    }
}


class Config(
        dict,
):
    """This class, derived from dict, holds the a local configuration state.
    """

    def __init__(
            self,
    ) -> None:
        """Initializes a new Config instance holding the base configuration.
        """
        super().__init__(copy.deepcopy(_BASECONFIG))

    @classmethod
    @tool.requires('yaml')
    def from_yaml(
            cls,
            pathfilename: str,
    ):
        """Creates a new instance of Config holding the base configuration and
        updated by the configuration items contained in the yaml file using the
        :meth:`dict.update` method.

        Parameters
        ----------
        pathfilename: str | None
            Path and filename to the yaml file containing the to-be-updated
            configuration items.
            If set to ``None``, nothing is done.

        Returns
        -------
        cfg : instance of Config
            The instance of Config holding the base configuration and updated by
            the configuration given in the yaml file.
        """
        cfg = cls()

        if pathfilename is None:
            return cfg

        yaml = tool.get('yaml')

        user_config_dict = yaml.load(
            open(pathfilename),
            Loader=yaml.SafeLoader)
        cfg.update(user_config_dict)

        return cfg

    @classmethod
    def from_dict(
            cls,
            user_dict: Dict[str, Any],
    ):
        """Creates a new instance of Config holding the base configuration and
        updated by the given configuration dictionary using the
        :meth:`dict.update` method.

        Parameters
        ----------
        user_dict: dict
            The dictionary containing the to-be-updated configuration items.

        Returns
        -------
        cfg : instance of Config
            The instance of Config holding the base configuration and updated by
            the given configuration dictionary.
        """
        cfg = cls()

        cfg.update(user_dict)

        return cfg

    @property
    def is_tracing_enabled(self):
        """``True``, if tracing mode is enabled, ``False`` otherwise.
        """
        return self['debugging']['enable_tracing']

    def disable_tracing(
            self,
    ):
        """Disables the tracing mode of SkyLLH.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        self['debugging']['enable_tracing'] = False

        return self

    def enable_tracing(
            self,
    ):
        """Enables the tracing mode of SkyLLH.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        self['debugging']['enable_tracing'] = True

        return self

    def get_wd(
            self,
    ):
        """Retrieves the absolute path to the working directory as configured in
        this configuration.

        Returns
        -------
        wd : str
            The absolute path to the project's working directory.
        """
        wd = os.path.abspath(self['project']['working_directory'])

        return wd

    def set_enable_tracing(
            self,
            flag,
    ):
        """Sets the setting for tracing.

        Parameters
        ----------
        flag : bool
            The flag if tracing should be enabled (``True``) or disabled
            (``False``).

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        self['debugging']['enable_tracing'] = flag

        return self

    def set_internal_units(
            self,
            angle_unit=None,
            energy_unit=None,
            length_unit=None,
            time_unit=None,
    ):
        """Sets the units used internally to compute quantities. These units
        must match the units used in the monte-carlo files.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The internal unit that should be used for angles.
            If set to ``None``, the unit is not changed.
        energy_unit : instance of astropy.units.UnitBase | None
            The internal unit that should be used for energy.
            If set to ``None``, the unit is not changed.
        length_unit : instance of astropy.units.UnitBase | None
            The internal unit that should be used for length.
            If set to ``None``, the unit is not changed.
        time_unit : instance of astropy.units.UnitBase | None
            The internal unit that should be used for time.
            If set to ``None``, the unit is not changed.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        if angle_unit is not None:
            if not isinstance(angle_unit, units.UnitBase):
                raise TypeError(
                    'The angle_unit argument must be an instance of '
                    'astropy.units.UnitBase!')
            self['units']['internal']['angle'] = angle_unit

        if energy_unit is not None:
            if not isinstance(energy_unit, units.UnitBase):
                raise TypeError(
                    'The energy_unit argument must be an instance of '
                    'astropy.units.UnitBase!')
            self['units']['internal']['energy'] = energy_unit

        if length_unit is not None:
            if not isinstance(length_unit, units.UnitBase):
                raise TypeError(
                    'The length_unit argument must be an instance of '
                    'astropy.units.UnitBase!')
            self['units']['internal']['length'] = length_unit

        if time_unit is not None:
            if not isinstance(time_unit, units.UnitBase):
                raise TypeError(
                    'The time_unit argument must be an instance of '
                    'astropy.units.UnitBase!')
            self['units']['internal']['time'] = time_unit

        return self

    def set_ncpu(
            self,
            ncpu,
    ):
        """Sets the global setting for the number of CPUs to use, when
        parallelization is available.

        Parameters
        ----------
        ncpu : int
            The number of CPUs.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        self['multiproc']['ncpu'] = ncpu

        return self

    def set_wd(
            self,
            path=None,
    ):
        """Sets the project's working directory configuration variable and adds
        it to the Python path variable.

        Parameters
        ----------
        cfg : instance of Config
            The instance of Config holding the local configuration.
        path : str | None
            The path of the project's working directory. This can be a path
            relative to the path given by ``os.path.getcwd``, the current
            working directory of the program.
            If set to ``None``, the path is taken from the working directory
            setting of the given configuration.

        Returns
        -------
        wd : str
            The absolute path to the project's working directory.
        """
        if path is None:
            path = self['project']['working_directory']

        if self['project']['working_directory'] in sys.path:
            sys.path.remove(self['project']['working_directory'])

        wd = os.path.abspath(path)
        self['project']['working_directory'] = wd
        sys.path.insert(0, wd)

        return wd

    def to_internal_time_unit(
            self,
            time_unit,
    ):
        """Calculates the conversion factor from the given time unit to the
        internal time unit specified by this local configuration.

        Parameters
        ----------
        time_unit : instance of astropy.units.UnitBase
            The time unit from which to convert to the internal time unit.
        """
        internal_time_unit = self['units']['internal']['time']
        factor = time_unit.to(internal_time_unit)

        return factor

    def wd_filename(self, filename):
        """Generates the fully qualified file name under the project's working
        directory of the given file.

        Parameters
        ----------
        filename : str
            The name of the file for which to generate the working directory
            path file name.

        Returns
        -------
        pathfilename : str
            The generated fully qualified path file name of ``filename`` with
            the project's working directory prefixed.
        """
        pathfilename = os.path.join(self.get_wd(), filename)

        return pathfilename


class HasConfig(
        object,
):
    """Classifier class defining the cfg property. Classes that derive from
    this class indicate, that they hold an instance of Config.
    """

    def __init__(
            self,
            cfg,
            *args,
            **kwargs,
    ):
        """Creates a new instance having the property ``cfg``.

        Parameters
        ----------
        cfg : instance of Config
            The instance of Config holding the local configuration.
        """
        super().__init__(
            *args,
            **kwargs)

        self.cfg = cfg

    @property
    def cfg(self):
        """The instance of Config holding the local configuration.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, c):
        if not isinstance(c, Config):
            raise TypeError(
                'The cfg property must be an instance of Config! '
                f'Currently its type is {classname(c)}!')
        self._cfg = c
