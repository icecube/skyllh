# -*- coding: utf-8 -*-

"""This file contains the global configuration dictionary, together with some
convenience utility functions to set different configuration settings.
"""

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
from skyllh.core.py import (
    issequenceof,
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
    },
    'units': {
        # Definition of the internal units to use. These must match with the
        # units of the monto-carlo data files.
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
    'dataset': {
        # Define the data field names of the data set's experimental data,
        # that are required by the analysis.
        'analysis_required_exp_field_names': [
            'run',
            'ra',
            'dec',
            'ang_err',
            'time',
            'log_energy',
        ],
        # Define the data field names of the data set's monte-carlo data,
        # that are required by the analysis.
        'analysis_required_mc_field_names': [
            'true_ra',
            'true_dec',
            'true_energy',
            'mcweight',
        ],
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
        super().__init__(_BASECONFIG)

    @tool.requires('yaml')
    @classmethod
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
            Path and filename to the yaml file containg the to-be-updated
            configuration items.
            If set to ``None``, nothing is done.

        Returns
        -------
        cfg : instance of Config
            The instance of Config holding the base configuration and updated by
            the configuration given in the yaml file.
        """
        if pathfilename is None:
            return

        yaml = tool.get('yaml')

        user_config_dict = yaml.load(
            open(pathfilename),
            Loader=yaml.SafeLoader)

        cfg = cls()

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
            The dictionary containg the to-be-updated configuration items.

        Returns
        -------
        cfg : instance of Config
            The instance of Config holding the base configuration and updated by
            the given configuration dictionary.
        """
        cfg = cls()

        cfg.update(user_dict)

        return cfg

    def add_analysis_required_exp_data_field_names(
            self,
            fieldnames,
    ):
        """Adds the given data field names to the set of data field names of the
        experimental data that are required by the analysis.

        Parameters
        ----------
        fieldnames : str | sequence of str
            The field name or sequence of field names that should get added for
            the experimental data.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]
        elif not issequenceof(fieldnames, str):
            raise TypeError(
                'The fieldnames argument must be an instance of str '
                'or a sequence of type str instances!')

        self['dataset']['analysis_required_exp_field_names'] = list(set(
            self['dataset']['analysis_required_exp_field_names'] + fieldnames))

        return self

    def add_analysis_required_mc_data_field_names(
            self,
            fieldnames,
    ):
        """Adds the given data field names to the set of data field names of the
        monte-carlo data that are required by the analysis.

        Parameters
        ----------
        fieldnames : str | sequence of str
            The field name or sequence of field names that should get added for
            the monto-carlo data.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]
        elif not issequenceof(fieldnames, str):
            raise TypeError(
                'The fieldnames argument must be an instance of str '
                'or a sequence of type str instances!')

        self['dataset']['analysis_required_mc_field_names'] = list(set(
            self['dataset']['analysis_required_mc_field_names'] + fieldnames))

        return self

    def get_wd(
            self,
    ):
        """Retrieves the absolut path to the working directoy as configured in
        this configuration.

        Returns
        -------
        wd : str
            The absolut path to the project's working directory.
        """
        wd = os.path.abspath(self['project']['working_directory'])

        return wd

    def set_analysis_required_exp_data_field_names(
            self,
            fieldnames,
    ):
        """Sets the data field names of the experimental data that are required
        by the analysis.

        Parameters
        ----------
        fieldnames : str | sequence of str
            The field name or sequence of field names for the experimental data.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]
        elif not issequenceof(fieldnames, str):
            raise TypeError(
                'The fieldnames argument must be an instance of str '
                'or a sequence of type str instances!')

        self['dataset']['analysis_required_exp_field_names'] = list(set(
            fieldnames))

        return self

    def set_analysis_required_mc_data_field_names(
            self,
            fieldnames,
    ):
        """Sets the data field names of the monte-carlo data that are required
        by the analysis.

        Parameters
        ----------
        fieldnames : str | sequence of str
            The field name or sequence of field names for the monte-carlo data.

        Returns
        -------
        self : instance of Config
            The updated instance of Config.
        """
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]
        elif not issequenceof(fieldnames, str):
            raise TypeError(
                'The fieldnames argument must be an instance of str '
                'or a sequence of type str instances!')

        self['dataset']['analysis_required_mc_field_names'] = list(set(
            fieldnames))

        return self

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
            The absolut path to the project's working directory.
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
