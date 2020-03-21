# -*- coding: utf-8 -*-

"""This file contains the global configuration dictionary, together with some
convenience utility functions to set different configuration settings.
"""

import os.path
import sys
from typing import Any, Dict, Iterator, KeysView, ItemsView, ValuesView

import yaml
from astropy import units  # type: ignore

from skyllh.core.py import issequenceof

_BASECONFIG = {
    'multiproc': {
        # The number of CPUs to use for functions that allow multi-processing.
        # If this setting is set to an int value in the range [1, N] this
        # setting will be used if a function's local ncpu setting is not
        # specified.
        'ncpu': None
    },
    'debugging': {
        # The default log format.
        'log_format': (
            '%(asctime)s %(processName)s %(name)s %(levelname)s: '
            '%(message)s')
    },
    'project': {
        # The project's working directory.
        'working_directory': '.'
    },
    'repository': {
        # A base path of repository datasets.
        'base_path': None
    },
    # Definition of the internal units to use. These must match with the units
    # from the monto-carlo data files.
    'internal_units': {
        'angle': units.radian,
        'energy': units.GeV,
        'length': units.cm,
        'time': units.s
    },
    'units': {
        'defaults': {
            # Definition of default units used for fluxes.
            'fluxes': {
                'angle': units.radian,
                'energy': units.GeV,
                'length': units.cm,
                'time': units.s
            }
        }
    },
    'dataset': {
        # Define the data field names of the data set's experimental data,
        # that are required by the analysis.
        'analysis_required_exp_field_names': [
            'run', 'ra', 'dec', 'ang_err', 'time', 'log_energy'
        ],
        # Define the data field names of the data set's monte-carlo data,
        # that are required by the analysis.
        'analysis_required_mc_field_names': [
            'true_ra', 'true_dec', 'true_energy', 'mcweight'
        ]
    }
}


class CFG:
    """
    This class holds the global config state

    The class behaves like a dict, delegating all methods of the dict
    interface to the underlying config dictionary
    """

    __config = dict(_BASECONFIG)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> None:
        """
        Update config with yaml file

        Parameters:
            yaml_file: str
                path to yaml file
        """

        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        cls.__config.update(yaml_config)

    @classmethod
    def from_dict(cls, user_dict: Dict[Any, Any]) -> None:
        """
        Creates a config from dictionary

        Parameters:
            user_dict: dict

        Returns:
            dict
        """
        cls.__config.update(user_dict)

    @classmethod
    def __getitem__(cls, key: Any) -> Any:
        """Get a config value"""
        if key not in cls.__config:
            raise KeyError("Key {} not in config".format(key))
        return cls.__config[key]

    @classmethod
    def __setitem__(cls, key: Any, val: Any) -> None:
        """Set a config value"""
        cls.__config[key] = val

    @classmethod
    def __iter__(cls) -> Iterator[Any]:
        """Get the underlying dicts iterator"""
        return cls.__config.__iter__()

    @classmethod
    def __contains__(cls, key: Any) -> bool:
        """Check if key is in underlying dict"""
        return key in cls.__config

    @classmethod
    def keys(cls) -> KeysView[Any]:
        """Get the underlying keys view"""
        return cls.__config.keys()

    @classmethod
    def items(cls) -> ItemsView[Any, Any]:
        """Get the underlying items view"""
        return cls.__config.items()

    @classmethod
    def values(cls) -> ValuesView[Any]:
        """Get the underlying values view"""
        return cls.__config.values()

    @classmethod
    def get(cls, key: Any) -> Any:
        """Delegates get call to the underlying dict"""
        return cls.__config.get(key)

    @classmethod
    def __eq__(cls, other: Any) -> bool:
        """Check if underlying dict is equal to `other`"""
        return cls.__eq__(other)

    @classmethod
    def __ne__(cls, other: Any) -> bool:
        """Check if underlying dict is not equal to `other`"""
        return cls.__ne__(other)


def set_internal_units(
        angle_unit=None, energy_unit=None, length_unit=None, time_unit=None):
    """Sets the units used internally to compute quantities. These units must
    match the units used in the monte-carlo files.

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
    """
    if(angle_unit is not None):
        if(not isinstance(angle_unit, units.UnitBase)):
            raise TypeError(
                'The angle_unit argument must be an instance of '
                'astropy.units.UnitBase!')
        CFG['internal_units']['angle'] = angle_unit

    if(energy_unit is not None):
        if(not isinstance(energy_unit, units.UnitBase)):
            raise TypeError(
                'The energy_unit argument must be an instance of '
                'astropy.units.UnitBase!')
        CFG['internal_units']['energy'] = energy_unit

    if(length_unit is not None):
        if(not isinstance(length_unit, units.UnitBase)):
            raise TypeError(
                'The length_unit argument must be an instance of '
                'astropy.units.UnitBase!')
        CFG['internal_units']['length'] = length_unit

    if(time_unit is not None):
        if(not isinstance(time_unit, units.UnitBase)):
            raise TypeError(
                'The time_unit argument must be an instance of '
                'astropy.units.UnitBase!')
        CFG['internal_units']['time'] = time_unit


def set_wd(path):
    """Sets the project's working directory configuration variable and adds it
    to the Python path variable.

    Parameters
    ----------
    path : str
        The path of the project's working directory. This can be a path
        relative to the path given by ``os.path.getcwd``, the current
        working directory of the program.

    Returns
    -------
    wd : str
        The project's working directory.
    """
    if(CFG['project']['working_directory'] in sys.path):
        sys.path.remove(CFG['project']['working_directory'])

    wd = os.path.abspath(path)
    CFG['project']['working_directory'] = wd
    sys.path.insert(0, wd)

    return wd


def set_analysis_required_exp_data_field_names(fieldnames):
    """Sets the data field names of the experimental data that are required by
    the analysis.

    Parameters
    ----------
    fieldnames : str | sequence of str
        The field name or sequence of field names for the experimental data.
    """
    if(isinstance(fieldnames, str)):
        fieldnames = [fieldnames]
    elif(not issequenceof(fieldnames, str)):
        raise TypeError(
            'The fieldnames argument must be an instance of str '
            'or a sequence of type str instances!')

    CFG['dataset']['analysis_required_exp_field_names'] = list(set(fieldnames))


def set_analysis_required_mc_data_field_names(fieldnames):
    """Sets the data field names of the monte-carlo data that are required by
    the analysis.

    Parameters
    ----------
    fieldnames : str | sequence of str
        The field name or sequence of field names for the monte-carlo data.
    """
    if(isinstance(fieldnames, str)):
        fieldnames = [fieldnames]
    elif(not issequenceof(fieldnames, str)):
        raise TypeError(
            'The fieldnames argument must be an instance of str '
            'or a sequence of type str instances!')

    CFG['dataset']['analysis_required_mc_field_names'] = list(set(fieldnames))
