from __future__ import annotations #  allows forwards referencing of type hints
from typing import Any, Dict
import yaml

class UserConfig:
    """Class for loading and storing user configurations"""

    def __init__(self, config: Dict[Any, Any]) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, yaml_file: str) -> UserConfig:
        """
        Create a Userconfig from yaml file

        Parameters:
            yaml_file: str
                path to yaml file

        Returns:
            UserConfig
        """

        config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        return cls(config)


    def __getitem__(self, key: str) -> Any:
        """
        Get item from underlying dict
        """
        return self.config[key]

