import warnings

from skyllh.datasets.i3 import (
    PublicData_10y_ps,
    PublicData_10y_ps_wMC,
    PublicData_14y_ps,
    TestData,
)

_DEPRECATED_KEYS = {
    'PublicData_10y_ps': 'IceTracks-DR1',
    'PublicData_10y_ps_wMC': 'IceTracks-DR1_wMC',
    'PublicData_14y_ps': 'IceTracks-DR2',
}


class _DataSamplesDict(dict):
    """Dict subclass that emits DeprecationWarnings for renamed data sample keys."""

    def __getitem__(self, key):
        if key in _DEPRECATED_KEYS:
            new_key = _DEPRECATED_KEYS[key]
            warnings.warn(
                f'Data sample name "{key}" is deprecated and will be removed in '
                f'a future release. Use "{new_key}" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            key = new_key
        return super().__getitem__(key)

    def __contains__(self, key):
        return super().__contains__(_DEPRECATED_KEYS.get(key, key))


data_samples = _DataSamplesDict(
    {
        'IceTracks-DR1': PublicData_10y_ps,
        'IceTracks-DR2': PublicData_14y_ps,
        # For internal use:
        'IceTracks-DR1_wMC': PublicData_10y_ps_wMC,
        'TestData': TestData,
    }
)
