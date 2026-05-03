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


def _warn_deprecated_key(key, new_key, stacklevel):
    warnings.warn(
        f'Data sample name "{key}" is deprecated and will be removed in a future release. Use "{new_key}" instead.',
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class _DataSamplesDict(dict):
    """Dict subclass that emits DeprecationWarnings for renamed data sample keys."""

    def __init__(self, data, deprecated_keys):
        super().__init__(data)
        self._deprecated_keys = deprecated_keys

    def _resolve_key(self, key):
        if key in self._deprecated_keys:
            new_key = self._deprecated_keys[key]
            # stacklevel=4: _warn_deprecated_key → _resolve_key → __getitem__/get → caller
            _warn_deprecated_key(key, new_key, stacklevel=4)
            return new_key
        return key

    def __getitem__(self, key):
        return super().__getitem__(self._resolve_key(key))

    def get(self, key, default=None):
        return super().get(self._resolve_key(key), default)

    def __contains__(self, key):
        return super().__contains__(self._deprecated_keys.get(key, key))


data_samples = _DataSamplesDict(
    {
        'IceTracks-DR1': PublicData_10y_ps,
        'IceTracks-DR2': PublicData_14y_ps,
        # For internal use:
        'IceTracks-DR1_wMC': PublicData_10y_ps_wMC,
        'TestData': TestData,
    },
    _DEPRECATED_KEYS,
)
