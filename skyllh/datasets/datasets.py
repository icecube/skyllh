import warnings

from .i3 import (
    _DEPRECATED_KEYS as _i3_DEPRECATED_KEYS,
)
from .i3 import (
    data_samples as _i3_data_samples,
)

# Merged deprecated-key registry. When a new experiment package is added,
# extend this dict with its own _DEPRECATED_KEYS.
_DEPRECATED_KEYS = {
    **_i3_DEPRECATED_KEYS,
}


def _warn_deprecated_key(key, new_key, stacklevel):
    warnings.warn(
        f'Data sample name "{key}" is deprecated and will be removed in a future release. Use "{new_key}" instead.',
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class _DataSamplesDict(dict):
    """Dict subclass that emits DeprecationWarnings for renamed data sample keys."""

    def _resolve_key(self, key):
        if key in _DEPRECATED_KEYS:
            new_key = _DEPRECATED_KEYS[key]
            # stacklevel=4: _warn_deprecated_key → _resolve_key → __getitem__/get → caller
            _warn_deprecated_key(key, new_key, stacklevel=4)
            return new_key
        return key

    def __getitem__(self, key):
        return super().__getitem__(self._resolve_key(key))

    def get(self, key, default=None):
        return super().get(self._resolve_key(key), default)

    def __contains__(self, key):
        return super().__contains__(_DEPRECATED_KEYS.get(key, key))


# Merged global registry. When a new experiment package is added,
# extend this dict with its own data_samples.
data_samples = _DataSamplesDict(
    {
        **_i3_data_samples,
    }
)


def create_datasets(sample_name, cfg, names=None, base_path=None, sub_path_fmt=None):
    """Creates a list of Dataset instances for a named data sample.

    Parameters
    ----------
    sample_name : str
        The name of the data sample. Available samples are the keys of
        ``skyllh.datasets.data_samples``.
    cfg : instance of Config
        The instance of Config holding the local configuration.
    names : sequence of str | None
        The dataset names to return. If None, the module's ``DATASET_NAMES``
        default is used (combined IC86 seasons where applicable).
    base_path : str | None
        The base path of the data files. If None, uses
        ``cfg['repository']['base_path']``.
    sub_path_fmt : str | None
        The sub path format override. If None, uses the module default.

    Returns
    -------
    datasets : list of Dataset
    """
    if sample_name in _DEPRECATED_KEYS:
        new_name = _DEPRECATED_KEYS[sample_name]
        # stacklevel=3: _warn_deprecated_key → create_datasets → caller
        _warn_deprecated_key(sample_name, new_name, stacklevel=3)
        sample_name = new_name
    if sample_name not in data_samples:
        available = ', '.join(f'"{n}"' for n in data_samples)
        raise KeyError(f'Unknown data sample "{sample_name}". Available samples: {available}')
    module = data_samples[sample_name]
    dsc = module.create_dataset_collection(cfg=cfg, base_path=base_path, sub_path_fmt=sub_path_fmt)
    if names is None:
        names = module.DATASET_NAMES
    elif isinstance(names, str):
        names = (names,)
    return dsc[names]
