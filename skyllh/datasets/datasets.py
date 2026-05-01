from .i3 import data_samples


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
    if sample_name not in data_samples:
        available = ', '.join(f'"{n}"' for n in data_samples)
        raise KeyError(f'Unknown data sample "{sample_name}". Available samples: {available}')
    module = data_samples[sample_name]
    dsc = module.create_dataset_collection(cfg=cfg, base_path=base_path, sub_path_fmt=sub_path_fmt)
    if names is None:
        names = module.DATASET_NAMES
    return dsc[names]
