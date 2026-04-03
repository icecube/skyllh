from typing import TypedDict

from skyllh.core.config import Config
from skyllh.core.dataset import (
    DatasetCollection,
)
from skyllh.i3.dataset import (
    I3Dataset,
)

DATASET_NAMES = ('TestData',)


class _DsKwargs(TypedDict):
    cfg: Config
    version: int
    verqualifiers: dict[str, int] | None
    base_path: str | None
    default_sub_path_fmt: str
    sub_path_fmt: str | None


def create_dataset_collection(
    cfg: Config,
    base_path: str | None = None,
    sub_path_fmt: str | None = None,
) -> DatasetCollection:
    """Defines a dataset collection with a test dataset.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    base_path
        The base path of the data files. The actual path of a data file is
        assumed to be of the structure <base_path>/<sub_path>/<file_name>.
        If ``None``, ``cfg['repository']['base_path']`` is used, which
        defaults to ``~/.cache/skyllh``.
    sub_path_fmt
        The sub path format of the data files of the public data sample.
        If None, use the default sub path format
        'testdata'.

    Returns
    -------
    dsc
        The dataset collection containing all the seasons as individual
        I3Dataset objects.
    """
    (version, verqualifiers) = (1, {'p': 0})

    default_sub_path_fmt = 'testdata'

    dsc = DatasetCollection('Test public data')

    dsc.description = r"""
    This dataset collection contains a test dataset which can be used for unit
    tests.
    """

    # Define the common keyword arguments for all data sets.
    ds_kwargs: _DsKwargs = {
        'cfg': cfg,
        'version': version,
        'verqualifiers': verqualifiers,
        'base_path': base_path,
        'default_sub_path_fmt': default_sub_path_fmt,
        'sub_path_fmt': sub_path_fmt,
    }

    TestData = I3Dataset(
        name='TestData',
        exp_pathfilenames='exp.npy',
        mc_pathfilenames='mc.npy',
        grl_pathfilenames='grl.npy',
        **ds_kwargs,
    )

    dsc.add_datasets((TestData,))

    return dsc
