from skyllh.core.dataset import (
    DatasetOrigin,
    URLRetrieveDatasetTransfer,
    post_transfer_unarchive,
)
from skyllh.datasets.i3 import (
    PublicData_14y_ps,
)

DATASET_NAMES = PublicData_14y_ps.DATASET_NAMES


def create_dataset_collection(
    cfg,
    base_path=None,
    sub_path_fmt=None,
):
    """Defines the dataset collection for IceCube's 14-year
    point-source public data, which is available at
    https://doi.org/10.7910/DVN/MMIIZA.

    Version 2: Fixes two binning issues identified in version 1.0:
    1. The IC86 Monte Carlo simulations extend only up to a true energy of
    log10(E/GeV) = 8.7, which falls inside the last true-energy bin,
    log10(E/GeV) = [8.6, 8.8). The resulting lack of simulated events
    artificially reduced the effective area in that bin at all declinations.
    Version 2.0 sets the range of this bin to log10(E/GeV) = [8.6, 8.7),
    matching the simulation boundary.
    2. The sin(declination) binning of the smearing matrices was set to
    IceCube's internal IC40 binning for all detector configurations (IC40,
    IC59, IC79, and IC86), rather than to the per-season internal binning
    described in the accompanying paper. Version 2.0 uses the corresponding
    binning for each season.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    base_path : str | None
        The base path of the data files. The actual path of a data file is
        assumed to be of the structure <base_path>/<sub_path>/<file_name>.
        If ``None``, ``cfg['repository']['base_path']`` is used, which
        defaults to ``~/.cache/skyllh``.
    sub_path_fmt : str | None
        The sub path format of the data files of the public data sample.
        If None, use the default sub path format 'icecube_14year_ps_v2'.

    Returns
    -------
    dsc : DatasetCollection
        The dataset collection containing all the seasons as individual
        I3Dataset objects.
    """
    # Reuse the version 1.0 dataset collection definition and only re-point it to the version 2.0 data release.
    dsc = PublicData_14y_ps.create_dataset_collection(
        cfg=cfg,
        base_path=base_path,
        sub_path_fmt=sub_path_fmt,
    )

    version = 2
    default_sub_path_fmt = 'icecube_14year_ps_v2'

    for name in dsc.dataset_names:
        ds = dsc.get_dataset(name)
        ds.version = version
        ds.default_sub_path_fmt = default_sub_path_fmt
        ds.origin = DatasetOrigin(
            url='https://dataverse.harvard.edu/api/access/dataset/:persistentId/versions/2.0'
            '?persistentId=doi:10.7910/DVN/MMIIZA',
            base_path='',
            sub_path=default_sub_path_fmt,
            filename='tmp.zip',
            transfer_func=URLRetrieveDatasetTransfer(protocol='https').transfer,
            post_transfer_func=post_transfer_unarchive,
        )

    return dsc
