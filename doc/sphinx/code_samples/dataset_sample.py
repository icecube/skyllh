# -*- coding: utf-8 -*-

import os.path

import numpy as np
from numpy.lib import recfunctions as np_rfn

from skyllh.core.dataset import (
    DatasetCollection,
    generate_data_file_path,
)
from skyllh.i3.dataset import (
    I3Dataset,
)


def create_dataset_collection(base_path=None, sub_path=None):
    """Defines the dataset collection for the GFU sample.

    Parameters
    ----------
    base_path : str | None
        The base path of the data files. The actual path of a data file is
        assumed to be of the structure <base_path>/<path>/<file_name>.
        If None, use the default path '/data/ana/analyses'.
    sub_path : str | None
        The sub path of the data files of the GFU sample.
        If None, use the default path 'gfu/version-%(version)03d-p%(p)02d'.

    Returns
    -------
    dsc : DatasetCollection
        The dataset collection containing all the seasons as individual
        I3Dataset objects.
    """

    # Define the version of the data sample (collection).
    (version, verqualifiers) = (2, dict(p=3))

    # Generate the path to the data files.
    path = generate_data_file_path(
        "/data/ana/analyses",
        "gfu/version-%(version)03d-p%(p)02d",
        version,
        verqualifiers,
        base_path,
        sub_path,
    )

    # We create a dataset collection that will hold the individual seasonal GFU
    # datasets (all of the same version!).
    dsc = DatasetCollection("GFU")

    dsc.description = """Real-time Gamma-Ray Follow-Up (GFU) Sample

    See: https://wiki.icecube.wisc.edu/images/7/7d/OnLineL2_GFU_TFT_proposal_2017.pdf

    NOTE #1:
        Binning comes from IC86, 2012 of PointSourceTracks since
        the GFU dataset is inspired by the point source track sample.

    NOTE #2:
        MC file works for all years >= 2012.

    v002p03 is updated to fix a leap second issue, in addition to
    standardizing grl field names
    """

    # Add the seasons to the dataset collection. Each season is an own dataset.
    IC86_2015 = I3Dataset(
        name="IC86, 2015",
        livetime=364.684,  # days
        exp_pathfilenames=os.path.join(path, "IC86_2015_data.npy"),
        mc_pathfilenames=os.path.join(path, "IC86_2015_MC.npy"),
        grl_pathfilenames=os.path.join(path, "GRL/IC86_2015_data.npy"),
        version=version,
        verqualifiers=verqualifiers,
    )

    IC86_2016 = I3Dataset(
        name="IC86, 2016",
        livetime=356.198,  # days
        exp_pathfilenames=os.path.join(path, "IC86_2016_data.npy"),
        mc_pathfilenames=IC86_2015.mc_pathfilename_list,
        grl_pathfilenames=os.path.join(path, "GRL/IC86_2016_data.npy"),
        version=version,
        verqualifiers=verqualifiers,
    )

    IC86_2017 = I3Dataset(
        name="IC86, 2017",
        livetime=165.443,  # days
        exp_pathfilenames=os.path.join(path, "IC86_2017_data.npy"),
        mc_pathfilenames=IC86_2015.mc_pathfilename_list,
        grl_pathfilenames=os.path.join(path, "GRL/IC86_2017_data.npy"),
        version=version,
        verqualifiers=verqualifiers,
    )

    seasons = [IC86_2015, IC86_2016, IC86_2017]
    IC86_2015_to_2017 = I3Dataset(
        name="IC86, 2015-2017",
        livetime=np.sum([s.livetime for s in seasons]),  # days
        exp_pathfilenames=[s.exp_pathfilename_list[0] for s in seasons],
        mc_pathfilenames=IC86_2015.mc_pathfilename_list,
        grl_pathfilenames=[
            os.path.join(path, "GRL/IC86_2015_data.npy"),
            os.path.join(path, "GRL/IC86_2016_data.npy"),
            os.path.join(path, "GRL/IC86_2017_data.npy"),
        ],
        version=version,
        verqualifiers=verqualifiers,
    )

    # Add all the datasets of the different seasons to the dataset collection.
    dsc.add_datasets((IC86_2015, IC86_2016, IC86_2017, IC86_2015_to_2017))

    # Define the data preparation function and add it to all datasets.
    def data_prep(exp, mc):
        # Remove events with very large uncertainties.
        exp = exp[exp["angErr"] < np.radians(15)]
        mc = mc[mc["angErr"] < np.radians(15)]
        return (exp, mc)

    dsc.add_data_preparation(data_prep)

    # Define a data preparation function to rename some of the data fields to
    # make it conform with the skyllh naming scheme.
    def format_data_keys(exp, mc):
        exp = np_rfn.rename_fields(
            exp, {"logE": "log_energy", "angErr": "ang_err"}
        )
        mc = np_rfn.rename_fields(
            mc,
            {
                "trueAzi": "true_azi",
                "trueZen": "true_zen",
                "logE": "log_energy",
                "angErr": "ang_err",
                "trueE": "true_energy",
                "trueRa": "true_ra",
                "trueDec": "true_dec",
                "ow": "mcweight",
            },
        )
        return (exp, mc)

    dsc.add_data_preparation(format_data_keys)

    # Define declination and energy binning and use it for all datasets.
    sin_dec_bins = np.unique(
        np.concatenate(
            [
                np.linspace(-1.0, -0.93, 4 + 1),
                np.linspace(-0.93, -0.3, 10 + 1),
                np.linspace(-0.3, 0.05, 9 + 1),
                np.linspace(0.05, 1.0, 18 + 1),
            ]
        )
    )
    energy_bins = np.arange(1.0, 9.5 + 0.01, 0.125)
    dsc.define_binning("sin_dec", sin_dec_bins)
    dsc.define_binning("log_energy", energy_bins)

    return dsc
