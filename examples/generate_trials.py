#!/usr/bin/env python

import os, sys

import logging
import numpy as np
import skyllh

from skyllh.core.config import CFG
from skyllh.core.random import RandomStateService
from skyllh.core.timing import TimeLord
from skyllh.core.minimizer import ScipyMinimizerImpl, LBFGSMinimizerImpl
from skyllh.physics.source import PointLikeSource

from i3skyllh.datasets import data_samples
from i3skyllh.datasets import repository
from i3skyllh.analyses.kdepdf_mcbg_ps.analysis import create_analysis

from skyllh.core.analysis_utils import create_trial_data_file

from skyllh.core.debugging import setup_logger, setup_console_handler
setup_logger('skyllh', logging.INFO)
setup_console_handler('skyllh', logging.INFO)

import argparse
parser=argparse.ArgumentParser()

parser.add_argument("--config", "-c", help="Path to user config", type=str,
                    required=True)
parser.add_argument("--gamma", "-g", help="spectral index", type=float, default=2.0)
parser.add_argument("--n_trials", "-nt", help="number of trials per job", type=int, default=10)
parser.add_argument("--mean_ns", "-ns", help="mean number of nsig injected per job", type=float, default=0.0)
parser.add_argument("--ncpus", "-nc", help="number of CPUs to use", type=int, default=1)
parser.add_argument("--rss_seed", "-rs", help="random_number_seed", type=int, default=5)
parser.add_argument("--gseed", "-gs", help="gamma seed for minimizer", type=float, default=3.0)

args=parser.parse_args()
print(args)

# DO NOT USE PHOTOSPLINE
skyllh.core.pdf.PHOTOSPLINE_LOADED = False

CFG.from_yaml(args.config)

# Define datasets (todo: generalize to other analyses)
bp = CFG["paths"]["base_path"]
dsc_name = CFG["analysis"]["name"]
dsc = data_samples[dsc_name].create_dataset_collection(base_path=bp)
ds_names = CFG["analysis"]["datasets"]
datasets = dsc.get_datasets(ds_names)

# Define the point source at TXS position.
src_ra  = np.radians(CFG["analysis"]["default_source"]["ra"])
src_dec = np.radians(CFG["analysis"]["default_source"]["dec"])
source = PointLikeSource(src_ra, src_dec)

tl = TimeLord()

# Create analysis.
yr=365.25 # days per year
livetime = CFG["analysis"]["livetime"] * yr

minimizers = {"COBYLA": ScipyMinimizerImpl(method="COBYLA"),
              "LBFGSB": LBFGSMinimizerImpl()}


for min_name, minimizer in minimizers.items():
    with tl.task_timer('Creating analysis.') as tt:
        ana = create_analysis(
            datasets,
            source,
            bkg_event_rate_field_names=['astro', 'conv'],
            refplflux_gamma=args.gamma,
            gamma_seed = args.gseed,
            fit_gamma=True,
            livetime_list=[livetime],
            compress_data=True,
            minimizer_impl=minimizer,
            tl=tl)

    print(tl)

    # Define a random state service.
    rss_seed = args.rss_seed
    rss = RandomStateService(rss_seed)

    trials_outfile = os.path.join(CFG["paths"]["output_dir"],
                                  dsc_name+'_atTXS'+'_gamma{:.2f}_mean_ns{}_gseed{:.2f}_ltime{:.0f}yr_rss{:}_mini{}'.format(args.gamma,
                                                                                                                            args.mean_ns,
                                                                                                                            args.gseed,
                                                                                                                            livetime,
                                                                                                                            args.rss_seed,
                                                                                                                            min_name))

    # Run trials.
    with tl.task_timer('Running trials.') as tt:
        (seed,mean_ns,mean_ns_null,trial_data)=create_trial_data_file(
            ana=ana,
            ncpu=args.ncpus,
            rss=rss,
            pathfilename=trials_outfile,
            n_trials=args.n_trials,
            mean_n_sig=args.mean_ns,
            tl=tl)

    print(tl)
