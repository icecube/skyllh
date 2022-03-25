# -*- coding: utf-8 -*-

"""The trad_ps analysis is a multi-dataset time-integrated single source
analysis with a two-component likelihood function using a spacial and an energy
event PDF.
"""

import argparse
import logging
import numpy as np

from skyllh.core.progressbar import ProgressBar

# Classes to define the source hypothesis.
from skyllh.physics.source import PointLikeSource
from skyllh.physics.flux import PowerLawFlux
from skyllh.core.source_hypo_group import SourceHypoGroup
from skyllh.core.source_hypothesis import SourceHypoGroupManager

# Classes to define the fit parameters.
from skyllh.core.parameters import (
    SingleSourceFitParameterMapper,
    FitParameter
)

# Classes for the minimizer.
from skyllh.core.minimizer import Minimizer, LBFGSMinimizerImpl

# Classes for utility functionality.
from skyllh.core.config import CFG
from skyllh.core.random import RandomStateService
from skyllh.core.optimize import SpatialBoxEventSelectionMethod
from skyllh.core.smoothing import BlockSmoothingFilter
from skyllh.core.timing import TimeLord
from skyllh.core.trialdata import TrialDataManager

# Classes for defining the analysis.
from skyllh.core.test_statistic import TestStatisticWilks
from skyllh.core.analysis import (
    TimeIntegratedMultiDatasetSingleSourceAnalysis as Analysis
)

# Classes to define the background generation.
from skyllh.core.scrambling import DataScrambler, UniformRAScramblingMethod
from skyllh.i3.background_generation import FixedScrambledExpDataI3BkgGenMethod

# Classes to define the detector signal yield tailored to the source hypothesis.
from skyllh.analyses.i3.trad_ps.detsigyield import (
    PublicDataPowerLawFluxPointLikeSourceI3DetSigYieldImplMethod
)


# Classes to define the signal and background PDFs.
from skyllh.core.signalpdf import GaussianPSFPointLikeSourceSignalSpatialPDF
from skyllh.i3.signalpdf import SignalI3EnergyPDFSet
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF,
    DataBackgroundI3EnergyPDF
)
# Classes to define the spatial and energy PDF ratios.
from skyllh.core.pdfratio import (
    SpatialSigOverBkgPDFRatio,
    Skylab2SkylabPDFRatioFillMethod
)
from skyllh.i3.pdfratio import I3EnergySigSetOverBkgPDFRatioSpline

from skyllh.i3.signal_generation import PointLikeSourceI3SignalGenerationMethod

# Analysis utilities.
from skyllh.core.analysis_utils import (
    pointlikesource_to_data_field_array
)

# Logging setup utilities.
from skyllh.core.debugging import (
    setup_logger,
    setup_console_handler,
    setup_file_handler
)

# The pre-defined data samples.
from skyllh.datasets.i3 import data_samples


def TXS_location():
    src_ra  = np.radians(77.358)
    src_dec = np.radians(5.693)
    return (src_ra, src_dec)

def create_analysis(
    datasets,
    source,
    refplflux_Phi0=1,
    refplflux_E0=1e3,
    refplflux_gamma=2,
    ns_seed=10.0,
    gamma_seed=3,
    compress_data=False,
    keep_data_fields=None,
    optimize_delta_angle=10,
    efficiency_mode=None,
    tl=None,
    ppbar=None
):
    """Creates the Analysis instance for this particular analysis.

    Parameters:
    -----------
    datasets : list of Dataset instances
        The list of Dataset instances, which should be used in the
        analysis.
    source : PointLikeSource instance
        The PointLikeSource instance defining the point source position.
    refplflux_Phi0 : float
        The flux normalization to use for the reference power law flux model.
    refplflux_E0 : float
        The reference energy to use for the reference power law flux model.
    refplflux_gamma : float
        The spectral index to use for the reference power law flux model.
    ns_seed : float
        Value to seed the minimizer with for the ns fit.
    gamma_seed : float | None
        Value to seed the minimizer with for the gamma fit. If set to None,
        the refplflux_gamma value will be set as gamma_seed.
    compress_data : bool
        Flag if the data should get converted from float64 into float32.
    keep_data_fields : list of str | None
        List of additional data field names that should get kept when loading
        the data.
    optimize_delta_angle : float
        The delta angle in degrees for the event selection optimization methods.
    efficiency_mode : str | None
        The efficiency mode the data should get loaded with. Possible values
        are:

            - 'memory':
                The data will be load in a memory efficient way. This will
                require more time, because all data records of a file will
                be loaded sequentially.
            - 'time':
                The data will be loaded in a time efficient way. This will
                require more memory, because each data file gets loaded in
                memory at once.

        The default value is ``'time'``. If set to ``None``, the default
        value will be used.
    tl : TimeLord instance | None
        The TimeLord instance to use to time the creation of the analysis.
    ppbar : ProgressBar instance | None
        The instance of ProgressBar for the optional parent progress bar.

    Returns
    -------
    analysis : SpatialEnergyTimeIntegratedMultiDatasetSingleSourceAnalysis
        The Analysis instance for this analysis.
    """
    # Define the flux model.
    fluxmodel = PowerLawFlux(
        Phi0=refplflux_Phi0, E0=refplflux_E0, gamma=refplflux_gamma)

    # Define the fit parameter ns.
    fitparam_ns = FitParameter('ns', 0, 1e3, ns_seed)

    # Define the gamma fit parameter.
    fitparam_gamma = FitParameter('gamma', valmin=1, valmax=4, initial=gamma_seed)

    # Define the detector signal efficiency implementation method for the
    # IceCube detector and this source and fluxmodel.
    # The sin(dec) binning will be taken by the implementation method
    # automatically from the Dataset instance.
    gamma_grid = fitparam_gamma.as_linear_grid(delta=0.1)
    detsigyield_implmethod = \
        PublicDataPowerLawFluxPointLikeSourceI3DetSigYieldImplMethod(
            gamma_grid)

    # Define the signal generation method.
    sig_gen_method = PointLikeSourceI3SignalGenerationMethod()

    # Create a source hypothesis group manager.
    src_hypo_group_manager = SourceHypoGroupManager(
        SourceHypoGroup(
            source, fluxmodel, detsigyield_implmethod, sig_gen_method))

    # Create a source fit parameter mapper and define the fit parameters.
    src_fitparam_mapper = SingleSourceFitParameterMapper()
    src_fitparam_mapper.def_fit_parameter(fitparam_gamma)

    # Define the test statistic.
    test_statistic = TestStatisticWilks()

    # Define the data scrambler with its data scrambling method, which is used
    # for background generation.
    data_scrambler = DataScrambler(UniformRAScramblingMethod())

    # Create background generation method.
    bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(data_scrambler)

    # Create the minimizer instance.
    minimizer = Minimizer(LBFGSMinimizerImpl())

    # Create the Analysis instance.
    analysis = Analysis(
        src_hypo_group_manager,
        src_fitparam_mapper,
        fitparam_ns,
        test_statistic,
        bkg_gen_method
    )

    # Define the event selection method for pure optimization purposes.
    # We will use the same method for all datasets.
    event_selection_method = SpatialBoxEventSelectionMethod(
        src_hypo_group_manager, delta_angle=np.deg2rad(optimize_delta_angle))

    # Add the data sets to the analysis.
    pbar = ProgressBar(len(datasets), parent=ppbar).start()
    for ds in datasets:
        # Load the data of the data set.
        data = ds.load_and_prepare_data(
            keep_fields=keep_data_fields,
            compress=compress_data,
            efficiency_mode=efficiency_mode,
            tl=tl)

        # Create a trial data manager and add the required data fields.
        tdm = TrialDataManager()
        tdm.add_source_data_field('src_array',
            pointlikesource_to_data_field_array)

        sin_dec_binning = ds.get_binning_definition('sin_dec')
        log_energy_binning = ds.get_binning_definition('log_energy')

        # Create the spatial PDF ratio instance for this dataset.
        spatial_sigpdf = GaussianPSFPointLikeSourceSignalSpatialPDF(
            dec_range=np.arcsin(sin_dec_binning.range))
        spatial_bkgpdf = DataBackgroundI3SpatialPDF(
            data.exp, sin_dec_binning)
        spatial_pdfratio = SpatialSigOverBkgPDFRatio(
            spatial_sigpdf, spatial_bkgpdf)

        # Create the energy PDF ratio instance for this dataset.
        smoothing_filter = BlockSmoothingFilter(nbins=1)
        energy_sigpdfset = SignalI3EnergyPDFSet(
            data.mc, log_energy_binning, sin_dec_binning, fluxmodel, gamma_grid,
            smoothing_filter, ppbar=pbar)
        energy_bkgpdf = DataBackgroundI3EnergyPDF(
            data.exp, log_energy_binning, sin_dec_binning, smoothing_filter)
        fillmethod = Skylab2SkylabPDFRatioFillMethod()
        energy_pdfratio = I3EnergySigSetOverBkgPDFRatioSpline(
            energy_sigpdfset, energy_bkgpdf,
            fillmethod=fillmethod,
            ppbar=pbar)

        pdfratios = [ spatial_pdfratio, energy_pdfratio ]

        analysis.add_dataset(
            ds, data, pdfratios, tdm, event_selection_method)

        pbar.increment()
    pbar.finish()

    analysis.llhratio = analysis.construct_llhratio(minimizer, ppbar=ppbar)

    analysis.construct_signal_generator()

    return analysis

if(__name__ == '__main__'):
    p = argparse.ArgumentParser(
        description = "Calculates TS for a given source location using 7-year "
                      "point source sample and 3-year GFU sample.",
        formatter_class = argparse.RawTextHelpFormatter
    )
    p.add_argument("--data_base_path", default=None, type=str,
        help='The base path to the data samples (default=None)'
    )
    p.add_argument("--ncpu", default=1, type=int,
        help='The number of CPUs to utilize where parallelization is possible.'
    )
    args = p.parse_args()

    # Setup `skyllh` package logging.
    # To optimize logging set the logging level to the lowest handling level.
    setup_logger('skyllh', logging.DEBUG)
    log_format = '%(asctime)s %(processName)s %(name)s %(levelname)s: '\
                 '%(message)s'
    setup_console_handler('skyllh', logging.INFO, log_format)
    setup_file_handler('skyllh', logging.DEBUG, log_format, 'debug.log')

    CFG['multiproc']['ncpu'] = args.ncpu

    sample_seasons = [
        ("PointSourceTracks", "IC40"),
        ("PointSourceTracks", "IC59"),
        ("PointSourceTracks", "IC79"),
        ("PointSourceTracks", "IC86, 2011"),
        ("PointSourceTracks", "IC86, 2012-2014"),
        ("GFU", "IC86, 2015-2017")
    ]

    datasets = []
    for (sample, season) in sample_seasons:
        # Get the dataset from the correct dataset collection.
        dsc = data_samples[sample].create_dataset_collection(args.data_base_path)
        datasets.append(dsc.get_dataset(season))

    rss_seed = 1
    # Define a random state service.
    rss = RandomStateService(rss_seed)

    # Define the point source.
    source = PointLikeSource(*TXS_location())

    tl = TimeLord()

    with tl.task_timer('Creating analysis.'):
        ana = create_analysis(
            datasets, source, compress_data=False, tl=tl)

    with tl.task_timer('Unblinding data.'):
        (TS, fitparam_dict, status) = ana.unblind(rss)

    #print('log_lambda_max: %g'%(log_lambda_max))
    print('TS = %g'%(TS))
    print('ns_fit = %g'%(fitparam_dict['ns']))
    print('gamma_fit = %g'%(fitparam_dict['gamma']))

    # Generate some signal events.
    with tl.task_timer('Generating signal events.'):
        (n_sig, signal_events_dict) = ana.sig_generator.generate_signal_events(rss, 100)

    print('n_sig: %d', n_sig)
    print('signal datasets: '+str(signal_events_dict.keys()))

    print(tl)
