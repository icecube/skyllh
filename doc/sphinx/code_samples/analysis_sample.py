# -*- coding: utf-8 -*-

"""The IC170922A_wGFU analysis is a multi-dataset time-integrated single source
analysis with a two-component likelihood function using a spacial and an energy
event PDF.
"""

import argparse
import numpy as np

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
from skyllh.core import multiproc
from skyllh.core.stopwatch import Stopwatch
from skyllh.core.random import RandomStateService
from skyllh.core.scrambling import DataScrambler, UniformRAScramblingMethod
from skyllh.core.optimize import SpatialBoxEventSelectionMethod
from skyllh.core.smoothing import BlockSmoothingFilter

# Classes for defining the analysis.
from skyllh.core.test_statistic import TestStatisticWilks
from skyllh.core.analysis import (
    MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis as Analysis
)

# Classes to define the detector signal efficiency tailored to the source
# hypothesis.
from skyllh.i3.detsigeff import PowerLawFluxPointLikeSourceI3DetSigEffImplMethod

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

# The pre-defined data samples.
from i3skyllh.datasets import data_samples

def TXS_location():
    src_ra  = np.radians(77.358)
    src_dec = np.radians(5.693)
    return (src_ra, src_dec)

def create_analysis(
    sample_seasons, source, rss_seed=None, data_base_path=None, compress_data=False,
    keep_data_fields=None
):
    """Creates the Analysis instance for this particular analysis.

    Parameters:
    -----------
    sample_seasons : list of 2-element tuples
        The list of (data sample, season) tuples, which should be used in the
        analysis. Each pair represents an individual dataset in the analysis.
    source : PointLikeSource instance
        The PointLikeSource instance defining the point source position.
    rss_seed : int | None
        The random state service seed number. Use None for random seed.
    data_base_path : str | None
        The base path of the data files (of all data samples).
    compress_data : bool
        Flag if the data should get converted from float64 into float32.
    keep_data_fields : list of str | None
        List of additional data field names that should get kept when loading
        the data.

    Returns
    -------
    analysis : MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis
        The Analysis instance for this analysis.
    """

    # Define a random state service.
    rss = RandomStateService(rss_seed)

    # Define the flux model.
    fluxmodel = PowerLawFlux(Phi0=1, E0=1e3, gamma=2)

    # Define the fit parameter ns.
    fitparam_ns = FitParameter('ns', 0, 1e3, 15)

    # Define the gamma fit parameter.
    fitparam_gamma = FitParameter('gamma', valmin=1, valmax=4, initial=2)

    # Define the detector signal efficiency implementation method for the
    # IceCube detector and this source and fluxmodel.
    # The sin(dec) binning will be taken by the implementation method
    # automatically from the Dataset instance.
    gamma_grid = fitparam_gamma.as_linear_grid(delta=0.1)
    detsigeff_implmethod = PowerLawFluxPointLikeSourceI3DetSigEffImplMethod(gamma_grid)

    # Define the signal generation method.
    sig_gen_method = PointLikeSourceI3SignalGenerationMethod()

    # Create a source hypothesis group manager.
    src_hypo_group_manager = SourceHypoGroupManager(
        SourceHypoGroup(source, fluxmodel, detsigeff_implmethod, sig_gen_method))

    # Create a source fit parameter mapper and define the fit parameters.
    src_fitparam_mapper = SingleSourceFitParameterMapper(rss)
    src_fitparam_mapper.def_fit_parameter(fitparam_gamma)

    # Define the test statistic.
    test_statistic = TestStatisticWilks()

    # Define the data scrambler with its data scrambling method.
    data_scrambler = DataScrambler(
        UniformRAScramblingMethod(rss), inplace_scrambling=True)

    # Define the event selection method for pure optimization purposes.
    event_selection_method = SpatialBoxEventSelectionMethod(
        src_hypo_group_manager, delta_angle=np.deg2rad(10))

    # Create the minimizer instance.
    minimizer = Minimizer(LBFGSMinimizerImpl())

    # Create the Analysis instance.
    analysis = Analysis(
        minimizer,
        src_hypo_group_manager,
        src_fitparam_mapper,
        fitparam_ns,
        test_statistic,
        data_scrambler,
        event_selection_method
    )

    # Add the datasets to the analysis.
    for (sample, season) in sample_seasons:
        # Get the dataset from the correct dataset collection.
        dsc = data_samples[sample].create_dataset_collection(data_base_path)
        ds = dsc.get_dataset(season)

        data = ds.load_and_prepare_data(
            keep_fields=keep_data_fields, compress=compress_data)

        sin_dec_binning = ds.get_binning_definition('sin_dec')
        log_energy_binning = ds.get_binning_definition('log_energy')

        # Create the spatial PDF ratio instance for this dataset.
        spatial_sigpdf = GaussianPSFPointLikeSourceSignalSpatialPDF(src_hypo_group_manager)
        spatial_bkgpdf = DataBackgroundI3SpatialPDF(
            data.exp, sin_dec_binning)
        spatial_pdfratio = SpatialSigOverBkgPDFRatio(
            spatial_sigpdf, spatial_bkgpdf)

        # Create the energy PDF ratio instance for this dataset.
        smoothing_filter = BlockSmoothingFilter(nbins=1)
        energy_sigpdfset = SignalI3EnergyPDFSet(
            data.mc, log_energy_binning, sin_dec_binning, fluxmodel, gamma_grid, smoothing_filter)
        energy_bkgpdf = DataBackgroundI3EnergyPDF(
            data.exp, log_energy_binning, sin_dec_binning, smoothing_filter)
        fillmethod = Skylab2SkylabPDFRatioFillMethod()
        energy_pdfratio = I3EnergySigSetOverBkgPDFRatioSpline(
            energy_sigpdfset, energy_bkgpdf,
            fillmethod=fillmethod)

        analysis.add_dataset(ds, data, spatial_pdfratio, energy_pdfratio)

    analysis.construct_llhratio()

    analysis.construct_signal_generator(rss)

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

    multiproc.NCPU = args.ncpu

    sample_seasons = [
        #("PointSourceTracks", "IC40"),
        #("PointSourceTracks", "IC59"),
        #("PointSourceTracks", "IC79"),
        #("PointSourceTracks", "IC86, 2011"),
        #("PointSourceTracks", "IC86, 2012-2014"),
        ("GFU", "IC86, 2015-2017")
    ]
    rss_seed = 1

    # Define the point source.
    source = PointLikeSource(*TXS_location())

    sw = Stopwatch().start()

    analysis = create_analysis(
        sample_seasons, source, rss_seed,
        args.data_base_path,
        compress_data=False)

    sw.take_lap('Created analysis.')

    (TS, fitparam_dict, status) = analysis.unblind()

    sw.take_lap('Unblinded data.')

    #print('log_lambda_max: %g'%(log_lambda_max))
    print('TS = %g'%(TS))
    print('ns_fit = %g'%(fitparam_dict['ns']))
    print('gamma_fit = %g'%(fitparam_dict['gamma']))

    # Generate some signal events.
    (n_signal, signal_events_dict) = analysis.signal_generator.generate_signal_events(100)
    sw.stop('Generated signal events.')

    print('n_signal: %d', n_signal)
    print('signal datasets: '+str(signal_events_dict.keys()))
    print(sw)
