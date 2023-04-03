# -*- coding: utf-8 -*-

"""The time_integrated_ps analysis is a multi-dataset time-integrated single source
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
from skyllh.core.minimizers.iminuit import IMinuitMinimizerImpl

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

# Classes to define the signal and background PDFs.
from skyllh.core.signalpdf import RayleighPSFPointSourceSignalSpatialPDF
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF
)

# Classes to define the spatial and energy PDF ratios.
from skyllh.core.pdfratio import SpatialSigOverBkgPDFRatio

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

# Pre-defined public IceCube data samples.
from skyllh.datasets.i3 import data_samples

# Analysis specific classes for working with the public data.
from skyllh.analyses.i3.publicdata_ps.signal_generator import PDSignalGenerator

from skyllh.analyses.i3.publicdata_ps.detsigyield import (
    PublicDataPowerLawFluxPointLikeSourceI3DetSigYieldImplMethod
)
from skyllh.analyses.i3.publicdata_ps.signalpdf import (
    PDSignalEnergyPDFSet
)
from skyllh.analyses.i3.publicdata_ps.pdfratio import (
    PDPDFRatio
)
from skyllh.analyses.i3.publicdata_ps.backgroundpdf import (
    PDDataBackgroundI3EnergyPDF
)
from skyllh.analyses.i3.publicdata_ps.utils import create_energy_cut_spline


def psi_func(tdm, src_hypo_group_manager, fitparams):
    """Function to calculate the opening angle between the source position
    and the event's reconstructed position.
    """
    ra = tdm.get_data('ra')
    dec = tdm.get_data('dec')

    # Make the source position angles two-dimensional so the PDF value
    # can be calculated via numpy broadcasting automatically for several
    # sources. This is useful for stacking analyses.
    src_ra = tdm.get_data('src_array')['ra'][:, np.newaxis]
    src_dec = tdm.get_data('src_array')['dec'][:, np.newaxis]

    delta_dec = np.abs(dec - src_dec)
    delta_ra = np.abs(ra - src_ra)
    x = (
        (np.sin(delta_dec / 2.))**2. + np.cos(dec) *
        np.cos(src_dec) * (np.sin(delta_ra / 2.))**2.
    )

    # Handle possible floating precision errors.
    x[x < 0.] = 0.
    x[x > 1.] = 1.

    psi = (2.0*np.arcsin(np.sqrt(x)))

    # For now we support only a single source, hence return psi[0].
    return psi[0, :]


def TXS_location():
    src_ra = np.radians(77.358)
    src_dec = np.radians(5.693)
    return (src_ra, src_dec)


def create_analysis(
    datasets,
    source,
    refplflux_Phi0=1,
    refplflux_E0=1e3,
    refplflux_gamma=2.0,
    ns_seed=100.0,
    ns_min=0.,
    ns_max=1e3,
    gamma_seed=3.0,
    gamma_min=1.,
    gamma_max=5.,
    kde_smoothing=False,
    minimizer_impl="LBFGS",
    cut_sindec=None,
    spl_smooth=None,
    cap_ratio=False,
    compress_data=False,
    keep_data_fields=None,
    optimize_delta_angle=10,
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
    ns_min : float
        Lower bound for ns fit.
    ns_max : float
        Upper bound for ns fit.
    gamma_seed : float | None
        Value to seed the minimizer with for the gamma fit. If set to None,
        the refplflux_gamma value will be set as gamma_seed.
    gamma_min : float
        Lower bound for gamma fit.
    gamma_max : float
        Upper bound for gamma fit.
    kde_smoothing : bool
        Apply a KDE-based smoothing to the data-driven background pdf.
        Default: False.
    minimizer_impl : str | "LBFGS"
        Minimizer implementation to be used. Supported options are "LBFGS"
        (L-BFG-S minimizer used from the :mod:`scipy.optimize` module), or
        "minuit" (Minuit minimizer used by the :mod:`iminuit` module).
        Default: "LBFGS".
    cut_sindec : list of float | None
        sin(dec) values at which the energy cut in the southern sky should
        start. If None, np.sin(np.radians([-2, 0, -3, 0, 0])) is used.
    spl_smooth : list of float
        Smoothing parameters for the 1D spline for the energy cut. If None,
        [0., 0.005, 0.05, 0.2, 0.3] is used.
    cap_ratio : bool
        If set to True, the energy PDF ratio will be capped to a finite value
        where no background energy PDF information is available. This will
        ensure that an energy PDF ratio is available for high energies where
        no background is available from the experimental data.
        If kde_smoothing is set to True, cap_ratio should be set to False!
        Default is False.
    compress_data : bool
        Flag if the data should get converted from float64 into float32.
    keep_data_fields : list of str | None
        List of additional data field names that should get kept when loading
        the data.
    optimize_delta_angle : float
        The delta angle in degrees for the event selection optimization methods.
    tl : TimeLord instance | None
        The TimeLord instance to use to time the creation of the analysis.
    ppbar : ProgressBar instance | None
        The instance of ProgressBar for the optional parent progress bar.

    Returns
    -------
    analysis : TimeIntegratedMultiDatasetSingleSourceAnalysis
        The Analysis instance for this analysis.
    """

    # Create the minimizer instance.
    if minimizer_impl == "LBFGS":
        minimizer = Minimizer(LBFGSMinimizerImpl())
    elif minimizer_impl == "minuit":
        minimizer = Minimizer(IMinuitMinimizerImpl(ftol=1e-8))
    else:
        raise NameError(
            f"Minimizer implementation `{minimizer_impl}` is not supported "
            "Please use `LBFGS` or `minuit`.")

    # Define the flux model.
    flux_model = PowerLawFlux(
        Phi0=refplflux_Phi0, E0=refplflux_E0, gamma=refplflux_gamma)

    # Define the fit parameter ns.
    fitparam_ns = FitParameter('ns', ns_min, ns_max, ns_seed)

    # Define the gamma fit parameter.
    fitparam_gamma = FitParameter(
        'gamma', valmin=gamma_min, valmax=gamma_max, initial=gamma_seed)

    # Define the detector signal efficiency implementation method for the
    # IceCube detector and this source and flux_model.
    # The sin(dec) binning will be taken by the implementation method
    # automatically from the Dataset instance.
    gamma_grid = fitparam_gamma.as_linear_grid(delta=0.1)
    detsigyield_implmethod = \
        PublicDataPowerLawFluxPointLikeSourceI3DetSigYieldImplMethod(
            gamma_grid)

    # Define the signal generation method.
    #sig_gen_method = PointLikeSourceI3SignalGenerationMethod()
    sig_gen_method = None

    # Create a source hypothesis group manager.
    src_hypo_group_manager = SourceHypoGroupManager(
        SourceHypoGroup(
            source, flux_model, detsigyield_implmethod, sig_gen_method))

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

    # Create the Analysis instance.
    analysis = Analysis(
        src_hypo_group_manager,
        src_fitparam_mapper,
        fitparam_ns,
        test_statistic,
        bkg_gen_method,
        sig_generator_cls=PDSignalGenerator
    )

    # Define the event selection method for pure optimization purposes.
    # We will use the same method for all datasets.
    event_selection_method = SpatialBoxEventSelectionMethod(
        src_hypo_group_manager, delta_angle=np.deg2rad(optimize_delta_angle))

    # Prepare the spline parameters.
    if cut_sindec is None:
        cut_sindec = np.sin(np.radians([-2, 0, -3, 0, 0]))
    if spl_smooth is None:
        spl_smooth = [0., 0.005, 0.05, 0.2, 0.3]
    if len(spl_smooth) < len(datasets) or len(cut_sindec) < len(datasets):
        raise AssertionError(
            "The length of the spl_smooth and of the cut_sindec must be equal "
            f"to the length of datasets: {len(datasets)}.")

    # Add the data sets to the analysis.
    pbar = ProgressBar(len(datasets), parent=ppbar).start()
    energy_cut_splines = []
    for idx, ds in enumerate(datasets):
        # Load the data of the data set.
        data = ds.load_and_prepare_data(
            keep_fields=keep_data_fields,
            compress=compress_data,
            tl=tl)

        # Create a trial data manager and add the required data fields.
        tdm = TrialDataManager()
        tdm.add_source_data_field('src_array',
                                  pointlikesource_to_data_field_array)
        tdm.add_data_field('psi', psi_func)

        sin_dec_binning = ds.get_binning_definition('sin_dec')
        log_energy_binning = ds.get_binning_definition('log_energy')

        # Create the spatial PDF ratio instance for this dataset.
        spatial_sigpdf = RayleighPSFPointSourceSignalSpatialPDF(
            dec_range=np.arcsin(sin_dec_binning.range))
        spatial_bkgpdf = DataBackgroundI3SpatialPDF(
            data.exp, sin_dec_binning)
        spatial_pdfratio = SpatialSigOverBkgPDFRatio(
            spatial_sigpdf, spatial_bkgpdf)

        # Create the energy PDF ratio instance for this dataset.
        energy_sigpdfset = PDSignalEnergyPDFSet(
            ds=ds,
            src_dec=source.dec,
            flux_model=flux_model,
            fitparam_grid_set=gamma_grid,
            ppbar=ppbar
        )
        smoothing_filter = BlockSmoothingFilter(nbins=1)
        energy_bkgpdf = PDDataBackgroundI3EnergyPDF(
            data.exp, log_energy_binning, sin_dec_binning,
            smoothing_filter, kde_smoothing)

        energy_pdfratio = PDPDFRatio(
            sig_pdf_set=energy_sigpdfset,
            bkg_pdf=energy_bkgpdf,
            cap_ratio=cap_ratio
        )

        pdfratios = [spatial_pdfratio, energy_pdfratio]

        analysis.add_dataset(
            ds, data, pdfratios, tdm, event_selection_method)

        energy_cut_spline = create_energy_cut_spline(
            ds, data.exp, spl_smooth[idx])
        energy_cut_splines.append(energy_cut_spline)

        pbar.increment()
    pbar.finish()

    analysis.llhratio = analysis.construct_llhratio(minimizer, ppbar=ppbar)
    analysis.construct_signal_generator(
        llhratio=analysis.llhratio, energy_cut_splines=energy_cut_splines,
        cut_sindec=cut_sindec)

    return analysis


if(__name__ == '__main__'):
    p = argparse.ArgumentParser(
        description='Calculates TS for a given source location using the '
        '10-year public point source sample.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        '--dec',
        default=23.8,
        type=float,
        help='The source declination in degrees.'
    )
    p.add_argument(
        '--ra',
        default=216.76,
        type=float,
        help='The source right-ascention in degrees.'
    )
    p.add_argument(
        '--gamma-seed',
        default=3,
        type=float,
        help='The seed value of the gamma fit parameter.'
    )
    p.add_argument(
        '--data_base_path',
        default=None,
        type=str,
        help='The base path to the data samples (default=None)'
    )
    p.add_argument(
        '--seed',
        default=1,
        type=int,
        help='The random number generator seed for the likelihood '
             'minimization.'
    )
    p.add_argument(
        '--ncpu',
        default=1,
        type=int,
        help='The number of CPUs to utilize where parallelization is possible.'
    )
    p.add_argument(
        '--cap-ratio',
        action='store_true',
        help='Switch to cap the energy PDF ratio.')
    p.set_defaults(cap_ratio=False)
    args = p.parse_args()

    # Setup `skyllh` package logging.
    # To optimize logging set the logging level to the lowest handling level.
    setup_logger('skyllh', logging.DEBUG)
    log_format = '%(asctime)s %(processName)s %(name)s %(levelname)s: '\
                 '%(message)s'
    setup_console_handler('skyllh', logging.INFO, log_format)
    setup_file_handler('skyllh', 'debug.log',
                       log_level=logging.DEBUG,
                       log_format=log_format)

    CFG['multiproc']['ncpu'] = args.ncpu

    sample_seasons = [
        ('PublicData_10y_ps', 'IC40'),
        ('PublicData_10y_ps', 'IC59'),
        ('PublicData_10y_ps', 'IC79'),
        ('PublicData_10y_ps', 'IC86_I'),
        ('PublicData_10y_ps', 'IC86_II-VII'),
    ]

    datasets = []
    for (sample, season) in sample_seasons:
        # Get the dataset from the correct dataset collection.
        dsc = data_samples[sample].create_dataset_collection(
            args.data_base_path)
        datasets.append(dsc.get_dataset(season))

    # Define a random state service.
    rss = RandomStateService(args.seed)

    # Define the point source.
    source = PointLikeSource(np.deg2rad(args.ra), np.deg2rad(args.dec))
    print('source: ', str(source))

    tl = TimeLord()

    with tl.task_timer('Creating analysis.'):
        ana = create_analysis(
            datasets,
            source,
            cap_ratio=args.cap_ratio,
            gamma_seed=args.gamma_seed,
            tl=tl)

    with tl.task_timer('Unblinding data.'):
        (TS, fitparam_dict, status) = ana.unblind(rss)

    print('TS = %g' % (TS))
    print('ns_fit = %g' % (fitparam_dict['ns']))
    print('gamma_fit = %g' % (fitparam_dict['gamma']))

    print(tl)
