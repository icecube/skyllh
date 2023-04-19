# -*- coding: utf-8 -*-

"""Setup the time-dependent analysis. For now this works on a single dataset.
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
    TimeIntegratedMultiDatasetSingleSourceAnalysis,
)

# Classes to define the background generation.
from skyllh.core.scrambling import DataScrambler
from skyllh.i3.scrambling import I3SeasonalVariationTimeScramblingMethod
from skyllh.i3.background_generation import FixedScrambledExpDataI3BkgGenMethod

# Classes to define the signal and background PDFs.
from skyllh.core.signalpdf import (
    RayleighPSFPointSourceSignalSpatialPDF,
    SignalBoxTimePDF,
    SignalGaussTimePDF
)
from skyllh.core.backgroundpdf import BackgroundUniformTimePDF
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF
)
from skyllh.core.pdf import TimePDF

# Classes to define the spatial and energy PDF ratios.
from skyllh.core.pdfratio import (
    SpatialSigOverBkgPDFRatio,
    SigOverBkgPDFRatio
)

# Analysis utilities.
from skyllh.core.analysis_utils import (
    pointlikesource_to_data_field_array
)

from skyllh.core.expectation_maximization import em_fit

# Analysis specific classes for working with the public data.
from skyllh.analyses.i3.publicdata_ps.signal_generator import (
    PDTimeDependentSignalGenerator
)
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
from skyllh.analyses.i3.publicdata_ps.utils import (
    create_energy_cut_spline,
)
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import (
    psi_func,
)


def change_time_pdf(analysis, gauss=None, box=None):
    """Changes the time pdf.

    Parameters
    ----------
    gauss : dict | None
        None or dictionary with {"mu": float, "sigma": float}.
    box : dict | None
        None or dictionary with {"start": float, "end": float}.
    """

    if gauss is None and box is None:
        raise TypeError("Either gauss or box have to be specified as time pdf.")

    grl = analysis._data_list[0].grl
    # redo this in case the background pdf was not calculated before
    time_bkgpdf = BackgroundUniformTimePDF(grl)
    if gauss is not None:
        time_sigpdf = SignalGaussTimePDF(grl, gauss['mu'], gauss['sigma'])
    elif box is not None:
        time_sigpdf = SignalBoxTimePDF(grl, box["start"], box["end"])

    time_pdfratio = SigOverBkgPDFRatio(
        sig_pdf=time_sigpdf,
        bkg_pdf=time_bkgpdf,
        pdf_type=TimePDF
    )

    # the next line seems to make no difference in the llh evaluation. We keep it for consistency
    analysis._llhratio.llhratio_list[0].pdfratio_list[2] = time_pdfratio
    # this line here is relevant for the llh evaluation
    analysis._llhratio.llhratio_list[0]._pdfratioarray._pdfratio_list[2] = time_pdfratio

    #  change detector signal yield with flare livetime in sample (1 / grl_norm in pdf),
    #  rebuild the histograms if it is changed...


def get_energy_spatial_signal_over_background(analysis, fitparams):
    """Returns the signal over background ratio for
    (spatial_signal * energy_signal) / (spatial_background * energy_background).

    Parameter
    ---------
    fitparams : dict
        Dictionary with {"gamma": float} for energy pdf.

    Returns
    -------
    ratio : 1d ndarray
        Product of spatial and energy signal over background pdfs.
    """

    ratio = analysis._llhratio.llhratio_list[0].pdfratio_list[0].get_ratio(analysis._tdm_list[0], fitparams)
    ratio *= analysis._llhratio.llhratio_list[0].pdfratio_list[1].get_ratio(analysis._tdm_list[0], fitparams)

    return ratio


def change_fluxmodel_gamma(analysis, gamma):
    """Set new gamma for the flux model.

    Parameter
    ---------
    gamma : float
        Spectral index for flux model.
    """

    analysis.src_hypo_group_manager.src_hypo_group_list[0].fluxmodel.gamma = gamma


def change_signal_time(analysis, gauss=None, box=None):
    """Change the signal injection to gauss or box.

    Parameters
    ----------
    gauss : dict | None
        None or dictionary {"mu": float, "sigma": float}.
    box : dict | None
        None or dictionary {"start" : float, "end" : float}.
    """

    analysis.sig_generator.set_flare(box=box, gauss=gauss)


def calculate_TS(analysis, em_results, rss):
    """Calculate the best TS value for the expectation maximization gamma scan.

    Parameters
    ----------
    em_results : 1d ndarray of tuples
        Gamma scan result.
    rss : instance of RandomStateService
        The instance of RandomStateService that should be used to generate
        random numbers from.

    Returns
    -------
    float maximized TS value
    tuple(gamma from em scan [float], best fit mean time [float], best fit width [float])
    (float ns, float gamma) fitparams from TS optimization
    """

    max_TS = 0
    best_time = None
    best_fitparams = None
    for index, result in enumerate(em_results):
        change_time_pdf(analysis,  gauss={"mu": result["mu"], "sigma": result["sigma"]})
        (fitparamset, log_lambda_max, fitparam_values, status) = analysis.maximize_llhratio(rss)
        TS = analysis.calculate_test_statistic(log_lambda_max, fitparam_values)
        if TS > max_TS:
            max_TS = TS
            best_time = result
            best_fitparams = fitparam_values

    return max_TS, best_time, best_fitparams


def run_gamma_scan_single_flare(analysis, remove_time=None, gamma_min=1, gamma_max=5, n_gamma=51):
    """Run em for different gammas in the signal energy pdf

    Parameters
    ----------
    remove_time : float
        Time information of event that should be removed.
    gamma_min : float
        Lower bound for gamma scan.
    gamma_max : float
        Upper bound for gamma scan.
    n_gamma : int
        Number of steps for gamma scan.

    Returns
    -------
    array with "gamma", "mu", "sigma", and scaling factor for flare "ns_em"
    """
    dtype = [("gamma", "f8"), ("mu", "f8"), ("sigma", "f8"), ("ns_em", "f8")]
    results = np.empty(n_gamma, dtype=dtype)

    time = analysis._tdm_list[0].get_data("time")

    for index, g in enumerate(np.linspace(gamma_min, gamma_max, n_gamma)):
        ratio = get_energy_spatial_signal_over_background(analysis, {"gamma": g})
        mu, sigma, ns = em_fit(time, ratio, n=1, tol=1.e-200, iter_max=500, weight_thresh=0,
                            initial_width=5000, remove_x=remove_time)
        results[index] = (g, mu[0], sigma[0], ns[0])

    return results


def unblind_flare(analysis, remove_time=None):
    """Run EM on unscrambled data. Similar to the original analysis, remove the alert event.

    Parameters
    ----------
    remove_time : float
        Time information of event that should be removed.
        In the case of the TXS analysis: remove_time=58018.8711856

    Returns
    -------
    array with "gamma", "mu", "sigma", and scaling factor for flare "ns_em"
    """

    # get the original unblinded data
    rss = RandomStateService(seed=1)
    analysis.unblind(rss)
    time_results = run_gamma_scan_single_flare(analysis, remove_time=remove_time)
    return time_results


def create_analysis(
    datasets,
    source,
    gauss=None,
    box=None,
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
    gauss : None or dictionary with mu, sigma
        None if no Gaussian time pdf. Else dictionary with {"mu": float, "sigma": float} of Gauss
    box : None or dictionary with start, end
        None if no Box shaped time pdf. Else dictionary with {"start": float, "end": float} of box.
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

    if len(datasets) != 1:
        raise RuntimeError(
            'This analysis supports only analyses with only single datasets '
            'at the moment!')

    if gauss is None and box is None:
        raise ValueError("No time pdf specified (box or gauss)")
    if gauss is not None and box is not None:
        raise ValueError(
            "Time PDF cannot be both Gaussian and box shaped. "
            "Please specify only one shape.")

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

    # Create the Analysis instance.
    analysis = TimeIntegratedMultiDatasetSingleSourceAnalysis(
        src_hypo_group_manager,
        src_fitparam_mapper,
        fitparam_ns,
        test_statistic,
        sig_generator_cls=PDTimeDependentSignalGenerator
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
    data_list = []
    energy_cut_splines = []
    for idx, ds in enumerate(datasets):
        # Load the data of the data set.
        data = ds.load_and_prepare_data(
            keep_fields=keep_data_fields,
            compress=compress_data,
            tl=tl)
        data_list.append(data)

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

        # Create the time PDF ratio instance for this dataset.
        if gauss is not None or box is not None:
            time_bkgpdf = BackgroundUniformTimePDF(data.grl)
            if gauss is not None:
                time_sigpdf = SignalGaussTimePDF(
                    data.grl, gauss['mu'], gauss['sigma'])
            elif box is not None:
                time_sigpdf = SignalBoxTimePDF(
                    data.grl, box["start"], box["end"])
            time_pdfratio = SigOverBkgPDFRatio(
                sig_pdf=time_sigpdf,
                bkg_pdf=time_bkgpdf,
                pdf_type=TimePDF
            )
            pdfratios.append(time_pdfratio)

        analysis.add_dataset(
            ds, data, pdfratios, tdm, event_selection_method)

        energy_cut_spline = create_energy_cut_spline(
            ds, data.exp, spl_smooth[idx])
        energy_cut_splines.append(energy_cut_spline)

        pbar.increment()
    pbar.finish()

    analysis.llhratio = analysis.construct_llhratio(minimizer, ppbar=ppbar)

    # Define the data scrambler with its data scrambling method, which is used
    # for background generation.

    # FIXME: Support multiple datasets for the DataScrambler.
    data_scrambler = DataScrambler(I3SeasonalVariationTimeScramblingMethod(data_list[0]))
    # Create background generation method.
    bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(data_scrambler)

    analysis.bkg_gen_method = bkg_gen_method
    analysis.construct_background_generator()

    analysis.construct_signal_generator(
        llhratio=analysis.llhratio, energy_cut_splines=energy_cut_splines,
        cut_sindec=cut_sindec, box=box, gauss=gauss)

    return analysis
