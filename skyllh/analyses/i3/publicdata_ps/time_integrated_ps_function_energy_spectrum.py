# -*- coding: utf-8 -*-

"""The time_integrated_ps analysis is a multi-dataset time-integrated single
source analysis with a two-component likelihood function using a spacial and an
energy event PDF.
"""

import numpy as np

from scipy.interpolate import splrep, BSpline

from skyllh.analyses.i3.publicdata_ps.backgroundpdf import (
    PDDataBackgroundI3EnergyPDF,
)
from skyllh.analyses.i3.publicdata_ps.detsigyield import (
    PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)
from skyllh.analyses.i3.publicdata_ps.pdfratio import (
    PDSigSetOverBkgPDFRatio,
)
from skyllh.analyses.i3.publicdata_ps.signal_generator import (
    PDDatasetSignalGenerator,
)
from skyllh.analyses.i3.publicdata_ps.signalpdf import (
    PDSignalEnergyPDFSet,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    create_energy_cut_spline,
)

from skyllh.core.analysis import (
    SingleSourceMultiDatasetLLHRatioAnalysis as Analysis,
)
from skyllh.core.background_generator import (
    DatasetBackgroundGenerator,
)
from skyllh.core.config import (
    Config,
)

from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.event_selection import (
    SpatialBoxEventSelectionMethod,
)
from skyllh.core.flux_model import (
    EpeakFunctionEnergyProfile,
    SteadyPointlikeFFM,
)
from skyllh.core.minimizer import (
    Minimizer,
    LBFGSMinimizerImpl,

)
from skyllh.core.minimizers.iminuit import (
    IMinuitMinimizerImpl,
)
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterModelMapper,
)
from skyllh.core.pdfratio import (
    SigOverBkgPDFRatio,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.scrambling import (
    DataScrambler,
    UniformRAScramblingMethod,
)
from skyllh.core.signal_generator import (
    MultiDatasetSignalGenerator,
)
from skyllh.core.signalpdf import (
    RayleighPSFPointSourceSignalSpatialPDF,
)
from skyllh.core.smoothing import (
    BlockSmoothingFilter,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.test_statistic import (
    WilksTestStatistic,
)
from skyllh.core.timing import (
    TimeLord,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)
from skyllh.core.utils.analysis import (
    create_trial_data_file,
    pointlikesource_to_data_field_array,
)

from skyllh.core.utils.tdm import (
    get_tdm_field_func_psi,
)

from skyllh.i3.config import (
    add_icecube_specific_analysis_required_data_fields,
)

from skyllh.datasets.i3 import (
    data_samples,
)

from skyllh.i3.background_generation import (
    FixedScrambledExpDataI3BkgGenMethod,
)
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF,
)

from skyllh.scripting.argparser import (
    create_argparser,
)
from skyllh.scripting.logging import (
    setup_logging,
)

cfg = Config()


def set_epeak(analysis, e_peak):
    """Change the peak energy. The shape stays the same but the spectrum is 
    moved to higher/lower energies. 

    Parameters
    ----------
    analysis : instance of SingleSourceMultiDatasetLLHRatioAnalysis
        Analysis instance with the defined flux model and signal generator
    e_peak : float
        Peak energy of the flux model (this defines the reference flux)
    """
    analysis.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile.e_peak = e_peak
    analysis.sig_generator.change_shg_mgr(analysis.shg_mgr)


def flux_from_ns(analysis, e_peak, ns):
    """Get the flux at e_eak for a certain flux model (defined by e_peak)
      for a mean number of signal neutrinos ns
    
    Parameters
    ----------
    analysis : instance of SingleSourceMultiDatasetLLHRatioAnalysis
        Analysis instance with the defined flux model and signal generator
    e_peak : float
        Peak energy of the flux model (this defines the reference flux)
    ns : float
        Mean number of detected signal neutrinos

    Returns
    -------
    flux : float
        Flux (dN / dE) in (GeV cm^2 s)^-1 at peak energy. 
    """
    #set the fluxmodel to e_peak
    set_epeak(analysis, e_peak)

    scaling_factor = analysis.calculate_fluxmodel_scaling_factor(ns, [ns, e_peak])

    return analysis.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile(E=10**e_peak).squeeze() * scaling_factor


def ns_from_flux(analysis, e_peak, flux):
    """Get the mean number of signal neutrinos ns for a certain flux model (defined by e_peak) for a flux at e_peak (1/GeV/cm2/s).
    
    Parameters
    ----------
    analysis : instance of SingleSourceMultiDatasetLLHRatioAnalysis
        Analysis instance with the defined flux model and signal generator
    e_peak : float
        Peak energy of the flux model (this defines the reference flux)
    flux : float
        Flux in 1/(GeV cm2 s)

    Returns
    -------
    ns : float
        Mean number of signal neutrinos
    """

    #set the fluxmodel to e_peak
    set_epeak(analysis, e_peak)

    # reference flux at e_peak
    reference_flux = analysis.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile(E=10**e_peak).squeeze()

    scaling_factor = flux / reference_flux

    scaling_factor_norm = analysis.calculate_fluxmodel_scaling_factor(1, [1, e_peak])
 
    return scaling_factor / scaling_factor_norm


def create_analysis(
    cfg,
    datasets,
    source,
    source_energies,
    source_energy_spectrum,
    refplflux_Phi0=1,
    ns_seed=10.0,
    ns_min=0.,
    ns_max=1e3,
    e_peak_signal=5,
    e_peak_seed=3,
    e_peak_min=1.06,
    e_peak_max=10.06,
    kde_smoothing=False,
    minimizer_impl='minuit',
    cut_sindec=None,
    spl_smooth=None,
    cap_ratio=False,
    compress_data=False,
    keep_data_fields=None,
    evt_sel_delta_angle_deg=10,
    construct_sig_generator=True,
    tl=None,
    ppbar=None,
    logger_name=None,
):
    """Creates the Analysis instance for this particular analysis.

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    datasets : list of Dataset instances
        The list of Dataset instances, which should be used in the
        analysis.
    source : PointLikeSource instance
        The PointLikeSource instance defining the point source position.
    source_energies : numpy array
        Energies in GeV for which source_energy_spectrum is given
    source_energy_spectrum : numpy array
        The energy spectrum in GeV / cm^2 / s 
    refplflux_Phi0 : float
        The flux normalization to use for the reference power law flux model.
    ns_seed : float
        Value to seed the minimizer with for the ns fit.
    ns_min : float
        Lower bound for ns fit.
    ns_max : float
        Upper bound for ns fit.
    e_peak_signal : float
        Default energy peak value for the signal generator.
    e_peak_seed : float
        Seed value for minimizer for fitting energy peak.
    e_peak_min : float
        Lower bound for energy peak fit.
    e_peak_max : float 
        Upper bound for energy peak fit,
    kde_smoothing : bool
        Apply a KDE-based smoothing to the data-driven background pdf.
        Default: False.
    minimizer_impl : str
        Minimizer implementation to be used. Supported options are ``"LBFGS"``
        (L-BFG-S minimizer used from the :mod:`scipy.optimize` module), or
        ``"minuit"`` (Minuit minimizer used by the :mod:`iminuit` module).
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
    evt_sel_delta_angle_deg : float
        The delta angle in degrees for the event selection optimization methods.
    construct_sig_generator : bool
        Flag if the signal generator should be constructed (``True``) or not
        (``False``).
    tl : TimeLord instance | None
        The TimeLord instance to use to time the creation of the analysis.
    ppbar : ProgressBar instance | None
        The instance of ProgressBar for the optional parent progress bar.
    logger_name : str | None
        The name of the logger to be used. If set to ``None``, ``__name__`` will
        be used.

    Returns
    -------
    ana : instance of SingleSourceMultiDatasetLLHRatioAnalysis
        The Analysis instance for this analysis.
    """

    add_icecube_specific_analysis_required_data_fields(cfg)

    # Remove run number from the dataset data field requirements.
    cfg['datafields'].pop('run', None)

    if logger_name is None:
        logger_name = __name__
    logger = get_logger(logger_name)


    # Create the minimizer instance.
    if minimizer_impl == "LBFGS":
        minimizer = Minimizer(LBFGSMinimizerImpl(cfg=cfg))
    elif minimizer_impl == "minuit":
        minimizer = Minimizer(IMinuitMinimizerImpl(cfg=cfg, ftol=1e-8))
    else:
        raise NameError(
            f"Minimizer implementation `{minimizer_impl}` is not supported "
            "Please use `LBFGS` or `minuit`.")

    dtc_dict = None
    dtc_except_fields = None
    if compress_data is True:
        dtc_dict = {np.dtype(np.float64): np.dtype(np.float32)}
        dtc_except_fields = ['mcweight', 'time']

    # Define the flux model.

    energy_spectrum_spline = splrep(source_energies, source_energy_spectrum / source_energies / source_energies, k=1)

    spline_eval = BSpline(*energy_spectrum_spline)

    e_peak = np.log10(source_energies[np.argmax(source_energy_spectrum)])


    fluxmodel = SteadyPointlikeFFM(
        Phi0=refplflux_Phi0,
        energy_profile=EpeakFunctionEnergyProfile(
            function=spline_eval,
            e_peak_orig=e_peak,
            e_peak_offset=e_peak_signal,
            cfg=cfg,
        ),
        cfg=cfg,
    )

    # Define the fit parameter ns.
    param_ns = Parameter(
        name='ns',
        initial=ns_seed,
        valmin=ns_min,
        valmax=ns_max)

    # Define the fit parameter e_peak.
    param_e_peak = Parameter(
        name='e_peak',
        initial=e_peak_seed,
        valmin=e_peak_min,
        valmax=e_peak_max)

    # Define the detector signal yield builder for the IceCube detector and this
    # source and flux model.
    # The sin(dec) binning will be taken by the builder automatically from the
    # Dataset instance.
    e_peak_grid = param_e_peak.as_linear_grid(delta=0.04) 
    detsigyield_builder =\
        PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
            cfg=cfg,
            param_grid=e_peak_grid)

    # Define the signal generation method.
    sig_gen_method = None

    # Create a source hypothesis group manager with a single source hypothesis
    # group for the single source.
    shg_mgr = SourceHypoGroupManager(
        SourceHypoGroup(
            sources=source,
            fluxmodel=fluxmodel,
            detsigyield_builders=detsigyield_builder,
            sig_gen_method=sig_gen_method))

    # Define a detector model for the ns fit parameter.
    detector_model = DetectorModel('IceCube')

    # Define the parameter model mapper for the analysis, which will map global
    # parameters to local source parameters.
    pmm = ParameterModelMapper(
        models=[detector_model, source])
    pmm.map_param(param_ns, models=detector_model)
    pmm.map_param(param_e_peak, models=source)

    logger.info(str(pmm))

    # Define the test statistic.
    test_statistic = WilksTestStatistic()

    # Define the data scrambler with its data scrambling method, which is used
    # for background generation.
    data_scrambler = DataScrambler(UniformRAScramblingMethod())

    # Create background generation method.
    bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(
        cfg=cfg,
        data_scrambler=data_scrambler)

    # Create the Analysis instance.
    ana = Analysis(
        cfg=cfg,
        shg_mgr=shg_mgr,
        pmm=pmm,
        test_statistic=test_statistic,
#        bkg_gen_method=bkg_gen_method,
        sig_generator_cls=MultiDatasetSignalGenerator,
    )

    # Define the event selection method for pure optimization purposes.
    # We will use the same method for all datasets.
    event_selection_method = SpatialBoxEventSelectionMethod(
        shg_mgr=shg_mgr,
        delta_angle=np.deg2rad(evt_sel_delta_angle_deg))

    # Prepare the spline parameters for the signal generator.
    if cut_sindec is None:
        cut_sindec = np.sin(np.radians([-2, 0, -3, 0, 0]))
    if spl_smooth is None:
        spl_smooth = [0., 0.005, 0.05, 0.2, 0.3]
    if len(spl_smooth) < len(datasets) or len(cut_sindec) < len(datasets):
        raise AssertionError(
            'The length of the spl_smooth and of the cut_sindec must be equal '
            f'to the length of datasets: {len(datasets)}.')

    # Add the data sets to the analysis.
    pbar = ProgressBar(len(datasets), parent=ppbar).start()
    for (ds_idx, ds) in enumerate(datasets):
        data = ds.load_and_prepare_data(
            keep_fields=keep_data_fields,
            dtc_dict=dtc_dict,
            dtc_except_fields=dtc_except_fields,
 #           compress=compress_data,
            tl=tl)

        sin_dec_binning = ds.get_binning_definition('sin_dec')
        log_energy_binning = ds.get_binning_definition('log_energy')

        # Create the spatial PDF ratio instance for this dataset.
        spatial_sigpdf = RayleighPSFPointSourceSignalSpatialPDF(
            cfg=cfg,
            dec_range=np.arcsin(sin_dec_binning.range))
        spatial_bkgpdf = DataBackgroundI3SpatialPDF(
            cfg=cfg,
            data_exp=data.exp,
            sin_dec_binning=sin_dec_binning)
        spatial_pdfratio = SigOverBkgPDFRatio(
            cfg=cfg,
            sig_pdf=spatial_sigpdf,
            bkg_pdf=spatial_bkgpdf)

        # Create the energy PDF ratio instance for this dataset.
        energy_sigpdfset = PDSignalEnergyPDFSet(
            cfg=cfg,
            ds=ds,
            src_dec=source.dec,
            fluxmodel=fluxmodel,
            param_grid_set=e_peak_grid,
            ppbar=ppbar
        )
        smoothing_filter = BlockSmoothingFilter(nbins=1)
        energy_bkgpdf = PDDataBackgroundI3EnergyPDF(
            cfg=cfg,
            data_exp=data.exp,
            logE_binning=log_energy_binning,
            sinDec_binning=sin_dec_binning,
            smoothing_filter=smoothing_filter,
            kde_smoothing=kde_smoothing)

        energy_pdfratio = PDSigSetOverBkgPDFRatio(
            cfg=cfg,
            sig_pdf_set=energy_sigpdfset,
            bkg_pdf=energy_bkgpdf,
            cap_ratio=cap_ratio)

        pdfratio = spatial_pdfratio * energy_pdfratio

        # Create a trial data manager and add the required data fields.
        tdm = TrialDataManager()
        tdm.add_source_data_field(
            name='src_array',
            func=pointlikesource_to_data_field_array)
        tdm.add_data_field(
            name='psi',
            func=get_tdm_field_func_psi(),
            dt='dec',
            is_srcevt_data=True)

        energy_cut_spline = create_energy_cut_spline(
            ds,
            data.exp,
            spl_smooth[ds_idx])

        bkg_generator = DatasetBackgroundGenerator(
            cfg=cfg,
            dataset=ds,
            data=data,
            bkg_gen_method=bkg_gen_method,
        )

        sig_generator = PDDatasetSignalGenerator(
            cfg=cfg,
            shg_mgr=shg_mgr,
            ds=ds,
            ds_idx=ds_idx,
            energy_cut_spline=energy_cut_spline,
            cut_sindec=cut_sindec[ds_idx],
        )

        ana.add_dataset(
            dataset=ds,
            data=data,
            pdfratio=pdfratio,
            tdm=tdm,
            event_selection_method=event_selection_method,
            bkg_generator=bkg_generator,
            sig_generator=sig_generator)

        pbar.increment()
    pbar.finish()

    ana.construct_services(
        ppbar=ppbar)

    ana.llhratio = ana.construct_llhratio(
        minimizer=minimizer,
        ppbar=ppbar)

    if construct_sig_generator is True:
        ana.construct_signal_generator()

    return ana

