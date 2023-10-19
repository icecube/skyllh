# -*- coding: utf-8 -*-

"""The time_integrated_ps analysis is a multi-dataset time-integrated single
source analysis with a two-component likelihood function using a spacial and an
energy event PDF.
"""

import numpy as np

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
    PowerLawEnergyFluxProfile,
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

from skyllh.datasets.i3 import (
    data_samples,
)

from skyllh.i3.background_generation import (
    FixedScrambledExpDataI3BkgGenMethod,
)
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF,
)
from skyllh.i3.config import (
    add_icecube_specific_analysis_required_data_fields,
)

from skyllh.scripting.argparser import (
    create_argparser,
)
from skyllh.scripting.logging import (
    setup_logging,
)


def create_analysis(
        cfg,
        datasets,
        source,
        refplflux_Phi0=1,
        refplflux_E0=1e3,
        refplflux_gamma=2.0,
        ns_seed=10.0,
        ns_min=0.,
        ns_max=1e3,
        gamma_seed=3.0,
        gamma_min=1.,
        gamma_max=5.,
        kde_smoothing=False,
        minimizer_impl='LBFGS',
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
        dtc_except_fields = ['mcweight']

    # Define the flux model.
    fluxmodel = SteadyPointlikeFFM(
        Phi0=refplflux_Phi0,
        energy_profile=PowerLawEnergyFluxProfile(
            E0=refplflux_E0,
            gamma=refplflux_gamma,
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

    # Define the fit parameter gamma.
    param_gamma = Parameter(
        name='gamma',
        initial=gamma_seed,
        valmin=gamma_min,
        valmax=gamma_max)

    # Define the detector signal yield builder for the IceCube detector and this
    # source and flux model.
    # The sin(dec) binning will be taken by the builder automatically from the
    # Dataset instance.
    gamma_grid = param_gamma.as_linear_grid(delta=0.1)
    detsigyield_builder =\
        PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
            cfg=cfg,
            param_grid=gamma_grid)

    # Create a source hypothesis group manager with a single source hypothesis
    # group for the single source.
    shg_mgr = SourceHypoGroupManager(
        SourceHypoGroup(
            sources=source,
            fluxmodel=fluxmodel,
            detsigyield_builders=detsigyield_builder,
        ))
    logger.info(str(shg_mgr))

    # Define a detector model for the ns fit parameter.
    detector_model = DetectorModel('IceCube')

    # Define the parameter model mapper for the analysis, which will map global
    # parameters to local source parameters.
    pmm = ParameterModelMapper(
        models=[detector_model, source])
    pmm.map_param(param_ns, models=detector_model)
    pmm.map_param(param_gamma, models=source)
    logger.info(str(pmm))

    # Define the test statistic.
    test_statistic = WilksTestStatistic()

    # Create the Analysis instance.
    ana = Analysis(
        cfg=cfg,
        shg_mgr=shg_mgr,
        pmm=pmm,
        test_statistic=test_statistic,
        sig_generator_cls=MultiDatasetSignalGenerator,
    )

    # Define the data scrambler with its data scrambling method, which is used
    # for background generation.
    data_scrambler = DataScrambler(UniformRAScramblingMethod())

    # Create background generation method, which will be used for all datasets.
    bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(
        cfg=cfg,
        data_scrambler=data_scrambler)

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
            param_grid_set=gamma_grid,
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


if __name__ == '__main__':
    parser = create_argparser(
        description='Calculates TS for a given source location using the '
                    '10-year public point source sample.',
    )

    parser.add_argument(
        '--dec',
        dest='dec',
        default=5.7,
        type=float,
        help='The source declination in degrees.'
    )
    parser.add_argument(
        '--ra',
        dest='ra',
        default=77.35,
        type=float,
        help='The source right-ascension in degrees.'
    )
    parser.add_argument(
        '--gamma-seed',
        dest='gamma_seed',
        default=3,
        type=float,
        help='The seed value of the gamma fit parameter.'
    )

    parser.add_argument(
        '--cap-ratio',
        dest='cap_ratio',
        default=False,
        action='store_true',
        help='Switch to cap the energy PDF ratio.')

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.set_enable_tracing(args.enable_tracing)
    cfg.set_ncpu(args.n_cpu)

    setup_logging(
        cfg=cfg,
        script_logger_name=__name__,
        debug_pathfilename=args.debug_logfile)

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
            cfg=cfg,
            base_path=args.data_basepath)
        datasets.append(dsc.get_dataset(season))

    # Define a random state service.
    rss = RandomStateService(args.seed)

    # Define the point source.
    source = PointLikeSource(
        name='My Point-Like-Source',
        ra=np.deg2rad(args.ra),
        dec=np.deg2rad(args.dec))
    print(f'source: {source}')

    tl = TimeLord()

    with tl.task_timer('Creating analysis.'):
        ana = create_analysis(
            cfg=cfg,
            datasets=datasets,
            source=source,
            gamma_seed=args.gamma_seed,
            cap_ratio=args.cap_ratio,
            tl=tl)

    with tl.task_timer('Unblinding data.'):
        (TS, param_dict, status) = ana.unblind(
            minimizer_rss=rss)

    print(f'TS = {TS:g}')
    print(f'ns_fit = {param_dict["ns"]:g}')
    print(f'gamma_fit = {param_dict["gamma"]:g}')
    print(f'minimizer status = {status}')

    print(tl)

    tl = TimeLord()
    rss = RandomStateService(seed=1)
    (_, _, _, trials) = create_trial_data_file(
        ana=ana,
        rss=rss,
        n_trials=10,
        mean_n_sig=0,
        pathfilename=None,
        ncpu=1,
        tl=tl)
    print(f'trials: {trials}')
    print(tl)
