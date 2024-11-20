# -*- coding: utf-8 -*-

import numpy as np

# Set to find the correct branch for the skyllh package
import sys

from scipy import (
    integrate,
)
from astropy import(
    units,
)
from skyllh.analyses.i3.publicdata_ps.aeff import (
    PDAeff,
)
from skyllh.datasets.i3.PublicData_10y_ps import(
    create_dataset_collection,
)
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix,
)
from skyllh.analyses.i3.publicdata_ps.time_integrated_stacked_ps import (
    create_analysis,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    FctSpline1D,
)
from skyllh.core.config import (
    Config,
)
from skyllh.core.catalog import (
    SourceCatalog,
)
from skyllh.core.flux_model import (
    PowerLawEnergyFluxProfile,
    SteadyPointlikeFFM,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.scripting.argparser import (
    create_argparser,
)

'''In this file we generate the expected astrophysical diffuse background.
This information will be added to compute the energy background pdf for 
public data.'''

def prob_dec_mu_from_cartesian(src_dec,psi,sin_dec_binning=None,
                               full_output=False):
    """Given a neutrino declination and the True Angular Separation psi it 
    returns the probability of each sin_dec bin for the muon.

    Parameters
    ----------
    src_dec: float
        The declination of the true neutrino direction, in radians.
    psi: float
        The True Angular Separation between the reconstructed muon direction
    and the true neutrino direction, in radians.
    sin_dec_binning: instance of ndarray, None
        The 1d ndarray with the binedges in sin dec for which the probability 
    will be computed. If None, it sets an equal spaced binning.
    full_output: bool
        If true, it also returns the muons declinations and right ascencions
    computed.
    """
    n_points = 10000 # How fine our sample distribution will be
    psi = np.atleast_1d(psi)

    # Transform everything in radians and convert the source declination
    # to source zenith angle
    a = psi
    b = np.pi/2 - src_dec
    
    # Random rotation angle for the 2D circle
    t = np.linspace(0, 2*np.pi, n_points)

    # Parametrize the circle. We set source RA to 0, since it allows
    # to simplify formulas.

    x = (
        (np.sin(a) * np.cos(b)) * np.cos(t) -
        (np.cos(a)*np.sin(b))
    )
    y = (
        (np.sin(a)) * np.sin(t) 
    )
    z = (
        (np.sin(a)*np.sin(b)) * np.cos(t) +
        (np.cos(a)*np.cos(b))
    )

    # Convert back to right-ascension and declination.
    # This is to distinguish between diametrically opposite directions.
    zen = np.arccos(z)
    azi = np.arctan2(y, x)

    dec = np.pi/2 - zen
    ra = np.pi - azi

    if sin_dec_binning is None:
        sin_dec_binning = np.linspace(-1,1,num=10)

    sindec_prob = np.histogram(np.sin(dec),
                               bins=sin_dec_binning,
                               )[0] / n_points
    
    if full_output:
        return sindec_prob,sin_dec_binning,dec,ra
    else:
        return sindec_prob,sin_dec_binning


def create_astro_diffuse_signal_2D_hist(ds,
                                        sindec_mu_binning,
                                        logE_reco_binning,
                                        astrodiffplflux_phi0=1.44e-18,
                                        astrodiffplflux_E0=1e5,
                                        astrodiffplflux_gamma=2.37):

    """Computes the expected events from the astrophysical diffuse background
    for the given dataset in IceCube, assuming a power law spectrum. 
    
        The first output is the # of expected neutrino events per (sindec_True,
    log10_E_True) bin. The second output is the # of expected muon events per
    (sindec_Reco, log10_E_Reco) bin. 

    Parameters
    ----------
    ds : instance of Dataset
        The instance of Dataset for which we want to compute the expected
        astrophysical diffuse background.
    sindec_mu_binning: instance of BinningDefinition
        The binning definition for the sin(declination) (reconstructed) axis.
    logE_reco_binning: instance of BinningDefinition
        The binning definition for the log10(E_reconstructed) axis.
    astrodiffplflux_phi0: float
        The flux normalization to use for the astrophysical diffuse power
        law flux model. Default value from: 
        [1] https://arxiv.org/pdf/2111.10299.pdf
    astrodiffplflux_E0 : float
        The reference energy to use for the astrophysical diffuse power 
        law flux model. Default value from: 
        [1] https://arxiv.org/pdf/2111.10299.pdf
    astrodiffplflux_gamma : float
        The spectral index to use for the astrophysical diffuse law 
        flux model. Default value from: 
        [1] https://arxiv.org/pdf/2111.10299.pdf

    """

    # We load the necesary data.
    ds_data = ds.load_and_prepare_data()

    aeff = PDAeff(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('eff_area_datafile')))

    sm = PDSmearingMatrix(
        pathfilenames=ds.get_abs_pathfilename_list(
            ds.get_aux_data_definition('smearing_datafile')))
    
    # Generate astrophysical neutrino flux model
    fluxmodel_astro =  SteadyPointlikeFFM(
        Phi0=astrodiffplflux_phi0,
        energy_profile=PowerLawEnergyFluxProfile(
            E0=astrodiffplflux_E0,
            gamma=astrodiffplflux_gamma,
            cfg=ds.cfg),
            cfg=ds.cfg)
    

    # Step 1. Get number of neutrinos

    # This creates a function Aeff(sindec_nu | log10 Enu) * cos(dec_nu) 
    # for each log10 Enu bincenter.
    # We add the extra cos factor for the later integration in colatitude

    aeff_1D_spl_sindec_list = [FctSpline1D(aeff.aeff_decnu_log10enu[:,log10_enu_idx] *\
                        np.cos(aeff.decnu_bincenters) ,
                        aeff.decnu_binedges) for 
                        log10_enu_idx in 
                        range(len(aeff.log10_enu_bincenters))]
    

    # This matrix contains the values for the aeff integrated for each
    # dec nu bin for each log10 True E bin.
    # The shape is (dec_nu bin, log10_enu bin)

    aeff_sindec_integrated = np.zeros((aeff.n_decnu_bins,aeff.n_log10_enu_bins))

    for j, aeff_sindec_spl in enumerate(aeff_1D_spl_sindec_list):
        for i in range(aeff.n_decnu_bins):
            aeff_sindec_integrated[i,j] =integrate.quad(aeff_sindec_spl,
                        aeff.decnu_binedges[i],
                        aeff.decnu_binedges[i+1],
                        limit = 200)[0]
            
    # We add the contribution from the azimuth integral
    aeff_sindec_integrated *= 2 * np.pi 

    # Now we create a list of splines ready to be used for the integration
    # in True Energy.
    # Each element of the list is a spline for a sindec_nu bin already 
    # integrated before.
    aeff_1D_spl_log10enu_sindec_integrated_list = [
        FctSpline1D(aeff_sindec_integrated[sindec_nu_idx,:],
                                aeff.log10_enu_binedges) for sindec_nu_idx in
                                range(aeff.n_decnu_bins) 
    ]


    # Now we integrate each one for each log 10 Enu bin convolved with the flux
    def aeff_times_flux(Enu,fluxmodel_astro,aeff_spl):
        aeff_times_flux = fluxmodel_astro(E=Enu) * aeff_spl(np.log10(Enu)) 
        return aeff_times_flux


    n_nu_sindec_nu_log10_enu_integrated = np.zeros((aeff.n_decnu_bins,
                                                sm.n_log10_true_e_bins))

    for j, aeff_1D_spl_log10enu_sindec_integrated in enumerate(
        aeff_1D_spl_log10enu_sindec_integrated_list):
        for i in range(sm.n_log10_true_e_bins):
                n_nu_sindec_nu_log10_enu_integrated[j,i] = integrate.quad(
                    aeff_times_flux,
                    np.power(10,sm.log10_true_enu_binedges[i]),
                    np.power(10,sm.log10_true_enu_binedges[i+1]),
                    args=(fluxmodel_astro,
                        aeff_1D_spl_log10enu_sindec_integrated),
                        limit = 200)[0]
        
    # This is the number of neutrinos (if we multiply by lifetime)
    # that we detect in the (sin_dec_nu bin , log10_true_enu bin)

    # We integrate for the full time livetime of the dataset in seconds.
    n_nu_sindec_nu_log10_enu_integrated *= (ds_data.livetime * 
                                            ds.cfg.to_internal_time_unit(
                                                time_unit=units.day))

    print(f'How many neutrinos we measure for the full dataset?\
        {n_nu_sindec_nu_log10_enu_integrated.sum()}')
    

    # STEP 2: Translate to muon variables.
    # Now we get the pdf f(log10 E recon, Psi | Log10E nu, delta_nu)

    log10_reco_e_bw = sm.reco_e_upper_edges - sm.reco_e_lower_edges

    # Divide the histogram bin probability values by their bin volume.
    # We do this only where the histogram actually has non-zero entries.
    prob_hist = np.copy(sm.histogram)
    prob_hist = prob_hist.sum(axis=-1) # With this we sum over all possible sigma values

    bkg_log10_E_reco_binning = logE_reco_binning
    prob_log10_recoE_psi = np.zeros((aeff.n_decnu_bins,
                                    sm.n_log10_true_e_bins,
                                    bkg_log10_E_reco_binning.nbins,
                                    sm.n_psi_bins))
    residues = np.zeros((aeff.n_decnu_bins,
                                    sm.n_log10_true_e_bins,
                                    bkg_log10_E_reco_binning.nbins,
                                    sm.n_psi_bins))

    def zero_func(x):
        return 0 

    for i in range(sm.n_log10_true_e_bins):
        for j in range(sm.n_true_dec_bins):
            for k in range(sm.n_psi_bins):
                if np.allclose(prob_hist[i,j,:,k],0,atol=1e-10):
                    spl_help = zero_func
                else:
                    spl_help = FctSpline1D(prob_hist[i,j,:,k]/log10_reco_e_bw[i,j],
                                            sm.log10_reco_e_binedges[i,j])
                for l in range(bkg_log10_E_reco_binning.nbins):
                    prob_log10_recoE_psi[i,j,l,k],residues[i,j,l,k] = integrate.quad(spl_help,
                                                            bkg_log10_E_reco_binning.binedges[l],
                                                            bkg_log10_E_reco_binning.binedges[l+1],
                                                            limit=200)

    # Probabilities per bin p(log10 E recon, Psi | Log10E nu, delta_nu). 
    # The dimensions are  (log10Enu bin, delta_nu bin,log10 E recon bin, Psi bin)

    # Now we multiply with the number of neutrinos
    bkg_log10_E_reco_binning = logE_reco_binning
    n_mu_integrated = np.zeros((aeff.n_decnu_bins,
                                    sm.n_log10_true_e_bins,
                                    bkg_log10_E_reco_binning.nbins,
                                    sm.n_psi_bins))

    for i in range(aeff.n_decnu_bins):
        i2 = sm.get_true_dec_idx(aeff.decnu_bincenters[i])
        for j in range(sm.n_log10_true_e_bins):
            n_nu = n_nu_sindec_nu_log10_enu_integrated[i,j]
            for k in range(bkg_log10_E_reco_binning.nbins):
                n_mu_integrated[i,j,k,:] = n_nu * prob_log10_recoE_psi[j,i2,k,:] 

    # Shape is (dec_nu bin(integrated), log10 E true bin (integrated), 
    # log10 reco bin(integrated), psi bin(integrated))

    # Now we go from psi to sindec_mu.

    # Now we have (dec_nu bin(integrated), log10 E true bin(integrated)
    # (log10 reco bin, sin dec_mu )
    bkg_sindec_mu_binning = sindec_mu_binning
    n_mu_sindec_mu_integrated = np.zeros((aeff.n_decnu_bins,
                                    sm.n_log10_true_e_bins,
                                    bkg_log10_E_reco_binning.nbins,
                                    bkg_sindec_mu_binning.nbins))


    for i in range(aeff.n_decnu_bins):
        dec_nu = aeff.decnu_bincenters[i]
        i2 = sm.get_true_dec_idx(aeff.decnu_bincenters[i])
        for j in range(sm.n_log10_true_e_bins):
            for l in range(sm.n_psi_bins):
                psi = 0.5 * (sm.psi_binedges[j,i2,0,l+1] - sm.psi_binedges[j,i2,0,l])
                sindecmu_prob,_ = prob_dec_mu_from_cartesian(
                    dec_nu,psi,bkg_sindec_mu_binning.binedges)
                for k in range(bkg_log10_E_reco_binning.nbins):           
                    n_mu_sindec_mu_integrated[i,j,k,:] += \
                        n_mu_integrated[i,j,k,l] * sindecmu_prob


    # We can now collapse in the first two dimensions.
    # Finally we have our desired shape for the bkg pdf in (log10_E_reco,sindec_mu)
    n_mu_sindec_mu_integrated_reduced = np.sum(n_mu_sindec_mu_integrated,
                                            axis=(0,1))

    return n_nu_sindec_nu_log10_enu_integrated,n_mu_sindec_mu_integrated_reduced


if __name__ == '__main__':
    parser = create_argparser(
        description='Calculates the number of neutrinos and muons expected'
        'for a given dataset from the astrophysical diffuse background.',
    )
    parser.add_argument(
        '--data_basepath',
        dest='data_basepath',
        type=str,
        help='The absolute path holding the dataset data files.'
    )
    parser.add_argument(
        '--bkg_phi0',
        dest='astrodiffplflux_phi0',
        default=1.44e-18,
        type=float,
        help='The flux normalization to use for the astrophysical diffuse'
        'power law flux model.'
    )
    parser.add_argument(
        '--bkg_E0',
        dest='astrodiffplflux_E0',
        default=1e5,
        type=float,
        help='The reference energy to use for the astrophysical diffuse'
        'power law flux model.'
    )
    parser.add_argument(
        '--bkg_gamma',
        dest='astrodiffplflux_gamma',
        default=2.37,
        type=float,
        help='The spectral index to use for the astrophysical diffuse'
        'power law flux model.'
    )
    parser.add_argument(
        '--save_output',
        dest='save_output',
        default=False,
        type=bool,
        help='Option to save the outputs in a file.'
    )

    args = parser.parse_args()

    # Run analysis instance to retrieve the bkg_pdf binning for each dataset.
    cfg = Config()
    dsc = create_dataset_collection(cfg=cfg,
                                    base_path=args.data_basepath)

    datasets = dsc['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II-VII']

    catalog = SourceCatalog(f'Catalogs ')
    catalog += (
        PointLikeSource(
        name='Source ',
        weight=1,
        ra=0,
        dec=0
    ))

    ana = create_analysis(cfg=cfg,
                      datasets=datasets,
                      catalog=catalog,
                      kde_smoothing=False,
                      numerical_stabilizer=None,
                      cap_ratio=False)
    
    # Initializate data with random trial so bkg pdf is initializated

    rss_new = RandomStateService(seed=311024) 

    events_exp = ana.generate_pseudo_data(rss=rss_new,
                            mean_n_sig=0,
                            )
    data_exp = events_exp[2]

    ana.initialize_trial(events_exp[2],events_exp[1])
    (log_lambda, fitparam_values, status) = ana._llhratio.maximize(
                    rss=1,
                    tl=None)
    ts = ana.calculate_test_statistic(log_lambda=log_lambda,
                                fitparam_values=fitparam_values)
    n_inj = events_exp[0]


    bkg_energy_pdf_dict = {dataset.name: None for dataset in datasets}
    bkg_energy_2D_hist_dict = {dataset.name: None for dataset in datasets}

    for ds_idx, dataset in enumerate(datasets):
        pdf_ratios = ana.llhratio.llhratio_list[ds_idx].pdfratio
        pdf_ratios.initialize_for_new_trial(tdm=ana.tdm_list[ds_idx])
        bkg_energy_pdf = pdf_ratios.pdfratio.pdfratio2.bkg_pdf

        bkg_energy_pdf_dict[dataset.name] = bkg_energy_pdf
        bkg_energy_2D_hist_dict[dataset.name] = bkg_energy_pdf._hist_logE_sinDec

    # Run the algorithm for all datasets and store the result array.

    for ds in datasets:
        print(f'Start dataset {ds.name}')
        sindec_mu_binning = bkg_energy_pdf_dict[ds.name].get_binning('sin_dec')
        logE_reco_binning = bkg_energy_pdf_dict[ds.name].get_binning('log_energy')
        astro_diff_nu_signal_2D,astro_diff_mu_signal_2D = create_astro_diffuse_signal_2D_hist(ds=ds,
                                            sindec_mu_binning=sindec_mu_binning,
                                            logE_reco_binning=logE_reco_binning,
                                            astrodiffplflux_phi0=args.astrodiffplflux_phi0,
                                            astrodiffplflux_E0=args.astrodiffplflux_E0,
                                            astrodiffplflux_gamma=args.astrodiffplflux_gamma)
        
        if args.save_output:
            np.save(f'./astro_bkg_flux/astro_diff_nu_signal_2D-{ds.name}-new.npy',
                    astro_diff_nu_signal_2D)
            np.save(f'./astro_bkg_flux/astro_diff_mu_signal_2D-{ds.name}-new.npy',
                    astro_diff_mu_signal_2D)
                
    if args.save_output:
        with open(f'./astro_bkg_flux/summary_parameters_new.txt','w') as file:
            s = 'Parameters for the astrophysical diffuse background' +\
            'power law:\n'
            s += f'Flux normalization phi_0: {args.astrodiffplflux_phi0}'
            s += f'GeV^-1 cm^-2 s^-1 sr^-1'
            s += f'Reference energy E_0: {args.astrodiffplflux_E0} GeV\n' 
            s += f'Spectral index gamma: {args.astrodiffplflux_gamma}\n'