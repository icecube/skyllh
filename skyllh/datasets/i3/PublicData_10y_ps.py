# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

import numpy as np

from skyllh.core.dataset import (
    DatasetCollection,
    DatasetOrigin,
    WGETDatasetTransfer,
)
from skyllh.i3.dataset import (
    I3Dataset,
)


def create_dataset_collection(
        cfg,
        base_path=None,
        sub_path_fmt=None,
):
    """Defines the dataset collection for IceCube's 10-year
    point-source public data, which is available at
    http://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.zip

    Parameters
    ----------
    cfg : instance of Config
        The instance of Config holding the local configuration.
    base_path : str | None
        The base path of the data files. The actual path of a data file is
        assumed to be of the structure <base_path>/<sub_path>/<file_name>.
        If None, use the default path ``cfg['repository']['base_path']``.
    sub_path_fmt : str | None
        The sub path format of the data files of the public data sample.
        If None, use the default sub path format
        'icecube_10year_ps'.

    Returns
    -------
    dsc : DatasetCollection
        The dataset collection containing all the seasons as individual
        I3Dataset objects.
    """
    # Define the version of the data sample (collection).
    (version, verqualifiers) = (1, dict(p=0))

    # Define the default sub path format.
    default_sub_path_fmt = 'icecube_10year_ps'

    # We create a dataset collection that will hold the individual seasonal
    # public data datasets (all of the same version!).
    dsc = DatasetCollection('Public Data 10-year point-source')

    dsc.description = r"""
    The events contained in this release correspond to the IceCube's
    time-integrated point source search with 10 years of data [2]. Please refer
    to the description of the sample and known changes in the text at [1].

    The data contained in this release of IceCube’s point source sample shows
    evidence of a cumulative excess of events from four sources (NGC 1068,
    TXS 0506+056, PKS 1424+240, and GB6 J1542+6129) from a catalogue of 110
    potential sources. NGC 1068 gives the largest excess and is coincidentally
    the hottest spot in the full Northern sky search [1].

    Data from IC86-2012 through IC86-2014 used in [2] use an updated selection
    and reconstruction compared to the 7 year time-integrated search [3] and the
    detection of the 2014-2015 neutrino flare from the direction of
    TXS 0506+056 [4]. The 7 year and 10 year versions of the sample show
    overlaps of between 80 and 90%.

    An a posteriori cross check of the updated sample has been performed on
    TXS 0506+056 showing two previously-significant cascade-like events removed
    in the newer sample. These two events occur near the blazar's position
    during the TXS flare and give large reconstructed energies, but are likely
    not well-modeled by the track-like reconstructions included in this
    selection. While the events are unlikely to be track-like, their
    contribution to previous results has been handled properly.

    While the significance of the 2014-2015 TXS 0505+56 flare has decreased from
    p=7.0e-5 to 8.1e-3, the change is a result of changes to the sample and not
    of increased data. No problems have been identified with the previously
    published results and since we have no reason a priori to prefer the new
    sample over the old sample, these results do not supercede those in [4].

    This release contains data beginning in 2008 (IC40) until the spring of 2018
    (IC86-2017). This release duplicates and supplants previously released data
    from 2012 and earlier. Events from this release cannot be combined with any
    other releases

    --------------------------
    # Experimental data events
    --------------------------
    The "events" folder contains the events observed in the 10 year sample of
    IceCube's point source neutrino selection. Each file corresponds to a single
    season of IceCube datataking, including roughly one year of data. For each
    event, reconstructed particle information is included.

    - MJD: The MJD time (ut1) of the event interaction given to 1e-8 days,
    corresponding to roughly millisecond precision.

    - log10(E/GeV): The reconstructed energy of a muon passing through the
    detector. The reconstruction follows the prescription for unfolding the
    given in Section 8 of [5].

    - AngErr[deg]: The estimated angular uncertainty on the reconstructed
    direction given in degrees. The angular uncertainty is assumed to be
    symmetric in azimuth and zenith and is used to calculate the signal spatial
    probabilities for each event following the procedure given in [6]. The
    errors are calibrated using simulated events so that they provide correct
    coverage for an E^{-2} power law flux. This sample assumes a lower limit on
    the estimated angular uncertainty of 0.2 degrees.

    - RA[deg], Dec[deg]: The right ascension and declination (J2000)
    corresponding to the particle's reconstructed origin. Given in degrees.

    - Azimuth[deg], Zenith[deg]: The local coordinates of the particle's
    reconstructed origin.

    The local coordinates may be necessary when searching for transient
    phenomena on timescales shorter than 1 day due to non-uniformity in the
    detector's response as a function of azimuth. In these cases, we recommend
    scrambling events in time, then using the local coordinates and time to
    calculate new RA and Dec values.

    Note that during the preparation of this data release, one duplicated event
    was discovered in the IC86-2015 season. This event has not contributed to
    any significant excesses.

    -----------------
    # Detector uptime
    -----------------
    In order to properly account for detector uptime, IceCube maintains
    "good run lists". These contain information about "good runs", periods of
    datataking useful for analysis. Data may be marked unusable for various
    reasons, including major construction or upgrade work, calibration runs, or
    other anomalies. The "uptime" folder contains lists of the good runs for
    each season.

    - MJD_start[days], MJD_stop[days]: The start and end times for each good run

    -------------------------------
    # Instrument response functions
    -------------------------------
    In order to best model the response of the IceCube detector to a given
    signal, Monte Carlo simulations are produced for each detector
    configuration. Events are sampled from these simulations to model the
    response of point sources from an arbitrary source and spectrum.

    We provide several binned responses for the detector in the "irfs" folder
    of this data release.

    -----------------
    # Effective Areas
    -----------------
    The effective area is a property of the detector and selection which, when
    convolved with a flux model, gives the expected rate of events in the
    detector. Here we release the muon neutrino effective areas for each season
    of data.

    The effective areas are averaged over bins using simulated muon neutrino
    events ranging from 100 GeV to 100 PeV. Because the response varies widely
    in both energy and declination, we provide the tabulated response in these
    two dimensions. Due to IceCube's unique position at the south pole, the
    effective area is uniform in right ascension for timescales longer than
    1 day. It varies by about 10% as a function of azimuth, an effect which may
    be important for shorter timescales. While the azimuthal effective areas are
    not included here, they are included in IceCube's internal analyses.
    These may be made available upon request.

    Tabulated versions of the effective area are included in csv files in the
    "irfs" folder. Plotted versions are included as pdf files in the same
    location. Because the detector configuration and selection were unchanged
    after the IC86-2012 season, the effective area for this season should be
    used for IC86-2012 through IC86-2017.

    - log10(E_nu/GeV)_min, log10(E_nu/GeV)_max: The minimum and maximum of the
    energy bin used to caclulate the average effective area. Note that this uses
    the neutrino's true energy and not the reconstructed muon energy.

    - Dec_nu_min[deg], Dec_nu_max[deg]: The minimum and maximum of the
    declination of the neutrino origin. Again, note that this is the true
    direction of the neutrino and not the reconstructed muon direction.

    - A_Eff[cm^2]: The average effective area across a bin.

    -------------------
    # Smearing Matrices
    -------------------
    IceCube has a nontrivial smearing matrix with correlations between the
    directional uncertainty, the point spread function, and the reconstructed
    muon energy. To provide the most complete set of information, we include
    tables of these responses for each season from IC40 through IC86-2012.
    Seasons after IC86-2012 reuse that season's response functions.

    The included smearing matrices take the form of 5D tables mapping a
    (E_nu, Dec_nu) bin in effective area to a 3D matrix of (E, PSF, AngErr).
    The contents of each 3D matrix bin give the fractional count of simulated
    events within the bin relative to all events in the (E_nu, Dec_nu) bin.

    Fractional_Counts = [Events in (E_nu, Dec_nu, E, PSF, AngErr)] /
                        [Events in (E_nu, Dec_nu)]

    The simulations statistics, while large enough for direct sampling, are
    limited when producing these tables, ranging from just 621,858 simulated
    events for IC40 to 11,595,414 simulated events for IC86-2012. In order to
    reduce statistical uncertainties in each 5D bin, bins are selected in each
    (E_nu, Dec_nu) bin independently. The bin edges are given in the smearing
    matrix files. All locations not given have a Fractional_Counts of 0.

    - log10(E_nu/GeV)_min, log10(E_nu/GeV)_max: The minimum and maximum of the
    energy bin used to caclulate the average effective area. Note that this uses
    the neutrino's true energy and not the reconstructed muon energy.

    - Dec_nu_min[deg], Dec_nu_max[deg]: The minimum and maximum of the
    declination of the neutrino origin. Again, note that this is the true
    direction of the neutrino and not the reconstructed muon direction.

    - log10(E/GeV): The reconstructed energy of a muon passing through the
    detector. The reconstruction follows the prescription for unfolding the
    given in Section 8 of [5].

    - PSF_min[deg], PSF_max[deg]: The minimum and maximum of the true angle
    between the neutrino origin and the reconstructed muon direction.

    - AngErr_min[deg], AngErr_max[deg]: The estimated angular uncertainty on the
    reconstructed direction given in degrees. The angular uncertainty is assumed
    to be symmetric in azimuth and zenith and is used to calculate the signal
    spatial probabilities for each event following the procedure given in [6].
    The errors are calibrated so that they provide correct coverage for an
    E^{-2} power law flux. This sample assumes a lower limit on the estimated
    angular uncertainty of 0.2 degrees.

    - Fractional_Counts: The fraction of simulated events falling within each
    5D bin relative to all events in the (E_nu, Dec_nu) bin.

    ------------
    # References
    ------------
    [1] IceCube Data for Neutrino Point-Source Searches: Years 2008-2018,
        [ArXiv link](https://arxiv.org/abs/2101.09836)
    [2] Time-integrated Neutrino Source Searches with 10 years of IceCube Data,
        Phys. Rev. Lett. 124, 051103 (2020)
    [3] All-sky search for time-integrated neutrino emission from astrophysical
        sources with 7 years of IceCube data,
        Astrophys. J., 835 (2017) no. 2, 151
    [4] Neutrino emission from the direction of the blazar TXS 0506+056 prior to
        the IceCube-170922A alert,
        Science 361, 147-151 (2018)
    [5] Energy Reconstruction Methods in the IceCube Neutrino Telescope,
        JINST 9 (2014), P03009
    [6] Methods for point source analysis in high energy neutrino telescopes,
        Astropart.Phys.29:299-305,2008

    -------------
    # Last Update
    -------------
    28 January 2021
    """

    # Define the origin of the dataset.
    origin = DatasetOrigin(
        base_path='data-releases',
        sub_path='',
        filename='20210126_PS-IC40-IC86_VII.zip',
        host='icecube.wisc.edu',
        transfer_func=WGETDatasetTransfer(protocol='http').transfer,
        post_transfer_func=WGETDatasetTransfer.post_transfer_unzip,
    )

    # Define the common keyword arguments for all data sets.
    ds_kwargs = dict(
        cfg=cfg,
        livetime=None,
        version=version,
        verqualifiers=verqualifiers,
        base_path=base_path,
        default_sub_path_fmt=default_sub_path_fmt,
        sub_path_fmt=sub_path_fmt,
        origin=origin,
    )

    grl_field_name_renaming_dict = {
        'MJD_start[days]': 'start',
        'MJD_stop[days]': 'stop',
    }

    # Define the datasets for the different seasons.
    # For the declination and energy binning we use the same binning as was
    # used in the original point-source analysis using the PointSourceTracks
    # dataset.

    # ---------- IC40 ----------------------------------------------------------
    IC40 = I3Dataset(
        name='IC40',
        exp_pathfilenames='events/IC40_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC40_exp.csv',
        **ds_kwargs,
    )
    IC40.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC40.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC40_effectiveArea.csv')
    IC40.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC40_smearing.csv')

    sin_dec_bins = np.unique(np.concatenate([
        np.linspace(-1., -0.25, 10 + 1),
        np.linspace(-0.25, 0.0, 10 + 1),
        np.linspace(0.0, 1., 10 + 1),
    ]))
    IC40.define_binning('sin_dec', sin_dec_bins)

    energy_bins = np.arange(2., 9.5 + 0.01, 0.125)
    IC40.define_binning('log_energy', energy_bins)

    # ---------- IC59 ----------------------------------------------------------
    IC59 = I3Dataset(
        name='IC59',
        exp_pathfilenames='events/IC59_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC59_exp.csv',
        **ds_kwargs,
    )
    IC59.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC59.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC59_effectiveArea.csv')
    IC59.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC59_smearing.csv')

    sin_dec_bins = np.unique(np.concatenate([
        np.linspace(-1., -0.95, 2 + 1),
        np.linspace(-0.95, -0.25, 25 + 1),
        np.linspace(-0.25, 0.05, 15 + 1),
        np.linspace(0.05, 1., 10 + 1),
    ]))
    IC59.define_binning('sin_dec', sin_dec_bins)

    energy_bins = np.arange(2., 9.5 + 0.01, 0.125)
    IC59.define_binning('log_energy', energy_bins)

    # ---------- IC79 ----------------------------------------------------------
    IC79 = I3Dataset(
        name='IC79',
        exp_pathfilenames='events/IC79_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC79_exp.csv',
        **ds_kwargs,
    )
    IC79.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC79.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC79_effectiveArea.csv')
    IC79.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC79_smearing.csv')

    sin_dec_bins = np.unique(np.concatenate([
        np.linspace(-1., -0.75, 10 + 1),
        np.linspace(-0.75, 0., 15 + 1),
        np.linspace(0., 1., 20 + 1),
    ]))
    IC79.define_binning('sin_dec', sin_dec_bins)

    energy_bins = np.arange(2., 9.5 + 0.01, 0.125)
    IC79.define_binning('log_energy', energy_bins)

    # ---------- IC86-I --------------------------------------------------------
    IC86_I = I3Dataset(
        name='IC86_I',
        exp_pathfilenames='events/IC86_I_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_I_exp.csv',
        **ds_kwargs,
    )
    IC86_I.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_I.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_I_effectiveArea.csv')
    IC86_I.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_I_smearing.csv')

    b = np.sin(np.radians(-5.))  # North/South transition boundary.
    sin_dec_bins = np.unique(np.concatenate([
        np.linspace(-1., -0.2, 10 + 1),
        np.linspace(-0.2, b, 4 + 1),
        np.linspace(b, 0.2, 5 + 1),
        np.linspace(0.2, 1., 10),
    ]))
    IC86_I.define_binning('sin_dec', sin_dec_bins)

    energy_bins = np.arange(1., 10.5 + 0.01, 0.125)
    IC86_I.define_binning('log_energy', energy_bins)

    # ---------- IC86-II -------------------------------------------------------
    IC86_II = I3Dataset(
        name='IC86_II',
        exp_pathfilenames='events/IC86_II_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_II_exp.csv',
        **ds_kwargs,
    )
    IC86_II.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_II.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_II.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    sin_dec_bins = np.unique(np.concatenate([
        np.linspace(-1., -0.93, 4 + 1),
        np.linspace(-0.93, -0.3, 10 + 1),
        np.linspace(-0.3, 0.05, 9 + 1),
        np.linspace(0.05, 1., 18 + 1),
    ]))
    IC86_II.define_binning('sin_dec', sin_dec_bins)

    energy_bins = np.arange(0.5, 9.5 + 0.01, 0.125)
    IC86_II.define_binning('log_energy', energy_bins)

    # ---------- IC86-III ------------------------------------------------------
    IC86_III = I3Dataset(
        name='IC86_III',
        exp_pathfilenames='events/IC86_III_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_III_exp.csv',
        **ds_kwargs,
    )
    IC86_III.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_III.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_III.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    IC86_III.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_III.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # ---------- IC86-IV -------------------------------------------------------
    IC86_IV = I3Dataset(
        name='IC86_IV',
        exp_pathfilenames='events/IC86_IV_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_IV_exp.csv',
        **ds_kwargs,
    )
    IC86_IV.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_IV.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_IV.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    IC86_IV.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_IV.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # ---------- IC86-V --------------------------------------------------------
    IC86_V = I3Dataset(
        name='IC86_V',
        exp_pathfilenames='events/IC86_V_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_V_exp.csv',
        **ds_kwargs,
    )
    IC86_V.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_V.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_V.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    IC86_V.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_V.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # ---------- IC86-VI -------------------------------------------------------
    IC86_VI = I3Dataset(
        name='IC86_VI',
        exp_pathfilenames='events/IC86_VI_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_VI_exp.csv',
        **ds_kwargs,
    )
    IC86_VI.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_VI.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_VI.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    IC86_VI.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_VI.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # ---------- IC86-VII ------------------------------------------------------
    IC86_VII = I3Dataset(
        name='IC86_VII',
        exp_pathfilenames='events/IC86_VII_exp.csv',
        mc_pathfilenames=None,
        grl_pathfilenames='uptime/IC86_VII_exp.csv',
        **ds_kwargs,
    )
    IC86_VII.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_VII.add_aux_data_definition(
        'eff_area_datafile', 'irfs/IC86_II_effectiveArea.csv')
    IC86_VII.add_aux_data_definition(
        'smearing_datafile', 'irfs/IC86_II_smearing.csv')

    IC86_VII.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_VII.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # ---------- IC86-II-VII ---------------------------------------------------
    ds_list = [
        IC86_II,
        IC86_III,
        IC86_IV,
        IC86_V,
        IC86_VI,
        IC86_VII,
    ]
    IC86_II_VII = I3Dataset(
        name='IC86_II-VII',
        exp_pathfilenames=I3Dataset.get_combined_exp_pathfilenames(ds_list),
        mc_pathfilenames=None,
        grl_pathfilenames=I3Dataset.get_combined_grl_pathfilenames(ds_list),
        **ds_kwargs,
    )
    IC86_II_VII.grl_field_name_renaming_dict = grl_field_name_renaming_dict
    IC86_II_VII.add_aux_data_definition(
        'eff_area_datafile',
        IC86_II.get_aux_data_definition('eff_area_datafile'))

    IC86_II_VII.add_aux_data_definition(
        'smearing_datafile',
        IC86_II.get_aux_data_definition('smearing_datafile'))

    IC86_II_VII.add_binning_definition(
        IC86_II.get_binning_definition('sin_dec'))
    IC86_II_VII.add_binning_definition(
        IC86_II.get_binning_definition('log_energy'))

    # --------------------------------------------------------------------------

    dsc.add_datasets((
        IC40,
        IC59,
        IC79,
        IC86_I,
        IC86_II,
        IC86_III,
        IC86_IV,
        IC86_V,
        IC86_VI,
        IC86_VII,
        IC86_II_VII,
    ))

    dsc.set_exp_field_name_renaming_dict({
        'MJD[days]':    'time',
        'log10(E/GeV)': 'log_energy',
        'AngErr[deg]':  'ang_err',
        'RA[deg]':      'ra',
        'Dec[deg]':     'dec',
        'Azimuth[deg]': 'azi',
        'Zenith[deg]':  'zen',
    })

    def convert_deg2rad(data):
        exp = data.exp
        exp['ang_err'] = np.deg2rad(exp['ang_err'])
        exp['ra'] = np.deg2rad(exp['ra'])
        exp['dec'] = np.deg2rad(exp['dec'])
        exp['azi'] = np.deg2rad(exp['azi'])
        exp['zen'] = np.deg2rad(exp['zen'])

    dsc.add_data_preparation(convert_deg2rad)

    return dsc
