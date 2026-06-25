The events contained in this release correspond to the IceCube's time-integrated point source search with 10 years of data [2]. Please refer to the description of the sample and known changes in the text at [1].

The data contained in this release of IceCube’s point source sample shows evidence of a cumulative excess of events from four sources (NGC 1068, TXS 0506+056, PKS 1424+240, and GB6 J1542+6129) from a catalogue of 110 potential sources. NGC 1068 gives the largest excess and is coincidentally the hottest spot in the full Northern sky search [1].

Data from IC86-2012 through IC86-2014 used in [2] use an updated selection and reconstruction compared to the 7 year time-integrated search [3] and the detection of the 2014-2015 neutrino flare from the direction of TXS 0506+056 [4]. The 7 year and 10 year versions of the sample show overlaps of between 80 and 90%.  

An a posteriori cross check of the updated sample has been performed on TXS 0506+056 showing two previously-significant cascade-like events removed in the newer sample. These two events occur near the blazar's position during the TXS flare and give large reconstructed energies, but are likely not well-modeled by the track-like reconstructions included in this selection. While the events are unlikely to be track-like, their contribution to previous results has been handled properly. 

While the significance of the 2014-2015 TXS 0505+56 flare has decreased from p=7.0e-5 to 8.1e-3, the change is a result of changes to the sample and not of increased data. No problems have been identified with the previously published results and since we have no reason a priori to prefer the new sample over the old sample, these results do not supercede those in [4].

This release contains data beginning in 2008 (IC40) until the spring of 2018 (IC86-2017). This release duplicates and supplants previously released data from 2012 and earlier. Events from this release cannot be combined with any other releases

-----------------------------------------
# Experimental data events
-----------------------------------------
The "events" folder contains the events observed in the 10 year sample of IceCube's point source neutrino selection. Each file corresponds to a single season of IceCube datataking, including roughly one year of data. For each event, reconstructed particle information is included.

- MJD: The MJD time (ut1) of the event interaction given to 1e-8 days, corresponding to roughly millisecond precision.

- log10(E/GeV): The reconstructed energy of a muon passing through the detector. The reconstruction follows the prescription for unfolding the given in Section 8 of [5].

- AngErr[deg]: The estimated angular uncertainty on the reconstructed direction given in degrees. The angular uncertainty is assumed to be symmetric in azimuth and zenith and is used to calculate the signal spatial probabilities for each event following the procedure given in [6]. The errors are calibrated using simulated events so that they provide correct coverage for an E^{-2} power law flux. This sample assumes a lower limit on the estimated angular uncertainty of 0.2 degrees. 

- RA[deg], Dec[deg]: The right ascension and declination (J2000) corresponding to the particle's reconstructed origin. Given in degrees.

- Azimuth[deg], Zenith[deg]: The local coordinates of the particle's reconstructed origin. 

The local coordinates may be necessary when searching for transient phenomena on timescales shorter than 1 day due to non-uniformity in the detector's response as a function of azimuth. In these cases, we recommend scrambling events in time, then using the local coordinates and time to calculate new RA and Dec values.

Note that during the preparation of this data release, one duplicated event was discovered in the IC86-2015 season. This event has not contributed to any significant excesses.

-----------------------------------------
# Detector uptime
-----------------------------------------
In order to properly account for detector uptime, IceCube maintains "good run lists". These contain information about "good runs", periods of datataking useful for analysis. Data may be marked unusable for various reasons, including major construction or upgrade work, calibration runs, or other anomalies. The "uptime" folder contains lists of the good runs for each season.

- MJD_start[days], MJD_stop[days]: The start and end times for each good run

-----------------------------------------
# Instrument response functions
-----------------------------------------
In order to best model the response of the IceCube detector to a given signal, Monte Carlo simulations are produced for each detector configuration. Events are sampled from these simulations to model the response of point sources from an arbitrary source and spectrum.

We provide several binned responses for the detector in the "irfs" folder of this data release.

------------------
# Effective Areas
------------------
The effective area is a property of the detector and selection which, when convolved with a flux model, gives the expected rate of events in the detector. Here we release the muon neutrino effective areas for each season of data. 

The effective areas are averaged over bins using simulated muon neutrino events ranging from 100 GeV to 100 PeV. Because the response varies widely in both energy and declination, we provide the tabulated response in these two dimensions. Due to IceCube's unique position at the south pole, the effective area is uniform in right ascension for timescales longer than 1 day. It varies by about 10% as a function of azimuth, an effect which may be important for shorter timescales. While the azimuthal effective areas are not included here, they are included in IceCube's internal analyses. These may be made available upon request.

Tabulated versions of the effective area are included in csv files in the "irfs" folder. Plotted versions are included as pdf files in the same location. Because the detector configuration and selection were unchanged after the IC86-2012 season, the effective area for this season should be used for IC86-2012 through IC86-2017. 

- log10(E_nu/GeV)_min, log10(E_nu/GeV)_max: The minimum and maximum of the energy bin used to caclulate the average effective area. Note that this uses the neutrino's true energy and not the reconstructed muon energy.

- Dec_nu_min[deg], Dec_nu_max[deg]: The minimum and maximum of the declination of the neutrino origin. Again, note that this is the true direction of the neutrino and not the reconstructed muon direction. 

- A_Eff[cm^2]: The average effective area across a bin.

------------------
# Smearing Matrices
------------------
IceCube has a nontrivial smearing matrix with correlations between the directional uncertainty, the point spread function, and the reconstructed muon energy. To provide the most complete set of information, we include tables of these responses for each season from IC40 through IC86-2012. Seasons after IC86-2012 reuse that season's response functions.

The included smearing matrices take the form of 5D tables mapping a (E_nu, Dec_nu) bin in effective area to a 3D matrix of (E, PSF, AngErr). The contents of each 3D matrix bin give the fractional count of simulated events within the bin relative to all events in the (E_nu, Dec_nu) bin.

Fractional_Counts = [Events in (E_nu, Dec_nu, E, PSF, AngErr)] / [Events in (E_nu, Dec_nu)]

The simulations statistics, while large enough for direct sampling, are limited when producing these tables, ranging from just 621,858 simulated events for IC40 to 11,595,414 simulated events for IC86-2012. In order to reduce statistical uncertainties in each 5D bin, bins are selected in each (E_nu, Dec_nu) bin independently. The bin edges are given in the smearing matrix files. All locations not given have a Fractional_Counts of 0.

- log10(E_nu/GeV)_min, log10(E_nu/GeV)_max: The minimum and maximum of the energy bin used to caclulate the average effective area. Note that this uses the neutrino's true energy and not the reconstructed muon energy.

- Dec_nu_min[deg], Dec_nu_max[deg]: The minimum and maximum of the declination of the neutrino origin. Again, note that this is the true direction of the neutrino and not the reconstructed muon direction. 

- log10(E/GeV): The reconstructed energy of a muon passing through the detector. The reconstruction follows the prescription for unfolding the given in Section 8 of [5].

- PSF_min[deg], PSF_max[deg]: The minimum and maximum of the true angle between the neutrino origin and the reconstructed muon direction. 

- AngErr_min[deg], AngErr_max[deg]: The estimated angular uncertainty on the reconstructed direction given in degrees. The angular uncertainty is assumed to be symmetric in azimuth and zenith and is used to calculate the signal spatial probabilities for each event following the procedure given in [6]. The errors are calibrated so that they provide correct coverage for an E^{-2} power law flux. This sample assumes a lower limit on the estimated angular uncertainty of 0.2 degrees. 

- Fractional_Counts: The fraction of simulated events falling within each 5D bin relative to all events in the (E_nu, Dec_nu) bin.

-----------------------------------------
# References
-----------------------------------------
[1] IceCube Data for Neutrino Point-Source Searches: Years 2008-2018, [[ArXiv link]]
[2] Time-integrated Neutrino Source Searches with 10 years of IceCube Data, Phys. Rev. Lett. 124, 051103 (2020)
[3] All-sky search for time-integrated neutrino emission from astrophysical sources with 7 years of IceCube data, Astrophys. J., 835 (2017) no. 2, 151
[4] Neutrino emission from the direction of the blazar TXS 0506+056 prior to the IceCube-170922A alert, Science 361, 147-151 (2018)
[5] Energy Reconstruction Methods in the IceCube Neutrino Telescope, JINST 9 (2014), P03009
[6] Methods for point source analysis in high energy neutrino telescopes, Astropart.Phys.29:299-305,2008

-----------------------------------------
# Last Update
-----------------------------------------
28 January 2021
