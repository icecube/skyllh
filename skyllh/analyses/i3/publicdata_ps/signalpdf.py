# -*- coding: utf-8 -*-

import numpy as np

import os
import pickle

from copy import deepcopy
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from itertools import product

from skyllh.core.py import module_classname
from skyllh.core.debugging import get_logger
from skyllh.core.timing import TaskTimer
from skyllh.core.binning import (
    BinningDefinition,
    UsesBinning,
    get_bincenters_from_binedges,
    get_bin_indices_from_lower_and_upper_binedges
)
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.pdf import (
    PDF,
    PDFAxis,
    PDFSet,
    IsSignalPDF,
    EnergyPDF
)
from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize
)
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet
)
from skyllh.core.smoothing import SmoothingFilter
from skyllh.i3.pdf import I3EnergyPDF
from skyllh.i3.dataset import I3Dataset
from skyllh.physics.flux import FluxModel

from skyllh.analyses.i3.publicdata_ps.pd_aeff import (
    PDAeff,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    FctSpline1D,
    create_unionized_smearing_matrix_array,
    load_smearing_histogram,
    psi_to_dec_and_ra,
    PublicDataSmearingMatrix,
    merge_reco_energy_bins
)



class PublicDataSignalGenerator(object):
    def __init__(self, ds, **kwargs):
        """Creates a new instance of the signal generator for generating signal
        events from the provided public data smearing matrix.
        """
        super().__init__(**kwargs)

        self.smearing_matrix = PublicDataSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('smearing_datafile')))

    def _generate_events(
            self, rss, src_dec, src_ra, dec_idx, flux_model, n_events):
        """Generates `n_events` signal events for the given source location
        and flux model.

        Note:
            Some values can be NaN in cases where a PDF was not available!

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for drawing random
            numbers.
        src_dec : float
            The declination of the source in radians.
        src_ra : float
            The right-ascention of the source in radians.

        Returns
        -------
        events : numpy record array of size `n_events`
            The numpy record array holding the event data.
            It contains the following data fields:
                - 'isvalid'
                - 'log_true_energy'
                - 'log_energy'
                - 'sin_dec'
            Single values can be NaN in cases where a pdf was not available.
        """
        # Create the output event array.
        out_dtype = [
            ('isvalid', np.bool_),
            ('log_true_energy', np.double),
            ('log_energy', np.double),
            ('sin_dec', np.double)
        ]
        events = np.empty((n_events,), dtype=out_dtype)

        sm = self.smearing_matrix

        # Determine the true energy range for which log_e PDFs are available.
        (min_log_true_e,
         max_log_true_e) = sm.get_true_log_e_range_with_valid_log_e_pdfs(
             dec_idx)

        # First draw a true neutrino energy from the hypothesis spectrum.
        log_true_e = np.log10(flux_model.get_inv_normed_cdf(
            rss.random.uniform(size=n_events),
            E_min=10**min_log_true_e,
            E_max=10**max_log_true_e
        ))

        events['log_true_energy'] = log_true_e

        log_true_e_idxs = (
            np.digitize(log_true_e, bins=sm.true_e_bin_edges) - 1
        )
        # Sample reconstructed energies given true neutrino energies.
        (log_e_idxs, log_e) = sm.sample_log_e(
            rss, dec_idx, log_true_e_idxs)
        events['log_energy'] = log_e

        # Sample reconstructed psi values given true neutrino energy and
        # reconstructed energy.
        (psi_idxs, psi) = sm.sample_psi(
            rss, dec_idx, log_true_e_idxs, log_e_idxs)

        # Sample reconstructed ang_err values given true neutrino energy,
        # reconstructed energy, and psi.
        (ang_err_idxs, ang_err) = sm.sample_ang_err(
            rss, dec_idx, log_true_e_idxs, log_e_idxs, psi_idxs)

        isvalid = np.invert(
            np.isnan(log_e) | np.isnan(psi) | np.isnan(ang_err))
        events['isvalid'] = isvalid

        # Convert the psf into a set of (r.a. and dec.). Only use non-nan
        # values.
        (dec, ra) = psi_to_dec_and_ra(rss, src_dec, src_ra, psi[isvalid])
        events['sin_dec'][isvalid] = np.sin(dec)

        return events

    def generate_signal_events(
            self, rss, src_dec, src_ra, flux_model, n_events):
        """Generates ``n_events`` signal events for the given source location
        and flux model.

        Returns
        -------
        events : numpy record array
            The numpy record array holding the event data.
            It contains the following data fields:
                - 'isvalid'
                - 'log_energy'
                - 'sin_dec'
        """
        sm = self.smearing_matrix

        # Find the declination bin index.
        dec_idx = sm.get_true_dec_idx(src_dec)

        events = None
        n_evt_generated = 0
        while n_evt_generated != n_events:
            n_evt = n_events - n_evt_generated

            events_ = self._generate_events(
                rss, src_dec, src_ra, dec_idx, flux_model, n_evt)

            # Cut events that failed to be generated due to missing PDFs.
            events_ = events_[events_['isvalid']]

            n_evt_generated += len(events_)
            if events is None:
                events = events_
            else:
                events = np.concatenate((events, events_))

        return events


class PublicDataSignalI3EnergyPDF(EnergyPDF, IsSignalPDF, UsesBinning):
    """Class that implements the enegry signal PDF for a given flux model given
    the public data.
    """

    def __init__(self, ds, flux_model, data_dict=None):
        """Constructs a new enegry PDF instance using the public IceCube data.

        Parameters
        ----------
        ds : instance of I3Dataset
            The I3Dataset instance holding the file name of the smearing data of
            the public data.
        flux_model : instance of FluxModel
            The flux model that should be used to calculate the energy signal
            pdf.
        data_dict : dict | None
            If not None, the histogram data and its bin edges can be provided.
            The dictionary needs the following entries:

            - 'histogram'
            - 'true_e_bin_edges'
            - 'true_dec_bin_edges'
            - 'reco_e_lower_edges'
            - 'reco_e_upper_edges'
        """
        super().__init__()

        if(not isinstance(ds, I3Dataset)):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')
        if(not isinstance(flux_model, FluxModel)):
            raise TypeError(
                'The flux_model argument must be an instance of FluxModel!')

        self._ds = ds
        self._flux_model = flux_model

        if(data_dict is None):
            (self.histogram,
             true_e_bin_edges,
             true_dec_bin_edges,
             self.reco_e_lower_edges,
             self.reco_e_upper_edges
             ) = load_smearing_histogram(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('smearing_datafile')))
        else:
            self.histogram = data_dict['histogram']
            true_e_bin_edges = data_dict['true_e_bin_edges']
            true_dec_bin_edges = data_dict['true_dec_bin_edges']
            self.reco_e_lower_edges = data_dict['reco_e_lower_edges']
            self.reco_e_upper_edges = data_dict['reco_e_upper_edges']

        # Get the number of bins for each of the variables in the matrix.
        # The number of bins for e_mu, psf, and ang_err for each true_e and
        # true_dec bin are equal.
        self.add_binning(BinningDefinition('true_e', true_e_bin_edges))
        self.add_binning(BinningDefinition('true_dec', true_dec_bin_edges))

        # Marginalize over the PSF and angular error axes.
        self.histogram = np.sum(self.histogram, axis=(3, 4))

        # Create a (prob vs E_reco) spline for each source declination bin.
        n_true_dec = len(true_dec_bin_edges) - 1
        true_e_binning = self.get_binning('true_e')
        self.spline_norm_list = []
        for true_dec_idx in range(n_true_dec):
            (spl, norm) = self.get_total_weighted_energy_pdf(
                true_dec_idx, true_e_binning)
            self.spline_norm_list.append((spl, norm))

    @property
    def ds(self):
        """(read-only) The I3Dataset instance for which this enegry signal PDF
        was constructed.
        """
        return self._ds

    @property
    def flux_model(self):
        """(read-only) The FluxModel instance for which this energy signal PDF
        was constructed.
        """
        return self._flux_model

    def _create_spline(self, bin_centers, values, order=1, smooth=0):
        """Creates a :class:`scipy.interpolate.UnivariateSpline` with the
        given order and smoothing factor.
        """
        spline = UnivariateSpline(
            bin_centers, values, k=order, s=smooth, ext='zeros'
        )

        return spline

    def get_weighted_energy_pdf_hist_for_true_energy_dec_bin(
            self, true_e_idx, true_dec_idx, flux_model, log_e_min=0):
        """Gets the reconstructed muon energy pdf histogram for a specific true
        neutrino energy and declination bin weighted with the assumed flux
        model.

        Parameters
        ----------
        true_e_idx : int
            The index of the true enegry bin.
        true_dec_idx : int
            The index of the true declination bin.
        flux_model : instance of FluxModel
            The FluxModel instance that represents the flux formula.
        log_e_min : float
            The minimal reconstructed energy in log10 to be considered for the
            PDF.

        Returns
        -------
        energy_pdf_hist : 1d ndarray | None
            The enegry PDF values.
            None is returned if all PDF values are zero.
        bin_centers : 1d ndarray | None
            The bin center values for the energy PDF values.
            None is returned if all PDF values are zero.
        """
        # Find the index of the true neutrino energy bin and the corresponding
        # distribution for the reconstructed muon energy.
        energy_pdf_hist = deepcopy(self.histogram[true_e_idx, true_dec_idx])

        # Check whether there is no pdf in the table for this neutrino energy.
        if(np.sum(energy_pdf_hist) == 0):
            return (None, None)

        # Get the reco energy bin centers.
        lower_binedges = self.reco_e_lower_edges[true_e_idx, true_dec_idx]
        upper_binedges = self.reco_e_upper_edges[true_e_idx, true_dec_idx]
        bin_centers = 0.5 * (lower_binedges + upper_binedges)

        # Convolve the reco energy pdf with the flux model.
        energy_pdf_hist *= flux_model.get_integral(
            np.power(10, lower_binedges), np.power(10, upper_binedges)
        )

        # Find where the reconstructed energy is below the minimal energy and
        # mask those values. We don't have any reco energy below the minimal
        # enegry in the data.
        mask = bin_centers >= log_e_min
        bin_centers = bin_centers[mask]
        bin_widths = upper_binedges[mask] - lower_binedges[mask]
        energy_pdf_hist = energy_pdf_hist[mask]

        # Re-normalize in case some bins were cut.
        energy_pdf_hist /= np.sum(energy_pdf_hist * bin_widths)

        return (energy_pdf_hist, bin_centers)

    def get_total_weighted_energy_pdf(
            self, true_dec_idx, true_e_binning, log_e_min=2, order=1, smooth=0):
        """Gets the reconstructed muon energy distribution weighted with the
        assumed flux model and marginalized over all possible true neutrino
        energies for a given true declination bin. The function generates a
        spline, and calculates its integral for later normalization.

        Parameters
        ----------
        true_dec_idx : int
            The index of the true declination bin.
        true_e_binning : instance of BinningDefinition
            The BinningDefinition instance holding the true energy binning
            information.
        log_e_min : float
            The log10 value of the minimal energy to be considered.
        order : int
            The order of the spline.
        smooth : int
            The smooth strength of the spline.

        Returns
        -------
        spline : instance of scipy.interpolate.UnivariateSpline
            The enegry PDF spline.
        norm : float
            The integral of the enegry PDF spline.
        """
        # Loop over the true energy bins and for each create a spline for the
        # reconstructed muon energy pdf.
        splines = []
        bin_centers = []
        for true_e_idx in range(true_e_binning.nbins):
            (e_pdf, e_pdf_bin_centers) =\
                self.get_weighted_energy_pdf_hist_for_true_energy_dec_bin(
                    true_e_idx, true_dec_idx, self.flux_model
            )
            if(e_pdf is None):
                continue
            splines.append(
                self._create_spline(e_pdf_bin_centers, e_pdf)
            )
            bin_centers.append(e_pdf_bin_centers)

        # Build a (non-normalized) spline for the total reconstructed muon
        # energy pdf by summing the splines corresponding to each true energy.
        # Take as x values for the spline all the bin centers of the single
        # reconstructed muon energy pdfs.
        spline_x_vals = np.sort(
            np.unique(
                np.concatenate(bin_centers)
            )
        )

        spline = self._create_spline(
            spline_x_vals,
            np.sum([spl(spline_x_vals) for spl in splines], axis=0),
            order=order,
            smooth=smooth
        )
        norm = spline.integral(
            np.min(spline_x_vals), np.max(spline_x_vals)
        )

        return (spline, norm)

    def calc_prob_for_true_dec_idx(self, true_dec_idx, log_energy, tl=None):
        """Calculates the PDF value for the given true declination bin and the
        given log10(E_reco) energy values.
        """
        (spline, norm) = self.spline_norm_list[true_dec_idx]
        with TaskTimer(tl, 'Evaluating logE spline.'):
            prob = spline(log_energy) / norm
        return prob

    def get_prob(self, tdm, fitparams=None, tl=None):
        """Calculates the energy probability (in log10(E)) of each event.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy' : float
                The 10-base logarithm of the energy value of the event.
            - 'src_array' : (n_sources,)-shaped record array with the follwing
                data fields:

                - 'dec' : float
                    The declination of the source.
        fitparams : None
            Unused interface parameter.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : 1D (N_events,) shaped ndarray
            The array with the energy probability for each event.
        """
        get_data = tdm.get_data

        src_array = get_data('src_array')
        if(len(src_array) != 1):
            raise NotImplementedError(
                'The PDF class "{}" is only implemneted for a single '
                'source! {} sources were defined!'.format(
                    self.__class__.name, len(src_array)))

        src_dec = get_data('src_array')['dec'][0]
        true_dec_binning = self.get_binning('true_dec')
        true_dec_idx = np.digitize(src_dec, true_dec_binning.binedges)

        log_energy = get_data('log_energy')

        prob = self.calc_prob_for_true_dec_idx(true_dec_idx, log_energy, tl=tl)

        return prob


class PublicDataSignalI3EnergyPDFSet(PDFSet, IsSignalPDF, IsParallelizable):
    """This is the signal energy PDF for IceCube using public data.
    It creates a set of PublicDataI3EnergyPDF objects for a discrete set of
    energy signal parameters.
    """

    def __init__(
            self,
            rss,
            ds,
            flux_model,
            fitparam_grid_set,
            n_events=int(1e6),
            smoothing_filter=None,
            ncpu=None,
            ppbar=None):
        """
        """
        if(isinstance(fitparam_grid_set, ParameterGrid)):
            fitparam_grid_set = ParameterGridSet([fitparam_grid_set])
        if(not isinstance(fitparam_grid_set, ParameterGridSet)):
            raise TypeError('The fitparam_grid_set argument must be an '
                            'instance of ParameterGrid or ParameterGridSet!')

        if((smoothing_filter is not None) and
           (not isinstance(smoothing_filter, SmoothingFilter))):
            raise TypeError('The smoothing_filter argument must be None or '
                            'an instance of SmoothingFilter!')

        # We need to extend the fit parameter grids on the lower and upper end
        # by one bin to allow for the calculation of the interpolation. But we
        # will do this on a copy of the object.
        fitparam_grid_set = fitparam_grid_set.copy()
        fitparam_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            pdf_type=I3EnergyPDF,
            fitparams_grid_set=fitparam_grid_set,
            ncpu=ncpu)

        def create_I3EnergyPDF(
                logE_binning, sinDec_binning, smoothing_filter,
                aeff, siggen, flux_model, n_events, gridfitparams, rss):
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            my_flux_model = flux_model.copy(newprop=gridfitparams)

            # Generate signal events for sources in every sin(dec) bin.
            # The physics weight is the effective area of the event given its
            # true energy and true declination.
            data_physicsweight = None
            events = None
            n_evts = int(np.round(n_events / sinDec_binning.nbins))
            for sin_dec in sinDec_binning.bincenters:
                src_dec = np.arcsin(sin_dec)
                events_ = siggen.generate_signal_events(
                    rss=rss,
                    src_dec=src_dec,
                    src_ra=np.radians(180),
                    flux_model=my_flux_model,
                    n_events=n_evts)
                data_physicsweight_ = aeff.get_aeff(
                    np.repeat(sin_dec, len(events_)),
                    events_['log_true_energy'])
                if events is None:
                    events = events_
                    data_physicsweight = data_physicsweight_
                else:
                    events = np.concatenate(
                        (events, events_))
                    data_physicsweight = np.concatenate(
                        (data_physicsweight, data_physicsweight_))

            data_logE = events['log_energy']
            data_sinDec = events['sin_dec']
            data_mcweight = np.ones((len(events),), dtype=np.double)

            epdf = I3EnergyPDF(
                data_logE=data_logE,
                data_sinDec=data_sinDec,
                data_mcweight=data_mcweight,
                data_physicsweight=data_physicsweight,
                logE_binning=logE_binning,
                sinDec_binning=sinDec_binning,
                smoothing_filter=smoothing_filter
            )

            return epdf

        print('Generate signal energy PDF for ds {} with {} CPUs'.format(
            ds.name, self.ncpu))

        # Create a signal generator for this dataset.
        siggen = PublicDataSignalGenerator(ds)

        aeff = PDAeff(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('eff_area_datafile')))

        logE_binning = ds.get_binning_definition('log_energy')
        sinDec_binning = ds.get_binning_definition('sin_dec')

        args_list = [
            ((logE_binning, sinDec_binning, smoothing_filter, aeff,
              siggen, flux_model, n_events, gridfitparams), {})
            for gridfitparams in self.gridfitparams_list
        ]

        epdf_list = parallelize(
            create_I3EnergyPDF,
            args_list,
            self.ncpu,
            rss=rss,
            ppbar=ppbar)

        # Save all the energy PDF objects in the PDFSet PDF registry with
        # the hash of the individual parameters as key.
        for (gridfitparams, epdf) in zip(self.gridfitparams_list, epdf_list):
            self.add_pdf(epdf, gridfitparams)

    def assert_is_valid_for_exp_data(self, data_exp):
        pass

    def get_prob(self, tdm, gridfitparams):
        """Calculates the signal energy probability (in logE) of each event for
        a given set of signal fit parameters on a grid.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy' : float
                The logarithm of the energy value of the event.
            - 'src_array' : 1d record array
                The source record array containing the declination of the
                sources.
        gridfitparams : dict
            The dictionary holding the signal parameter values for which the
            signal energy probability should be calculated. Note, that the
            parameter values must match a set of parameter grid values for which
            a PublicDataSignalI3EnergyPDF object has been created at
            construction time of this PublicDataSignalI3EnergyPDFSet object.
            There is no interpolation method defined
            at this point to allow for arbitrary parameter values!

        Returns
        -------
        prob : 1d ndarray
            The array with the signal energy probability for each event.

        Raises
        ------
        KeyError
            If no energy PDF can be found for the given signal parameter values.
        """
        epdf = self.get_pdf(gridfitparams)

        prob = epdf.get_prob(tdm)
        return prob


class PDSignalEnergyPDF_old(PDF, IsSignalPDF):
    """This class provides a signal energy PDF for a spectrial index value.
    """

    def __init__(
            self, f_e, log_e_edges, **kwargs):
        """Creates a new signal energy PDF instance for a particular spectral
        index value.
        """
        super().__init__(**kwargs)

        self.f_e = f_e

        self.log_e_lower_edges = log_e_edges[:-1]
        self.log_e_upper_edges = log_e_edges[1:]

        # Add the PDF axes.
        self.add_axis(PDFAxis(
            name='log_energy',
            vmin=self.log_e_lower_edges[0],
            vmax=self.log_e_upper_edges[-1])
        )

        # Check integrity.
        integral = np.sum(self.f_e * np.diff(log_e_edges))
        if not np.isclose(integral, 1):
            raise ValueError(
                'The integral over log10_E of the energy term must be unity! '
                'But it is {}!'.format(integral))

        # Create a spline of the PDF.
        self._create_spline(order=1, s=0)

    def _create_spline(self, order=1, s=0):
        """Creates the spline representation of the energy PDF.
        """
        log10_e_bincenters = 0.5*(
            self.log_e_lower_edges + self.log_e_upper_edges)
        self.spl_rep = interpolate.splrep(
            log10_e_bincenters, self.f_e,
            xb=self.log_e_lower_edges[0],
            xe=self.log_e_upper_edges[-1],
            k=order,
            s=s
        )
        self.spl_norm = integrate.quad(
            self._eval_spline,
            self.log_e_lower_edges[0], self.log_e_upper_edges[-1],
            limit=200, full_output=1)[0]

    def _eval_spline(self, x):
        return interpolate.splev(x, self.spl_rep, der=0)

    def assert_is_valid_for_trial_data(self, tdm):
        pass

    def get_splined_pd_by_log10_e(self, log10_e, tl=None):
        """Calculates the probability density for the given log10(E/GeV)
        values using the spline representation of the PDF.


        """
        # Select events that actually have a signal enegry PDF.
        # All other events will get zero signal probability.
        m = (
            (log10_e >= self.log_e_lower_edges[0]) &
            (log10_e < self.log_e_upper_edges[-1])
        )

        pd = np.zeros((len(log10_e),), dtype=np.double)

        pd[m] = self._eval_spline(log10_e[m]) / self.spl_norm

        return pd

    def get_pd_by_log10_e(self, log10_e, tl=None):
        """Calculates the probability density for the given log10(E/GeV)
        values.

        Parameters
        ----------
        log10_e : (n_events,)-shaped 1D numpy ndarray
            The numpy ndarray holding the log10(E/GeV) values.
        tl : TimeLord | None
            The optional TimeLord instance to measure code timing information.
        """
        # Select events that actually have a signal enegry PDF.
        # All other events will get zero signal probability.
        m = (
            (log10_e >= self.log_e_lower_edges[0]) &
            (log10_e < self.log_e_upper_edges[-1])
        )

        log_e_idxs = get_bin_indices_from_lower_and_upper_binedges(
            self.log_e_lower_edges, self.log_e_upper_edges, log10_e[m])

        pd = np.zeros((len(log10_e),), dtype=np.double)
        pd[m] = self.f_e[log_e_idxs]

        return pd

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability density for the events given by the
        TrialDataManager.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the data events for which the
            probability should be looked up. The following data fields are
            required:
                - 'log_energy'
                    The log10 of the reconstructed energy.
                - 'psi'
                    The opening angle from the source to the event in radians.
                - 'ang_err'
                    The angular error of the event in radians.
        params : dict | None
            The dictionary containing the parameter names and values for which
            the probability should get calculated.
            By definition this PDF does not depend on parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability density for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.
        """
        log10_e = tdm.get_data('log_energy')

        pd = self.get_pd_by_log10_e(log10_e, tl=tl)

        return (pd, None)


class PDSignalEnergyPDFSet_old(PDFSet, IsSignalPDF, IsParallelizable):
    """This class provides a signal energy PDF set for the public data.
    It creates a set of PDSignalEnergyPDF instances, one for each spectral
    index value on a grid.
    """

    def __init__(
            self,
            ds,
            src_dec,
            flux_model,
            fitparam_grid_set,
            union_sm_arr_pathfilename=None,
            smoothing=1,
            ncpu=None,
            ppbar=None,
            **kwargs):
        """Creates a new PDSignalEnergyPDFSet instance for the public data.

        Parameters
        ----------
        ds : I3Dataset instance
            The I3Dataset instance that defines the public data dataset.
        src_dec : float
            The declination of the source in radians.
        flux_model : FluxModel instance
            The FluxModel instance that defines the source's flux model.
        fitparam_grid_set : ParameterGrid | ParameterGridSet instance
            The parameter grid set defining the grids of the fit parameters.
        union_sm_arr_pathfilename : str | None
            The pathfilename of the unionized smearing matrix array file from
            which the unionized smearing matrix array should get loaded from.
            If None, the unionized smearing matrix array will be created.
        smoothing : int
            The number of bins to combine to create a smoother energy pdf.
            Eight seems to produce good results.
        """
        self._logger = get_logger(module_classname(self))

        # Check for the correct types of the arguments.
        if not isinstance(ds, I3Dataset):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')

        if not isinstance(flux_model, FluxModel):
            raise TypeError(
                'The flux_model argument must be an instance of FluxModel!')

        if (not isinstance(fitparam_grid_set, ParameterGrid)) and\
           (not isinstance(fitparam_grid_set, ParameterGridSet)):
            raise TypeError(
                'The fitparam_grid_set argument must be an instance of type '
                'ParameterGrid or ParameterGridSet!')

        # Extend the fitparam_grid_set to allow for parameter interpolation
        # values at the grid edges.
        fitparam_grid_set = fitparam_grid_set.copy()
        fitparam_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            pdf_type=PDF,
            fitparams_grid_set=fitparam_grid_set,
            ncpu=ncpu
        )

        # Load the unionized smearing matrix array or create it if no one was
        # specified.
        if ((union_sm_arr_pathfilename is not None) and
                os.path.exists(union_sm_arr_pathfilename)):
            self._logger.info(
                'Loading unionized smearing matrix from file "{}".'.format(
                    union_sm_arr_pathfilename))
            with open(union_sm_arr_pathfilename, 'rb') as f:
                data = pickle.load(f)
        else:
            pathfilenames = ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('smearing_datafile'))
            self._logger.info(
                'Creating unionized smearing matrix from smearing matrix file '
                '"{}".'.format(
                    pathfilenames))
            sm = PublicDataSmearingMatrix(
                pathfilenames=pathfilenames)
            data = create_unionized_smearing_matrix_array(sm, src_dec)
            if union_sm_arr_pathfilename is not None:
                self._logger.info(
                    'Saving unionized smearing matrix to file "{}".'.format(
                        union_sm_arr_pathfilename))
                with open(union_sm_arr_pathfilename, 'wb') as f:
                    pickle.dump(data, f)
            del(sm)
        union_arr = data['union_arr']
        log10_true_e_binedges = data['log10_true_e_binedges']
        log10_reco_e_edges = data['log10_reco_e_binedges']
        psi_edges = data['psi_binedges']
        ang_err_edges = data['ang_err_binedges']
        del(data)

        # Merge small energy bins.
        bw_th = 0.1
        max_bw = 0.2
        (union_arr, log10_reco_e_edges) = merge_reco_energy_bins(
            union_arr, log10_reco_e_edges, bw_th, max_bw)

        true_e_binedges = np.power(10, log10_true_e_binedges)
        nbins_true_e = len(true_e_binedges) - 1

        # Calculate the neutrino enegry bin widths in GeV.
        dE_nu = np.diff(true_e_binedges)
        self._logger.debug(
            'dE_nu = {}'.format(dE_nu)
        )

        # Load the effective area.
        aeff = PDAeff(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('eff_area_datafile')))

        # Calculate the detector's neutrino energy detection probability to
        # detect a neutrino of energy E_nu given a neutrino declination:
        # p(E_nu|dec)
        det_prob = np.empty((len(dE_nu),), dtype=np.double)
        for i in range(len(dE_nu)):
            det_prob[i] = aeff.get_detection_prob_for_sin_true_dec(
                sin_true_dec=np.sin(src_dec),
                true_e_min=true_e_binedges[i],
                true_e_max=true_e_binedges[i+1],
                true_e_range_min=true_e_binedges[0],
                true_e_range_max=true_e_binedges[-1]
            )

        self._logger.debug('det_prob = {}, sum = {}'.format(
            det_prob, np.sum(det_prob)))

        if not np.isclose(np.sum(det_prob), 1):
            self._logger.warn(
                'The sum of the detection probabilities is not unity! It is '
                '{}.'.format(np.sum(det_prob)))

        log10_reco_e_bw = np.diff(log10_reco_e_edges)
        psi_edges_bw = np.diff(psi_edges)
        ang_err_bw = np.diff(ang_err_edges)

        bin_volumes = (
            log10_reco_e_bw[:, np.newaxis, np.newaxis] *
            psi_edges_bw[np.newaxis, :, np.newaxis] *
            ang_err_bw[np.newaxis, np.newaxis, :])

        # Create the energy pdf for different gamma values.
        def create_energy_pdf(union_arr, flux_model, gridfitparams):
            """Creates an energy pdf for a specific gamma value.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            my_flux_model = flux_model.copy(newprop=gridfitparams)

            E_nu_min = true_e_binedges[:-1]
            E_nu_max = true_e_binedges[1:]

            self._logger.debug(
                'Generate signal energy PDF for parameters {} in {} E_nu '
                'bins.'.format(
                    gridfitparams, nbins_true_e)
            )

            # Calculate the flux probability p(E_nu|gamma).
            flux_prob = (
                my_flux_model.get_integral(E_nu_min, E_nu_max) /
                my_flux_model.get_integral(
                    true_e_binedges[0],
                    true_e_binedges[-1]
                )
            )
            if not np.isclose(np.sum(flux_prob), 1):
                self._logger.warn(
                    'The sum of the flux probabilities is not unity! It is '
                    '{}.'.format(np.sum(flux_prob)))

            self._logger.debug(
                'flux_prob = {}, sum = {}'.format(
                    flux_prob, np.sum(flux_prob))
            )

            p = flux_prob * det_prob

            true_e_prob = p / np.sum(p)

            self._logger.debug(
                'true_e_prob = {}'.format(
                    true_e_prob))

            transfer = np.copy(union_arr)
            for true_e_idx in range(nbins_true_e):
                transfer[true_e_idx] *= true_e_prob[true_e_idx]
            pdf_arr = np.sum(transfer, axis=0)
            del(transfer)

            # Normalize the pdf, which is the probability per bin volume.
            norm = np.sum(pdf_arr)
            if norm == 0:
                raise ValueError(
                    'The signal PDF is empty for {}! This should '
                    'not happen. Check the parameter ranges!'.format(
                        str(gridfitparams)))
            pdf_arr /= norm
            pdf_arr /= bin_volumes

            # Create the enegry PDF f_e = P(log10_E_reco|dec) =
            # \int dPsi dang_err P(E_reco,Psi,ang_err).
            f_e = np.sum(
                pdf_arr * psi_edges_bw[np.newaxis, :, np.newaxis] *
                ang_err_bw[np.newaxis, np.newaxis, :],
                axis=(1, 2))

            del(pdf_arr)

            # Combine always step bins to smooth out the pdf.
            step = smoothing
            n = len(log10_reco_e_edges)-1
            n_new = int(np.ceil((len(log10_reco_e_edges)-1)/step,))
            f_e_new = np.zeros((n_new,), dtype=np.double)
            log10_reco_e_edges_new = np.zeros(
                (n_new+1), dtype=np.double)
            start = 0
            k = 0
            while start <= n-1:
                end = np.min([start+step, n])

                v = np.sum(f_e[start:end]) / (end - start)
                f_e_new[k] = v
                log10_reco_e_edges_new[k] = log10_reco_e_edges[start]

                start += step
                k += 1
            log10_reco_e_edges_new[-1] = log10_reco_e_edges[-1]

            # Re-normalize the PDF.
            f_e_new = f_e_new / np.sum(f_e_new) / \
                np.diff(log10_reco_e_edges_new)

            pdf = PDSignalEnergyPDF(f_e_new, log10_reco_e_edges_new)

            return pdf

        args_list = [
            ((union_arr, flux_model, gridfitparams), {})
            for gridfitparams in self.gridfitparams_list
        ]

        pdf_list = parallelize(
            create_energy_pdf,
            args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        del(union_arr)

        # Save all the energy PDF objects in the PDFSet PDF registry with
        # the hash of the individual parameters as key.
        for (gridfitparams, pdf) in zip(self.gridfitparams_list, pdf_list):
            self.add_pdf(pdf, gridfitparams)

    def get_prob(self, tdm, gridfitparams, tl=None):
        """Calculates the signal probability density of each event for the
        given set of signal fit parameters on a grid.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy'
                The log10 of the reconstructed energy.
            - 'psi'
                The opening angle from the source to the event in radians.
            - 'ang_err'
                The angular error of the event in radians.
        gridfitparams : dict
            The dictionary holding the signal parameter values for which the
            signal energy probability should be calculated. Note, that the
            parameter values must match a set of parameter grid values for which
            a PDSignalPDF object has been created at construction time of this
            PDSignalPDFSet object.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure time.

        Returns
        -------
        prob : 1d ndarray
            The array with the signal energy probability for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.

        Raises
        ------
        KeyError
            If no energy PDF can be found for the given signal parameter values.
        """
        # print('Getting signal PDF for gridfitparams={}'.format(
        #    str(gridfitparams)))
        pdf = self.get_pdf(gridfitparams)

        (prob, grads) = pdf.get_prob(tdm, tl=tl)

        return (prob, grads)


#def eval_spline(x, spl):
    #values = spl(x)
    #values = np.nan_to_num(values, nan=0)
    #return values


#def create_spline(log10_e_bincenters, f_e, norm=False):
    #"""Creates the spline representation of the energy PDF.
    #"""

    #spline = interpolate.PchipInterpolator(
        #log10_e_bincenters, f_e, extrapolate=False
    #)

    #if norm:
        #spl_norm = integrate.quad(
            #eval_spline,
            #log10_e_bincenters[0], log10_e_bincenters[-1],
            #args=(spline,),
            #limit=200, full_output=1)[0]

        #return spline, spl_norm

    #else:
        #return spline


class PDSignalEnergyPDF(PDF, IsSignalPDF):
    """This class provides a signal energy PDF for a spectrial index value.
    """
    def __init__(
            self, f_e_spl, **kwargs):
        """Creates a new signal energy PDF instance for a particular spectral
        index value.

        Parameters
        ----------
        f_e_spl : FctSpline1D instance
            The FctSpline1D instance representing the spline of the energy PDF.
        """
        super().__init__(**kwargs)

        if not isinstance(f_e_spl, FctSpline1D):
            raise TypeError(
                'The f_e_spl argument must be an instance of FctSpline1D!')

        self.f_e_spl = f_e_spl

        self.log10_reco_e_lower_binedges = self.f_e_spl.x_binedges[:-1]
        self.log10_reco_e_upper_binedges = self.f_e_spl.x_binedges[1:]

        self.log10_reco_e_min = self.log10_reco_e_lower_binedges[0]
        self.log10_reco_e_max = self.log10_reco_e_upper_binedges[-1]

        # Add the PDF axes.
        self.add_axis(PDFAxis(
            name='log_energy',
            vmin=self.log10_reco_e_min,
            vmax=self.log10_reco_e_max)
        )

        # Check integrity.
        integral = integrate.quad(
            self.f_e_spl.evaluate,
            self.log10_reco_e_min,
            self.log10_reco_e_max,
            limit=200,
            full_output=1
        )[0] / self.f_e_spl.norm
        if not np.isclose(integral, 1):
            raise ValueError(
                'The integral over log10_reco_e of the energy term must be '
                'unity! But it is {}!'.format(integral))

    def assert_is_valid_for_trial_data(self, tdm):
        pass

    def get_pd_by_log10_reco_e(self, log10_reco_e, tl=None):
        """Calculates the probability density for the given log10(E_reco/GeV)
        values using the spline representation of the PDF.

        Parameters
        ----------
        log10_reco_e : (n_log10_reco_e,)-shaped 1D numpy ndarray
            The numpy ndarray holding the log10(E_reco/GeV) values for which
            the energy PDF should get evaluated.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.
        """
        # Select events that actually have a signal enegry PDF.
        # All other events will get zero signal probability density.
        m = (
            (log10_reco_e >= self.log10_reco_e_min) &
            (log10_reco_e < self.log10_reco_e_max)
        )

        with TaskTimer(tl, 'Evaluate PDSignalEnergyPDF'):
            pd = np.zeros((len(log10_reco_e),), dtype=np.double)
            pd[m] = self.f_e_spl(log10_reco_e[m]) / self.f_e_spl.norm

        return pd

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability density for the events given by the
        TrialDataManager.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the data events for which the
            probability should be looked up. The following data fields are
            required:
                - 'log_energy'
                    The log10 of the reconstructed energy.
        params : dict | None
            The dictionary containing the parameter names and values for which
            the probability should get calculated.
            By definition this PDF does not depend on parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability density for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.
        """
        log10_reco_e = tdm.get_data('log_energy')

        pd = self.get_pd_by_log10_reco_e(log10_reco_e, tl=tl)

        return (pd, None)


class PDSignalEnergyPDFSet(PDFSet, IsSignalPDF, IsParallelizable):
    """This class provides a signal energy PDF set for the public data.
    It creates a set of PDSignalEnergyPDF instances, one for each spectral
    index value on a grid.
    """
    def __init__(
            self,
            ds,
            src_dec,
            flux_model,
            fitparam_grid_set,
            ncpu=None,
            ppbar=None,
            **kwargs):
        """Creates a new PDSignalEnergyPDFSet instance for the public data.

        Parameters
        ----------
        ds : I3Dataset instance
            The I3Dataset instance that defines the dataset of the public data.
        src_dec : float
            The declination of the source in radians.
        flux_model : FluxModel instance
            The FluxModel instance that defines the source's flux model.
        fitparam_grid_set : ParameterGrid | ParameterGridSet instance
            The parameter grid set defining the grids of the fit parameters.
        """
        self._logger = get_logger(module_classname(self))

        # Check for the correct types of the arguments.
        if not isinstance(ds, I3Dataset):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')

        if not isinstance(flux_model, FluxModel):
            raise TypeError(
                'The flux_model argument must be an instance of FluxModel!')

        if (not isinstance(fitparam_grid_set, ParameterGrid)) and\
           (not isinstance(fitparam_grid_set, ParameterGridSet)):
            raise TypeError(
                'The fitparam_grid_set argument must be an instance of type '
                'ParameterGrid or ParameterGridSet!')

        # Extend the fitparam_grid_set to allow for parameter interpolation
        # values at the grid edges.
        fitparam_grid_set = fitparam_grid_set.copy()
        fitparam_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            pdf_type=PDF,
            fitparams_grid_set=fitparam_grid_set,
            ncpu=ncpu
        )

        # Load the smearing matrix.
        sm = PublicDataSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('smearing_datafile')))

        # Select the slice of the smearing matrix corresponding to the
        # source declination band.
        # Note that we take the pdfs of the reconstruction calculated
        # from the smearing matrix here.
        true_dec_idx = sm.get_true_dec_idx(src_dec)
        sm_pdf = sm.pdf[:, true_dec_idx]

        # Only look at true neutrino energies for which a recostructed
        # muon energy distribution exists in the smearing matrix.
        (min_log_true_e,
         max_log_true_e) = sm.get_true_log_e_range_with_valid_log_e_pdfs(
            true_dec_idx)
        log_true_e_mask = np.logical_and(
            sm.log10_true_enu_binedges >= min_log_true_e,
            sm.log10_true_enu_binedges <= max_log_true_e)
        true_enu_binedges = np.power(
            10, sm.log10_true_enu_binedges[log_true_e_mask])
        true_enu_binedges_lower = true_enu_binedges[:-1]
        true_enu_binedges_upper = true_enu_binedges[1:]
        valid_true_e_idxs = [sm.get_log10_true_e_idx(0.5 * (he + le))
            for he,le in zip(
                sm.log10_true_enu_binedges[log_true_e_mask][1:],
                sm.log10_true_enu_binedges[log_true_e_mask][:-1])
            ]

        # Define the values at which to evaluate the splines.
        # Some bins might have zero bin widths.
        # m = (sm.log10_reco_e_binedges_upper[valid_true_e_idxs, true_dec_idx] -
        #      sm.log10_reco_e_binedges_lower[valid_true_e_idxs, true_dec_idx]) > 0
        # le = sm.log10_reco_e_binedges_lower[valid_true_e_idxs, true_dec_idx][m]
        # ue = sm.log10_reco_e_binedges_upper[valid_true_e_idxs, true_dec_idx][m]
        # min_log10_reco_e = np.min(le)
        # max_log10_reco_e = np.max(ue)
        # d_log10_reco_e = np.min(ue - le) / 20
        # n_xvals = int((max_log10_reco_e - min_log10_reco_e) / d_log10_reco_e)
        # xvals_binedges = np.linspace(
        #     min_log10_reco_e,
        #     max_log10_reco_e,
        #     n_xvals+1
        # )
        # xvals = get_bincenters_from_binedges(xvals_binedges)

        xvals_binedges = ds.get_binning_definition('log_energy').binedges
        xvals = get_bincenters_from_binedges(xvals_binedges)

        # Calculate the neutrino enegry bin widths in GeV.
        d_enu = np.diff(true_enu_binedges)
        self._logger.debug(
            'dE_nu = {}'.format(d_enu)
        )

        # Load the effective area.
        aeff = PDAeff(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('eff_area_datafile')))

        # Calculate the detector's neutrino energy detection probability to
        # detect a neutrino of energy E_nu given a neutrino declination:
        # p(E_nu|dec)
        det_prob = aeff.get_detection_prob_for_decnu(
            decnu=src_dec,
            enu_min=true_enu_binedges[:-1],
            enu_max=true_enu_binedges[1:],
            enu_range_min=true_enu_binedges[0],
            enu_range_max=true_enu_binedges[-1]
        )

        self._logger.debug('det_prob = {}, sum = {}'.format(
            det_prob, np.sum(det_prob)))

        if not np.isclose(np.sum(det_prob), 1):
            self._logger.warn(
                'The sum of the detection probabilities is not unity! It is '
                '{}.'.format(np.sum(det_prob)))

        psi_edges_bw = sm.psi_upper_edges - sm.psi_lower_edges
        ang_err_bw = sm.ang_err_upper_edges - sm.ang_err_lower_edges

        # Create the energy pdf for different gamma values.
        def create_energy_pdf(sm_pdf, flux_model, gridfitparams):
            """Creates an energy pdf for a specific gamma value.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            my_flux_model = flux_model.copy(newprop=gridfitparams)

            self._logger.debug(
                'Generate signal energy PDF for parameters {} in {} E_nu '
                'bins.'.format(
                    gridfitparams, len(valid_true_e_idxs))
            )

            # Calculate the flux probability p(E_nu|gamma).
            flux_prob = (
                my_flux_model.get_integral(
                    true_enu_binedges_lower,
                    true_enu_binedges_upper
                ) /
                my_flux_model.get_integral(
                    true_enu_binedges[0],
                    true_enu_binedges[-1]
                )
            )
            if not np.isclose(np.sum(flux_prob), 1):
                self._logger.warn(
                    'The sum of the flux probabilities is not unity! It is '
                    '{}.'.format(np.sum(flux_prob)))

            self._logger.debug(
                'flux_prob = {}, sum = {}'.format(
                    flux_prob, np.sum(flux_prob))
            )

            p = flux_prob * det_prob

            true_e_prob = p / np.sum(p)

            self._logger.debug(
                'true_e_prob = {}'.format(
                    true_e_prob))

            def create_reco_e_pdf_for_true_e(idx, true_e_idx):
                """This functions creates a spline for the reco energy
                distribution given a true neutrino engery.
                """
                # Create the enegry PDF f_e = P(log10_E_reco|dec) =
                # \int dPsi dang_err P(E_reco,Psi,ang_err).
                f_e = np.sum(
                    sm_pdf[true_e_idx] *
                    psi_edges_bw[true_e_idx, true_dec_idx, :, :, np.newaxis] *
                    ang_err_bw[true_e_idx, true_dec_idx, :, :, :],
                    axis=(-1, -2)
                )

                # Now build the spline to use it in the sum over the true
                # neutrino energy. At this point, add the weight of the pdf
                # with the true neutrino energy probability.
                log10_reco_e_binedges = sm.log10_reco_e_binedges[
                    true_e_idx, true_dec_idx]

                p = f_e * true_e_prob[idx]

                spline = FctSpline1D(p, log10_reco_e_binedges)

                return spline(xvals)

            # Integrate over the true neutrino energy and spline the output.
            sum_pdf = np.sum([
                create_reco_e_pdf_for_true_e(i, true_e_idx)
                for i,true_e_idx in enumerate(valid_true_e_idxs)
            ], axis=0)

            spline = FctSpline1D(sum_pdf, xvals_binedges, norm=True)

            pdf = PDSignalEnergyPDF(spline)

            return pdf

        args_list = [
            ((sm_pdf, flux_model, gridfitparams), {})
            for gridfitparams in self.gridfitparams_list
        ]

        pdf_list = parallelize(
            create_energy_pdf,
            args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        del(sm_pdf)

        # Save all the energy PDF objects in the PDFSet PDF registry with
        # the hash of the individual parameters as key.
        for (gridfitparams, pdf) in zip(self.gridfitparams_list, pdf_list):
            self.add_pdf(pdf, gridfitparams)

    def get_prob(self, tdm, gridfitparams, tl=None):
        """Calculates the signal probability density of each event for the
        given set of signal fit parameters on a grid.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy'
                The log10 of the reconstructed energy.
            - 'psi'
                The opening angle from the source to the event in radians.
            - 'ang_err'
                The angular error of the event in radians.
        gridfitparams : dict
            The dictionary holding the signal parameter values for which the
            signal energy probability should be calculated. Note, that the
            parameter values must match a set of parameter grid values for which
            a PDSignalPDF object has been created at construction time of this
            PDSignalPDFSet object.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure time.

        Returns
        -------
        prob : 1d ndarray
            The array with the signal energy probability for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.

        Raises
        ------
        KeyError
            If no energy PDF can be found for the given signal parameter values.
        """
        # print('Getting signal PDF for gridfitparams={}'.format(
        #    str(gridfitparams)))
        pdf = self.get_pdf(gridfitparams)

        (prob, grads) = pdf.get_prob(tdm, tl=tl)

        return (prob, grads)


class PDSignalPDF_unionized_matrix(PDF, IsSignalPDF):
    """This class provides a signal pdf for a given spectrial index value.
    """

    def __init__(
            self, f_s, f_e, log_e_edges, psi_edges, ang_err_edges,
            true_e_prob, **kwargs):
        """Creates a new signal PDF for the public data.

        Parameters
        ----------
        f_s : (n_e_reco, n_psi, n_ang_err)-shaped 3D numpy ndarray
            The conditional PDF array P(Psi|E_reco,ang_err).

        """
        super().__init__(**kwargs)

        self.f_s = f_s
        self.f_e = f_e

        self.log_e_lower_edges = log_e_edges[:-1]
        self.log_e_upper_edges = log_e_edges[1:]

        self.psi_lower_edges = psi_edges[:-1]
        self.psi_upper_edges = psi_edges[1:]

        self.ang_err_lower_edges = ang_err_edges[:-1]
        self.ang_err_upper_edges = ang_err_edges[1:]

        self.true_e_prob = true_e_prob

        # Add the PDF axes.
        self.add_axis(PDFAxis(
            name='log_energy',
            vmin=self.log_e_lower_edges[0],
            vmax=self.log_e_upper_edges[-1])
        )
        self.add_axis(PDFAxis(
            name='psi',
            vmin=self.psi_lower_edges[0],
            vmax=self.psi_lower_edges[-1])
        )
        self.add_axis(PDFAxis(
            name='ang_err',
            vmin=self.ang_err_lower_edges[0],
            vmax=self.ang_err_upper_edges[-1])
        )

        # Check integrity.
        integral = np.sum(
            # 1/(2*np.pi*np.sin(0.5*(psi_edges[None,1:,None]+
            #                       psi_edges[None,:-1,None])
            #                 )) *
            self.f_s * np.diff(psi_edges)[None, :, None], axis=1)
        if not np.all(np.isclose(integral[integral > 0], 1)):
            raise ValueError(
                'The integral over Psi of the spatial term must be unity! '
                'But it is {}!'.format(integral[integral > 0]))
        integral = np.sum(
            self.f_e * np.diff(log_e_edges)
        )
        if not np.isclose(integral, 1):
            raise ValueError(
                'The integral over log10_E of the energy term must be unity! '
                'But it is {}!'.format(integral))

    def assert_is_valid_for_trial_data(self, tdm):
        pass

    def get_prob(self, tdm, params=None, tl=None):
        """Calculates the probability density for the events given by the
        TrialDataManager.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The TrialDataManager instance holding the data events for which the
            probability should be looked up. The following data fields are
            required:
                - 'log_energy'
                    The log10 of the reconstructed energy.
                - 'psi'
                    The opening angle from the source to the event in radians.
                - 'ang_err'
                    The angular error of the event in radians.
        params : dict | None
            The dictionary containing the parameter names and values for which
            the probability should get calculated.
            By definition this PDF does not depend on parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : (N_events,)-shaped numpy ndarray
            The 1D numpy ndarray with the probability density for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.
        """
        log_e = tdm.get_data('log_energy')
        psi = tdm.get_data('psi')
        ang_err = tdm.get_data('ang_err')

        # Select events that actually have a signal PDF.
        # All other events will get zero signal probability.
        m = (
            (log_e >= self.log_e_lower_edges[0]) &
            (log_e < self.log_e_upper_edges[-1]) &
            (psi >= self.psi_lower_edges[0]) &
            (psi < self.psi_upper_edges[-1]) &
            (ang_err >= self.ang_err_lower_edges[0]) &
            (ang_err < self.ang_err_upper_edges[-1])
        )

        log_e_idxs = get_bin_indices_from_lower_and_upper_binedges(
            self.log_e_lower_edges, self.log_e_upper_edges, log_e[m])
        psi_idxs = get_bin_indices_from_lower_and_upper_binedges(
            self.psi_lower_edges, self.psi_upper_edges, psi[m])
        ang_err_idxs = get_bin_indices_from_lower_and_upper_binedges(
            self.ang_err_lower_edges, self.ang_err_upper_edges, ang_err[m])

        pd_spatial = np.zeros((len(psi),), dtype=np.double)
        pd_spatial[m] = (
            1/(2*np.pi * np.sin(psi[m])) *
            self.f_s[(log_e_idxs, psi_idxs, ang_err_idxs)]
        )

        pd_energy = np.zeros((len(log_e),), dtype=np.double)
        pd_energy[m] = self.f_e[log_e_idxs]

        return (pd_spatial * pd_energy, None)


class PDSignalPDFSet_unionized_matrix(PDFSet, IsSignalPDF, IsParallelizable):
    """This class provides a signal PDF set for the public data.
    """

    def __init__(
            self,
            ds,
            src_dec,
            flux_model,
            fitparam_grid_set,
            union_sm_arr_pathfilename=None,
            ncpu=None,
            ppbar=None,
            **kwargs):
        """Creates a new PDSignalPDFSet instance for the public data.

        Parameters
        ----------
        ds : I3Dataset instance
            The I3Dataset instance that defines the public data dataset.
        src_dec : float
            The declination of the source in radians.
        flux_model : FluxModel instance
            The FluxModel instance that defines the source's flux model.
        """
        self._logger = get_logger(module_classname(self))

        # Check for the correct types of the arguments.
        if not isinstance(ds, I3Dataset):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')

        if not isinstance(flux_model, FluxModel):
            raise TypeError(
                'The flux_model argument must be an instance of FluxModel!')

        # Extend the fitparam_grid_set to allow for parameter interpolation
        # values at the grid edges.
        fitparam_grid_set = fitparam_grid_set.copy()
        fitparam_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            pdf_type=PDF,
            fitparams_grid_set=fitparam_grid_set,
            ncpu=ncpu
        )

        if(union_sm_arr_pathfilename is not None):
            with open(union_sm_arr_pathfilename, 'rb') as f:
                data = pickle.load(f)
        else:
            sm = PublicDataSmearingMatrix(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('smearing_datafile')))
            data = create_unionized_smearing_matrix_array(sm, src_dec)
            del(sm)
        union_arr = data['union_arr']
        log_true_e_binedges = data['log10_true_e_binedges']
        reco_e_edges = data['log10_reco_e_binedges']
        psi_edges = data['psi_binedges']
        ang_err_edges = data['ang_err_binedges']
        del(data)

        true_e_bincenters = np.power(
            10,
            0.5*(log_true_e_binedges[:-1] + log_true_e_binedges[1:]))

        true_e_binedges = np.power(10, log_true_e_binedges)

        # Calculate the neutrino enegry bin widths in GeV.
        dE_nu = np.diff(true_e_binedges)
        self._logger.debug(
            'dE_nu = {}'.format(dE_nu)
        )

        # Load the effective area.
        aeff = PDAeff(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('eff_area_datafile')))

        # Calculate the detector's neutrino energy detection probability to
        # detect a neutrino of energy E_nu given a neutrino declination:
        # p(E_nu|dec)
        det_prob = np.empty((len(dE_nu),), dtype=np.double)
        for i in range(len(dE_nu)):
            det_prob[i] = aeff.get_detection_prob_for_sin_true_dec(
                sin_true_dec=np.sin(src_dec),
                true_e_min=true_e_binedges[i],
                true_e_max=true_e_binedges[i+1]
            )

        self._logger.debug('det_prob = {}, sum = {}'.format(
            det_prob, np.sum(det_prob)))

        if not np.isclose(np.sum(det_prob), 1, rtol=0.06):
            raise ValueError(
                'The sum of the detection probabilities is not unity! It is '
                '{}.'.format(np.sum(det_prob)))

        reco_e_bw = np.diff(reco_e_edges)
        psi_edges_bw = np.diff(psi_edges)
        ang_err_bw = np.diff(ang_err_edges)

        bin_volumes = (
            reco_e_bw[:, np.newaxis, np.newaxis] *
            psi_edges_bw[np.newaxis, :, np.newaxis] *
            ang_err_bw[np.newaxis, np.newaxis, :])

        # Create the pdf in gamma for different gamma values.

        def create_pdf(union_arr, flux_model, gridfitparams):
            """Creates a pdf for a specific gamma value.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            my_flux_model = flux_model.copy(newprop=gridfitparams)

            E_nu_min = np.power(10, log_true_e_binedges[:-1])
            E_nu_max = np.power(10, log_true_e_binedges[1:])

            nbins_log_true_e = len(log_true_e_binedges) - 1

            self._logger.debug(
                'Generate signal PDF for parameters {} in {} E_nu bins.'.format(
                    gridfitparams, nbins_log_true_e)
            )

            # Calculate the flux probability p(E_nu|gamma).
            flux_prob = (
                my_flux_model.get_integral(E_nu_min, E_nu_max) /
                my_flux_model.get_integral(
                    np.power(10, log_true_e_binedges[0]),
                    np.power(10, log_true_e_binedges[-1])
                )
            )
            if not np.isclose(np.sum(flux_prob), 1):
                raise ValueError(
                    'The sum of the flux probabilities is not unity!')

            self._logger.debug(
                'flux_prob = {}'.format(flux_prob)
            )

            p = flux_prob * det_prob
            self._logger.debug(
                'p = {}, sum(p)={}'.format(p, sum(p))
            )

            true_e_prob = p / np.sum(p)

            self._logger.debug(
                f'true_e_prob = {true_e_prob}')

            transfer = np.copy(union_arr)
            for true_e_idx in range(nbins_log_true_e):
                transfer[true_e_idx] *= true_e_prob[true_e_idx]
            pdf_arr = np.sum(transfer, axis=0)
            del(transfer)

            # Normalize the pdf, which is the probability per bin volume.
            norm = np.sum(pdf_arr)
            if norm == 0:
                raise ValueError(
                    'The signal PDF is empty for {}! This should '
                    'not happen. Check the parameter ranges!'.format(
                        str(gridfitparams)))
            pdf_arr /= norm
            pdf_arr /= bin_volumes

            # Create the spatial PDF f_s = P(Psi|E_reco,ang_err) =
            # P(E_reco,Psi,ang_err) / \int dPsi P(E_reco,Psi,ang_err).
            marg_pdf = np.sum(
                pdf_arr * psi_edges_bw[np.newaxis, :, np.newaxis],
                axis=1,
                keepdims=True
            )
            f_s = pdf_arr / marg_pdf
            f_s[np.isnan(f_s)] = 0

            # Create the enegry PDF f_e = P(log10_E_reco|dec) =
            # \int dPsi dang_err P(E_reco,Psi,ang_err).
            f_e = np.sum(
                pdf_arr * psi_edges_bw[np.newaxis, :, np.newaxis] *
                ang_err_bw[np.newaxis, np.newaxis, :],
                axis=(1, 2))

            del(pdf_arr)

            pdf = PDSignalPDF_unionized_matrix(
                f_s, f_e, reco_e_edges, psi_edges, ang_err_edges,
                true_e_prob)

            return pdf

        args_list = [
            ((union_arr, flux_model, gridfitparams), {})
            for gridfitparams in self.gridfitparams_list
        ]

        pdf_list = parallelize(
            create_pdf,
            args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        del(union_arr)

        # Save all the energy PDF objects in the PDFSet PDF registry with
        # the hash of the individual parameters as key.
        for (gridfitparams, pdf) in zip(self.gridfitparams_list, pdf_list):
            self.add_pdf(pdf, gridfitparams)

    def get_prob(self, tdm, gridfitparams, tl=None):
        """Calculates the signal probability density of each event for the
        given set of signal fit parameters on a grid.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy'
                The log10 of the reconstructed energy.
            - 'psi'
                The opening angle from the source to the event in radians.
            - 'ang_err'
                The angular error of the event in radians.
        gridfitparams : dict
            The dictionary holding the signal parameter values for which the
            signal energy probability should be calculated. Note, that the
            parameter values must match a set of parameter grid values for which
            a PDSignalPDF object has been created at construction time of this
            PDSignalPDFSet object.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure time.

        Returns
        -------
        prob : 1d ndarray
            The array with the signal energy probability for each event.
        grads : (N_fitparams,N_events)-shaped ndarray | None
            The 2D numpy ndarray holding the gradients of the PDF w.r.t.
            each fit parameter for each event. The order of the gradients
            is the same as the order of floating parameters specified through
            the ``param_set`` property.
            It is ``None``, if this PDF does not depend on any parameters.

        Raises
        ------
        KeyError
            If no energy PDF can be found for the given signal parameter values.
        """
        print('Getting signal PDF for gridfitparams={}'.format(
            str(gridfitparams)))
        pdf = self.get_pdf(gridfitparams)

        (prob, grads) = pdf.get_prob(tdm, tl=tl)

        return (prob, grads)
