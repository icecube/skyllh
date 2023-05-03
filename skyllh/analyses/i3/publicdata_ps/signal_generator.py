# -*- coding: utf-8 -*-

import numpy as np
from scipy import (
    interpolate,
)
import scipy.stats

from skyllh.analyses.i3.publicdata_ps.aeff import (
    PDAeff,
)
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    get_times_in_grl_mask,
    psi_to_dec_and_ra,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    module_classname,
)
from skyllh.core.signal_generator import (
    MultiDatasetSignalGenerator,
    SignalGenerator,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)


class PDDatasetSignalGenerator(
        SignalGenerator):
    """This class implements a signal generator for a single public data
    dataset.
    """
    def __init__(
            self,
            shg_mgr,
            ds,
            ds_idx,
            energy_cut_spline=None,
            cut_sindec=None,
            **kwargs):
        """Creates a new instance of the signal generator for generating
        signal events from a specific public data dataset.

        Parameters:
        -----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager defining the source
            hypothesis groups.
        ds : instance of Dataset
            The instance of Dataset for which signal events should get
            generated.
        ds_idx : int
            The index of the dataset.
        energy_cut_spline : scipy.interpolate.UnivariateSpline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : float
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            **kwargs)

        self._logger = get_logger(module_classname(self))

        self.ds_idx = ds_idx
        self.energy_cut_spline = energy_cut_spline
        self.cut_sindec = cut_sindec

        self.sm = PDSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('smearing_datafile')))

        # Get the right-ascention, declination, and the effective area for each
        # source.
        n_sources = self._shg_mgr.n_sources
        self._src_ra_arr = np.empty((n_sources,), dtype=np.float64)
        self._src_dec_arr = np.empty((n_sources,), dtype=np.float64)
        self._effA_arr = np.empty((n_sources,), dtype=np.object_)
        for (i, src) in enumerate(self._shg_mgr.source_list):
            self._src_ra_arr[i] = src.ra
            self._src_dec_arr[i] = src.dec

            dec_idx = self.sm.get_true_dec_idx(src.dec)
            (min_log_true_e,
             max_log_true_e) =\
                self.sm.get_true_log_e_range_with_valid_log_e_pdfs(
                    dec_idx)

            self._effA_arr[i] = PDAeff(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('eff_area_datafile')),
                src_dec=src.dec,
                min_log10enu=min_log_true_e,
                max_log10enu=max_log_true_e)

    @staticmethod
    def _eval_spline(x, spl):
        """Evaluates the given spline at the given coordinates.
        """
        x = np.asarray(x)
        if (x.any() < 0) or (x.any() > 1):
            raise ValueError(
                f'{x} is outside of the valid spline range. '
                'The valid range is [0,1].')

        values = interpolate.splev(x, spl, ext=3)

        return values

    def _create_inv_cdf_spline(
            self,
            src_idx,
            fluxmodel,
            log_e_min,
            log_e_max):
        """Creates a spline for the inverse cumulative distribution function of
        the detectable true energy probability distribution.
        """
        effA = self._effA_arr[src_idx]

        m = (effA.log10_enu_bincenters >= log_e_min) & (
             effA.log10_enu_bincenters < log_e_max)
        bin_centers = effA.log10_enu_bincenters[m]
        low_bin_edges = effA.log10_enu_binedges_lower[m]
        high_bin_edges = effA.log10_enu_binedges_upper[m]

        # Flux probability P(E_nu | gamma) per bin.
        flux_prob = fluxmodel.energy_profile.get_integral(
            E1=10**low_bin_edges,
            E2=10**high_bin_edges
        ) / fluxmodel.energy_profile.get_integral(
            E1=10**low_bin_edges[0],
            E2=10**high_bin_edges[-1]
        )

        # Do the product and normalize again to a probability per bin.
        product = flux_prob * effA.det_prob
        prob_per_bin = product / np.sum(product)

        # The probability per bin cannot be zero, otherwise the cumulative
        # sum would not be increasing monotonically. So we set zero bins to
        # 1000 times smaller than the smallest non-zero bin.
        m = prob_per_bin == 0
        prob_per_bin[m] = np.min(prob_per_bin[np.invert(m)]) / 1000
        to_keep = prob_per_bin > 1e-15  # For numerical stability.
        prob_per_bin = prob_per_bin[to_keep]
        prob_per_bin /= np.sum(prob_per_bin)

        # Compute the cumulative distribution CDF.
        cum_per_bin = [
            np.sum(prob_per_bin[:i])
            for i in range(prob_per_bin.size+1)
        ]
        if np.any(np.diff(cum_per_bin) == 0):
            raise ValueError(
                'The cumulative sum of the true energy probability is not '
                'monotonically increasing! Values of the cumsum are '
                f'{cum_per_bin}.')

        bin_centers = bin_centers[to_keep]
        bin_centers = np.concatenate(([low_bin_edges[0]], bin_centers))

        # Build a spline for the inverse CDF.
        return interpolate.splrep(cum_per_bin, bin_centers, k=1, s=0)

    def _draw_signal_events_for_source(
            self,
            rss,
            src_dec,
            src_ra,
            dec_idx,
            log10_true_e_inv_cdf_spl,
            n_events):
        """Generates `n_events` signal events for the given source location
        given the given inverse cumulative density function for the
        log10(E_true/GeV) distribution.

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
        dec_idx : int
            The SM's declination bin index of the source's declination.
        log10_true_e_inv_cdf_spl : instance of scipy.interpolate.splrep
            The linear spline interpolation representation of the inverse
            cummulative density function of the log10(E_true/GeV) distribution.
        n_events : int
            The number of events to generate.

        Returns
        -------
        events : instance of DataFieldRecordArray of size `n_events`
            The instance of DataFieldRecordArray of length `n_events` holding
            the event data. It contains the following data fields:

                ``'isvalid'``
                ``'log_true_energy'``
                ``'log_energy'``
                ``'dec'``
                ``'sin_dec'``
                ``'ang_err'``
                ``'time'``
                ``'azi'``
                ``'zen'``
                ``'run'``

            Single values can be NaN in cases where a pdf was not available.
        """
        # Create the output event DataFieldRecordArray.
        out_dtype = [
            ('isvalid', np.bool_),
            ('log_true_energy', np.double),
            ('log_energy', np.double),
            ('dec', np.double),
            ('ra', np.double),
            ('sin_dec', np.double),
            ('ang_err', np.double),
            ('time', int),
            ('azi', np.double),
            ('zen', np.double),
            ('run', int)
        ]

        data = dict(
            [(out_dt[0], np.empty(
                (n_events,),
                dtype=out_dt[1])
              ) for out_dt in out_dtype]
        )

        events = DataFieldRecordArray(data, copy=False)

        sm = self.sm

        log_true_e = self._eval_spline(
            rss.random.uniform(size=n_events), log10_true_e_inv_cdf_spl)

        events['log_true_energy'] = log_true_e

        log_true_e_idxs = (
            np.digitize(log_true_e, bins=sm.true_e_bin_edges) - 1
        )

        # Sample reconstructed energies given true neutrino energies.
        (log_e_idxs, log_e) = sm.sample_log_e(rss, dec_idx, log_true_e_idxs)
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
        events['ra'][isvalid] = ra
        events['dec'][isvalid] = dec
        events['sin_dec'][isvalid] = np.sin(dec)

        # Add an angular error. Only use non-nan values.
        events['ang_err'][isvalid] = ang_err[isvalid]

        # Add fields required by the framework.
        events['time'] = np.ones(n_events)
        events['azi'] = np.ones(n_events)
        events['zen'] = np.ones(n_events)
        events['run'] = -1 * np.ones(n_events)

        return events

    @staticmethod
    def create_energy_filter_mask(
            events,
            spline,
            cut_sindec,
            logger):
        """Creates a mask for cutting all events below ``cut_sindec``
        that have an energy smaller than the energy spline at their
        declination.

        Paramters
        ---------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the generated signal
            events.
        spline : instance of scipy.interpolate.UnivariateSpline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : float
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        logger : instance of logging.Logger
            The Logger instance.

        Returns
        filter_mask : instance of numpy ndarray
            The (len(events),)-shaped numpy ndarray with the mask of the events
            to cut.
        """
        if cut_sindec is None:
            logger.warn(
                'No `cut_sindec` has been specified. The energy cut will be '
                'applied in [-90, 0] deg.')
            cut_sindec = 0.

        filter_mask = np.logical_and(
            events['sin_dec'] < cut_sindec,
            events['log_energy'] < spline(events['sin_dec']))

        return filter_mask

    def generate_signal_events_for_source(
            self,
            rss,
            src_idx,
            src_dec,
            src_ra,
            fluxmodel,
            n_events):
        """Generates ``n_events`` signal events for the given source location
        and flux model.

        Paramters
        ---------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        src_idx : int
            The index of the source.
        src_dec : float
            Declination coordinate of the injection point.
        src_ra : float
            Right ascension coordinate of the injection point.
        fluxmodel : instance of FactorizedFluxModel
            The instance of FactorizedFluxModel representing the flux model of
            the source.
        n_events : int
            Number of signal events to be generated.

        Returns
        -------
        events : instance of DataFieldRecordArray
            The numpy record array holding the event data.
            It contains the following data fields:
                - 'isvalid'
                - 'log_true_energy'
                - 'log_energy'
                - 'dec'
                - 'ra'
                - 'ang_err'
        """
        sm = self.sm

        # Find the declination bin index.
        dec_idx = sm.get_true_dec_idx(src_dec)

        # Determine the true energy range for which log_e PDFs are available.
        (min_log_true_e,
         max_log_true_e) = sm.get_true_log_e_range_with_valid_log_e_pdfs(
             dec_idx)
        # Build the spline for the inverse CDF and draw a true neutrino
        # energy from the hypothesis spectrum.
        log10_true_e_inv_cdf_spl = self._create_inv_cdf_spline(
            src_idx=src_idx,
            fluxmodel=fluxmodel,
            log_e_min=min_log_true_e,
            log_e_max=max_log_true_e)

        events = None

        n_events_generated = 0
        while n_events_generated < n_events:
            n_evt = n_events - n_events_generated

            events_ = self._draw_signal_events_for_source(
                rss=rss,
                src_dec=src_dec,
                src_ra=src_ra,
                dec_idx=dec_idx,
                log10_true_e_inv_cdf_spl=log10_true_e_inv_cdf_spl,
                n_events=n_evt)

            # Cut events that failed to be generated due to missing PDFs.
            # Also cut low energy events if generating in the southern sky.
            events_ = events_[events_['isvalid']]

            if self.energy_cut_spline is not None:
                cut_mask = self.create_energy_filter_mask(
                    events=events_,
                    spline=self.energy_cut_spline,
                    cut_sindec=self.cut_sindec,
                    logger=self._logger)
                events_ = events_[~cut_mask]

            if len(events_) > 0:
                n_events_generated += len(events_)
                if events is None:
                    events = events_
                else:
                    events.append(events_)

        return events

    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            src_detsigyield_weights_service=None,
            **kwargs):
        """Generates `mean` number of signal events.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        mean : int | float
            The mean number of signal events. If the ``poisson`` argument is set
            to True, the actual number of generated signal events will be drawn
            from a Poisson distribution with this given mean value of signal
            events.
        poisson : bool
            If set to True, the actual number of generated signal events will
            be drawn from a Poisson distribution with the given mean value of
            signal events.
            If set to False, the argument ``mean`` specifies the actual number
            of generated signal events.
        src_detsigyield_weights_service : instance of SrcDetSigYieldWeightsService
            The instance of SrcDetSigYieldWeightsService providing the weighting
            of the sources within the detector.

        Returns
        -------
        n_signal : int
            The number of generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        if poisson:
            n_events = rss.random.poisson(
                float_cast(
                    mean,
                    'The `mean` argument must be castable to type of float!'))

        n_events = int_cast(
            mean,
            'The `mean` argument must be castable to type of int!')

        if src_detsigyield_weights_service is None:
            raise ValueError(
                'The src_detsigyield_weights_service argument must be provided '
                f'for the signal generator {classname(self)}!')

        (a_jk, a_jk_grads) = src_detsigyield_weights_service.get_weights()

        a_k = np.copy(a_jk[self.ds_idx])
        a_k /= np.sum(a_k)

        n_signal = 0
        signal_events_dict = {}

        # Loop over the sources and generate signal events according to the
        # weights of the sources.
        src_idx = 0
        for shg in self._shg_mgr.shg_list:
            fluxmodel = shg.fluxmodel
            for (shg_src_idx, src) in enumerate(shg.source_list):
                n_events_src = int(np.round(n_events * a_k[src_idx], 0))

                src_dec = self._src_dec_arr[src_idx]
                src_ra = self._src_ra_arr[src_idx]

                src_events = self.generate_signal_events_for_source(
                    rss=rss,
                    src_idx=src_idx,
                    src_dec=src_dec,
                    src_ra=src_ra,
                    fluxmodel=fluxmodel,
                    n_events=n_events_src,
                )
                if src_events is None:
                    continue

                n_signal += len(src_events)

                if self.ds_idx not in signal_events_dict:
                    signal_events_dict[self.ds_idx] = src_events
                else:
                    signal_events_dict[self.ds_idx].append(src_events)

                src_idx += shg_src_idx

        return (n_signal, signal_events_dict)


class TimeDependentMultiDatasetSignalGenerator(
        MultiDatasetSignalGenerator):
    """ The time dependent signal generator works so far only for one single
    dataset. For multi datasets one needs to adjust the dataset weights
    accordingly (scaling of the effective area with livetime of the flare in
    the dataset).
    """
    def __init__(
            self,
            shg_mgr,
            dataset_list,
            data_list=None,
            sig_generator_list=None,
            ds_sig_weight_factors_service=None,
            gauss=None,
            box=None,
            **kwargs):
        """
        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            source hypothesis groups, i.e. the list of sources.
        dataset_list : list of instance of Dataset
            The list of instance of Dataset for which signal events should get
            generated.
        data_list : list of instance of DatasetData
            The list of instance of DatasetData holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        sig_generator_list : list of instance of SignalGenerator | None
            The optional list of instance of SignalGenerator holding
            signal generator instances for each individual dataset. This can be
            ``None`` if this signal generator does not require individual signal
            generators for each dataset.
        ds_sig_weight_factors_service : instance of DatasetSignalWeightFactorsService
            The instance of DatasetSignalWeightFactorsService providing the
            dataset signal weight factor service for calculating the dataset
            signal weights.
        gauss : dict | None
            None or dictionary with {"mu": float, "sigma": float}.
        box : dict | None
            None or dictionary with {"start": float, "end": float}.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            dataset_list=dataset_list,
            data_list=data_list,
            sig_generator_list=sig_generator_list,
            ds_sig_weight_factors_service=ds_sig_weight_factors_service,
        )

        if gauss is None and box is None:
            raise ValueError(
                'Either box or gauss arguments must define the neutrino flare!')
        if gauss is not None and box is not None:
            raise ValueError(
                'Either box or gauss arguments must define the neutrino flare, '
                'cannot use both!')

        self.box = box
        self.gauss = gauss

        self.time_pdf = self._get_time_pdf()

    def _get_time_pdf(self):
        """Get the neutrino flare time pdf given parameters.
        Will be used to generate random numbers by calling `rvs()` method.

        Returns
        -------
        time_pdf : instance of scipy.stats.rv_continuous base class
            Has to base scipy.stats.rv_continuous.
        """
        # Make sure flare is in dataset.
        for data_list in self.data_list:
            grl = data_list.grl

            if self.gauss is not None:
                if (self.gauss["mu"] - 4 * self.gauss["sigma"] > grl["stop"][-1]) or (
                        self.gauss["mu"] + 4 * self.gauss["sigma"] < grl["start"][0]):
                    raise ValueError(
                        f"Gaussian {str(self.gauss)} flare is not in dataset.")

            if self.box is not None:
                if (self.box["start"] > grl["stop"][-1]) or (
                        self.box["end"] < grl["start"][0]):
                    raise ValueError(
                        f"Box {str(self.box)} flare is not in dataset.")

        # Create `time_pdf`.
        if self.gauss is not None:
            time_pdf = scipy.stats.norm(self.gauss["mu"], self.gauss["sigma"])
        if self.box is not None:
            time_pdf = scipy.stats.uniform(
                self.box["start"],
                self.box["end"] - self.box["start"]
            )

        return time_pdf

    def set_flare(
            self,
            gauss=None,
            box=None):
        """Set the neutrino flare given parameters.

        Parameters
        ----------
        gauss : dict | None
            None or dictionary with {"mu": float, "sigma": float}.
        box : dict | None
             None or dictionary with {"start": float, "end": float}.
        """
        if gauss is None and box is None:
            raise ValueError(
                "Either box or gauss keywords must define the neutrino flare.")
        if gauss is not None and box is not None:
            raise ValueError(
                "Either box or gauss keywords must define the neutrino flare, "
                "cannot use both.")

        self.box = box
        self.gauss = gauss

        self.time_pdf = self._get_time_pdf()

    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            **kwargs):
        """Same as in PDSignalGenerator, but we assign times here.
        """
        # Call method from the parent class to generate signal events.
        (n_signal, signal_events_dict) = super().generate_signal_events(
            rss=rss,
            mean=mean,
            poisson=poisson,
            **kwargs)

        # Assign times for flare. We can also use inverse transform
        # sampling instead of the lazy version implemented here.
        for (ds_idx, events) in signal_events_dict.items():
            grl = self.data_list[ds_idx].grl

            # Optimized time injection version, based on csky implementation.
            # https://github.com/icecube/csky/blob/7e969639c5ef6dbb42872dac9b761e1e8b0ccbe2/csky/inj.py#L1122
            times = np.array([])
            n_events = len(events)
            while len(times) < n_events:
                new_times = self.time_pdf.rvs(
                    n_events - len(times),
                    random_state=rss.random)
                mask = get_times_in_grl_mask(times=new_times, grl=grl)
                new_times = new_times[mask]

                times = np.concatenate((times, new_times))

            events['time'] = times

        return (n_signal, signal_events_dict)
