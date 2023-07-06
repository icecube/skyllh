# -*- coding: utf-8 -*-

import numpy as np

from scipy import (
    interpolate,
)

from skyllh.analyses.i3.publicdata_ps.aeff import (
    PDAeff,
)
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    psi_to_dec_and_ra,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    module_classname,
)
from skyllh.core.signal_generator import (
    SignalGenerator,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.utils.flux_model import (
    create_scipy_stats_rv_continuous_from_TimeFluxProfile,
)


class PDDatasetSignalGenerator(
        SignalGenerator,
):
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
            **kwargs,
    ):
        """Creates a new instance of the signal generator for generating
        signal events from a specific public data dataset.

        Parameters
        ----------
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

        self.ds = ds
        self.ds_idx = ds_idx
        self.energy_cut_spline = energy_cut_spline
        self.cut_sindec = cut_sindec

        self.sm = PDSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('smearing_datafile')))

        self._create_source_dependent_data_structures()

    def _create_source_dependent_data_structures(self):
        """Creates the source dependent data structures needed by this signal
        generator. These are:

            - source location in ra and dec
            - effective area
            - log10 true energy inv CDF spline

        """
        n_sources = self.shg_mgr.n_sources

        self._src_ra_arr = np.empty(
            (n_sources,),
            dtype=np.float64)
        self._src_dec_arr = np.empty(
            (n_sources,),
            dtype=np.float64)
        self._effA_arr = np.empty(
            (n_sources,),
            dtype=np.object_)
        self._log10_true_e_inv_cdf_spl_arr = np.empty(
            (n_sources,),
            dtype=np.object_)

        for (src_idx, src) in enumerate(self._shg_mgr.source_list):
            self._src_ra_arr[src_idx] = src.ra
            self._src_dec_arr[src_idx] = src.dec

            dec_idx = self.sm.get_true_dec_idx(src.dec)
            (min_log_true_e,
             max_log_true_e) =\
                self.sm.get_true_log_e_range_with_valid_log_e_pdfs(
                    dec_idx)

            self._effA_arr[src_idx] = PDAeff(
                pathfilenames=self.ds.get_abs_pathfilename_list(
                    self.ds.get_aux_data_definition('eff_area_datafile')),
                src_dec=src.dec,
                min_log10enu=min_log_true_e,
                max_log10enu=max_log_true_e)

            # Build the spline for the inverse CDF of the source flux's true
            # energy probability distribution.
            fluxmodel = self.shg_mgr.get_fluxmodel_by_src_idx(src_idx=src_idx)
            self._log10_true_e_inv_cdf_spl_arr[src_idx] =\
                self._create_inv_cdf_spline(
                    src_idx=src_idx,
                    fluxmodel=fluxmodel,
                    log_e_min=min_log_true_e,
                    log_e_max=max_log_true_e)

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
        events['time'] = np.full((n_events,), np.nan, dtype=np.float64)
        events['azi'] = np.full((n_events,), np.nan, dtype=np.float64)
        events['zen'] = np.full((n_events,), np.nan, dtype=np.float64)
        events['run'] = np.full((n_events,), -1, dtype=np.int64)

        return events

    def change_shg_mgr(
            self,
            shg_mgr):
        """Changes the source hypothesis group manager. This will recreate the
        internal source dependent data structures.
        """
        super().change_shg_mgr(
            shg_mgr=shg_mgr)

        self._create_source_dependent_data_structures()

    @staticmethod
    def create_energy_filter_mask(
            events,
            spline,
            cut_sindec,
            logger):
        """Creates a mask for cutting all events below ``cut_sindec``
        that have an energy smaller than the energy spline at their
        declination.

        Parameters
        ----------
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
        -------
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
            n_events):
        """Generates ``n_events`` signal events for the given source location
        and flux model.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        src_idx : int
            The index of the source.
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

        src_dec = self._src_dec_arr[src_idx]
        src_ra = self._src_ra_arr[src_idx]

        log10_true_e_inv_cdf_spl = self._log10_true_e_inv_cdf_spl_arr[src_idx]

        # Find the declination bin index.
        dec_idx = sm.get_true_dec_idx(src_dec)

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
        """Generates ``mean`` number of signal events.

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
        for src_idx in range(self.shg_mgr.n_sources):
            n_events_src = int(np.round(n_events * a_k[src_idx], 0))

            src_events = self.generate_signal_events_for_source(
                rss=rss,
                src_idx=src_idx,
                n_events=n_events_src,
            )
            if src_events is None:
                continue

            n_signal += len(src_events)

            if self.ds_idx not in signal_events_dict:
                signal_events_dict[self.ds_idx] = src_events
            else:
                signal_events_dict[self.ds_idx].append(src_events)

        return (n_signal, signal_events_dict)


class TimeDependentPDDatasetSignalGenerator(
        PDDatasetSignalGenerator,
):
    """This time dependent signal generator for a public PS dataset generates
    events using the
    :class:`~skyllh.analyses.i3.publicdata_ps.signal_generator.PDDatasetSignalGenerator`
    class. It then draws times for each event and adds them to the event array.
    """

    def __init__(
            self,
            shg_mgr,
            ds,
            ds_idx,
            livetime,
            time_flux_profile,
            energy_cut_spline=None,
            cut_sindec=None,
            **kwargs,
    ):
        """
        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            source hypothesis groups, i.e. the list of sources.
        ds : instance of Dataset
            The instance of Dataset for which signal events should get
            generated.
        ds_idx : int
            The index of the dataset.
        livetime : instance of Livetime
            The instance of Livetime providing the live-time information of the
            dataset.
        time_flux_profile : instance of TimeFluxProfile
            The instance of TimeFluxProfile providing the time profile of the
            source(s).

            Note:

                At this time the some time profile will be used for all
                sources!

        energy_cut_spline : scipy.interpolate.UnivariateSpline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : float
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            ds=ds,
            ds_idx=ds_idx,
            energy_cut_spline=energy_cut_spline,
            cut_sindec=cut_sindec,
            **kwargs)

        if not isinstance(time_flux_profile, TimeFluxProfile):
            raise TypeError(
                'The time_flux_profile argument must be an instance of '
                'TimeFluxProfile! '
                f'Its current type is {classname(time_flux_profile)}!')

        self.livetime = livetime
        self._time_flux_profile = time_flux_profile

    @property
    def livetime(self):
        """The instance of Livetime providing the live-time information of the
        dataset.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if not isinstance(lt, Livetime):
            raise TypeError(
                'The livetime property must be an instance of Livetime! '
                f'Its current type is {classname(lt)}!')

    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            src_detsigyield_weights_service=None,
            **kwargs,
    ):
        """Generates ``mean`` number of signal events with times.

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
        (n_signal, signal_events_dict) = super().generate_signal_events(
            rss=rss,
            mean=mean,
            poisson=poisson,
            src_detsigyield_weights_service=src_detsigyield_weights_service,
            **kwargs)

        # Create a scipy.stats.rv_continuous instance for the time flux profile.
        time_rv = create_scipy_stats_rv_continuous_from_TimeFluxProfile(
            profile=self._time_flux_profile)

        # Optimized time injection version, based on csky implementation.
        # https://github.com/icecube/csky/blob/7e969639c5ef6dbb42872dac9b761e1e8b0ccbe2/csky/inj.py#L1122
        events = signal_events_dict[self.ds_idx]
        times = np.array([], dtype=np.float64)
        n_events = len(events)
        while len(times) < n_events:
            new_times = time_rv.rvs(
                size=(n_events - len(times)),
                random_state=rss.random)
            mask = self._livetime.is_on(
                mjd=new_times)
            new_times = new_times[mask]

            times = np.concatenate((times, new_times))
        events['time'] = times

        return (n_signal, signal_events_dict)
