from typing import Any, cast

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
from skyllh.core.dataset import Dataset
from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.logging import (
    get_logger,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    issequence,
    module_classname,
)
from skyllh.core.random import RandomStateService
from skyllh.core.services import SrcDetSigYieldWeightsService
from skyllh.core.signal_generation import (
    HasEnergyRange,
)
from skyllh.core.signal_generator import (
    SignalGenerator,
)
from skyllh.core.source_hypo_grouping import SourceHypoGroupManager
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.utils.flux_model import (
    create_scipy_stats_rv_continuous_from_TimeFluxProfile,
)


class PDDatasetSignalGenerator(
    SignalGenerator,
    HasEnergyRange,
):
    """This class implements a signal generator for a single public data
    dataset.
    """

    def __init__(
        self,
        shg_mgr: SourceHypoGroupManager,
        ds: Dataset,
        ds_idx: int,
        energy_cut_spline: interpolate.UnivariateSpline | None = None,
        cut_sindec: float | None = None,
        energy_range: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Creates a new instance of the signal generator for generating
        signal events from a specific public data dataset.

        Parameters
        ----------
        shg_mgr
            The instance of SourceHypoGroupManager defining the source
            hypothesis groups.
        ds
            The instance of Dataset for which signal events should get
            generated.
        ds_idx
            The index of the dataset.
        energy_cut_spline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        energy_range
            The energy range in which signal events should be generated.
            If set to None, the full energy range (1e2 - 1e9 GeV) is used.
        """
        super().__init__(shg_mgr=shg_mgr, **kwargs)

        self._logger = get_logger(module_classname(self))

        self.ds = ds
        self.ds_idx = ds_idx
        self.energy_cut_spline = energy_cut_spline
        self.cut_sindec = cut_sindec
        self.sm = PDSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(ds.get_aux_data_definition('smearing_datafile'))
        )

        # Cache effective-area data reused for correction-factor calculations.
        self._aeff_pathfilenames = ds.get_abs_pathfilename_list(ds.get_aux_data_definition('eff_area_datafile'))
        self._aeff_full = None
        self._aeff_for_src_cache = {}

        self.energy_range = energy_range

    @property
    def energy_range(self) -> tuple[float, float]:
        """The configured true-energy range for signal generation in GeV.

        It is a 2-element tuple ``(E_min, E_max)`` in GeV. If no explicit
        range was configured, the full available true-energy range of the
        smearing matrix is returned in GeV.
        """
        if self._input_range is None:
            return (10 ** self.sm.true_e_bin_edges[0], 10 ** self.sm.true_e_bin_edges[-1])
        return self._input_range

    @property
    def _log10_energy_range(self):
        """Internal true-energy range in log10(E/GeV) used for calculations.

        For the public data analysis, this matches bin edges of the smearing matrices.
        It is not simply the log10 of the input energy range!
        """
        if self._energy_range_log10 is None:
            return (self.sm.true_e_bin_edges[0], self.sm.true_e_bin_edges[-1])
        return self._energy_range_log10

    @energy_range.setter
    def energy_range(self, r):
        if hasattr(self, '_input_range') and self._input_range == r:
            return

        if r is not None:
            if not issequence(r) or len(r) != 2:
                raise ValueError(
                    f'The energy_range property must be a 2-element sequence of floats! Its current value is {r}!'
                )
            r = (
                float_cast(r[0], 'The first element of the energy_range sequence must be castable to type of float!'),
                float_cast(r[1], 'The second element of the energy_range sequence must be castable to type of float!'),
            )

            if r[0] <= 0 or r[1] <= 0:
                raise ValueError('Both energy_range values must be strictly positive!')
            if r[0] >= r[1]:
                raise ValueError(
                    'The first element of the energy_range sequence must be strictly smaller than the second element!'
                )

            self._input_range = r

            # Convert the energy boundaries to the closest SM bin edges.
            idx0 = self.sm.get_log10_true_e_idx(np.log10(r[0]))
            idx1 = self.sm.get_log10_true_e_idx(np.log10(r[1]), upper_edge=True)
            r_log10 = (self.sm.true_e_bin_edges[idx0], self.sm.true_e_bin_edges[idx1])

            if r_log10[0] >= r_log10[1]:
                raise ValueError(
                    'The first element of the energy_range sequence must be strictly smaller than the second element!'
                )

            self._logger.info(
                f'Energy range for signal generation set to {self._input_range} GeV '
                f'(effective bin-aligned range: {r_log10} in log10(E/GeV)).'
            )

            self._energy_range_log10 = r_log10
        else:
            self._input_range = None
            self._energy_range_log10 = None

        self._create_source_dependent_data_structures()

    def _create_source_dependent_data_structures(self):
        """Creates the source dependent data structures needed by this signal generator. These are:

        - source location in ra and dec
        - effective area
        - log10 true energy inv CDF spline
        - energy range correction factors

        """
        n_sources = self.shg_mgr.n_sources

        self._src_ra_arr = np.empty((n_sources,), dtype=np.float64)
        self._src_dec_arr = np.empty((n_sources,), dtype=np.float64)
        self._effA_arr = np.empty((n_sources,), dtype=np.object_)
        self._log10_true_e_inv_cdf_spl_arr = np.empty((n_sources,), dtype=np.object_)

        for src_idx, src in enumerate(self._shg_mgr.source_list):
            self._src_ra_arr[src_idx] = src.ra
            self._src_dec_arr[src_idx] = src.dec

            dec_idx = self.sm.get_true_dec_idx(src.dec)
            (min_log_true_e, max_log_true_e) = self.sm.get_true_log_e_range_with_valid_log_e_pdfs(dec_idx)

            if self._energy_range_log10 is not None:
                min_log_true_e = max(min_log_true_e, self._log10_energy_range[0])
                max_log_true_e = min(max_log_true_e, self._log10_energy_range[1])

            self._effA_arr[src_idx] = PDAeff(
                pathfilenames=self._aeff_pathfilenames,
                src_dec=src.dec,
                min_log10enu=min_log_true_e,
                max_log10enu=max_log_true_e,
            )

            # Build the spline for the inverse CDF of the source flux's true energy probability distribution.
            fluxmodel = self.shg_mgr.get_fluxmodel_by_src_idx(src_idx=src_idx)
            self._log10_true_e_inv_cdf_spl_arr[src_idx] = self._create_inv_cdf_spline(
                src_idx=src_idx, fluxmodel=fluxmodel, log_e_min=min_log_true_e, log_e_max=max_log_true_e
            )

        # Cache correction factors so they are in sync with the current energy_range and source hypothesis.
        self._energy_range_correction_factors = self._calculate_energy_range_correction_factors()

    @staticmethod
    def _eval_spline(x, spl) -> np.ndarray:
        """Evaluates the given spline at the given coordinates."""
        x = np.asarray(x)
        if (x.any() < 0) or (x.any() > 1):
            raise ValueError(f'{x} is outside of the valid spline range. The valid range is [0,1].')

        values = np.asarray(interpolate.splev(x, spl, ext=3))

        return values

    def _create_inv_cdf_spline(self, src_idx: int, fluxmodel, log_e_min, log_e_max):
        """Creates a spline for the inverse cumulative distribution function of
        the detectable true energy probability distribution.
        """
        effA = self._effA_arr[src_idx]

        m = (effA.log10_enu_bincenters >= log_e_min) & (effA.log10_enu_bincenters < log_e_max)
        bin_centers = effA.log10_enu_bincenters[m]
        low_bin_edges = effA.log10_enu_binedges_lower[m]
        high_bin_edges = effA.log10_enu_binedges_upper[m]

        # Flux probability P(E_nu | gamma) per bin.
        flux_prob = fluxmodel.energy_profile.get_integral(
            E1=10**low_bin_edges, E2=10**high_bin_edges
        ) / fluxmodel.energy_profile.get_integral(E1=10 ** low_bin_edges[0], E2=10 ** high_bin_edges[-1])

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
        cum_per_bin = [np.sum(prob_per_bin[:i]) for i in range(prob_per_bin.size + 1)]
        if np.any(np.diff(cum_per_bin) == 0):
            raise ValueError(
                'The cumulative sum of the true energy probability is not '
                'monotonically increasing! Values of the cumsum are '
                f'{cum_per_bin}.'
            )

        bin_centers = bin_centers[to_keep]
        bin_centers = np.concatenate(([low_bin_edges[0]], bin_centers))

        # Build a spline for the inverse CDF.
        return interpolate.splrep(cum_per_bin, bin_centers, k=1, s=0)

    def _draw_signal_events_for_source(
        self,
        rss: RandomStateService,
        src_dec: float,
        src_ra: float,
        dec_idx: int,
        log10_true_e_inv_cdf_spl,
        n_events: int,
    ) -> DataFieldRecordArray:
        """Generates `n_events` signal events for the given source location
        given the given inverse cumulative density function for the
        log10(E_true/GeV) distribution.

        Note:
            Some values can be NaN in cases where a PDF was not available!

        Parameters
        ----------
        rss
            The instance of RandomStateService to use for drawing random
            numbers.
        src_dec
            The declination of the source in radians.
        src_ra
            The right-ascention of the source in radians.
        dec_idx
            The SM's declination bin index of the source's declination.
        log10_true_e_inv_cdf_spl
            The linear spline interpolation representation of the inverse
            cummulative density function of the log10(E_true/GeV) distribution.
        n_events
            The number of events to generate.

        Returns
        -------
        events
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
            ('run', int),
        ]

        data = dict([(out_dt[0], np.empty((n_events,), dtype=out_dt[1])) for out_dt in out_dtype])

        events = DataFieldRecordArray(data, copy=False)

        sm = self.sm

        log_true_e = self._eval_spline(rss.random.uniform(size=n_events), log10_true_e_inv_cdf_spl)

        events['log_true_energy'] = log_true_e

        log_true_e_idxs = np.digitize(log_true_e, bins=sm.true_e_bin_edges) - 1

        # Sample reconstructed energies given true neutrino energies.
        (log_e_idxs, log_e) = sm.sample_log_e(rss, dec_idx, log_true_e_idxs)
        events['log_energy'] = log_e

        # Sample reconstructed psi values given true neutrino energy and
        # reconstructed energy.
        (psi_idxs, psi) = sm.sample_psi(rss, dec_idx, log_true_e_idxs, log_e_idxs)

        # Sample reconstructed ang_err values given true neutrino energy,
        # reconstructed energy, and psi.
        (_, ang_err) = sm.sample_ang_err(rss, dec_idx, log_true_e_idxs, log_e_idxs, psi_idxs)

        isvalid = np.invert(np.isnan(log_e) | np.isnan(psi) | np.isnan(ang_err))
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

    def change_shg_mgr(self, shg_mgr):
        """Changes the source hypothesis group manager. This will recreate the
        internal source dependent data structures.
        """
        super().change_shg_mgr(shg_mgr=shg_mgr)

        # Source indices/declinations can change with a new hypothesis.
        self._aeff_for_src_cache = {}

        self._create_source_dependent_data_structures()

    @staticmethod
    def create_energy_filter_mask(
        events: DataFieldRecordArray, spline: interpolate.UnivariateSpline, cut_sindec: float | None, logger
    ) -> np.ndarray:
        """Creates a mask for cutting all events below ``cut_sindec``
        that have an energy smaller than the energy spline at their
        declination.

        Parameters
        ----------
        events
            The instance of DataFieldRecordArray holding the generated signal
            events.
        spline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        logger
            The Logger instance.

        Returns
        -------
        filter_mask
            The (len(events),)-shaped numpy ndarray with the mask of the events
            to cut.
        """
        if cut_sindec is None:
            logger.warning('No `cut_sindec` has been specified. The energy cut will be applied in [-90, 90] deg.')
            cut_sindec = np.sin(np.radians(90.1))

        filter_mask = np.logical_and(events['sin_dec'] < cut_sindec, events['log_energy'] < spline(events['sin_dec']))

        return filter_mask

    def generate_signal_events_for_source(
        self, rss: RandomStateService, src_idx: int, n_events: int
    ) -> DataFieldRecordArray | None:
        """Generates ``n_events`` signal events for the given source location
        and flux model.

        Parameters
        ----------
        rss
            The instance of RandomStateService providing the random number
            generator state.
        src_idx
            The index of the source.
        n_events
            Number of signal events to be generated.

        Returns
        -------
        events
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
                n_events=n_evt,
            )

            # Cut events that failed to be generated due to missing PDFs.
            # Also cut low energy events if generating in the southern sky.
            events_ = events_[events_['isvalid']]

            if self.energy_cut_spline is not None:
                cut_mask = self.create_energy_filter_mask(
                    events=events_, spline=self.energy_cut_spline, cut_sindec=self.cut_sindec, logger=self._logger
                )
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
        rss: RandomStateService,
        mean: int | float,
        poisson: bool = True,
        src_detsigyield_weights_service: SrcDetSigYieldWeightsService | None = None,
        **kwargs,
    ) -> tuple[int, dict[int, DataFieldRecordArray]]:
        """Generates ``mean`` number of signal events.

        Parameters
        ----------
        rss
            The instance of RandomStateService providing the random number
            generator state.
        mean
            The mean number of signal events. If the ``poisson`` argument is set
            to True, the actual number of generated signal events will be drawn
            from a Poisson distribution with this given mean value of signal
            events.
        poisson
            If set to True, the actual number of generated signal events will
            be drawn from a Poisson distribution with the given mean value of
            signal events.
            If set to False, the argument ``mean`` specifies the actual number
            of generated signal events.
        src_detsigyield_weights_service
            The instance of SrcDetSigYieldWeightsService providing the weighting
            of the sources within the detector.

        Returns
        -------
        n_signal
            The number of generated signal events.
        signal_events_dict
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        _mean = rss.random.poisson(float(mean)) if poisson else mean

        n_events = int_cast(_mean, 'The `mean` argument must be castable to type of int!')

        if src_detsigyield_weights_service is None:
            raise ValueError(
                'The src_detsigyield_weights_service argument must be provided '
                f'for the signal generator {classname(self)}!'
            )

        (a_jk, _) = src_detsigyield_weights_service.get_weights()

        assert a_jk is not None
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

    def _calculate_flux_weight(self, src_idx, log_e_min, log_e_max):
        """Calculates an unnormalized detectable flux weight for one source
        and one true-energy range.
        """
        src = self._shg_mgr.source_list[src_idx]
        fluxmodel = self.shg_mgr.get_fluxmodel_by_src_idx(src_idx=src_idx)

        effA = self._get_cached_full_aeff()

        low_all = effA.log10_enu_binedges_lower
        high_all = effA.log10_enu_binedges_upper

        # Compute overlap of each Aeff bin with the chosen E range to include partial boundary bins consistently.
        overlap_low = np.maximum(low_all, log_e_min)
        overlap_high = np.minimum(high_all, log_e_max)
        m = overlap_high > overlap_low
        if not np.any(m):
            return 0.0

        flux_integral = cast(Any, fluxmodel).energy_profile.get_integral(
            E1=10 ** overlap_low[m], E2=10 ** overlap_high[m]
        )

        aeff_for_dec = self._get_cached_aeff_for_source(src_idx=src_idx, src_dec=src.dec)[m]

        return float(np.sum(flux_integral * aeff_for_dec))

    def _get_cached_full_aeff(self):
        """Returns the dataset-level effective area, loading it only once."""
        if self._aeff_full is None:
            # Re-load it here because in _create_source_dependent_data_structures it was loaded
            # for a specific source declination and energy range only.
            self._aeff_full = PDAeff(pathfilenames=self._aeff_pathfilenames)
        return self._aeff_full

    def _get_cached_aeff_for_source(self, src_idx, src_dec):
        """Returns cached Aeff(log10E) values for one source declination."""
        cached = self._aeff_for_src_cache.get(src_idx)
        if (cached is not None) and (cached[0] == src_dec):
            return cached[1]

        aeff_for_dec = self._get_cached_full_aeff().get_aeff_for_decnu(src_dec)
        self._aeff_for_src_cache[src_idx] = (src_dec, aeff_for_dec)
        return aeff_for_dec

    def _calculate_energy_range_correction_factors(self):
        """Calculates per-source correction factors for the configured
        ``energy_range``.

        The factors are 1 if no additional true-energy cut is applied. If a
        cut is configured, each factor is the ratio between detectable flux in
        the cut range and detectable flux in the full valid range of the
        smearing matrix.
        """
        n_sources = self.shg_mgr.n_sources
        correction_factors = np.ones((n_sources,), dtype=np.float64)

        for src_idx, src in enumerate(self._shg_mgr.source_list):
            dec_idx = self.sm.get_true_dec_idx(src.dec)
            (valid_min_log_e, valid_max_log_e) = self.sm.get_true_log_e_range_with_valid_log_e_pdfs(dec_idx)

            cut_min_log_e = valid_min_log_e
            cut_max_log_e = valid_max_log_e
            if self._energy_range_log10 is not None:
                cut_min_log_e = max(cut_min_log_e, self._log10_energy_range[0])
                cut_max_log_e = min(cut_max_log_e, self._log10_energy_range[1])

            if cut_min_log_e >= cut_max_log_e:
                correction_factors[src_idx] = 0.0
                continue

            full_weight = self._calculate_flux_weight(
                src_idx=src_idx,
                log_e_min=valid_min_log_e,
                log_e_max=valid_max_log_e,
            )
            if full_weight <= 0:  # This should never happen, but well...
                correction_factors[src_idx] = 0.0
                continue

            cut_weight = self._calculate_flux_weight(
                src_idx=src_idx,
                log_e_min=cut_min_log_e,
                log_e_max=cut_max_log_e,
            )

            ratio = cut_weight / full_weight
            if ratio > 1.0 + 1e-6 or ratio < -1e-6:
                self._logger.warning(
                    'Energy-range correction factor outside [0,1] for source index '
                    f'{src_idx}: {ratio:.6g}. Clipping to [0,1].'
                )
            correction_factors[src_idx] = np.clip(ratio, 0.0, 1.0)

        return correction_factors

    def get_energy_range_correction_factors(self):
        """Returns per-source correction factors for the configured
        ``energy_range``.

        The factors are computed when source-dependent structures are rebuilt
        (e.g. after changing ``energy_range`` or source hypotheses) and cached
        here for repeated use.
        """
        if not hasattr(self, '_energy_range_correction_factors'):
            self._energy_range_correction_factors = self._calculate_energy_range_correction_factors()
        return np.copy(self._energy_range_correction_factors)


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
        shg_mgr: SourceHypoGroupManager,
        ds: Dataset,
        ds_idx: int,
        livetime: Livetime,
        time_flux_profile: TimeFluxProfile,
        energy_cut_spline=None,
        cut_sindec: float | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        shg_mgr
            The instance of SourceHypoGroupManager that defines the list of
            source hypothesis groups, i.e. the list of sources.
        ds
            The instance of Dataset for which signal events should get
            generated.
        ds_idx
            The index of the dataset.
        livetime
            The instance of Livetime providing the live-time information of the
            dataset.
        time_flux_profile
            The instance of TimeFluxProfile providing the time profile of the
            source(s).

            Note:

                At this time the some time profile will be used for all
                sources!

        energy_cut_spline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec
            The sine of the declination to start applying the energy cut.
            The cut will be applied from this declination down.
        """
        super().__init__(
            shg_mgr=shg_mgr, ds=ds, ds_idx=ds_idx, energy_cut_spline=energy_cut_spline, cut_sindec=cut_sindec, **kwargs
        )

        if not isinstance(time_flux_profile, TimeFluxProfile):
            raise TypeError(
                'The time_flux_profile argument must be an instance of '
                'TimeFluxProfile! '
                f'Its current type is {classname(time_flux_profile)}!'
            )

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
                f'The livetime property must be an instance of Livetime! Its current type is {classname(lt)}!'
            )
        self._livetime = lt

    def generate_signal_events(
        self,
        rss: RandomStateService,
        mean: int | float,
        poisson: bool = True,
        src_detsigyield_weights_service: SrcDetSigYieldWeightsService | None = None,
        **kwargs,
    ) -> tuple[int, dict[int, DataFieldRecordArray]]:
        """Generates ``mean`` number of signal events with times.

        Parameters
        ----------
        rss
            The instance of RandomStateService providing the random number
            generator state.
        mean
            The mean number of signal events. If the ``poisson`` argument is set
            to True, the actual number of generated signal events will be drawn
            from a Poisson distribution with this given mean value of signal
            events.
        poisson
            If set to True, the actual number of generated signal events will
            be drawn from a Poisson distribution with the given mean value of
            signal events.
            If set to False, the argument ``mean`` specifies the actual number
            of generated signal events.
        src_detsigyield_weights_service
            The instance of SrcDetSigYieldWeightsService providing the weighting
            of the sources within the detector.

        Returns
        -------
        n_signal
            The number of generated signal events.
        signal_events_dict
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        (n_signal, signal_events_dict) = super().generate_signal_events(
            rss=rss,
            mean=mean,
            poisson=poisson,
            src_detsigyield_weights_service=src_detsigyield_weights_service,
            **kwargs,
        )

        # Create a scipy.stats.rv_continuous instance for the time flux profile.
        time_rv = create_scipy_stats_rv_continuous_from_TimeFluxProfile(profile=self._time_flux_profile)

        # Optimized time injection version, based on csky implementation.
        # https://github.com/icecube/csky/blob/7e969639c5ef6dbb42872dac9b761e1e8b0ccbe2/csky/inj.py#L1122
        events = signal_events_dict[self.ds_idx]
        times = np.array([], dtype=np.float64)
        n_events = len(events)
        while len(times) < n_events:
            new_times = time_rv.rvs(size=(n_events - len(times)), random_state=rss.random)
            mask = self._livetime.is_on(mjd=new_times)
            new_times = new_times[mask]

            times = np.concatenate((times, new_times))
        events['time'] = times

        return (n_signal, signal_events_dict)
