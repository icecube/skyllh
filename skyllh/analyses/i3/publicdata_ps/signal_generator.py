# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
import scipy.stats

from skyllh.core.py import (
    issequenceof,
    float_cast,
    int_cast
)
from skyllh.core.py import module_classname
from skyllh.core.debugging import get_logger
from skyllh.core.signal_generator import SignalGeneratorBase
from skyllh.core.llhratio import LLHRatio
from skyllh.core.dataset import Dataset
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.storage import DataFieldRecordArray

from skyllh.analyses.i3.publicdata_ps.utils import psi_to_dec_and_ra
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix
)
from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff


class PDDatasetSignalGenerator(object):
    """This class provides a signal generation method for a point-like source
    seen in the IceCube detector using one dataset of the 10 years public data
    release. It is used by the PDSignalGenerator class in a loop over all the
    datasets that have been added to the analysis.
    """

    def __init__(self, ds, src_dec, effA=None, sm=None, **kwargs):
        """Creates a new instance of the signal generator for generating
        signal events from a specific public data dataset.

        Parameters:
        -----------
        ds : Dataset instance
            Dataset instance for which signal events should get
            generated for.
        src_dec : float
            The declination of the source in radians.
        effA : PDAeff | None
            Representation of the effective area provided by the public data.
        sm : PDSmearingMatrix | None
            Representation of the smearing matrix provided by the public data.
        """
        super().__init__(**kwargs)

        self._logger = get_logger(module_classname(self))

        if sm is None:
            self.smearing_matrix = PDSmearingMatrix(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('smearing_datafile')))
        else:
            self.smearing_matrix = sm

        if effA is None:
            dec_idx = self.smearing_matrix.get_true_dec_idx(src_dec)
            (min_log_true_e,
             max_log_true_e) = \
                self.smearing_matrix.get_true_log_e_range_with_valid_log_e_pdfs(
                    dec_idx)
            kwargs = {
                'src_dec': src_dec,
                'min_log10enu': min_log_true_e,
                'max_log10enu': max_log_true_e
            }
            self.effA = PDAeff(
                pathfilenames=ds.get_abs_pathfilename_list(
                    ds.get_aux_data_definition('eff_area_datafile')),
                **kwargs)

        else:
            self.effA = effA

    def _generate_inv_cdf_spline(self, flux_model, log_e_min,
                                 log_e_max):
        """Sample the true neutrino energy from the power-law
        re-weighted with the detection probability.
        """
        m = (self.effA.log10_enu_bincenters >= log_e_min) & (
            self.effA.log10_enu_bincenters < log_e_max)
        bin_centers = self.effA.log10_enu_bincenters[m]
        low_bin_edges = self.effA._log10_enu_binedges_lower[m]
        high_bin_edges = self.effA._log10_enu_binedges_upper[m]

        # Flux probability P(E_nu | gamma) per bin.
        flux_prob = flux_model.get_integral(
            10**low_bin_edges, 10**high_bin_edges
        ) / flux_model.get_integral(
            10**low_bin_edges[0], 10**high_bin_edges[-1]
        )

        # Do the product and normalize again to a probability per bin.
        product = flux_prob * self.effA.det_prob
        prob_per_bin = product / np.sum(product)

        # The probability per bin cannot be zero, otherwise the cumulative
        # sum would not be increasing monotonically. So we set zero bins to
        # 1000 times smaller than the smallest non-zero bin.
        m = prob_per_bin == 0
        prob_per_bin[m] = np.min(prob_per_bin[np.invert(m)]) / 1000
        to_keep = np.where(prob_per_bin > 1e-15)[0]  # For numerical stability
        prob_per_bin = prob_per_bin[to_keep]
        prob_per_bin /= np.sum(prob_per_bin)

        # Compute the cumulative distribution CDF.
        cum_per_bin = [np.sum(prob_per_bin[:i])
                       for i in range(prob_per_bin.size+1)]
        if np.any(np.diff(cum_per_bin) == 0):
            raise ValueError(
                'The cumulative sum of the true energy probability is not '
                'monotonically increasing! Values of the cumsum are '
                f'{cum_per_bin}.')

        bin_centers = bin_centers[to_keep]
        bin_centers = np.concatenate(([low_bin_edges[0]], bin_centers))

        # Build a spline for the inverse CDF.
        return interpolate.splrep(cum_per_bin, bin_centers, k=1, s=0)

    @staticmethod
    def _eval_spline(x, spl):
        x = np.asarray(x)
        if (x.any() < 0 or x.any() > 1):
            raise ValueError(
                f'{x} is outside of the valid spline range. '
                'The valid range is [0,1].')
        values = interpolate.splev(x, spl, ext=3)
        return values

    def _generate_events(
            self, rss, src_dec, src_ra, dec_idx,
            log_true_e_inv_cdf_spl, n_events):
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

        sm = self.smearing_matrix

        log_true_e = self._eval_spline(
            rss.random.uniform(size=n_events), log_true_e_inv_cdf_spl)

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

        # Add fields required by the framework
        events['time'] = np.ones(n_events)
        events['azi'] = np.ones(n_events)
        events['zen'] = np.ones(n_events)
        events['run'] = -1 * np.ones(n_events)

        return events

    @staticmethod
    @np.vectorize
    def energy_filter(events, spline, cut_sindec, logger):
        """The energy filter will select all events below `cut_sindec`
        that have an energy smaller than the energy spline at their
        declination.

        Paramters
        ---------
        events : numpy record array
            Numpy record array with the generated signal events.
        energy_cut_splines : scipy.interpolate.UnivariateSpline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : float
            The sine of the declination to start applying the energy cut. 
            The cut will be applied from this declination down.
        logger : logging.Logger
            The Logger instance.

        Returns
        energy_filter : (len(events),)-shaped numpy ndarray
            A mask of shape `len(events)` of the events to be cut.
        """
        if cut_sindec is None:
            logger.warn(
                'No `cut_sindec` has been specified. The energy cut will be '
                'applied in [-90, 0] deg.')
            cut_sindec = 0.
        energy_filter = np.logical_and(
            events['sin_dec'] < cut_sindec,
            events['log_energy'] < spline(events['sin_dec']))

        return energy_filter

    def generate_signal_events(
            self, rss, src_dec, src_ra, flux_model, n_events,
            energy_cut_spline=None, cut_sindec=None):
        """Generates ``n_events`` signal events for the given source location
        and flux model.

        Paramters
        ---------
        rss : RandomStateService
        src_dec : float
            Declination coordinate of the injection point.
        src_ra : float
            Right ascension coordinate of the injection point.
        flux_model : FluxModel
            Instance of the `FluxModel` class.
        n_events : int
            Number of signal events to be generated.
        energy_cut_splines : scipy.interpolate.UnivariateSpline
            A spline of E(sin_dec) that defines the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : float
            The sine of the declination to start applying the energy cut. 
            The cut will be applied from this declination down.

        Returns
        -------
        events : numpy record array
            The numpy record array holding the event data.
            It contains the following data fields:
                - 'isvalid'
                - 'log_true_energy'
                - 'log_energy'
                - 'dec'
                - 'ra'
                - 'ang_err'
        """
        sm = self.smearing_matrix

        # Find the declination bin index.
        dec_idx = sm.get_true_dec_idx(src_dec)

        # Determine the true energy range for which log_e PDFs are available.
        (min_log_true_e,
         max_log_true_e) = sm.get_true_log_e_range_with_valid_log_e_pdfs(
             dec_idx)
        # Build the spline for the inverse CDF and draw a true neutrino
        # energy from the hypothesis spectrum.
        log_true_e_inv_cdf_spl = self._generate_inv_cdf_spline(
            flux_model, min_log_true_e, max_log_true_e)

        events = None
        n_evt_generated = 0
        while n_evt_generated != n_events:
            n_evt = n_events - n_evt_generated

            events_ = self._generate_events(
                rss, src_dec, src_ra, dec_idx, log_true_e_inv_cdf_spl, n_evt)

            # Cut events that failed to be generated due to missing PDFs.
            # Also cut low energy events if generating in the southern sky.
            events_ = events_[events_['isvalid']]
            if energy_cut_spline is not None:
                to_cut = self.energy_filter(
                    events_, energy_cut_spline, cut_sindec, self._logger)
                events_ = events_[~to_cut]
            if not len(events_) == 0:
                n_evt_generated += len(events_)
                if events is None:
                    events = events_
                else:
                    events.append(events_)

        return events


class PDSignalGenerator(SignalGeneratorBase):
    """This class provides a signal generation method for a point-like source
    seen in the IceCube detector using the 10 years public data release.
    """

    def __init__(self, src_hypo_group_manager, dataset_list, data_list=None,
                 llhratio=None, energy_cut_splines=None, cut_sindec=None):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source hypothesis
            groups.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        llhratio : LLHRatio
            The likelihood ratio object contains the datasets signal weights
            needed for distributing the event generation among the different
            datasets.
        energy_cut_splines : list of UnivariateSpline
            A list of splines of E(sin_dec) used to define the declination
            dependent energy cut in the IceCube southern sky.
        cut_sindec : list of float
            The sine of the declination to start applying the energy cut. 
            The cut will be applied from this declination down.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self.dataset_list = dataset_list
        self.data_list = data_list
        self.llhratio = llhratio
        self.effA = [None] * len(self._dataset_list)
        self.sm = [None] * len(self._dataset_list)
        self.splines = energy_cut_splines
        self.cut_sindec = cut_sindec

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance defining the source groups with
        their spectra.
        """
        return self._src_hypo_group_manager

    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                            'instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    @property
    def dataset_list(self):
        """The list of Dataset instances for which signal events should get
        generated for.
        """
        return self._dataset_list

    @dataset_list.setter
    def dataset_list(self, datasets):
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The dataset_list property must be a sequence of '
                            'Dataset instances!')
        self._dataset_list = list(datasets)

    @property
    def llhratio(self):
        """The log-likelihood ratio function for the analysis.
        """
        return self._llhratio

    @llhratio.setter
    def llhratio(self, llhratio):
        if llhratio is not None:
            if(not isinstance(llhratio, LLHRatio)):
                raise TypeError('The llratio property must be an instance of '
                                'LLHRatio!')
        self._llhratio = llhratio

    def generate_signal_events(self, rss, mean, poisson=True):
        shg_list = self._src_hypo_group_manager.src_hypo_group_list
        # Only supports a single source hypothesis group. Raise an error
        # if more than one shg is in the source hypo group manager.
        if len(shg_list) > 1:
            raise RuntimeError(
                'Signal injection for multiple source hypothesis groups is '
                'not supported yet.')

        tot_n_events = 0
        signal_events_dict = {}

        for shg in shg_list:
            # Only supports single point source signal injection. Raise
            # an error if more than one source is in the source hypo group.
            if len(shg.source_list) > 1:
                raise RuntimeError(
                    'Signal injection for multiple sources within a source '
                    'hypothesis group is not supported yet.')
            # This only works with power-laws for now.
            # Each source hypo group can have a different power-law
            gamma = shg.fluxmodel.gamma
            weights, _ = self.llhratio.dataset_signal_weights([mean, gamma])
            for (ds_idx, w) in enumerate(weights):
                w_mean = mean * w
                if(poisson):
                    n_events = rss.random.poisson(
                        float_cast(
                            w_mean,
                            '`mean` must be castable to type of float!'
                        )
                    )
                else:
                    n_events = int_cast(
                        w_mean,
                        '`mean` must be castable to type of int!'
                    )
                tot_n_events += n_events

                events_ = None
                for (shg_src_idx, src) in enumerate(shg.source_list):
                    ds = self._dataset_list[ds_idx]
                    sig_gen = PDDatasetSignalGenerator(
                        ds, src.dec, self.effA[ds_idx], self.sm[ds_idx])
                    if self.effA[ds_idx] is None:
                        self.effA[ds_idx] = sig_gen.effA
                    if self.sm[ds_idx] is None:
                        self.sm[ds_idx] = sig_gen.smearing_matrix
                    # ToDo: here n_events should be split according to some
                    # source weight
                    events_ = sig_gen.generate_signal_events(
                        rss,
                        src.dec,
                        src.ra,
                        shg.fluxmodel,
                        n_events,
                        energy_cut_spline=self.splines[ds_idx],
                        cut_sindec=self.cut_sindec[ds_idx]
                    )
                    if events_ is None:
                        continue

                    if shg_src_idx == 0:
                        signal_events_dict[ds_idx] = events_
                    else:
                        signal_events_dict[ds_idx].append(events_)

        return tot_n_events, signal_events_dict


class PDTimeDependentSignalGenerator(PDSignalGenerator):
    """ The time dependent signal generator works so far only for one single
    dataset. For multi datasets one needs to adjust the dataset weights
    accordingly (scaling of the effective area with livetime of the flare in
    the dataset).
    """

    def __init__(self, src_hypo_group_manager, dataset_list, data_list=None,
                 llhratio=None, energy_cut_splines=None, cut_sindec=None,
                 gauss=None, box=None):
        """
        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        llhratio : LLHRatio
            The likelihood ratio object contains the datasets signal weights
            needed for distributing the event generation among the different
            datsets.
        energy_cut_splines : list of UnivariateSpline
        cut_sindec : float
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

        super().__init__(src_hypo_group_manager, dataset_list, data_list,
                         llhratio, energy_cut_splines, cut_sindec)
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

    def set_flare(self, gauss=None, box=None):
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

    def is_in_grl(self, time, grl):
        """Helper function to check if given times are in the grl ontime.

        Parameters
        ----------
        time : 1d ndarray
            Time values.
        grl : ndarray
            Array of the detector good run list.

        Returns
        -------
        is_in_grl : 1d ndarray
            Boolean mask of `time` in grl ontime.
        """
        def f(time, grl):
            return np.any((grl["start"] <= time) & (time <= grl["stop"]))

        # Vectorize `f`, but exclude `grl` argument from vectorization.
        # This is needed to support `time` as an array argument.
        f_v = np.vectorize(f, excluded=[1])
        is_in_grl = f_v(time, grl)

        return is_in_grl

    def generate_signal_events(self, rss, mean, poisson=True):
        """Same as in PDSignalGenerator, but we assign times here. 
        """
        # Call method from the parent class to generate signal events.
        (tot_n_events, signal_events_dict) = super().generate_signal_events(
            rss, mean, poisson=poisson)

        # Assign times for flare. We can also use inverse transform
        # sampling instead of the lazy version implemented here.
        for (ds_idx, events_) in signal_events_dict.items():
            grl = self.data_list[ds_idx].grl

            # Optimized time injection version, based on csky implementation.
            # https://github.com/icecube/csky/blob/7e969639c5ef6dbb42872dac9b761e1e8b0ccbe2/csky/inj.py#L1122
            times = np.array([])
            n_events = len(events_)
            while len(times) < n_events:
                times = np.concatenate(
                    (times, self.time_pdf.rvs(n_events - len(times),
                                              random_state=rss.random))
                )
                # Check if times is in grl.
                is_in_grl_mask = self.is_in_grl(times, grl)
                times = times[is_in_grl_mask]

            events_["time"] = times
        return tot_n_events, signal_events_dict
