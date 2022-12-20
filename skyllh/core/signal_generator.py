# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np

from skyllh.core.py import (
    issequenceof,
    float_cast,
    int_cast,
    get_smallest_numpy_int_type
)
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.storage import DataFieldRecordArray
from skyllh.physics.flux import (
    get_conversion_factor_to_internal_flux_unit
)


class SignalGeneratorBase(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for all signal generator classes in
    SkyLLH. It defines the interface for signal generators.
    """
    def __init__(self, src_hypo_group_manager, dataset_list, data_list,
                 *args, **kwargs):
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
        """
        super().__init__(*args, **kwargs)

        self.src_hypo_group_manager = src_hypo_group_manager
        self.dataset_list = dataset_list
        self.data_list = data_list

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance defining the source hypothesis
        groups.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError(
                'The src_hypo_group_manager property must be an instance of '
                'SourceHypoGroupManager!')
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
            raise TypeError(
                'The dataset_list property must be a sequence of Dataset '
                'instances!')
        self._dataset_list = list(datasets)

    @property
    def data_list(self):
        """The list of DatasetData instances holding the actual data of each
        dataset. The order must match the order of the ``dataset_list``
        property.
        """
        return self._data_list
    @data_list.setter
    def data_list(self, datas):
        if(not issequenceof(datas, DatasetData)):
            raise TypeError(
                'The data_list property must be a sequence of DatasetData '
                'instances!')
        self._data_list = datas

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the source hypothesis group manager. Derived classes can
        reimplement this method but this method of the base class must still be
        called by the derived class.
        """
        self.src_hypo_group_manager = src_hypo_group_manager

    @abc.abstractmethod
    def generate_signal_events(self, rss, mean, poisson=True):
        """This abstract method must be implemented by the derived class to
        generate a given number of signal events.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        mean : float
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

        Returns
        -------
        n_signal : int
            The number of generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instancs with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        pass


class SignalGenerator(SignalGeneratorBase):
    """This is the general signal generator class. It does not depend on the
    detector or source hypothesis, because these dependencies are factored out
    into the signal generation method. In fact the construction within this
    class depends on the construction of the signal generation method. In case
    of multiple sources the handling here is very suboptimal. Therefore the
    MultiSourceSignalGenerator should be used instead!
    """
    def __init__(self, src_hypo_group_manager, dataset_list, data_list,
                 *args, **kwargs):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        """
        super().__init__(
            *args,
            src_hypo_group_manager=src_hypo_group_manager,
            dataset_list=dataset_list,
            data_list=data_list,
            **kwargs)

        self._construct_signal_candidates()

    def _construct_signal_candidates(self):
        """Constructs an array holding pointer information of signal candidate
        events pointing into the real MC dataset(s).
        """
        n_datasets = len(self._dataset_list)
        n_sources = self._src_hypo_group_manager.n_sources
        shg_list = self._src_hypo_group_manager.src_hypo_group_list
        sig_candidates_dtype = [
            ('ds_idx', get_smallest_numpy_int_type((0, n_datasets))),
            ('ev_idx', get_smallest_numpy_int_type(
                [0]+[len(data.mc) for data in self._data_list])),
            ('shg_idx', get_smallest_numpy_int_type((0, n_sources))),
            ('shg_src_idx', get_smallest_numpy_int_type(
                [0]+[shg.n_sources for shg in shg_list])),
            ('weight', np.float64)
        ]
        self._sig_candidates = np.empty(
            (0,), dtype=sig_candidates_dtype, order='F')

        # Go through the source hypothesis groups to get the signal event
        # candidates.
        for ((shg_idx,shg), (j,(ds,data))) in itertools.product(
            enumerate(shg_list), enumerate(zip(self._dataset_list, self._data_list))):
            sig_gen_method = shg.sig_gen_method
            if(sig_gen_method is None):
                raise ValueError('No signal generation method has been '
                    'specified for the %dth source hypothesis group!'%(shg_idx))
            data_mc = data.mc
            (ev_indices_list, flux_list) = sig_gen_method.calc_source_signal_mc_event_flux(
                data_mc, shg
            )
            for (k, (ev_indices, flux)) in enumerate(zip(ev_indices_list, flux_list)):
                ev = data_mc[ev_indices]
                # The weight of the event specifies the number of signal events
                # this one event corresponds to for the given reference flux.
                # [weight] = GeV cm^2 sr * s * 1/(GeV cm^2 s sr)
                weight = ev['mcweight'] * data.livetime * 86400 * flux

                sig_candidates = np.empty(
                    (len(ev_indices),), dtype=sig_candidates_dtype, order='F'
                )
                sig_candidates['ds_idx'] = j
                sig_candidates['ev_idx'] = ev_indices
                sig_candidates['shg_idx'] = shg_idx
                sig_candidates['shg_src_idx'] = k
                sig_candidates['weight'] = weight

                self._sig_candidates = np.append(self._sig_candidates, sig_candidates)

        # Normalize the signal candidate weights.
        self._sig_candidates_weight_sum = np.sum(self._sig_candidates['weight'])
        self._sig_candidates['weight'] /= self._sig_candidates_weight_sum

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Recreates the signal candidates with the changed source hypothesis
        group manager.
        """
        super().change_source_hypo_group_manager(src_hypo_group_manager)

        self._construct_signal_candidates()

    def mu2flux(self, mu, per_source=False):
        """Translate the mean number of signal events `mu` into the
        corresponding flux. The unit of the returned flux is 1/(GeV cm^2 s).

        Parameters
        ----------
        mu : float
            The mean number of expected signal events for which to get the flux.
        per_source : bool
            Flag if the flux should be returned for each source individually
            (True), or as the sum of all these fluxes (False). The default is
            False.

        Returns
        -------
        mu_flux : float | (n_sources,)-shaped numpy ndarray
            The total flux for all sources (if `per_source = False`) that would
            correspond to the given mean number of detected signal events `mu`.
            If `per_source` is set to True, a numpy ndarray is returned that
            contains the flux for each individual source.
        """
        # Calculate the expected mean number of signal events for each source
        # of the source hypo group manager. For each source we can calculate the
        # flux that would correspond to the given mean number of signal events
        # `mu`. The total flux for all sources is then just the sum.

        # The ref_N variable describes how many total signal events are expected
        # on average for the reference fluxes.
        ref_N = self._sig_candidates_weight_sum

        # The mu_fluxes array is the flux of each source for mu mean detected
        # signal events.
        n_sources = self._src_hypo_group_manager.n_sources
        mu_fluxes = np.empty((n_sources,), dtype=np.float64)

        shg_list = self._src_hypo_group_manager.src_hypo_group_list
        mu_fluxes_idx_offset = 0
        for (shg_idx,shg) in enumerate(shg_list):
            fluxmodel = shg.fluxmodel
            # Calculate conversion factor from the flux model unit into the
            # internal flux unit GeV^-1 cm^-2 s^-1.
            toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)
            for k in range(shg.n_sources):
                mask = ((self._sig_candidates['shg_idx'] == shg_idx) &
                        (self._sig_candidates['shg_src_idx'] == k))
                ref_N_k = np.sum(self._sig_candidates[mask]['weight']) * ref_N
                mu_flux_k = mu / ref_N * (ref_N_k / ref_N) * fluxmodel.Phi0*toGeVcm2s
                mu_fluxes[mu_fluxes_idx_offset + k] = mu_flux_k
            mu_fluxes_idx_offset += shg.n_sources

        if(per_source):
            return mu_fluxes

        mu_flux = np.sum(mu_fluxes)
        return mu_flux

    def generate_signal_events(self, rss, mean, poisson=True):
        """Generates a given number of signal events from the signal candidate
        monte-carlo events.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        mean : float
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

        Returns
        -------
        n_signal : int
            The number of generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instancs with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        if(poisson):
            mean = rss.random.poisson(float_cast(
                mean, 'The mean argument must be castable to type of float!'))

        n_signal = int_cast(
            mean, 'The mean argument must be castable to type of int!')

        # Draw n_signal signal candidates according to their weight.
        sig_events_meta = rss.random.choice(
            self._sig_candidates,
            size=n_signal,
            p=self._sig_candidates['weight']
        )
        # Get the list of unique dataset and source hypothesis group indices of
        # the drawn signal events.
        # Note: This code does not assume the same format for each of the
        #       individual MC dataset numpy record arrays, thus might be a bit
        #       slower. If one could assume the same MC dataset format, one
        #       could gather all the MC events of all the datasets first and do
        #       the signal event post processing for all datasets at once.
        signal_events_dict = dict()
        ds_idxs = np.unique(sig_events_meta['ds_idx'])
        for ds_idx in ds_idxs:
            mc = self._data_list[ds_idx].mc
            ds_mask = sig_events_meta['ds_idx'] == ds_idx
            n_sig_events_ds = np.count_nonzero(ds_mask)

            data = dict(
                [(fname, np.empty(
                    (n_sig_events_ds,),
                    dtype=mc.get_field_dtype(fname))
                 ) for fname in mc.field_name_list])
            sig_events = DataFieldRecordArray(data, copy=False)

            fill_start_idx = 0
            # Get the list of unique source hypothesis group indices for the
            # current dataset.
            shg_idxs = np.unique(sig_events_meta[ds_mask]['shg_idx'])
            for shg_idx in shg_idxs:
                shg = self._src_hypo_group_manager.src_hypo_group_list[shg_idx]
                shg_mask = sig_events_meta['shg_idx'] == shg_idx
                # Get the MC events for the drawn signal events.
                ds_shg_mask = ds_mask & shg_mask
                shg_sig_events_meta = sig_events_meta[ds_shg_mask]
                n_shg_sig_events = len(shg_sig_events_meta)
                ev_idx = shg_sig_events_meta['ev_idx']
                # Get the signal MC events of the current dataset and source
                # hypothesis group.
                shg_sig_events = mc.get_selection(ev_idx)

                # Do the signal event post sampling processing.
                shg_sig_events = shg.sig_gen_method.signal_event_post_sampling_processing(
                    shg, shg_sig_events_meta, shg_sig_events)

                indices = np.indices((n_shg_sig_events,))[0] + fill_start_idx
                sig_events.set_selection(indices, shg_sig_events)

                #sig_events[fill_start_idx:fill_start_idx+n_shg_sig_events] = shg_sig_events
                fill_start_idx += n_shg_sig_events

            signal_events_dict[ds_idx] = sig_events

        return (n_signal, signal_events_dict)


class MultiSourceSignalGenerator(SignalGenerator):
    """More optimal signal generator for multiple sources.
    """
    def __init__(self, src_hypo_group_manager, dataset_list, data_list):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        """
        super(MultiSourceSignalGenerator, self).__init__(
            src_hypo_group_manager, dataset_list, data_list)

    def _construct_signal_candidates(self):
        """Constructs an array holding pointer information of signal candidate
        events pointing into the real MC dataset(s).
        """
        n_datasets = len(self._dataset_list)
        n_sources = self._src_hypo_group_manager.n_sources
        shg_list = self._src_hypo_group_manager.src_hypo_group_list
        sig_candidates_dtype = [
            ('ds_idx', get_smallest_numpy_int_type((0, n_datasets))),
            ('ev_idx', get_smallest_numpy_int_type(
                [0]+[len(data.mc) for data in self._data_list])),
            ('shg_idx', get_smallest_numpy_int_type((0, n_sources))),
            ('shg_src_idx', get_smallest_numpy_int_type(
                [0]+[shg.n_sources for shg in shg_list])),
            ('weight', np.float64)
        ]
        self._sig_candidates = np.empty(
            (0,), dtype=sig_candidates_dtype, order='F')

        # Go through the source hypothesis groups to get the signal event
        # candidates.
        for ((shg_idx, shg), (j, (ds, data))) in itertools.product(
                enumerate(shg_list),
                enumerate(zip(self._dataset_list, self._data_list))):
            sig_gen_method = shg.sig_gen_method
            if(sig_gen_method is None):
                raise ValueError(
                    'No signal generation method has been specified '
                    'for the %dth source hypothesis group!' % (shg_idx))
            data_mc = data.mc
            (ev_indices, src_indices, flux) = sig_gen_method.calc_source_signal_mc_event_flux(
                data_mc, shg)

            sig_candidates = np.empty(
                    (len(ev_indices),), dtype=sig_candidates_dtype, order='F'
                )
            sig_candidates['ds_idx'] = j
            sig_candidates['ev_idx'] = ev_indices
            sig_candidates['shg_idx'] = shg_idx
            sig_candidates['shg_src_idx'] = src_indices
            sig_candidates['weight'] = data_mc[ev_indices]['mcweight'] * data.livetime * 86400 * flux

            self._sig_candidates = np.append(self._sig_candidates, sig_candidates)
            del sig_candidates

        # Normalize the signal candidate weights.
        self._sig_candidates_weight_sum = np.sum(self._sig_candidates['weight'])
        self._sig_candidates['weight'] /= self._sig_candidates_weight_sum
