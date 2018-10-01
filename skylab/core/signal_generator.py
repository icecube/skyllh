# -*- coding: utf-8 -*-

import itertools
import numpy as np

from skylab.core.py import (
    issequenceof,
    float_cast,
    int_cast,
    get_smallest_numpy_int_type
)
from skylab.core.random import RandomStateService
from skylab.core.dataset import Dataset, DatasetData
from skylab.core.source_hypothesis import SourceHypoGroupManager


class SignalGenerator(object):
    """This is the general signal generator class. It does not depend on the
    detector or source hypothesis, because these dependencies are factored out
    into the signal generation method.
    """
    def __init__(self, rss, src_hypo_group_manager, dataset_list, data_list):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
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
        super(SignalGenerator, self).__init__()

        self.rss = rss
        self.src_hypo_group_manager = src_hypo_group_manager

        self.dataset_list = dataset_list
        self.data_list = data_list

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
            ('weight', np.float)
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
            (ev_indices_list, flux_list) = sig_gen_method.calc_source_signal_mc_event_flux(
                data.mc, shg
            )
            for (k, (ev_indices, flux)) in enumerate(zip(ev_indices_list, flux_list)):
                ev = data.mc[ev_indices]
                # The weight of the event specifies the number of signal events
                # this one event corresponds to.
                # [weight] = GeV cm^2 sr * s * 1/(GeV cm^2 s sr)
                weight = ev['mcweight'] * ds.livetime * 86400 * flux

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

    @property
    def rss(self):
        """The RandomStateService instance providing the state of the random
        number generator.
        """
        return self._rss
    @rss.setter
    def rss(self, rss):
        if(not isinstance(rss, RandomStateService)):
            raise TypeError('The rss property must be an instance of '
                'RandomStateService!')
        self._rss = rss

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
    def data_list(self):
        """The list of DatasetData instances holding the actual data of each
        dataset. The order must match the order of the ``dataset_list``
        property.
        """
        return self._data_list
    @data_list.setter
    def data_list(self, datas):
        if(not issequenceof(datas, DatasetData)):
            raise TypeError('The data_list property must be a sequence of '
                'DatasetData instances!')
        self._data_list = datas

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Recreates the signal candidates with the changed source hypothesis
        group manager.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._construct_signal_candidates()

    def generate_signal_events(self, mean, poisson=True):
        """Generates a given number of signal events from the signal candidate
        monte-carlo events.

        Parameters
        ----------
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
        signal_events_dict : dict of numpy record array
            The dictionary holding the numpy record arrays holding the generated
            signal events. Each key of this dictionary represents the dataset
            index for which the signal events have been generated.
        """
        if(poisson):
            mean = self._rss.random.poisson(float_cast(
                mean, 'The mean argument must be castable to type of float!'))

        n_signal = int_cast(
            mean, 'The mean argument must be castable to type of int!')

        # Draw n_signal signal candidates according to their weight.
        sig_events_meta = self._rss.random.choice(
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
            ds_mask = sig_events_meta['ds_idx'] == ds_idx
            sig_events = np.empty(
                (np.count_nonzero(ds_mask),), dtype=self._data_list[ds_idx].mc.dtype)
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
                # hypothesis group. We have to create a copy to not alter the
                # original MC events.
                shg_sig_events = np.copy(self._data_list[ds_idx].mc[ev_idx])

                # Do the signal event post sampling processing.
                shg_sig_events = shg.sig_gen_method.signal_event_post_sampling_processing(
                    shg, shg_sig_events_meta, shg_sig_events)

                sig_events[fill_start_idx:fill_start_idx+n_shg_sig_events] = shg_sig_events
                fill_start_idx += n_shg_sig_events

            signal_events_dict[ds_idx] = sig_events

        return (n_signal, signal_events_dict)
