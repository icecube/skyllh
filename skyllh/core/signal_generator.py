# -*- coding: utf-8 -*-

import abc
import itertools

import numpy as np

from astropy import (
    units,
)

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.py import (
    classname,
    issequenceof,
    float_cast,
    int_cast,
    get_smallest_numpy_int_type,
)
from skyllh.core.random import (
    RandomChoice,
)
from skyllh.core.services import (
    DatasetSignalWeightFactorsService,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)


class SignalGenerator(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """This is the abstract base class for all signal generator classes in
    SkyLLH. It defines the interface for a signal generator.
    """
    def __init__(
            self,
            shg_mgr,
            **kwargs,
    ):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The SourceHypoGroupManager instance defining the source hypothesis
            groups.
        """
        super().__init__(
            **kwargs)

        self.shg_mgr = shg_mgr

    @property
    def shg_mgr(self):
        """The SourceHypoGroupManager instance defining the source hypothesis
        groups.
        """
        return self._shg_mgr

    @shg_mgr.setter
    def shg_mgr(self, manager):
        if not isinstance(manager, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager!')
        self._shg_mgr = manager

    def create_src_params_recarray(
            self,
            src_detsigyield_weights_service):
        """Creates the src_params_recarray structured ndarray of length
        N_sources holding the local source parameter names and values needed for
        the calculation of the detector signal yields.

        Parameters
        ----------
        src_detsigyield_weights_service : instance of SrcDetSigYieldWeightsService
            The instance of SrcDetSigYieldWeightsService providing the product
            of the source weights with the detector signal yield.

        Returns
        -------
        src_params_recarray : instance of numpy structured ndarray
            The structured numpy ndarray of length N_sources, holding the local
            parameter names and values of each source needed to calculate the
            detector signal yield.
        """
        # Get the parameter names needed for the detector signal yield
        # calculation.
        param_names = []
        for detsigyield in src_detsigyield_weights_service.detsigyield_arr.flat:
            param_names.extend(detsigyield.param_names)
        param_names = set(param_names)

        # Create an empty structured ndarray of length N_sources.
        dt = []
        for pname in param_names:
            dt.extend([
                (pname, np.float64),
                (f'{pname}:gpidx', np.int32)
            ])
        src_params_recarray = np.empty((self._shg_mgr.n_sources,), dtype=dt)

        sidx = 0
        for (shg_idx, shg) in enumerate(self._shg_mgr.shg_list):

            shg_n_src = shg.n_sources

            shg_src_slice = slice(sidx, sidx+shg_n_src)

            pvalues = []
            for pname in param_names:
                pvalues.extend([
                    shg.fluxmodel.get_param(pname),
                    0
                ])

            src_params_recarray[shg_src_slice] = tuple(pvalues)

            sidx += shg_n_src

        return src_params_recarray

    def change_shg_mgr(
            self,
            shg_mgr):
        """Changes the source hypothesis group manager. Derived classes can
        reimplement this method but this method of the base class must still be
        called by the derived class.
        """
        self.shg_mgr = shg_mgr

    @abc.abstractmethod
    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            src_detsigyield_weights_service=None):
        """This abstract method must be implemented by the derived class to
        generate a given number of signal events.

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
        src_detsigyield_weights_service : instance of SrcDetSigYieldWeightsService | None
            The instance of SrcDetSigYieldWeightsService providing the weighting
            of the sources within the detector. This can be ``None`` if this
            signal generator does not need this information.

        Returns
        -------
        n_signal : int
            The number of generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        pass


class MultiDatasetSignalGenerator(
        SignalGenerator,
):
    """This is a signal generator class handling multiple datasets by using the
    individual signal generator instances for each dataset. This is the most
    general way to support multiple datasets of different formats and signal
    generation.
    """
    def __init__(
            self,
            shg_mgr,
            dataset_list,
            data_list,
            sig_generator_list=None,
            ds_sig_weight_factors_service=None,
            **kwargs):
        """Constructs a new signal generator handling multiple datasets.

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
        """
        super().__init__(
            shg_mgr=shg_mgr,
            **kwargs)

        self.dataset_list = dataset_list
        self.data_list = data_list
        self.sig_generator_list = sig_generator_list
        self.ds_sig_weight_factors_service = ds_sig_weight_factors_service

        self._src_params_recarray = None

    @property
    def dataset_list(self):
        """The list of Dataset instances for which signal events should get
        generated for.
        """
        return self._dataset_list

    @dataset_list.setter
    def dataset_list(self, datasets):
        if not issequenceof(datasets, Dataset):
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
        if not issequenceof(datas, DatasetData):
            raise TypeError(
                'The data_list property must be a sequence of DatasetData '
                'instances!')
        self._data_list = list(datas)

    @property
    def sig_generator_list(self):
        """The list of instance of SignalGenerator holding signal generator
        instances for each individual dataset.
        """
        return self._sig_generator_list

    @sig_generator_list.setter
    def sig_generator_list(self, generators):
        if generators is not None:
            if not issequenceof(generators, (SignalGenerator, type(None))):
                raise TypeError(
                    'The sig_generator_list property must be a sequence of '
                    'SignalGenerator instances!')
            generators = list(generators)
        self._sig_generator_list = generators

    @property
    def ds_sig_weight_factors_service(self):
        """The instance of DatasetSignalWeightFactorsService providing the
        dataset signal weight factor service for calculating the dataset
        signal weights.
        """
        return self._ds_sig_weight_factors_service

    @ds_sig_weight_factors_service.setter
    def ds_sig_weight_factors_service(self, service):
        if not isinstance(service, DatasetSignalWeightFactorsService):
            raise TypeError(
                'The ds_sig_weight_factors_service property must be an '
                'instance of DatasetSignalWeightFactorsService!')
        self._ds_sig_weight_factors_service = service

    @property
    def n_datasets(self):
        """(read-only) The number of datasets.
        """
        return len(self._dataset_list)

    def change_shg_mgr(
            self,
            shg_mgr):
        """Changes the source hypothesis group manager. This will recreate the
        src_params_recarray needed for calculating the detector signal yields.
        Also it calls the ``change_shg_mgr`` methods of the signal generators of
        the individual datasets.
        """
        super().change_shg_mgr(
            shg_mgr=shg_mgr)

        src_detsigyield_weights_service =\
            self.ds_sig_weight_factors_service.src_detsigyield_weights_service
        self._src_params_recarray = self.create_src_params_recarray(
            src_detsigyield_weights_service=src_detsigyield_weights_service)

        if self.sig_generator_list is None:
            return

        for sig_generator in filter(None, self.sig_generator_list):
            sig_generator.change_shg_mgr(
                shg_mgr=shg_mgr)

    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            **kwargs):
        """Generates a given number of signal events distributed across the
        individual datasets.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        mean : float | int
            The mean number of signal events. If the ``poisson`` argument is set
            to True, the actual number of generated signal events will be drawn
            from a Poisson distribution with this given mean value of signal
            events.
        poisson : bool
            If set to True, the actual number of generated signal events will
            be drawn from a Poisson distribution with the given mean value of
            signal events.
            If set to False, the argument ``mean`` must be an integer and
            specifies the actual number of generated signal events.

        Returns
        -------
        n_signal : int
            The number of actual generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        if poisson:
            mean = rss.random.poisson(
                float_cast(
                    mean,
                    'The mean argument must be cast-able to type of float!'))

        mean = int_cast(
            mean,
            'The mean argument must be cast-able to type of int!')

        src_detsigyield_weights_service =\
            self.ds_sig_weight_factors_service.src_detsigyield_weights_service

        # Calculate the dataset weights to distribute the signal events over the
        # datasets.
        if self._src_params_recarray is None:
            self._src_params_recarray = self.create_src_params_recarray(
                src_detsigyield_weights_service=src_detsigyield_weights_service)

        src_detsigyield_weights_service.calculate(
            src_params_recarray=self._src_params_recarray)

        self._ds_sig_weight_factors_service.calculate()
        (ds_weights, _) = self._ds_sig_weight_factors_service.get_weights()

        # Calculate the number of events that need to be generated for each
        # individual dataset. Due to rounding errors, it could happen that the
        # sum of the events is less or greater than ``mean``.
        n_events_arr = np.round(mean * ds_weights, 0).astype(np.int_)

        # Correct the n_events_arr array based on the dataset weights if
        # necessary.
        sum_n_events_arr = np.sum(n_events_arr)
        if sum_n_events_arr != mean:
            (ds_idxs, counts) = np.unique(
                rss.random.choice(
                    np.arange(len(n_events_arr)),
                    size=np.abs(mean - sum_n_events_arr),
                    p=ds_weights,
                ),
                return_counts=True
            )
            if sum_n_events_arr < mean:
                n_events_arr[ds_idxs] += counts
            elif sum_n_events_arr > mean:
                n_events_arr[ds_idxs] -= counts

        n_signal = 0
        signal_events_dict = {}

        for (n_events, ds_sig_generator) in zip(
                n_events_arr,
                self._sig_generator_list):

            (ds_n_signal, ds_sig_events_dict) =\
                ds_sig_generator.generate_signal_events(
                    rss=rss,
                    mean=n_events,
                    poisson=False,
                    src_detsigyield_weights_service=src_detsigyield_weights_service,
                )

            n_signal += ds_n_signal

            for (k, v) in ds_sig_events_dict.items():
                if k not in signal_events_dict:
                    signal_events_dict[k] = v
                else:
                    signal_events_dict[k].append(v)

        return (n_signal, signal_events_dict)


class MCMultiDatasetSignalGenerator(
        MultiDatasetSignalGenerator,
):
    """This is a signal generator class, which handles multiple datasets with
    monte-carlo (MC). It uses the MC events of all datasets to determine the
    possible signal events for a source.
    It does not depend on the detector or source hypothesis, because these
    dependencies are factored out into the signal generation method.
    In fact the construction within this class depends on the construction of
    the signal generation method.
    """
    def __init__(
            self,
            shg_mgr,
            dataset_list,
            data_list,
            valid_event_field_ranges_dict_list=None,
            **kwargs,
    ):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The SourceHypoGroupManager instance defining the source hypothesis
            groups.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        valid_event_field_ranges_dict_list : list of dict | None
            If not ``None``, it specifies for each dataset event fields (key)
            and their valid value range as a 2-element tuple (value). If a
            generated signal event does not fall into a given field range, the
            signal event will be discarded and a new signal event will be drawn.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            dataset_list=dataset_list,
            data_list=data_list,
            **kwargs)

        if valid_event_field_ranges_dict_list is None:
            valid_event_field_ranges_dict_list = [dict()]*len(self.dataset_list)
        if not isinstance(valid_event_field_ranges_dict_list, list):
            raise TypeError(
                'The `valid_event_field_ranges_dict_list` argument must be a list.'
            )
        if len(valid_event_field_ranges_dict_list) != len(self.dataset_list):
            raise ValueError(
                'The valid_event_field_ranges_dict_list argument must be a '
                f'list of length {len(self.dataset_list)}, but it is of length '
                f'{len(valid_event_field_ranges_dict_list)}!')
        self.valid_event_field_ranges_dict_list =\
            valid_event_field_ranges_dict_list

        self._construct_signal_candidates()

    @property
    def valid_event_field_ranges_dict_list(self):
        """The list of dictionary holding the event data fields (key) and their
        valid value range as 2-element tuple (value).
        """
        return self._valid_event_field_ranges_dict_list

    @valid_event_field_ranges_dict_list.setter
    def valid_event_field_ranges_dict_list(self, dict_list):
        if not isinstance(dict_list, list):
            raise TypeError(
                'The valid_event_field_ranges_dict_list must be an instance of '
                'list! '
                f'Its current type is {classname(dict_list)}!')
        for d in dict_list:
            for (k, v) in d.items():
                if not isinstance(k, str):
                    raise TypeError(
                        'Each key of the dictionary of the '
                        'valid_event_field_ranges_dict property must be an '
                        'instance of str! '
                        f'But the type of one of the keys is {classname(k)}!')
                if not isinstance(v, tuple):
                    raise TypeError(
                        'Each value of the dictionary of the '
                        'valid_event_field_ranges_dict property must be an '
                        'instance of tuple! '
                        f'But the value type for the event field "{k}" is '
                        f'{classname(v)}!')
                if len(v) != 2:
                    raise ValueError(
                        f'The tuple for the event field {k} must be of length '
                        f'2! Its current length is {len(v)}!')
        self._valid_event_field_ranges_dict_list = dict_list

    def _construct_signal_candidates(self):
        """Constructs an array holding pointer information of signal candidate
        events pointing into the real MC dataset(s).
        """
        n_datasets = len(self._dataset_list)
        n_sources = self._shg_mgr.n_sources
        shg_list = self._shg_mgr.shg_list
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

        to_internal_time_unit_factor = self._cfg.to_internal_time_unit(
            time_unit=units.day
        )

        # Go through the source hypothesis groups to get the signal event
        # candidates.
        for ((shg_idx, shg), (j, data)) in itertools.product(
                enumerate(shg_list),
                enumerate(self._data_list)):
            sig_gen_method = shg.sig_gen_method
            if sig_gen_method is None:
                raise ValueError(
                    'No signal generation method has been specified for the '
                    f'source hypothesis group with index {shg_idx}!')
            data_mc = data.mc

            (ev_idx_arr, src_idx_arr, flux_arr) =\
                sig_gen_method.calc_source_signal_mc_event_flux(
                    data_mc=data_mc,
                    shg=shg)

            livetime_days = Livetime.get_integrated_livetime(data.livetime)

            weight = (
                data_mc[ev_idx_arr]['mcweight'] *
                flux_arr *
                livetime_days*to_internal_time_unit_factor
            )

            sig_candidates = np.empty(
                (len(ev_idx_arr),),
                dtype=sig_candidates_dtype,
                order='F'
            )
            sig_candidates['ds_idx'] = j
            sig_candidates['ev_idx'] = ev_idx_arr
            sig_candidates['shg_idx'] = shg_idx
            sig_candidates['shg_src_idx'] = src_idx_arr
            sig_candidates['weight'] = weight

            self._sig_candidates = np.append(
                self._sig_candidates, sig_candidates)
            del sig_candidates

        # Normalize the signal candidate weights.
        self._sig_candidates_weight_sum = np.sum(self._sig_candidates['weight'])
        self._sig_candidates['weight'] /= self._sig_candidates_weight_sum

        # Create new RandomChoice instance for the signal candidates.
        self._sig_candidates_random_choice = RandomChoice(
            items=self._sig_candidates,
            probabilities=self._sig_candidates['weight'])

    def _get_invalid_events_mask(
            self,
            events,
            valid_event_field_ranges_dict,
    ):
        """Determines a boolean mask to select invalid events, which do not
        fulfill the given valid event field ranges.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray of length N_events holding the
            events to check.
        valid_event_field_ranges_dict : dict
            The dictionary holding the data field names (key) and their valid
            value ranges (value).

        Raises
        ------
        KeyError
            If one of the event field does not exist in ``events``.

        Returns
        -------
        mask : instance of numpy.ndarray
            The (N_events,)-shaped numpy.ndarray of bool, holding the mask of
            the invalid events.
        """
        mask = np.zeros((len(events),), dtype=np.bool_)

        for (field_name, min_max) in valid_event_field_ranges_dict.items():
            if field_name not in events:
                raise KeyError(
                    f'The event data field "{field_name}" specified in the '
                    'valid_event_field_ranges_dict does not exist in the event '
                    'data!')
            field_values = events[field_name]
            mask |= (field_values < min_max[0]) | (field_values > min_max[1])

        return mask

    def _draw_valid_sig_events_for_dataset_and_shg(
            self,
            rss,
            mc,
            n_signal,
            ds_idx,
            valid_event_field_ranges_dict,
            shg,
            shg_idx,
    ):
        """Draws n_signal valid signal events for the given dataset and source
        hypothesis group.

        Signal events will be drawn until all events match the event field
        ranges, i.e. are valid signal events.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService which should be used to draw
            random numbers from.
        mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monte-carlo events.
        n_signal : int
            The number of signal events to draw.
        ds_idx : int
            The index of the dataset.
        valid_event_field_ranges_dict : dict
            The dictionary holding the data field names (key) and their valid
            value ranges (value) for the requested dataset.
        shg : instance of SourceHypothesisGroup
            The instance of SourceHypothesisGroup for which signal events should
            get drawn.
        shg_idx : int
            The index of the source hypothesis group.

        Returns
        -------
        sig_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the drawn valid signal
            events.
        """
        sig_events = None

        n = 0
        while n < n_signal:
            events_meta = self._sig_candidates_random_choice(
                rss=rss,
                size=n_signal-n,
            )
            m = (events_meta['ds_idx'] == ds_idx) &\
                (events_meta['shg_idx'] == shg_idx)
            events = mc[events_meta['ev_idx'][m]]
            if len(events) > 0:
                events = shg.sig_gen_method.\
                    signal_event_post_sampling_processing(
                        shg, events_meta, events)

                valid_events_mask = np.invert(
                    self._get_invalid_events_mask(
                        events,
                        valid_event_field_ranges_dict,
                    )
                )
                events = events[valid_events_mask]
                if len(events) > 0:
                    if sig_events is None:
                        sig_events = events
                    else:
                        sig_events.append(events)
                    n = len(sig_events)

        return sig_events

    def change_shg_mgr(
            self,
            shg_mgr):
        """Recreates the signal candidates with the changed source hypothesis
        group manager.
        """
        super().change_shg_mgr(
            shg_mgr=shg_mgr)

        self._construct_signal_candidates()

    def mu2flux(
            self,
            mu,
            per_source=False):
        """Translate the mean number of signal events `mu` into the
        corresponding flux. The unit of the returned flux is the internally used
        flux unit.

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
        n_sources = self._shg_mgr.n_sources
        mu_fluxes = np.empty((n_sources,), dtype=np.float64)

        shg_list = self._shg_mgr.shg_list
        mu_fluxes_idx_offset = 0
        for (shg_idx, shg) in enumerate(shg_list):
            fluxmodel = shg.fluxmodel
            # Calculate conversion factor from the flux model unit into the
            # internal flux unit.
            to_internal_flux_unit =\
                fluxmodel.to_internal_flux_unit()
            for k in range(shg.n_sources):
                mask = ((self._sig_candidates['shg_idx'] == shg_idx) &
                        (self._sig_candidates['shg_src_idx'] == k))
                ref_N_k = np.sum(self._sig_candidates[mask]['weight']) * ref_N
                mu_flux_k = (
                    (mu / ref_N) *
                    (ref_N_k / ref_N) *
                    fluxmodel.Phi0 * to_internal_flux_unit)
                mu_fluxes[mu_fluxes_idx_offset + k] = mu_flux_k
            mu_fluxes_idx_offset += shg.n_sources

        if per_source:
            return mu_fluxes

        mu_flux = np.sum(mu_fluxes)
        return mu_flux

    def generate_signal_events(
            self,
            rss,
            mean,
            poisson=True,
            **kwargs):
        """Generates a given number of signal events from the signal candidate
        monte-carlo events.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService providing the random number
            generator state.
        mean : float | int
            The mean number of signal events. If the ``poisson`` argument is set
            to True, the actual number of generated signal events will be drawn
            from a Poisson distribution with this given mean value of signal
            events.
        poisson : bool
            If set to True, the actual number of generated signal events will
            be drawn from a Poisson distribution with the given mean value of
            signal events.
            If set to False, the argument ``mean`` must be an integer and
            specifies the actual number of generated signal events.

        Returns
        -------
        n_signal : int
            The number of actual generated signal events.
        signal_events_dict : dict of DataFieldRecordArray
            The dictionary holding the DataFieldRecordArray instances with the
            generated signal events. Each key of this dictionary represents the
            dataset index for which the signal events have been generated.
        """
        if poisson:
            mean = rss.random.poisson(
                float_cast(
                    mean,
                    'The mean argument must be cast-able to type of float!'))

        n_signal = int_cast(
            mean,
            'The mean argument must be cast-able to type of int!')

        # Draw n_signal signal candidates according to their weight.
        sig_events_meta = self._sig_candidates_random_choice(
            rss=rss,
            size=n_signal,
        )

        # Get the list of unique dataset and source hypothesis group indices of
        # the drawn signal events.
        # Note: This code does not assume the same format for each of the
        #       individual MC datasets, thus might be a bit slower.
        #       If one could assume the same MC dataset format, one
        #       could gather all the MC events of all the datasets first and do
        #       the signal event post processing for all datasets at once.
        signal_events_dict = dict()
        ds_idxs = np.unique(sig_events_meta['ds_idx'])
        for ds_idx in ds_idxs:
            valid_event_field_ranges_dict =\
                self.valid_event_field_ranges_dict_list[ds_idx]
            mc = self._data_list[ds_idx].mc
            ds_mask = sig_events_meta['ds_idx'] == ds_idx
            n_sig_events_ds = np.count_nonzero(ds_mask)

            data = dict([
                (
                    fname,
                    np.empty(
                        (n_sig_events_ds,),
                        dtype=mc.get_field_dtype(fname))
                )
                for fname in mc.field_name_list
            ])
            sig_events = DataFieldRecordArray(data, copy=False)

            fill_start_idx = 0
            # Get the list of unique source hypothesis group indices for the
            # current dataset.
            shg_idxs = np.unique(sig_events_meta[ds_mask]['shg_idx'])
            for shg_idx in shg_idxs:
                shg = self._shg_mgr.shg_list[shg_idx]
                shg_mask = sig_events_meta['shg_idx'] == shg_idx
                # Get the MC events for the drawn signal events.
                ds_shg_mask = ds_mask & shg_mask
                shg_sig_events_meta = sig_events_meta[ds_shg_mask]
                n_shg_sig_events = len(shg_sig_events_meta)
                ev_idx = shg_sig_events_meta['ev_idx']
                # Get the signal MC events of the current dataset and source
                # hypothesis group.
                shg_sig_events = mc[ev_idx]

                # Do the signal event post sampling processing.
                shg_sig_events = shg.sig_gen_method.\
                    signal_event_post_sampling_processing(
                        shg, shg_sig_events_meta, shg_sig_events)

                # Determine the signal events, which do not fulfill the valid
                # event field ranges for this dataset.
                invalid_events_mask = self._get_invalid_events_mask(
                    shg_sig_events,
                    valid_event_field_ranges_dict)
                n_redraw_events = np.count_nonzero(invalid_events_mask)
                if n_redraw_events > 0:
                    # Re-draw n_redraw_events signal events for this dataset
                    # and SHG.
                    redrawn_shg_sig_events =\
                        self._draw_valid_sig_events_for_dataset_and_shg(
                            rss=rss,
                            mc=mc,
                            n_signal=n_redraw_events,
                            ds_idx=ds_idx,
                            valid_event_field_ranges_dict=valid_event_field_ranges_dict,
                            shg=shg,
                            shg_idx=shg_idx,
                        )
                    shg_sig_events[invalid_events_mask] = redrawn_shg_sig_events

                indices = np.indices((n_shg_sig_events,))[0] + fill_start_idx
                sig_events.set_selection(indices, shg_sig_events)

                fill_start_idx += n_shg_sig_events

            signal_events_dict[ds_idx] = sig_events

        return (n_signal, signal_events_dict)
