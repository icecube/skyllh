# -*- coding: utf-8 -*-

import abc

import numpy as np

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.datafields import (
    DataFields,
    DataFieldStages as DFS,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.event_selection import (
    AllEventSelectionMethod,
    EventSelectionMethod,
)
from skyllh.core.py import (
    classname,
    float_cast,
    func_has_n_args,
    issequenceof,
)
from skyllh.core.random import (
    RandomChoice,
)
from skyllh.core.scrambling import (
    DataScrambler,
)
from skyllh.core.timing import (
    TaskTimer,
)


logger = get_logger(__name__)


class BackgroundGenerationMethod(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """This is the abstract base class for a detector specific background
    generation method.
    """

    def __init__(
            self,
            **kwargs,
    ):
        """Constructs a new background generation method instance.
        """
        super().__init__(**kwargs)

    def change_shg_mgr(self, shg_mgr):
        """Notifies the background generation method about an updated
        SourceHypoGroupManager instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        pass

    @abc.abstractmethod
    def generate_events(
            self,
            rss,
            dataset,
            data,
            mean,
            tl=None,
            **kwargs,
    ):
        """This method is supposed to generate a `mean` number of background
        events for the given dataset and its data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        dataset : instance of Dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data : instance of DatasetData
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.
        mean : float
            The mean number of background events to generate.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.
        **kwargs
            Additional keyword arguments, which might be required for a
            particular background generation method.

        Returns
        -------
        n_bkg : int
            The number of generated background events.
        bkg_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the generated
            background events. The number of events in this array might be less
            than ``n_bkg`` if an event selection method was used for
            optimization purposes. The difference ``n_bkg - len(bkg_events)`` is
            then the number of pure background events in the generated
            background event sample.
        """
        pass


class MCDataSamplingBkgGenMethod(
        BackgroundGenerationMethod,
):
    """This class implements the method to generate background events from
    monte-carlo (MC) data by sampling events from the MC data set according to a
    probability value given for each event. Functions can be provided to get the
    mean number of background events and the probability of each monte-carlo
    event.
    """
    def __init__(
            self,
            get_event_prob_func,
            get_mean_func=None,
            data_scrambler=None,
            keep_mc_data_fields=None,
            pre_event_selection_method=None,
            **kwargs,
    ):
        """Creates a new instance of the MCDataSamplingBkgGenMethod class.

        Parameters
        ----------
        get_event_prob_func : callable
            The function to get the background probability of each monte-carlo
            event. The call signature of this function must be

                __call__(dataset, data, events)

            where ``dataset`` is an instance of Dataset and ``data`` is an
            instance of DatasetData of the data set for which background events
            needs to get generated. The ``events`` argument holds the actual
            set of events, for which the background event probabilities need to
            get calculated.
        get_mean_func : callable | None
            The function to get the mean number of background events.
            The call signature of this function must be

                __call__(dataset, data, events)

            where ``dataset`` is an instance of Dataset and ``data`` is an
            instance of DatasetData of the data set for which background events
            needs to get generated. The `events` argument holds the actual set
            of events, for which the mean number of background events should get
            calculated. This argument can be `None`, which means that the mean
            number of background events to generate needs to get specified
            through the ``generate_events`` method. However, if a pre event
            selection method is provided, this argument cannot be ``None``!
        data_scrambler : instance of DataScrambler | None
            If set to an instance of DataScrambler, the drawn monte-carlo
            background events will get scrambled. This can ensure more
            independent data trials. It is especially important when monte-carlo
            statistics are low.
        keep_mc_data_fields : str | list of str | None
            The MC data field names that should be kept in order to be able to
            calculate the background events rates by the functions
            ``get_event_prob_func`` and ``get_mean_func``. All other MC fields
            will get dropped due to computational efficiency reasons.
        pre_event_selection_method : instance of EventSelectionMethod | None
            If set to an instance of EventSelectionMethod, this method will
            pre-select the MC events that will be used for later background
            event generation. Using this pre-selection a large portion of the
            MC data can be reduced prior to background event generation.
        """
        super().__init__(
            **kwargs)

        self.get_event_prob_func = get_event_prob_func
        self.get_mean_func = get_mean_func
        self.data_scrambler = data_scrambler
        self.keep_mc_data_field_names = keep_mc_data_fields
        self.pre_event_selection_method = pre_event_selection_method

        if (pre_event_selection_method is not None) and (get_mean_func is None):
            raise ValueError(
                'If an event pre-selection method is provided, a '
                'get_mean_func needs to be provided as well!')

        # Define cache members to cache the background probabilities for each
        # monte-carlo event. The probabilities change only if the data changes.
        self._cache_data_id = None
        self._cache_mc = None
        self._cache_mc_event_bkg_prob = None
        self._cache_mean = None
        self._cache_mean_pre_selected = None
        self._cache_random_choice = None

    @property
    def get_event_prob_func(self):
        """The function to obtain the background probability for each
        monte-carlo event of the data set.
        """
        return self._get_event_prob_func

    @get_event_prob_func.setter
    def get_event_prob_func(self, func):
        if not callable(func):
            raise TypeError(
                'The get_event_prob_func property must be a callable! '
                f'Its current type is {classname(func)}.')
        if not func_has_n_args(func, 3):
            raise TypeError(
                'The function provided for the get_event_prob_func property '
                'must have 3 arguments!')
        self._get_event_prob_func = func

    @property
    def get_mean_func(self):
        """The function to obtain the mean number of background events for the
        data set. This can be None, which means that the mean number of
        background events to generate needs to be specified through the
        `generate_events` method.
        """
        return self._get_mean_func

    @get_mean_func.setter
    def get_mean_func(self, func):
        if func is None:
            self._get_mean_func = None
            return

        if not callable(func):
            raise TypeError(
                'The get_mean_func property must be a callable! '
                f'Its current type is {classname(func)}.')
        if not func_has_n_args(func, 3):
            raise TypeError(
                'The function provided for the get_mean_func property must '
                'have 3 arguments!')
        self._get_mean_func = func

    @property
    def data_scrambler(self):
        """The DataScrambler instance that should be used to scramble the drawn
        monte-carlo background events to ensure more independent data trials.
        This is especially important when monte-carlo statistics are low. It is
        `None`, if no data scrambling should be used.
        """
        return self._data_scrambler

    @data_scrambler.setter
    def data_scrambler(self, scrambler):
        if scrambler is None:
            self._data_scrambler = None
            return

        if not isinstance(scrambler, DataScrambler):
            raise TypeError(
                'The data_scrambler property must be an instance of '
                'DataScrambler! '
                f'Its current type is {classname(scrambler)}.')
        self._data_scrambler = scrambler

    @property
    def keep_mc_data_field_names(self):
        """The MC data field names that should be kept in order to be able to
        calculate the background events rates by the functions
        ``get_event_prob_func`` and ``get_mean_func``. All other MC fields
        will get dropped due to computational efficiency reasons.
        """
        return self._keep_mc_data_field_names

    @keep_mc_data_field_names.setter
    def keep_mc_data_field_names(self, names):
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        elif not issequenceof(names, str):
            raise TypeError(
                'The keep_mc_data_field_names must be None, an instance of '
                'type str, or a sequence of instances of type str!')
        self._keep_mc_data_field_names = names

    @property
    def pre_event_selection_method(self):
        """The instance of EventSelectionMethod that pre-selects the MC events,
        which can be considered for background event generation.
        """
        return self._pre_event_selection_method

    @pre_event_selection_method.setter
    def pre_event_selection_method(self, method):
        if method is None:
            self._pre_event_selection_method = None
            return

        if not isinstance(method, EventSelectionMethod):
            raise TypeError(
                'The pre_event_selection_method property must be None, or an '
                'instance of EventSelectionMethod!')

        # If the event selection method selects all events, it's equivalent
        # to have it set to None, because then no operation has to be
        # performed.
        if isinstance(method, AllEventSelectionMethod):
            method = None

        self._pre_event_selection_method = method

    def change_shg_mgr(self, shg_mgr):
        """Changes the instance of SourceHypoGroupManager of the
        pre-event-selection method. Also it invalidates the data cache of this
        background generation method.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        if self._pre_event_selection_method is not None:
            self._pre_event_selection_method.change_shg_mgr(
                shg_mgr=shg_mgr)

        # Invalidate the data cache.
        self._cache_data_id = None

    def generate_events(
            self,
            rss,
            dataset,
            data,
            mean=None,
            poisson=True,
            tl=None,
    ):
        """Generates a ``mean`` number of background events for the given
        dataset and its data.

        The procedure is as follows:

            1. If the dataset has changed, calculate the background event
               probability for each MC event and cache it.
            1.1. Calculate the mean number of bkg events for the entire MC.
            1.2. Pre-select MC events if a pre event selection method is set.
            1.3. Calculate the background drawing probability of each selected
                 MC event.
            1.4. Calculate the mean number of bkg events for the selected MC
                 events.
            2. Draw the bkg events based on the calculated bkg event
               probability.
            3. Scramble the drawn background events if a data scrambler is set.
            4. Remove any MC specific data fields from the background events
               DataFieldRecordArray instance.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        dataset : instance of Dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data : instance of DatasetData
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.
        mean : float | None
            The mean number of background events to generate.
            Can be `None`. In that case the mean number of background events is
            obtained through the `get_mean_func` function.
        poisson : bool
            If set to ``True`` (default), the actual number of generated
            background events will be drawn from a Poisson distribution with the
            given mean number of background events.
            If set to ``False``, the argument ``mean`` specifies the actual
            number of generated background events.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        n_bkg : int
            The number of generated background events for the data set.
        bkg_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the generated
            background events. The number of events can be less than `n_bkg`
            if an event selection method is used.
        """
        tracing = self._cfg['debugging']['enable_tracing']

        # Check if the data set has changed. In that case need to get new
        # background probabilities for each monte-carlo event and a new mean
        # number of background events.
        data_id = id(data)
        if self._cache_data_id != data_id:
            if tracing:
                logger.debug(
                    f'DatasetData instance id of dataset "{dataset.name}" '
                    f'changed from {self._cache_data_id} to {data_id}')
            # Cache the current id of the data.
            self._cache_data_id = data_id

            # Create a copy of the MC data with all MC data fields removed,
            # except the specified MC data fields to keep for the
            # ``get_mean_func`` and ``get_event_prob_func`` functions.
            keep_field_names = list(set(
                DataFields.get_joint_names(
                    datafields=self._cfg['datafields'],
                    stages=(
                        DFS.ANALYSIS_EXP
                    )
                ) +
                data.exp_field_names +
                self._keep_mc_data_field_names
            ))
            self._cache_mc = data.mc.copy(keep_fields=keep_field_names)

            if self._get_mean_func is not None:
                with TaskTimer(
                        tl,
                        'Calculate total MC background mean.'):
                    self._cache_mean = self._get_mean_func(
                        dataset=dataset,
                        data=data,
                        events=self._cache_mc)

            if self._pre_event_selection_method is not None:
                with TaskTimer(
                        tl,
                        'Pre-select MC events.'):
                    (self._cache_mc, _) =\
                        self._pre_event_selection_method.select_events(
                            events=self._cache_mc,
                            ret_original_evt_idxs=False,
                            tl=tl)

                with TaskTimer(tl, 'Calculate selected MC background mean.'):
                    self._cache_mean_pre_selected = self._get_mean_func(
                        dataset=dataset,
                        data=data,
                        events=self._cache_mc)

            with TaskTimer(
                    tl,
                    'Calculate MC background event probability cache.'):
                self._cache_mc_event_bkg_prob = self._get_event_prob_func(
                    dataset=dataset,
                    data=data,
                    events=self._cache_mc)

            with TaskTimer(
                    tl,
                    'Create RandomChoice for MC background events.'):
                self._cache_random_choice = RandomChoice(
                    items=self._cache_mc.indices,
                    probabilities=self._cache_mc_event_bkg_prob)

        if mean is None:
            if self._cache_mean is None:
                raise ValueError(
                    'No mean number of background events and no '
                    'get_mean_func were specified! One of the two must be '
                    'specified!')
            mean = self._cache_mean
        else:
            mean = float_cast(
                mean,
                'The mean number of background events must be cast-able to '
                'type float!')

        # Draw the number of background events from a poisson distribution with
        # the given mean number of background events. This will be the number of
        # background events for this data set.
        n_bkg = (int(rss.random.poisson(mean)) if poisson else
                 int(np.round(mean, 0)))

        # Calculate the mean number of background events for the pre-selected
        # MC events.
        if self._pre_event_selection_method is None:
            # No selection at all, use the total mean.
            mean_pre_selected = mean
        else:
            mean_pre_selected = self._cache_mean_pre_selected

        # Calculate the actual number of background events for the selected
        # events.
        n_bkg_selected = int(np.around(n_bkg * mean_pre_selected / mean, 0))

        # Draw the actual background events from the selected events of the
        # monte-carlo data set.
        with TaskTimer(tl, 'Draw MC background indices.'):
            bkg_event_indices = self._cache_random_choice(
                rss=rss,
                size=n_bkg_selected)

        with TaskTimer(tl, 'Select MC background events from indices.'):
            bkg_events = self._cache_mc[bkg_event_indices]

        # Scramble the drawn MC events if requested.
        if self._data_scrambler is not None:
            with TaskTimer(tl, 'Scramble MC background data.'):
                bkg_events = self._data_scrambler.scramble_data(
                    rss=rss,
                    dataset=dataset,
                    data=bkg_events,
                    copy=False)

        # Remove MC specific data fields from the background events record
        # array. So the result contains only experimental data fields. The list
        # of experimental data fields is defined as the unique set of the
        # required experimental data fields defined by the data set, and the
        # actual experimental data fields (in case there are additional kept
        # data fields by the user).
        with TaskTimer(tl, 'Remove MC specific data fields from MC events.'):
            exp_field_names = list(set(
                DataFields.get_joint_names(
                    datafields=self._cfg['datafields'],
                    stages=(
                        DFS.ANALYSIS_EXP
                    )
                ) +
                data.exp_field_names))
            bkg_events.tidy_up(exp_field_names)

        return (n_bkg, bkg_events)


class CompositeMCDataSamplingBkgGenMethod(
    MCDataSamplingBkgGenMethod,
):
    """This background generation method samples background events from the
    monte-carlo (MC) data. It supports a list of background components, which
    depend on the sky position and hence whose event rates need to be calculated
    for each individual trial after scrambling the MC sample.
    """
    def __init__(
            self,
            bkg_component_rate_calc_func_dict,
            get_event_prob_func,
            get_mean_func=None,
            data_scrambler=None,
            keep_mc_data_fields=None,
            pre_event_selection_method=None,
            **kwargs,
    ):
        """Creates a new instance of CompositeMCDataSamplingBkgGenMethod.

        Parameters
        ----------
        bkg_component_rate_calc_func_dict : dict
            The dictionary holding the name of the background component as key,
            e.g. "gp", and the background rate calculation function for that
            component as value.
            The call signature of each rate calculation function must be

                __call__(dataset, data, events)

            where ``dataset`` is an instance of Dataset and ``data`` is an
            instance of DatasetData of the data set for which background events
            needs to get generated. The ``events`` argument holds the actual
            set of events, for which the background rate needs to get
            calculated.
        get_event_prob_func : callable
            The function to get the background probability of each monte-carlo
            event. The call signature of this function must be

                __call__(dataset, data, events)

            where ``dataset`` is an instance of Dataset and ``data`` is an
            instance of DatasetData of the data set for which background events
            needs to get generated. The ``events`` argument holds the actual
            set of events, for which the background event probabilities need to
            get calculated.
        get_mean_func : callable | None
            The function to get the mean number of background events.
            The call signature of this function must be

                __call__(dataset, data, events)

            where ``dataset`` is an instance of Dataset and ``data`` is an
            instance of DatasetData of the data set for which background events
            needs to get generated. The `events` argument holds the actual set
            of events, for which the mean number of background events should get
            calculated. This argument can be `None`, which means that the mean
            number of background events to generate needs to get specified
            through the ``generate_events`` method. However, if a pre event
            selection method is provided, this argument cannot be ``None``!
        data_scrambler : instance of DataScrambler | None
            If set to an instance of DataScrambler, the monte-carlo events will
            get scrambled before drawing the background events. This can ensure
            more independent data trials. It is especially important when
            monte-carlo statistics are low.
        keep_mc_data_fields : str | list of str | None
            The MC data field names that should be kept in order to be able to
            calculate the background events rates by the functions
            ``get_event_prob_func`` and ``get_mean_func``. All other MC fields
            will get dropped due to computational efficiency reasons.
        pre_event_selection_method : instance of EventSelectionMethod | None
            If set to an instance of EventSelectionMethod, this method will
            pre-select the MC events that will be used for later background
            event generation. Using this pre-selection a large portion of the
            MC data can be reduced prior to background event generation.
        """
        super().__init__(
            get_event_prob_func=get_event_prob_func,
            get_mean_func=get_mean_func,
            data_scrambler=data_scrambler,
            keep_mc_data_fields=keep_mc_data_fields,
            pre_event_selection_method=pre_event_selection_method,
            **kwargs)

        self.bkg_component_rate_calc_func_dict = bkg_component_rate_calc_func_dict

    @property
    def bkg_component_rate_calc_func_dict(self):
        """The dictionary holding the background components (as key) and their
        rate calculation functions (as value).
        """
        return self._bkg_component_rate_calc_func_dict

    @bkg_component_rate_calc_func_dict.setter
    def bkg_component_rate_calc_func_dict(self, d):
        if not isinstance(d, dict):
            raise TypeError(
                'The bkg_component_rate_calc_func_dict property must be an '
                'instance of dict! '
                f'Its current type is {classname(d)}.')
        for (name, func) in d.items():
            if not isinstance(name, str):
                raise TypeError(
                    'The keys of the dictionary of the '
                    'bkg_component_rate_calc_func_dict property must be '
                    'instances of type str! At least one of the keys are of '
                    f'type {classname(name)}!')
            if not func_has_n_args(func, 3):
                raise TypeError(
                    'The function provided for the background component '
                    f'"{name}"  must have 3 arguments!')
        self._bkg_component_rate_calc_func_dict = d

    def generate_events(
            self,
            rss,
            dataset,
            data,
            mean=None,
            poisson=True,
            tl=None,
    ):
        """Generates a ``mean`` number of background events for the given
        monte-carlo dataset and its data.

        The procedure is as follows:

            1. Scramble all MC events if a data scrambler is set.
            2. Calculate the rate for each background component for each MC
               event of the entire MC.
            3. Calculate the mean number of bkg events for the entire MC.
            4. Pre-select MC events if a pre event selection method is set.
            5. Calculate the drawing probability of each selected MC event.
            6. Calculate the mean number of bkg events for the selected MC
               events.
            7. Draw the bkg events based on the calculated bkg event
               probability.
            8. Remove any MC specific data fields.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        dataset : instance of Dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data : instance of DatasetData
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.
        mean : float | None
            The mean number of background events to generate.
            Can be `None`. In that case the mean number of background events is
            obtained through the `get_mean_func` function.
        poisson : bool
            If set to ``True`` (default), the actual number of generated
            background events will be drawn from a Poisson distribution with the
            given mean number of background events.
            If set to ``False``, the argument ``mean`` specifies the actual
            number of generated background events.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        n_bkg : int
            The number of generated background events for the data set.
        bkg_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the generated
            background events. The number of events can be less than `n_bkg`
            if an event selection method is used.
        """
        # Create a copy of the MC data with all MC data fields removed,
        # except the specified MC data fields to keep for the
        # ``bkg_component_rate_calc_func``, ``get_mean_func`` and
        # ``get_event_prob_func`` functions.
        keep_field_names = list(set(
            DataFields.get_joint_names(
                datafields=self._cfg['datafields'],
                stages=(
                    DFS.ANALYSIS_EXP
                )
            ) +
            data.exp_field_names +
            self._keep_mc_data_field_names
        ))
        data_mc = data.mc.copy(keep_fields=keep_field_names)

        # Scramble the MC events if requested.
        if self._data_scrambler is not None:
            with TaskTimer(tl, 'Scramble MC events.'):
                data_mc = self._data_scrambler.scramble_data(
                    rss=rss,
                    dataset=dataset,
                    data=data_mc,
                    copy=False)

        # Calculate the rate for each background component.
        # Note: In Python3 the order of the dictionary entries is defined by
        #       the order they were added to the dictionary, hence the order is
        #       preserved.
        for (name, rate_calc_func) in self._bkg_component_rate_calc_func_dict.items():
            rate = rate_calc_func(dataset, data, data_mc)
            data_mc[name] = rate

        # Calculate the mean number of bkg events for the entire MC.
        mean_mc = None
        if self._get_mean_func is not None:
            with TaskTimer(
                    tl,
                    'Calculate total MC background mean.'):
                mean_mc = self._get_mean_func(
                    dataset=dataset,
                    data=data,
                    events=data_mc)

        # Pre-select MC events if a pre event selection method is set.
        if self._pre_event_selection_method is not None:
            with TaskTimer(
                    tl,
                    'Pre-select MC events.'):
                (data_mc, _) =\
                    self._pre_event_selection_method.select_events(
                        events=data_mc,
                        ret_original_evt_idxs=False,
                        tl=tl)

            with TaskTimer(tl, 'Calculate selected MC background mean.'):
                mean_mc_pre_selected = self._get_mean_func(
                    dataset=dataset,
                    data=data,
                    events=data_mc)

        # Calculate the drawing probability of each selected MC event.
        with TaskTimer(
                tl,
                'Calculate MC background event probability.'):
            mc_event_bkg_prob = self._get_event_prob_func(
                dataset=dataset,
                data=data,
                events=data_mc)

        with TaskTimer(
                tl,
                'Create RandomChoice for MC background events.'):
            random_choice = RandomChoice(
                items=data_mc.indices,
                probabilities=mc_event_bkg_prob)

        # Select the correct mean value.
        if mean is None:
            if mean_mc is None:
                raise ValueError(
                    'No mean number of background events and no '
                    'get_mean_func were specified! One of the two must be '
                    'specified!')
            mean = mean_mc
        else:
            mean = float_cast(
                mean,
                'The mean number of background events must be cast-able to '
                'type float!')

        # Draw the number of background events from a poisson distribution with
        # the given mean number of background events. This will be the number of
        # background events for this data set.
        n_bkg = (int(rss.random.poisson(mean)) if poisson else
                 int(np.round(mean, 0)))

        # Calculate the mean number of background events for the pre-selected
        # MC events.
        if self._pre_event_selection_method is None:
            # No selection at all, use the total mean.
            mean_pre_selected = mean
        else:
            mean_pre_selected = mean_mc_pre_selected

        # Calculate the actual number of background events for the selected
        # events.
        n_bkg_selected = int(np.around(n_bkg * mean_pre_selected / mean, 0))

        # Draw the actual background events from the selected events of the
        # monte-carlo data set.
        with TaskTimer(tl, 'Draw MC background indices.'):
            bkg_event_indices = random_choice(
                rss=rss,
                size=n_bkg_selected)

        with TaskTimer(tl, 'Select MC background events from indices.'):
            bkg_events = data_mc[bkg_event_indices]

        # Remove MC specific data fields from the background events record
        # array. So the result contains only experimental data fields. The list
        # of experimental data fields is defined as the unique set of the
        # required experimental data fields defined by the data set, and the
        # actual experimental data fields (in case there are additional kept
        # data fields by the user).
        with TaskTimer(tl, 'Remove MC specific data fields from MC events.'):
            exp_field_names = list(set(
                DataFields.get_joint_names(
                    datafields=self._cfg['datafields'],
                    stages=(
                        DFS.ANALYSIS_EXP
                    )
                ) +
                data.exp_field_names))
            bkg_events.tidy_up(exp_field_names)

        return (n_bkg, bkg_events)
