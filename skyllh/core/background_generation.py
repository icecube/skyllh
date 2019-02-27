# -*- coding: utf-8 -*-

import abc
import numpy as np

from skyllh.core.py import (
    float_cast,
    func_has_n_args
)
from skyllh.core.scrambling import DataScrambler


class BackgroundGenerationMethod(object):
    """This is the abstract base class for a detector specific background
    generation method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Constructs a new background generation method instance.
        """
        super(BackgroundGenerationMethod, self).__init__()

    @abc.abstractmethod
    def generate_events(self, rss, dataset, data, mean, **kwargs):
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
        **kwargs
            Additional keyword arguments, which might be required for a
            particular background generation method.

        Returns
        -------
        bkg_events : numpy record array
            The numpy record arrays holding the generated background events.
        """
        pass


class MCDataSamplingBkgGenMethod(BackgroundGenerationMethod):
    """This class implements the method to generate background events from
    monte-carlo (MC) data by sampling events from the MC data set according to a
    probability value given for each event. Functions can be provided to get the
    mean number of background events and the probability of each monte-carlo
    event.
    """
    def __init__(self, get_event_prob_func, get_mean_func=None,
                 unique_events=False, data_scrambler=None):
        """Creates a new instance of the MCDataSamplingBkgGenMethod class.

        Parameters
        ----------
        get_event_prob_func : callable
            The function to get the background probability of each monte-carlo
            event. The call signature of this function must be
                `__call__(dataset, data)`,
            where `dataset` and `data` are `Dataset` and `DatasetData` instances
            of the data set for which background events needs to get generated.
        get_mean_func : callable | None
            The function to get the mean number of background events.
            The call signature of this function must be
                `__call__(dataset, data)`,
            where `dataset` and `data` are `Dataset` and `DatasetData` instances
            of the data set for which background events needs to get generated.
            This argument can be `None`, which means that the mean number of
            background events to generate needs to get specified through the
            `generate_events` method.
        unique_events : bool
            Flag if unique events should be drawn from the monte-carlo (True),
            or if events can be drawn several times (False). Default is False.
        data_scrambler : instance of DataScrambler | None
            If set to an instance of DataScrambler, the drawn monte-carlo
            background events will get scrambled. This can ensure more
            independent data trials. It is especially important when monte-carlo
            statistics are low.
        """
        super(MCDataSamplingBkgGenMethod, self).__init__()

        self.get_event_prob_func = get_event_prob_func
        self.get_mean_func = get_mean_func
        self.unique_events = unique_events
        self.data_scrambler = data_scrambler

        # Define cache members to cache the background probabilities for each
        # monte-carlo event. The probabilities change only if the data changes.
        self._cache_data_id = None
        self._cache_mc_event_bkg_prob = None
        self._cache_mean = None

    @property
    def get_event_prob_func(self):
        """The function to obtain the background probability for each
        monte-carlo event of the data set.
        """
        return self._get_event_prob_func
    @get_event_prob_func.setter
    def get_event_prob_func(self, func):
        if(not callable(func)):
            raise TypeError('The get_event_prob_func property must be a '
                'callable!')
        if(not func_has_n_args(func, 2)):
            raise TypeError('The function provided for the get_event_prob_func '
                'property must have 2 arguments!')
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
        if(func is not None):
            if(not callable(func)):
                raise TypeError('The get_mean_func property must be a '
                    'callable!')
            if(not func_has_n_args(func, 2)):
                raise TypeError('The function provided for the get_mean_func '
                    'property must have 2 arguments!')
        self._get_mean_func = func

    @property
    def unique_events(self, ):
        """Flag if unique events should be drawn from the monto-carlo (True),
        or if the same event can be drawn multiple times from the monte-carlo.
        """
        return self._unique_events
    @unique_events.setter
    def unique_events(self, b):
        if(not isinstance(b, bool)):
            raise TypeError('The unique_events property must be of type bool!')
        self._unique_events = b

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
        if(scrambler is not None):
            if(not isinstance(scrambler, DataScrambler)):
                raise TypeError('The data_scrambler property must be an instance '
                    'of DataScrambler!')
        self._data_scrambler = scrambler

    def generate_events(self, rss, dataset, data, mean):
        """Generates background events  a `mean` number of background
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
        mean : float | None
            The mean number of background events to generate.
            Can be `None`. In that case the mean number of background events is
            obtained through the `get_mean_func` function.

        Returns
        -------
        bkg_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the generated
            background events.
        """
        data_id = id(data)
        if(self._cache_data_id != data_id):
            # Dataset has changed. We need to get new background probabilities
            # for each monte-carlo event and a new mean number of background
            # events.
            self._cache_data_id = data_id
            self._cache_mc_event_bkg_prob = self._get_event_prob_func(dataset, data)
            if(self._get_mean_func is not None):
                self._cache_mean = self._get_mean_func(dataset, data)

        if(mean is None):
            if(self._cache_mean is None):
                raise ValueError('No mean number of background events and no '
                    'get_mean_func were specified! One of the two must be '
                    'specified!')
            mean = self._cache_mean

        mean = float_cast(mean, 'The mean number of background events must be '
            'castable to type float!')

        # Draw the number of background events from a poisson distribution with
        # the given mean number of background events.
        n_bkg = rss.random.poisson(mean)

        # Draw the actual background events from the monto-carlo data set.
        bkg_event_indices = rss.random.choice(
            data.mc.indices, size=n_bkg, p=self._cache_mc_event_bkg_prob,
            replace=(not self._unique_events))
        bkg_events = data.mc[bkg_event_indices]

        # Remove MC specific data fields from the background events record
        # array. So the result contains only experimental data fields.
        bkg_events.tidy_up(data.exp_field_names)

        # Scramble the background events if requested.
        if(self._data_scrambler is not None):
            bkg_events = self._data_scrambler.scramble_data(rss, bkg_events)

        return bkg_events
