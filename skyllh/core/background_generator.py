# -*- coding: utf-8 -*-

import abc

from skyllh.core.background_generation import (
    BackgroundGenerationMethod,
)
from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.py import (
    classname,
    issequenceof,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.timing import (
    TaskTimer,
)


class BackgroundGenerator(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """This is the abstract base class for all background generator classes in
    SkyLLH. It defines the interface for a background generator.
    """
    def __init__(
            self,
            **kwargs,
    ):
        """Constructs a new instance of BackgroundGenerator.

        Parameters
        ----------
        bkg_gen_method : instance of BackgroundGenerationMethod
            The optional background event generation method, which should be
            used to generate events.
        """
        super().__init__(**kwargs)

    def change_shg_mgr(self, shg_mgr):
        """This method should be reimplemented when the background generator
        depends on the sources.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        pass

    @abc.abstractmethod
    def generate_background_events(
            self,
            rss,
            tl=None,
            **kwargs,
    ):
        """This method is supposed to generate a mean number of background
        events for the datasets of this background generator.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.
        **kwargs
            Additional keyword arguments, which will be passed to the
            ``generate_events`` method of the background generation method
            instance. Usual keyword arguments are `mean`, the mean number of
            background events to generate, and `poisson`, the flag if the number
            of background events should get drawn from a Poisson distribution
            with the given mean number of background events as mean.

        Returns
        -------
        n_bkg_list : list of int
            The number of generated background events.
        bkg_events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray holding the generated
            background events. The number of events can be less than stated in
            `n_bkg_list` if an event selection method is used.
        """
        pass


class DatasetBackgroundGenerator(
        BackgroundGenerator,
):
    """This class provides a background generator for a particular dataset.
    It holds a background generation method which is used to generate the
    background events.
    """

    def __init__(
            self,
            dataset,
            data,
            bkg_gen_method,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset = dataset
        self.data = data
        self.bkg_gen_method = bkg_gen_method

    @property
    def dataset(self):
        """The instance of Dataset for which background events should get
        generated.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, ds):
        if not isinstance(ds, Dataset):
            raise TypeError(
                'The dataset property must be an instance of Dataset! '
                f'Its current type is {classname(ds)}.')
        self._dataset = ds

    @property
    def data(self):
        """The instance of DatasetData holding the experimental and simulation
        data of the dataset.
        """
        return self._data

    @data.setter
    def data(self, d):
        if not isinstance(d, DatasetData):
            raise TypeError(
                'The data property must be an instance of DatasetData! '
                f'Its current type is {classname(d)}.')
        self._data = d

    @property
    def bkg_gen_method(self):
        """The instance of BackgroundGenerationMethod which should be used to
        generate background events. This can be ``None``.
        """
        return self._bkg_gen_method

    @bkg_gen_method.setter
    def bkg_gen_method(self, method):
        if method is not None:
            if not isinstance(method, BackgroundGenerationMethod):
                raise TypeError(
                    'The bkg_gen_method property must be an instance of '
                    'BackgroundGenerationMethod! '
                    f'Its current type is {classname(method)}.')
        self._bkg_gen_method = method

    def change_shg_mgr(self, shg_mgr):
        """Changes the SourceHypoGroupManager instance of the background
        generation method.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        self._bkg_gen_method.change_shg_mgr(
            shg_mgr=shg_mgr)

    def generate_background_events(
            self,
            rss,
            tl=None,
            **kwargs,
    ):
        """Generates a mean number of background events for the dataset of this
        background generator.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.
        **kwargs
            Additional keyword arguments, which will be passed to the
            ``generate_events`` method of the background generation method
            instance. Usual keyword arguments are `mean`, the mean number of
            background events to generate, and `poisson`, the flag if the number
            of background events should get drawn from a Poisson distribution
            with the given mean number of background events as mean.

        Returns
        -------
        n_bkg_list : list of int
            The list of length 1 holding the number of generated background
            events for the dataset.
        bkg_events_list : list of instance of DataFieldRecordArray
            The list of length 1 holding the instance of DataFieldRecordArray
            holding the generated background events. The number of events can be
            less than stated in ``n_bkg_list`` if an event selection method is
            used.
        """
        (n_bkg, bkg_events) = self._bkg_gen_method.generate_events(
            rss=rss,
            dataset=self._dataset,
            data=self._data,
            tl=tl,
            **kwargs)

        return ([n_bkg], [bkg_events])


class MultiDatasetBackgroundGenerator(
        BackgroundGenerator,
):
    """This is a background generator class handling multiple datasets by using
    the individual background generator instances for each dataset. This is the
    most general way to support multiple datasets of different formats and
    background generation.
    """

    def __init__(
            self,
            dataset_list,
            data_list,
            bkg_generator_list,
            **kwargs,
    ):
        """Constructs a new instance of MultiDatasetBackgroundGenerator.

        Parameters
        ----------
        dataset_list : list of instance of Dataset
            The list of Dataset instances for which background events should get
            generated for.
        data_list : list of instance of DatasetData
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        bkg_generator_list : list of instance of BackgroundGenerator
            The list of BackgroundGenerator instances, one for each dataset.
            The order must match the order of ``dataset_list``.
        """
        super().__init__(
            **kwargs)

        self.dataset_list = dataset_list
        self.data_list = data_list
        self.bkg_generator_list = bkg_generator_list

    @property
    def dataset_list(self):
        """The list of Dataset instances for which background events should get
        generated for.
        """
        return self._dataset_list

    @dataset_list.setter
    def dataset_list(self, datasets):
        if not issequenceof(datasets, Dataset):
            raise TypeError(
                'The dataset_list property must be a sequence of Dataset '
                'instances! '
                f'Its current type is {classname(datasets)}.')
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
                'instances! '
                f'Its current type is {classname(datas)}.')
        self._data_list = datas

    @property
    def bkg_generator_list(self):
        """The list of instance of BackgroundGenerator, one for each dataset.
        The order must match the order of the ``dataset_list`` property.
        """
        return self._bkg_generator_list

    @bkg_generator_list.setter
    def bkg_generator_list(self, generators):
        if not issequenceof(generators, BackgroundGenerator):
            raise TypeError(
                'The bkg_generator_list property must be a sequence of '
                'BackgroundGenerator instances! '
                f'Its current type is {classname(generators)}.')
        self._bkg_generator_list = generators

    def change_shg_mgr(
            self,
            shg_mgr,
    ):
        """Calls the ``change_shg_mgr`` method of each individual dataset
        background generator.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        for bkg_generator in self._bkg_generator_list:
            bkg_generator.change_shg_mgr(
                shg_mgr=shg_mgr)

    def generate_background_events(
            self,
            rss,
            mean_n_bkg_list=None,
            tl=None,
            **kwargs,
    ):
        """Generates a mean number of background events for each individual
        dataset of this multi-dataset background generator.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the individual
            background generator instance needs to obtain this number itself.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.
        **kwargs
            Additional keyword arguments, which will be passed to the
            ``generate_background_events`` method of the individual background
            generator instance.

        Returns
        -------
        n_bkg_events_list : list of int
            The list holding the number of generated background events for each
            dataset.
        bkg_events_list : list of instance of DataFieldRecordArray
            The list holding the instance of DataFieldRecordArray holding the
            generated background events of each dataset. The number of events
            can be less than stated in ``n_bkg_list`` if an event selection
            method is used.
        """
        if not isinstance(rss, RandomStateService):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService! '
                f'Its current type is {classname(rss)}.')

        if mean_n_bkg_list is None:
            mean_n_bkg_list = [None] * len(self._bkg_generator_list)
        if not issequenceof(mean_n_bkg_list, (type(None), float)):
            raise TypeError(
                'The mean_n_bkg_list argument must be a sequence of None '
                'and/or floats! '
                f'Its current type is {classname(mean_n_bkg_list)}.')

        if kwargs is None:
            kwargs = dict()

        n_bkg_events_list = []
        bkg_events_list = []
        for (ds, bkg_generator, mean_n_bkg) in zip(
                self._dataset_list, self._bkg_generator_list, mean_n_bkg_list):
            kwargs.update(mean=mean_n_bkg)
            with TaskTimer(
                    tl,
                    f'Generating background events for dataset "{ds.name}".'):
                (n_bkg_events_list_, bkg_events_list_) =\
                    bkg_generator.generate_background_events(
                        rss=rss,
                        tl=tl,
                        **kwargs)
            n_bkg_events_list += n_bkg_events_list_
            bkg_events_list += bkg_events_list_

        return (n_bkg_events_list, bkg_events_list)
