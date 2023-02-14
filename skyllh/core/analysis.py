# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc
import numpy as np


from skyllh.core.background_generation import (
    BackgroundGenerationMethod,
)
from skyllh.core.background_generator import (
    BackgroundGenerator,
    BackgroundGeneratorBase,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.llhratio import (
    LLHRatio,
    MultiDatasetTCLLHRatio,
    ZeroSigH0SingleDatasetTCLLHRatio,
)
from skyllh.core.model import (
    SourceModel,
)
from skyllh.core.multiproc import (
    get_ncpu,
    parallelize,
)
from skyllh.core.optimize import (
    EventSelectionMethod,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.pdfratio import (
    PDFRatio,
    SourceWeightedPDFRatio,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
    issequenceof,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.signal_generator import (
    SignalGeneratorBase,
    SignalGenerator,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.test_statistic import (
    TestStatistic,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)
from skyllh.core.weights import (
    DatasetSignalWeightFactorsService,
    SrcDetSigYieldWeightsService,
)


logger = get_logger(__name__)


class Analysis(
        object,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for all analysis classes.
    It contains common properties required by all analyses and defines the
    overall analysis interface how to setup and run an analysis.
    """
    def __init__(
            self,
            shg_mgr,
            pmm,
            test_statistic,
            bkg_gen_method=None,
            bkg_generator_cls=None,
            sig_generator_cls=None,
            **kwargs):
        """Constructor of the analysis base class.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            yield implementation method.
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance managing the global set of
            parameters and their relation to individual models, e.g. sources.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that should be used to
            generate background events for pseudo data. This can be set to None,
            if there is no need to generate background events.
        bkg_generator_cls : class of BackgroundGeneratorBase | None
            The background generator class used to create the background
            generator instance.
            If set to ``None``, the
            :class:`skyllh.core.background_generator.BackgroundGenerator` class
            is used.
        sig_generator_cls : class of SignalGeneratorBase | None
            The signal generator class used to create the signal generator
            instance.
            If set to ``None``, the
            :class:`skyllh.core.signal_generator.SignalGenerator` class
            is used.
        """
        super().__init__(**kwargs)

        self.shg_mgr = shg_mgr
        self.pmm = pmm
        self.test_statistic = test_statistic
        self.bkg_gen_method = bkg_gen_method
        self.bkg_generator_cls = bkg_generator_cls
        self.sig_generator_cls = sig_generator_cls

        self._dataset_list = []
        self._data_list = []
        self._tdm_list = []
        self._event_selection_method_list = []

        self._bkg_generator = None
        self._sig_generator = None

    @property
    def shg_mgr(self):
        """The SourceHypoGroupManager instance, which defines the groups of
        source hypothesis, their flux model, and their detector signal
        yield implementation method.
        """
        return self._shg_mgr

    @shg_mgr.setter
    def shg_mgr(self, mgr):
        if not isinstance(mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager! '
                f'Its current type is {classname(mgr)}.')
        self._shg_mgr = mgr

    @property
    def pmm(self):
        """The ParameterModelMapper instance that manages the global set of
        parameters and their relation to individual models, e.g. sources.
        """
        return self._pmm

    @pmm.setter
    def pmm(self, mapper):
        if not isinstance(mapper, ParameterModelMapper):
            raise TypeError(
                'The pmm property must be an instance of '
                'ParameterModelMapper! '
                f'Its current type is {classname(mapper)}.')
        self._pmm = mapper

    @property
    def test_statistic(self):
        """The TestStatistic instance that defines the test-statistic function
        of the analysis.
        """
        return self._test_statistic

    @test_statistic.setter
    def test_statistic(self, ts):
        if not isinstance(ts, TestStatistic):
            raise TypeError(
                'The test_statistic property must be an instance of '
                'TestStatistic! '
                f'Its current type is {classname(ts)}.')
        self._test_statistic = ts

    @property
    def bkg_gen_method(self):
        """The BackgroundGenerationMethod instance that implements the
        background event generation. This can be None if no background
        generation method has been defined.
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

    @property
    def bkg_generator_cls(self):
        """The background generator class that should be used to construct the
        background generator instance.
        """
        return self._bkg_generator_cls

    @bkg_generator_cls.setter
    def bkg_generator_cls(self, cls):
        if cls is None:
            cls = BackgroundGenerator
        if not issubclass(cls, BackgroundGeneratorBase):
            raise TypeError(
                'The bkg_generator_cls property must be a subclass of '
                'BackgroundGeneratorBase! '
                f'Its current type is {classname(cls)}.')
        self._bkg_generator_cls = cls

    @property
    def dataset_list(self):
        """The list of Dataset instances.
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
        """The list of DatasetData instances holding the original data of the
        dataset.
        """
        return self._data_list

    @data_list.setter
    def data_list(self, datas):
        if not issequenceof(datas, DatasetData):
            raise TypeError(
                'The data_list property must be a sequence of DatasetData '
                'instances! '
                f'Its current type is {classname(datas)}.')
        self._data_list = list(datas)

    @property
    def n_datasets(self):
        """(read-only) The number of datasets used in this analysis.
        """
        return len(self._dataset_list)

    @property
    def bkg_generator(self):
        """(read-only) The background generator instance. Is None of the
        background generator has not been constructed via the
        `construct_background_generator` method.
        """
        return self._bkg_generator

    @property
    def sig_generator_cls(self):
        """The signal generator class that should be used to construct the
        signal generator instance.
        """
        return self._sig_generator_cls

    @sig_generator_cls.setter
    def sig_generator_cls(self, cls):
        if cls is None:
            cls = SignalGenerator
        if not issubclass(cls, SignalGeneratorBase):
            raise TypeError(
                'The sig_generator_cls property must be a subclass of '
                'SignalGeneratorBase! '
                f'Its current type is {classname(cls)}.')
        self._sig_generator_cls = cls

    @property
    def sig_generator(self):
        """(read-only) The signal generator instance. Is None if the signal
        generator has not been constructed via the
        `construct_signal_generator` method.
        """
        return self._sig_generator

    @property
    def total_livetime(self):
        """(read-only) The total live-time in days of the loaded data.
        """
        livetime = 0
        for data in self._data_list:
            livetime += data.livetime
        return livetime

    def construct_detsigyield_arr(self, ppbar=None):
        """Creates a (N_datasets,N_source_hypo_groups)-shaped numpy ndarray of
        object holding the constructed DetSigYield instances of the analysis.

        Parameters
        ----------
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield_arr : instance of numpy ndarray
            The (N_datasets,N_source_hypo_groups)-shaped numpy ndarray of
            object holding the constructed DetSigYield instances of the
            analysis.
        """
        detsigyield_arr = np.empty(
            (len(self._dataset_list),
             self._shg_mgr.n_src_hypo_groups),
            dtype=object
        )

        pbar = ProgressBar(
            self._shg_mgr.n_src_hypo_groups * len(self._dataset_list),
            parent=ppbar).start()
        for (g, shg) in enumerate(self._shg_mgr.src_hypo_group_list):
            fluxmodel = shg.fluxmodel
            detsigyield_builder_list = shg.detsigyield_builder_list

            if (len(detsigyield_builder_list) != 1) and\
               (len(detsigyield_builder_list) != self.n_datasets):
                raise ValueError(
                    'The number of detector signal yield builders is not 1 and '
                    'does not match the number of used datasets in the '
                    'analysis!')

            for (j, (dataset, data)) in enumerate(zip(self.dataset_list,
                                                      self.data_list)):
                if len(detsigyield_builder_list) == 1:
                    # Only one detsigyield builder was defined,
                    # so we use it for all datasets.
                    detsigyield_builder = detsigyield_builder_list[0]
                else:
                    detsigyield_builder = detsigyield_builder_list[j]

                detsigyield = detsigyield_builder.construct_detsigyield(
                    dataset=dataset,
                    data=data,
                    fluxmodel=fluxmodel,
                    livetime=data.livetime,
                    ppbar=pbar)
                detsigyield_arr[j, g] = detsigyield

                pbar.increment()
        pbar.finish()

        return detsigyield_arr

    def add_dataset(
            self,
            dataset,
            data,
            tdm=None,
            event_selection_method=None):
        """Adds the given dataset to the list of datasets for this analysis.

        Parameters
        ----------
        dataset : instance of Dataset
            The Dataset instance that should get added.
        data : instance of DatasetData
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        tdm : instance of TrialDataManager | None
            The TrialDataManager instance managing the trial data and additional
            data fields of the data set. If set to None, it means that no
            additional data fields are defined.
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod to use to select only
            signal-like events from the data. All other events
            will be treated as pure background events. This reduces the amount
            of log-likelihood-ratio function evaluations. If set to None, all
            events will be evaluated.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'The dataset argument must be an instance of Dataset!')

        if not isinstance(data, DatasetData):
            raise TypeError(
                'The data argument must be an instance of DatasetData!')

        if tdm is None:
            tdm = TrialDataManager()
        if not isinstance(tdm, TrialDataManager):
            raise TypeError(
                'The tdm argument must be None or an instance of '
                'TrialDataManager!')

        if event_selection_method is not None:
            if not isinstance(event_selection_method, EventSelectionMethod):
                raise TypeError(
                    'The event_selection_method argument must be None or an '
                    'instance of EventSelectionMethod!')

        self._dataset_list.append(dataset)
        self._data_list.append(data)
        self._tdm_list.append(tdm)
        self._event_selection_method_list.append(event_selection_method)

    def calculate_test_statistic(
            self,
            log_lambda,
            fitparam_values,
            **kwargs):
        """Calculates the test statistic value by calling the ``evaluate``
        method of the TestStatistic class with the given log_lambda value and
        fit parameter values.

        Parameters
        ----------
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D ndarray holding the global
            fit parameter values of the log-likelihood ratio function for
            the given log_lambda value.
        **kwargs
            Any additional keyword arguments are passed to the
            ``__call__`` method of the TestStatistic instance.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        return self._test_statistic(
            pmm=self._pmm,
            log_lambda=log_lambda,
            fitparam_values=fitparam_values,
            **kwargs)

    def construct_background_generator(self):
        """Constructs the background generator for all added datasets.
        This method must be called after all the datasets were added via the
        add_dataset method. It sets the `bkg_generator` property of this
        Analysis class instance.
        """
        if self._bkg_gen_method is None:
            raise RuntimeError(
                'No background generation method has been '
                f'defined for this analysis ({classname(self)})!')

        self._bkg_generator = self.bkg_generator_cls(
            bkg_gen_method=self._bkg_gen_method,
            dataset_list=self._dataset_list,
            data_list=self._data_list)

    def construct_signal_generator(self):
        """Constructs the signal generator for all added datasets.
        This method must be called after all the datasets were added via the
        add_dataset method. It sets the `sig_generator` property of this
        Analysis class instance. The signal generation method has to be set
        through the source hypothesis group.
        """
        self._sig_generator = self.sig_generator_cls(
            shg_mgr=self._shg_mgr,
            dataset_list=self._dataset_list,
            data_list=self._data_list)

    @abc.abstractmethod
    def initialize_trial(
            self,
            events_list,
            n_events_list=None):
        """This method is supposed to initialize the log-likelihood ratio
        function with a new set of given trial data. This is a low-level method.
        For convenient methods see the `unblind` and `do_trial` methods.

        Parameters
        ----------
        events_list : list of numpy record ndarray
            The list of data events to use for the log-likelihood function
            evaluation. The data arrays for the datasets must be in the same
            order than the added datasets.
        n_events_list : list of int | None
            The list of the number of events of each data set. These numbers
            can be larger than the number of events given by the `events_list`
            argument in cases where an event selection method was already used.
            If set to None, the number of events is taken from the given
            `events_list` argument.
        """
        pass

    @abc.abstractmethod
    def unblind(
            self,
            rss,
            tl=None):
        """This method is supposed to run the analysis on the experimental data,
        i.e. unblinds the data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used draw random
            numbers from. It can be used to generate random initial values for
            fit parameters.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time the
            maximization of the LLH ratio function.

        Returns
        -------
        TS : float
            The test-statistic value.
        global_params_dict : dict
            The dictionary holding the global parameter names and their
            best fit values. It includes fixed and floating parameters.
        status : dict
            The status dictionary with information about the performed
            minimization process of the analysis.
        """
        pass

    @abc.abstractmethod
    def do_trial_with_given_pseudo_data(
            self,
            rss,
            mean_n_sig,
            n_sig,
            n_events_list,
            events_list,
            minimizer_status_dict=None,
            tl=None,
            **kwargs):
        """This method is supposed to perform an analysis trial on a given
        pseudo data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for generating random
            numbers.
        mean_n_sig : float
            The mean number of signal events the pseudo data was generated with.
        n_sig : int
            The total number of actual signal events in the pseudo data.
        n_events_list : list of int
            The total number of events for each data set of the pseudo data.
        events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray containing the pseudo
            data events for each data sample. The number of events for each
            data sample can be less than the number of events given by
            ``n_events_list`` if an event selection method was already utilized
            when generating background events.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks.

        Returns
        -------
        recarray : instance of numpy record ndarray
            The numpy record ndarray holding the result of the trial. It must
            contain the following data fields:

            rss_seed : int
                The RandomStateService seed.
            mean_n_sig : float
                The mean number of signal events.
            n_sig : int
                The actual number of injected signal events.
            ts : float
                The test-statistic value.
            [<global_param_name> : float ]
                Any additional parameters of the analysis.
        """
        pass

    def change_shg_mgr(
            self,
            shg_mgr):
        """If the SourceHypoGroupManager instance changed, this method needs to
        be called to propagate the change to all components of the analysis.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        # Change the source hypo group manager of the EventSelectionMethod
        # instance.
        for evt_selection_method in self._event_selection_method_list:
            if evt_selection_method is not None:
                evt_selection_method.change_shg_mgr(
                    shg_mgr=shg_mgr)

        # Change the source hypo group manager of TrialDataManager for all
        # data sets.
        for tdm in self._tdm_list:
            tdm.change_shg_mgr(
                shg_mgr=shg_mgr)

        # Change the source hypo group manager of the background generator
        # instance.
        if self._bkg_generator is not None:
            self._bkg_generator.change_shg_mgr(
                shg_mgr=shg_mgr)

        # Change the source hypo group manager of the signal generator instance.
        if self._sig_generator is not None:
            self._sig_generator.change_shg_mgr(
                shg_mgr=shg_mgr)

    def do_trial_with_given_bkg_and_sig_pseudo_data(
            self,
            rss,
            mean_n_sig,
            n_sig,
            n_bkg_events_list,
            n_sig_events_list,
            bkg_events_list,
            sig_events_list,
            minimizer_status_dict=None,
            tl=None,
            **kwargs):
        """Performs an analysis trial on the given background and signal pseudo
        data. This method merges the background and signal pseudo events and
        calls the ``do_trial_with_given_pseudo_data`` method of this class.

        Note
        ----
        This method alters the DataFieldRecordArray instances of the
        bkg_events_list argument!

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService instance to use for generating
            random numbers.
        mean_n_sig : float
            The mean number of signal events the pseudo data was generated with.
        n_sig : int
            The total number of actual signal events in the pseudo data.
        n_bkg_events_list : list of int
            The total number of background events for each data set of the
            pseudo data.
        n_sig_events_list : list of int
            The total number of signal events for each data set of the
            pseudo data.
        bkg_events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray containing the
            background pseudo data events for each data set.
        sig_events_list : list of instance of DataFieldRecordArray | None
            The list of instance of DataFieldRecordArray containing the signal
            pseudo data events for each data set. If a particular dataset has
            no signal events, the entry for that dataset can be ``None``.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.
        **kwargs : dict
            Additional keyword arguments are passed to the
            :meth:`~skyllh.core.analysis.Analysis.do_trial_with_given_pseudo_data`
            method.

        Returns
        -------
        recarray : instance of numpy record ndarray
            The numpy record ndarray holding the result of the trial.
            See the documentation of the
            :meth:`~skyllh.core.analysis.Analysis.do_trial_with_given_pseudo_data`
            method for further information.
        """
        n_events_list = list(
            np.array(n_bkg_events_list) +
            np.array(n_sig_events_list)
        )

        events_list = bkg_events_list

        # Add potential signal events to the background events.
        for ds_idx in range(len(events_list)):
            if sig_events_list[ds_idx] is not None:
                if events_list[ds_idx] is None:
                    events_list[ds_idx] = sig_events_list[ds_idx]
                else:
                    events_list[ds_idx].append(sig_events_list[ds_idx])

        recarray = self.do_trial_with_given_pseudo_data(
            rss=rss,
            mean_n_sig=mean_n_sig,
            n_sig=n_sig,
            n_events_list=n_events_list,
            events_list=events_list,
            minimizer_status_dict=minimizer_status_dict,
            tl=tl,
            **kwargs)

        return recarray

    def generate_background_events(
            self,
            rss,
            mean_n_bkg_list=None,
            bkg_kwargs=None,
            tl=None):
        """Generates background events utilizing the background generator.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for generating random
            numbers.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the background
            generation method needs to obtain this number itself.
        bkg_kwargs : dict | None
            Optional keyword arguments for the ``generate_background_events``
            method of the background generator.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks of this method.

        Returns
        -------
        n_events_list : list of int
            The list of the number of events that have been generated for each
            pseudo data set.
        events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray containing the pseudo
            data events for each data sample. The number of events for each
            data set can be less than the number of events given by
            ``n_events_list`` if an event selection method was already utilized
            when generating background events.
        """
        n_datasets = self.n_datasets

        if not isinstance(rss, RandomStateService):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService! '
                f'Its current type is {classname(rss)}.')

        if mean_n_bkg_list is None:
            mean_n_bkg_list = [None] * n_datasets
        if not issequenceof(mean_n_bkg_list, (type(None), float)):
            raise TypeError(
                'The mean_n_bkg_list argument must be a sequence of None '
                'and/or floats! '
                f'Its current type is {classname(mean_n_bkg_list)}.')

        if bkg_kwargs is None:
            bkg_kwargs = dict()

        # Construct the background event generator in case it's not constructed
        # yet.
        if self._bkg_generator is None:
            self.construct_background_generator()

        n_events_list = []
        events_list = []
        for ds_idx in range(n_datasets):
            bkg_kwargs.update(mean=mean_n_bkg_list[ds_idx])
            with TaskTimer(
                    tl,
                    f'Generating background events for data set {ds_idx}.'):
                (n_bkg, bkg_events) =\
                    self._bkg_generator.generate_background_events(
                        rss=rss,
                        dataset_idx=ds_idx,
                        tl=tl,
                        **bkg_kwargs)
            n_events_list.append(n_bkg)
            events_list.append(bkg_events)

        return (n_events_list, events_list)

    def _assert_input_arguments_of_generate_signal_events(
            self,
            rss,
            n_events_list,
            events_list):
        """Checks the input arguments of the ``generate_signal_events`` method
        for correct type and value.
        """
        n_datasets = self.n_datasets

        if not isinstance(rss, RandomStateService):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService! '
                f'Its current type is {classname(rss)}.')

        if not issequenceof(n_events_list, int):
            raise TypeError(
                'The n_events_list argument must be a sequence of '
                'instances of type int! '
                f'Its current type is {classname(n_events_list)}.')
        if len(n_events_list) != n_datasets:
            raise ValueError(
                'The n_events_list argument must be a list of int of '
                f'length {n_datasets}! Currently it is of length '
                f'{len(n_events_list)}.')

        if not issequenceof(events_list, (type(None), DataFieldRecordArray)):
            raise TypeError(
                'The events_list argument must be a sequence of '
                'instances of type DataFieldRecordArray! '
                f'Its current type is {classname(events_list)}.')
        if len(events_list) != n_datasets:
            raise ValueError(
                'The events_list argument must be a list of instances of '
                f'type DataFieldRecordArray with a length of {n_datasets}! '
                f'Currently it is of length {len(events_list)}.')

    def generate_signal_events(
            self,
            rss,
            mean_n_sig,
            sig_kwargs=None,
            n_events_list=None,
            events_list=None,
            tl=None):
        """Generates signal events utilizing the signal generator.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for generating random
            numbers.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        sig_kwargs : dict | None
            Additional keyword arguments for the ``generate_signal_events``
            method of the ``sig_generator_cls`` class. An usual keyword argument
            is ``poisson``.
        n_events_list : list of int | None
            If given, it specifies the number of events of each data set already
            present and the number of signal events will be added.
        events_list : list of instance of DataFieldRecordArray | None
            If given, it specifies the events of each data set already present
            and the signal events will be added.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks of this method.

        Returns
        -------
        n_sig : int
            The actual number of injected signal events.
        n_events_list : list of int
            The list of the number of signal events that have been generated for
            each data set.
        events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray containing the
            signal data events for each data set. An entry is None, if no signal
            events were generated for this particular data set.
        """
        if sig_kwargs is None:
            sig_kwargs = dict()

        if n_events_list is None:
            n_events_list = [0] * self.n_datasets

        if events_list is None:
            events_list = [None] * self.n_datasets

        self._assert_input_arguments_of_generate_signal_events(
            rss=rss,
            n_events_list=n_events_list,
            events_list=events_list)

        n_sig = 0

        if mean_n_sig == 0:
            return (n_sig, n_events_list, events_list)

        # Construct the signal generator if not done yet.
        if self._sig_generator is None:
            with TaskTimer(tl, 'Constructing signal generator.'):
                self.construct_signal_generator()

        # Generate signal events with the given mean number of signal
        # events.
        sig_kwargs.update(mean=mean_n_sig)
        with TaskTimer(tl, 'Generating signal events.'):
            (n_sig, ds_sig_events_dict) =\
                self._sig_generator.generate_signal_events(
                    rss=rss,
                    **sig_kwargs)

        # Inject the signal events to the generated background data.
        for (ds_idx, sig_events) in ds_sig_events_dict.items():
            n_events_list[ds_idx] += len(sig_events)
            if events_list[ds_idx] is None:
                events_list[ds_idx] = sig_events
            else:
                events_list[ds_idx].append(sig_events)

        return (n_sig, n_events_list, events_list)

    def generate_pseudo_data(
            self,
            rss,
            mean_n_bkg_list=None,
            mean_n_sig=0,
            bkg_kwargs=None,
            sig_kwargs=None,
            tl=None):
        """Generates pseudo data with background and possible signal
        events for each data set using the background and signal generation
        methods of the analysis.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for generating random
            numbers.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the background
            generation method needs to obtain this number itself.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        bkg_kwargs : dict | None
            Additional keyword arguments for the `generate_events` method of the
            background generation method class. An usual keyword argument is
            `poisson`.
        sig_kwargs : dict | None
            Additional keyword arguments for the `generate_signal_events` method
            of the `SignalGenerator` class. An usual keyword argument is
            `poisson`.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks of this method.

        Returns
        -------
        n_sig : int
            The actual number of injected signal events.
        n_events_list : list of int
            The list of the number of events that have been generated for each
            pseudo data set.
        events_list : list of instance of DataFieldRecordArray
            The list of DataFieldRecordArray instances containing the pseudo
            data events for each data sample. The number of events for each
            data set can be less than the number of events given by
            `n_events_list` if an event selection method was already utilized
            when generating background events.
        """
        # Generate background events for each dataset.
        (n_events_list, events_list) = self.generate_background_events(
            rss=rss,
            mean_n_bkg_list=mean_n_bkg_list,
            bkg_kwargs=bkg_kwargs,
            tl=tl)

        # Generate signal events and add them to the already generated
        # background events.
        (n_sig, n_events_list, events_list) = self.generate_signal_events(
            rss=rss,
            mean_n_sig=mean_n_sig,
            sig_kwargs=sig_kwargs,
            n_events_list=n_events_list,
            events_list=events_list,
            tl=tl)

        return (n_sig, n_events_list, events_list)

    def do_trial(
            self,
            rss,
            mean_n_bkg_list=None,
            mean_n_sig=0,
            bkg_kwargs=None,
            sig_kwargs=None,
            minimizer_status_dict=None,
            tl=None,
            **kwargs):
        """This method performs an analysis trial by generating a
        pseudo data sample with background events and possible signal events
        via the :meth:`generate_pseudo_data` method, and performs the analysis
        on that random pseudo data sample by calling the
        :meth:`do_trial_with_given_pseudo_data` method.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService instance to use for generating
            random numbers.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the background
            generation method needs to obtain this number itself.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial.
        bkg_kwargs : dict | None
            Additional keyword arguments for the `generate_events` method of the
            background generation method class. An usual keyword argument is
            `poisson`.
        sig_kwargs : dict | None
            Additional keyword arguments for the `generate_signal_events` method
            of the `SignalGenerator` class. An usual keyword argument is
            `poisson`.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks.
        **kwargs : dict
            Additional keyword arguments are passed to the
            :meth:`do_trial_with_given_pseudo_data` method.

        Returns
        -------
        recarray : instance of numpy record ndarray
            The numpy record ndarray holding the result of the trial.
            See the documentation of the
            :py:meth:`~skyllh.core.analysis.Analysis.do_trial_with_given_pseudo_data`
            method for further information.
        """
        with TaskTimer(tl, 'Generating pseudo data.'):
            (n_sig, n_events_list, events_list) = self.generate_pseudo_data(
                rss=rss,
                mean_n_bkg_list=mean_n_bkg_list,
                mean_n_sig=mean_n_sig,
                bkg_kwargs=bkg_kwargs,
                sig_kwargs=sig_kwargs,
                tl=tl)

        recarray = self.do_trial_with_given_pseudo_data(
            rss=rss,
            mean_n_sig=mean_n_sig,
            n_sig=n_sig,
            n_events_list=n_events_list,
            events_list=events_list,
            minimizer_status_dict=minimizer_status_dict,
            tl=tl,
            **kwargs)

        return recarray

    def do_trials(
            self,
            rss,
            n,
            ncpu=None,
            tl=None,
            ppbar=None,
            **kwargs):
        """Executes the :meth:`do_trial` method ``n`` times with possible
        multi-processing.

        Parameters
        ----------
        rss : instance of RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        n : int
            Number of trials to generate using the `do_trial` method.
        ncpu : int | None
            The number of CPUs to use, i.e. the number of subprocesses to
            spawn. If set to None, the global setting will be used.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks.
        ppbar : instance of ProgressBar | None
            The possible parent ProgressBar instance.
        **kwargs
            Additional keyword arguments are passed to the :meth:`do_trial`
            method. See the documentation of that method for allowed keyword
            arguments.

        Returns
        -------
        recarray : numpy record ndarray
            The numpy record ndarray holding the result of all trials.
            See the documentation of the
            :py:meth:`~skyllh.core.analysis.Analysis.do_trial` method for the
            list of data fields.
        """
        ncpu = get_ncpu(ncpu)

        args_list = [((), kwargs) for i in range(n)]
        result_list = parallelize(
            func=self.do_trial,
            args_list=args_list,
            ncpu=ncpu,
            rss=rss,
            tl=tl,
            ppbar=ppbar)

        recarray_dtype = result_list[0].dtype
        recarray = np.empty(n, dtype=recarray_dtype)
        recarray[:] = result_list[:]

        return recarray


class LLHRatioAnalysis(
        Analysis,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for all log-likelihood ratio analysis
    classes. It requires a mathematical log-likelihood ratio function.

    To set-up and run an analysis the following procedure applies:

        1. Create an Analysis instance.
        2. Add the datasets and their PDF ratio instances via the
           :meth:`skyllh.core.analysis.Analysis.add_dataset` method.
        3. Construct the log-likelihood ratio function via the
           :meth:`construct_llhratio` method.
        4. Initialize a trial via the :meth:`initialize_trial` method.
        5. Fit the global fit parameters to the trial data via the
           :meth:`maximize` method of the ``llhratio`` property.

    Alternatively, one can use the convenient methods :meth:`do_trial` or
    :meth:`unblind` to perform a random trial or to unblind the data,
    respectively. Both methods will fit the global fit parameters using the set
    up data. Finally, the test statistic is calculated via the
    :meth:`calculate_test_statistic` method.

    In order to calculate sensitivities and discovery potentials, analysis
    trials have to be performed on random data samples with injected signal
    events. To perform a trial with injected signal events, the signal generator
    has to be constructed via the :meth:`construct_signal_generator` method
    before any random trial data can be generated.
    """

    def __init__(
            self,
            shg_mgr,
            pmm,
            test_statistic,
            bkg_gen_method=None,
            bkg_generator_cls=None,
            sig_generator_cls=None,
            **kwargs):
        """Constructs a new instance of LLHRatioAnalysis.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            yield implementation method.
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance managing the global set of
            parameters and their relation to individual models, e.g. sources.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that should be used to
            generate background events for pseudo data. This can be set to None,
            if there is no need to generate background events.
        bkg_generator_cls : class of BackgroundGeneratorBase | None
            The background generator class used to create the background
            generator instance.
            If set to ``None``, the
            :class:`skyllh.core.background_generator.BackgroundGenerator` class
            is used.
        sig_generator_cls : class of SignalGeneratorBase | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            pmm=pmm,
            test_statistic=test_statistic,
            bkg_gen_method=bkg_gen_method,
            bkg_generator_cls=bkg_generator_cls,
            sig_generator_cls=sig_generator_cls,
            **kwargs)

        # Define the member variable for the list of PDFRatio instances, one for
        # each dataset.
        self._pdfratio_list = []

        self._llhratio = None

    @property
    def llhratio(self):
        """The log-likelihood ratio function instance. It is None, if it has
        not been constructed yet.
        """
        if self._llhratio is None:
            raise RuntimeError(
                'The log-likelihood ratio function is not defined yet. '
                'Call the "construct_llhratio" method first!')
        return self._llhratio

    @llhratio.setter
    def llhratio(self, obj):
        if not isinstance(obj, LLHRatio):
            raise TypeError(
                'The llhratio property must be an instance of LLHRatio! '
                f'Its current type is {classname(obj)}.')
        self._llhratio = obj

    @abc.abstractmethod
    def construct_llhratio(
            self,
            minimizer,
            ppbar=None):
        """This method is supposed to construct the LLH ratio function.

        Returns
        -------
        llhratio : instance of LLHRatio
            The instance of LLHRatio that implements the
            log-likelihood-ratio function of this LLH ratio analysis.
        """
        pass

    def add_dataset(
            self,
            dataset,
            data,
            pdfratio,
            tdm=None,
            event_selection_method=None):
        """Adds a dataset with its PDF ratio instances to the analysis.

        Parameters
        ----------
        dataset : instance of Dataset
            The instance of Dataset that should get added.
        data : instance of DatasetData
            The instance of DatasetData holding the original (prepared) data of
            the dataset.
        pdfratio : instance of PDFRatio
            The instance of PDFRatio for the to-be-added data set.
        tdm : instance of TrialDataManager | None
            The TrialDataManager instance that manages the trial data and
            additional data fields for this data set.
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod to use to select only
            signal-like events from the trial data. All other events
            will be treated as pure background events. This reduces the amount
            of log-likelihood-ratio function evaluations. If set to None, all
            events will be evaluated.
        """
        super().add_dataset(
            dataset=dataset,
            data=data,
            tdm=tdm,
            event_selection_method=event_selection_method)

        if not isinstance(pdfratio, PDFRatio):
            raise TypeError(
                'The pdfratio argument must be an instance of PDFRatio! '
                f'Its current type is {classname(pdfratio)}')

        self._pdfratio_list.append(pdfratio)

    def change_shg_mgr(
            self,
            shg_mgr):
        """If the SourceHypoGroupManager instance changed, this method needs to
        be called to propagate the change to all components of the analysis.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        if self._llhratio is None:
            raise RuntimeError(
                'The LLH ratio function has to be constructed '
                'before the `change_shg_mgr` method can be called!')

        super().change_shg_mgr(
            shg_mgr=shg_mgr)

        # Change the source hypo group manager of the LLH ratio function
        # instance.
        self._llhratio.change_shg_mgr(
            shg_mgr=shg_mgr)

    def initialize_trial(
            self,
            events_list,
            n_events_list=None,
            tl=None):
        """This method initializes the log-likelihood ratio
        function with a new set of given trial data. This is a low-level method.
        For convenient methods see the ``unblind`` and ``do_trial`` methods.

        Parameters
        ----------
        events_list : list of DataFieldRecordArray instances
            The list of DataFieldRecordArray instances holding the data events
            to use for the log-likelihood function evaluation. The data arrays
            for the datasets must be in the same order than the added datasets.
        n_events_list : list of int | None
            The list of the number of events of each data set. If set to None,
            the number of events is taken from the size of the given events
            arrays.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used for timing
            measurements.
        """
        if n_events_list is None:
            n_events_list = [None] * len(events_list)

        # If there is more than a single source, we need to store the source
        # and event indices within the TrialDataManager instance.
        store_src_evt_idxs = self.shg_mgr.n_sources > 1

        for (tdm, events, n_events, evt_sel_method) in zip(
                self._tdm_list, events_list, n_events_list,
                self._event_selection_method_list):

            # Initialize the trial data manager with the given raw events.
            tdm.initialize_trial(
                shg_mgr=self._shg_mgr,
                pmm=self._pmm,
                events=events,
                n_events=n_events,
                evt_sel_method=evt_sel_method,
                store_src_evt_idxs=store_src_evt_idxs,
                tl=tl)

        self._llhratio.initialize_for_new_trial(
            tl=tl)

    def unblind(
            self,
            rss,
            tl=None):
        """Evaluates the unscrambled data, i.e. unblinds the data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used draw random
            numbers from.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time the
            maximization of the LLH ratio function.

        Returns
        -------
        TS : float
            The test-statistic value.
        global_params_dict : dict
            The dictionary holding the global parameter names and their
            best fit values. It includes fixed and floating parameters.
        status : dict
            The status dictionary with information about the performed
            minimization process of the negative of the log-likelihood ratio
            function.
        """
        events_list = [data.exp for data in self._data_list]
        self.initialize_trial(events_list)

        (log_lambda, fitparam_values, status) = self._llhratio.maximize(
            rss=rss,
            tl=tl)

        TS = self.calculate_test_statistic(
            log_lambda=log_lambda,
            fitparam_values=fitparam_values)

        global_params_dict = self._pmm.create_global_params_dict(
            gflp_values=fitparam_values)

        return (TS, global_params_dict, status)

    def do_trial_with_given_pseudo_data(
            self,
            rss,
            mean_n_sig,
            n_sig,
            n_events_list,
            events_list,
            minimizer_status_dict=None,
            tl=None,
            mean_n_sig_0=None):
        """Performs an analysis trial on the given pseudo data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService to use for generating random
            numbers.
        mean_n_sig : float
            The mean number of signal events the pseudo data was generated with.
        n_sig : int
            The total number of actual signal events in the pseudo data.
        n_events_list : list of int
            The total number of events for each data set of the pseudo data.
        events_list : list of instance of DataFieldRecordArray
            The list of instance of DataFieldRecordArray containing the pseudo
            data events for each data sample. The number of events for each
            data sample can be less than the number of events given by
            ``n_events_list`` if an event selection method was already utilized
            when generating background events.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to time
            individual tasks.
        mean_n_sig_0 : float | None
            The fixed mean number of signal events for the null-hypothesis,
            when using a ns-profile log-likelihood-ratio function.
            If set to None, this argument is interpreted as 0.

        Returns
        -------
        recarray : instance of numpy record ndarray
            The numpy record ndarray holding the result of the trial. It
            contains the following data fields:

            rss_seed : int
                The RandomStateService seed.
            mean_n_sig : float
                The mean number of signal events.
            n_sig : int
                The actual number of injected signal events.
            mean_n_sig_0 : float
                The fixed mean number of signal events for the null-hypothesis.
            ts : float
                The test-statistic value.
            [<global_param_name> : float ]
                Any additional parameters of the LLH ratio function.
        """
        if mean_n_sig_0 is None:
            mean_n_sig_0 = 0

        self._llhratio.mean_n_sig_0 = mean_n_sig_0

        with TaskTimer(tl, 'Initializing trial.'):
            self.initialize_trial(events_list, n_events_list)

        with TaskTimer(tl, 'Maximizing LLH ratio function.'):
            (log_lambda, fitparam_values, status) = self._llhratio.maximize(
                rss=rss,
                tl=tl)
        if isinstance(minimizer_status_dict, dict):
            minimizer_status_dict.update(status)

        with TaskTimer(tl, 'Calculating test statistic.'):
            ts = self.calculate_test_statistic(
                log_lambda=log_lambda,
                fitparam_values=fitparam_values)

        # Get the dictionary holding all floating and fixed parameter names
        # and values.
        global_params_dict = self._pmm.create_global_params_dict(
            gflp_values=fitparam_values)

        # Create the structured array data type for the result array.
        recarray_dtype = [
            ('seed', np.int64),
            ('mean_n_sig', np.float64),
            ('n_sig', np.int64),
            ('mean_n_sig_0', np.float64),
            ('ts', np.float64)
        ] + [
            (param_name, np.float64)
            for param_name in global_params_dict.keys()
        ]
        recarray = np.empty((1,), dtype=recarray_dtype)
        recarray['seed'] = rss.seed
        recarray['mean_n_sig'] = mean_n_sig
        recarray['n_sig'] = n_sig
        recarray['mean_n_sig_0'] = mean_n_sig_0
        recarray['ts'] = ts
        for (param_name, param_value) in global_params_dict.items():
            recarray[param_name] = param_value

        return recarray


class MultiSourceTimeIntegratedMultiDatasetLLHRatioAnalysis(
        LLHRatioAnalysis):
    """This is the abstract base class for all log-likelihood ratio analysis
    classes that use multiple dataset, i.e. multiple LLH-ratio functions, one
    for each dataset and multiple sources.

    For more information how to construct and run the analysis see the
    documentation of the :class:`~skyllh.core.analysis.LLHRatioAnalysis` class.
    """
    def __init__(
            self,
            shg_mgr,
            pmm,
            test_statistic,
            bkg_gen_method=None,
            bkg_generator_cls=None,
            sig_generator_cls=None,
            **kwargs):
        """Constructs a new instance of MultiDatasetLLHRatioAnalysis.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            yield implementation method.
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance managing the global set of
            parameters and their relation to individual models, e.g. sources.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that should be used to
            generate background events for pseudo data. This can be set to None,
            if there is no need to generate background events.
        bkg_generator_cls : class of BackgroundGeneratorBase | None
            The background generator class used to create the background
            generator instance.
            If set to ``None``, the
            :class:`skyllh.core.background_generator.BackgroundGenerator` class
            is used.
        sig_generator_cls : class of SignalGeneratorBase | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            pmm=pmm,
            test_statistic=test_statistic,
            bkg_gen_method=bkg_gen_method,
            bkg_generator_cls=bkg_generator_cls,
            sig_generator_cls=sig_generator_cls,
            **kwargs)

    def construct_llhratio(
            self,
            minimizer,
            ppbar=None):
        """Constructs the log-likelihood (LLH) ratio function of the analysis.
        This setups all the necessary analysis objects like detector signal
        yields and dataset signal weights, constructs the log-likelihood ratio
        functions for each dataset and the final composite LLH ratio function.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The instance of Minimizer that should be used to minimize the
            negative of the log-likelihood ratio function.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        llhratio : instance of MultiDatasetTCLLHRatio
            The instance of MultiDatasetTCLLHRatio that implements the
            log-likelihood-ratio function of the analysis.
        """
        # Create the detector signal yield instances for each dataset and
        # source hypothesis group.
        detsigyield_arr = self.construct_detsigyield_arr(ppbar=ppbar)

        src_detsigyield_weights_service = SrcDetSigYieldWeightsService(
            shg_mgr=self._shg_mgr,
            detsigyields=detsigyield_arr)

        ds_sig_weight_factors_service = DatasetSignalWeightFactorsService(
            src_detsigyield_weights_service=src_detsigyield_weights_service)

        # Create the list of log-likelihood ratio functions, one for each
        # dataset.
        llhratio_list = [
            ZeroSigH0SingleDatasetTCLLHRatio(
                pmm=self._pmm,
                minimizer=minimizer,
                shg_mgr=self._shg_mgr,
                tdm=tdm,
                pdfratio=SourceWeightedPDFRatio(
                    dataset_idx=dataset_idx,
                    src_detsigyield_weights_service=src_detsigyield_weights_service,
                    pdfratio=pdfratio)
            )
            for (dataset_idx, (tdm, pdfratio)) in enumerate(
                zip(self._tdm_list,
                    self._pdfratio_list))
        ]

        # Create the final multi-dataset log-likelihood ratio function.
        llhratio = MultiDatasetTCLLHRatio(
            pmm=self._pmm,
            minimizer=minimizer,
            src_detsigyield_weights_service=src_detsigyield_weights_service,
            ds_sig_weight_factors_service=ds_sig_weight_factors_service,
            llhratio_list=llhratio_list)

        return llhratio


class SingleSourceTimeIntegratedMultiDatasetLLHRatioAnalysis(
        MultiSourceTimeIntegratedMultiDatasetLLHRatioAnalysis):
    """This is an analysis class that implements a time-integrated
    log-likelihood ratio analysis for multiple datasets assuming a single
    source. It is a special case of the multi-source analysis.

    For more information how to construct and run the analysis see the
    documentation of the :class:`~skyllh.core.analysis.LLHRatioAnalysis` class.
    """
    def __init__(
            self,
            shg_mgr,
            pmm,
            test_statistic,
            bkg_gen_method=None,
            bkg_generator_cls=None,
            sig_generator_cls=None,
            **kwargs):
        """Creates a new time-integrated point-like source analysis assuming a
        single source.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            efficiency implementation method.
        pmm : instance of ParameterModelMapper
            The ParameterModelMapper instance managing the global set of
            parameters and their relation to individual models, e.g. sources.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that will be used to
            generate background events for a new analysis trial. This can be set
            to None, if no background events have to get generated.
        bkg_generator_cls : class of BackgroundGeneratorBase | None
            The background generator class used to create the background
            generator instance.
            If set to ``None``, the
            :class:`skyllh.core.background_generator.BackgroundGenerator` class
            is used.
        sig_generator_cls : SignalGeneratorBase class | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            pmm=pmm,
            test_statistic=test_statistic,
            bkg_gen_method=bkg_gen_method,
            bkg_generator_cls=bkg_generator_cls,
            sig_generator_cls=sig_generator_cls,
            **kwargs)

    def change_source(
            self,
            source):
        """Changes the source of the analysis to the given source. It makes the
        necessary changes to all the objects of the analysis.

        Parameters
        ----------
        source : instance of SourceModel
            The instance of SourceModel describing the new source.
        """
        if not isinstance(source, SourceModel):
            raise TypeError(
                'The source argument must be an instance of SourceModel! '
                f'Its current type is {classname(source)}.')

        # Change the source in the SourceHypoGroupManager instance.
        # Because this is a single source analysis, there can only be one source
        # hypothesis group defined.
        self._shg_mgr.src_hypo_group_list[0].source_list[0] = source

        self.change_shg_mgr(
            shg_mgr=self._shg_mgr)

    def calculate_fluxmodel_scaling_factor(
            self,
            mean_ns,
            fitparam_values):
        """Calculates the factor the source's fluxmodel has to be scaled
        in order to obtain the given mean number of signal events in the
        detector.

        Parameters
        ----------
        mean_ns : float
            The mean number of signal events in the detector for which the
            scaling factor is calculated.
        fitparam_values : instance of numpy ndarray
            The (N_fitparam,)-shaped 1D ndarray holding the values of the global
            fit parameters, that should be used for the flux calculation.
            The order of the values must match the order the fit parameters were
            defined in the parameter model mapper.

        Returns
        -------
        factor : float
            The factor the given fluxmodel needs to be scaled in order to obtain
            the given mean number of signal events in the detector.
        """
        src_params_recarray =\
            self._pmm.create_src_params_recarray(
                gflp_values=fitparam_values)

        ds_sig_weight_factors_service =\
            self._llhratio.ds_sig_weight_factors_service
        src_detsigyield_weights_service =\
            ds_sig_weight_factors_service.src_detsigyield_weights_service

        # Calculate the detector signal yield, i.e. the mean number of signal
        # events in the detector, for the given reference flux model.
        mean_ns_ref = 0

        detsigyields = src_detsigyield_weights_service.detsigyield_arr[:, 0]
        for (j, detsigyield) in enumerate(detsigyields):
            src_recarray =\
                src_detsigyield_weights_service.src_recarray_list_list[j][0]
            (Yj, Yj_grads) = detsigyield(
                src_recarray=src_recarray,
                src_params_recarray=src_params_recarray)
            mean_ns_ref += Yj[0]

        factor = mean_ns / mean_ns_ref

        return factor
