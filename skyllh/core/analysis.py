# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc
import numpy as np
import pickle

from skyllh.core.py import (
    classname,
    issequenceof
)
from skyllh.core.debugging import get_logger
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.dataset import (
    Dataset,
    DatasetData
)
from skyllh.core.parameters import (
    FitParameter,
    SourceFitParameterMapper,
    SingleSourceFitParameterMapper
)
from skyllh.core.pdf import (
    EnergyPDF,
    SpatialPDF
)
from skyllh.core.pdfratio import PDFRatio
from skyllh.core.progressbar import ProgressBar
from skyllh.core.random import RandomStateService
from skyllh.core.llhratio import (
    LLHRatio,
    MultiDatasetTCLLHRatio,
    SingleSourceDatasetSignalWeights,
    SingleSourceZeroSigH0SingleDatasetTCLLHRatio,
    MultiSourceZeroSigH0SingleDatasetTCLLHRatio,
    MultiSourceDatasetSignalWeights
)
from skyllh.core.scrambling import DataScramblingMethod
from skyllh.core.timing import TaskTimer
from skyllh.core.trialdata import TrialDataManager
from skyllh.core.optimize import (
    EventSelectionMethod,
    AllEventSelectionMethod
)
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.test_statistic import TestStatistic
from skyllh.core.multiproc import get_ncpu, parallelize
from skyllh.core.background_generation import BackgroundGenerationMethod
from skyllh.core.background_generator import BackgroundGenerator
from skyllh.core.signal_generator import (
    SignalGeneratorBase,
    SignalGenerator
)
from skyllh.physics.source import SourceModel


logger = get_logger(__name__)


class Analysis(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for all analysis classes. It contains
    common properties required by all analyses and defines the overall analysis
    interface howto set-up and run an analysis.

    Note: This analysis base class assumes the analysis to be a log-likelihood
          ratio test, i.e. requires a mathematical log-likelihood ratio
          function.

    To set-up and run an analysis the following procedure applies:

        1. Create an analysis instance.
        2. Add the datasets and their PDF ratio instances via the
           :meth:`.add_dataset` method.
        3. Construct the log-likelihood ratio function via the
           :meth:`.construct_llhratio` method.
        4. Call the :meth:`do_trial` or :meth:`unblind` method to perform a
           random trial or to unblind the data. Both methods will fit the global
           fit parameters using the set up data. Finally, the test statistic
           is calculated via the :meth:`.calculate_test_statistic` method.

    In order to calculate sensitivities and discovery potentials, analysis
    trials have to be performed on random data samples with injected signal
    events. To perform a trial with injected signal events, the signal generator
    has to be constructed via the ``construct_signal_generator`` method before
    any random trial data can be generated.
    """

    def __init__(self, src_hypo_group_manager, src_fitparam_mapper,
                 test_statistic, bkg_gen_method=None, sig_generator_cls=None):
        """Constructor of the analysis base class.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            efficiency implementation method.
        src_fitparam_mapper : instance of SourceFitParameterMapper
            The SourceFitParameterMapper instance managing the global fit
            parameters and their relation to the individual sources.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that should be used to
            generate background events for pseudo data. This can be set to None,
            if there is no need to generate background events.
        sig_generator_cls : SignalGeneratorBase class | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        # Call the super function to allow for multiple class inheritance.
        super(Analysis, self).__init__()

        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.test_statistic = test_statistic
        self.bkg_gen_method = bkg_gen_method
        self.sig_generator_cls = sig_generator_cls

        self._dataset_list = []
        self._data_list = []
        self._tdm_list = []
        self._event_selection_method_list = []

        # Predefine the variable for the global fit parameter set, which holds
        # all the global fit parameters.
        self._fitparamset = None

        # Predefine the variable for the log-likelihood ratio function.
        self._llhratio = None

        # Predefine the variable for the background and signal generators.
        self._bkg_generator = None
        self._sig_generator = None

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance, which defines the groups of
        source hypothesis, their flux model, and their detector signal
        efficiency implementation method.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                'instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    @property
    def src_fitparam_mapper(self):
        """The SourceFitParameterMapper instance that manages the global fit
        parameters and their relation to the sources.
        """
        return self._src_fitparam_mapper
    @src_fitparam_mapper.setter
    def src_fitparam_mapper(self, mapper):
        if(not isinstance(mapper, SourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper property must be an '
                'instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def test_statistic(self):
        """The TestStatistic instance that defines the test-statistic function
        of the analysis.
        """
        return self._test_statistic
    @test_statistic.setter
    def test_statistic(self, ts):
        if(not isinstance(ts, TestStatistic)):
            raise TypeError('The test_statistic property must be an instance '
                'of TestStatistic, but is %s!'%(classname(ts)))
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
        if(method is not None):
            if(not isinstance(method, BackgroundGenerationMethod)):
                raise TypeError('The bkg_gen_method property must be an '
                    'instance of BackgroundGenerationMethod!')
        self._bkg_gen_method = method

    @property
    def dataset_list(self):
        """The list of Dataset instances.
        """
        return self._dataset_list
    @dataset_list.setter
    def dataset_list(self, datasets):
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The dataset_list property must be a sequence '
                'of Dataset instances!')
        self._dataset_list = list(datasets)

    @property
    def data_list(self):
        """The list of DatasetData instances holding the original data of the
        dataset.
        """
        return self._data_list
    @data_list.setter
    def data_list(self, datas):
        if(not issequenceof(datas, DatasetData)):
            raise TypeError('The data_list property must be a sequence '
                'of DatasetData instances!')
        self._data_list = list(datas)

    @property
    def n_datasets(self):
        """(read-only) The number of datasets used in this analysis.
        """
        return len(self._dataset_list)

    @property
    def fitparamset(self):
        """(read-only) The instance of FitParameterSet holding all the global
        fit parameters of the log-likelihood ratio function.
        """
        return self._fitparamset

    @property
    def llhratio(self):
        """The log-likelihood ratio function instance. It is None, if it has
        not been constructed yet.
        """
        if(self._llhratio is None):
            raise RuntimeError('The log-likelihood ratio function is not '
                'defined yet. Call the construct_analysis method first!')
        return self._llhratio
    @llhratio.setter
    def llhratio(self, obj):
        if(not isinstance(obj, LLHRatio)):
            raise TypeError('The llhratio property must be an instance of '
                'LLHRatio!')
        self._llhratio = obj

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
                'The sig_generator_cls property must be an subclass of '
                'SignalGeneratorBase!')
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

    def add_dataset(self, dataset, data, tdm=None, event_selection_method=None):
        """Adds the given dataset to the list of datasets for this analysis.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance that should get added.
        data : DatasetData instance
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        tdm : TrialDataManager instance | None
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
        if(not isinstance(dataset, Dataset)):
            raise TypeError(
                'The dataset argument must be an instance of Dataset!')

        if(not isinstance(data, DatasetData)):
            raise TypeError(
                'The data argument must be an instance of DatasetData!')

        if(tdm is None):
            tdm = TrialDataManager()
        if(not isinstance(tdm, TrialDataManager)):
            raise TypeError(
                'The tdm argument must be None or an instance of '
                'TrialDataManager!')

        if(event_selection_method is not None):
            if(not isinstance(event_selection_method, EventSelectionMethod)):
                raise TypeError(
                    'The event_selection_method argument must be None or an '
                    'instance of EventSelectionMethod!')

        self._dataset_list.append(dataset)
        self._data_list.append(data)
        self._tdm_list.append(tdm)
        self._event_selection_method_list.append(event_selection_method)

    def calculate_test_statistic(
            self, log_lambda, fitparam_values, *args, **kwargs):
        """Calculates the test statistic value by calling the ``evaluate``
        method of the TestStatistic class with the given log_lambda value and
        fit parameter values.

        Parameters
        ----------
        log_lambda : float
            The value of the log-likelihood ratio function. Usually, this is its
            maximum.
        fitparam_values : (N_fitparam+1)-shaped 1D ndarray
            The 1D ndarray holding the fit parameter values of the
            log-likelihood ratio function for the given log_lambda value.

        Additional arguments and keyword arguments
        ------------------------------------------
        Any additional arguments and keyword arguments are passed to the
        evaluate method of the TestStatistic class instance.

        Returns
        -------
        TS : float
            The calculated test-statistic value.
        """
        return self._test_statistic.evaluate(
            self._llhratio, log_lambda, fitparam_values, *args, **kwargs)

    @abc.abstractmethod
    def construct_llhratio(self):
        """This method is supposed to construct the log-likelihood ratio
        function and sets it as the _llhratio property.
        """
        pass

    def construct_background_generator(self):
        """Constructs the background generator for all added datasets.
        This method must be called after all the datasets were added via the
        add_dataset method. It sets the `bkg_generator` property of this
        Analysis class instance.
        """
        if(self._bkg_gen_method is None):
            raise RuntimeError('No background generation method has been '
                'defined for this analysis!')

        self._bkg_generator = BackgroundGenerator(
            self._bkg_gen_method, self._dataset_list, self._data_list)

    def construct_signal_generator(self):
        """Constructs the signal generator for all added datasets.
        This method must be called after all the datasets were added via the
        add_dataset method. It sets the `sig_generator` property of this
        Analysis class instance. The signal generation method has to be set
        through the source hypothesis group.
        """
        self._sig_generator = self.sig_generator_cls(
            src_hypo_group_manager=self._src_hypo_group_manager,
            dataset_list=self._dataset_list,
            data_list=self._data_list)

    @abc.abstractmethod
    def initialize_trial(self, events_list, n_events_list=None):
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
    def maximize_llhratio(self, rss, tl=None):
        """This method is supposed to maximize the log-likelihood ratio
        function, by calling the ``maximize`` method of the LLHRatio class.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance to draw random numbers from.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization of the LLH ratio function.

        Returns
        -------
        fitparamset : FitParameterSet instance
            The instance of FitParameterSet holding the global fit parameter
            definitions used in the maximization process.
        log_lambda_max : float
            The value of the log-likelihood ratio function at its maximum.
        fitparam_values : (N_fitparam,)-shaped 1D ndarray
            The ndarray holding the global fit parameter values.
            By definition, the first element is the value of the fit parameter
            ns.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        pass

    def unblind(self, rss):
        """Evaluates the unscrambled data, i.e. unblinds the data.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used draw random
            numbers from.

        Returns
        -------
        TS : float
            The test-statistic value.
        fitparam_dict : dict
            The dictionary holding the global fit parameter names and their best
            fit values.
        status : dict
            The status dictionary with information about the performed
            minimization process of the negative of the log-likelihood ratio
            function.
        """
        events_list = [ data.exp for data in self._data_list ]
        self.initialize_trial(events_list)
        (fitparamset, log_lambda_max, fitparam_values, status) = self.maximize_llhratio(rss)
        TS = self.calculate_test_statistic(log_lambda_max, fitparam_values)

        fitparam_dict = fitparamset.fitparam_values_to_dict(fitparam_values)

        return (TS, fitparam_dict, status)

    def generate_background_events(
            self, rss, mean_n_bkg_list=None, bkg_kwargs=None, tl=None):
        """Generates background events utilizing the background generator.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the background
            generation method needs to obtain this number itself.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks of this method.

        Returns
        -------
        n_events_list : list of int
            The list of the number of events that have been generated for each
            pseudo data set.
        events_list : list of DataFieldRecordArray instances
            The list of DataFieldRecordArray instances containing the pseudo
            data events for each data sample. The number of events for each
            data set can be less than the number of events given by
            `n_events_list` if an event selection method was already utilized
            when generating background events.
        """
        n_datasets = self.n_datasets

        if(not isinstance(rss, RandomStateService)):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService!')

        if(mean_n_bkg_list is None):
            mean_n_bkg_list = [ None ] * n_datasets
        if(not issequenceof(mean_n_bkg_list, (type(None), float))):
            raise TypeError(
                'The mean_n_bkg_list argument must be a sequence of None '
                'and/or floats!')

        if(bkg_kwargs is None):
            bkg_kwargs = dict()

        # Construct the background event generator in case it's not constructed
        # yet.
        if(self._bkg_generator is None):
            self.construct_background_generator()

        n_events_list = []
        events_list = []
        for ds_idx in range(n_datasets):
            bkg_kwargs.update(mean=mean_n_bkg_list[ds_idx])
            with TaskTimer(tl, 'Generating background events for data set '
                '{:d}.'.format(ds_idx)):
                (n_bkg, bkg_events) = self._bkg_generator.generate_background_events(
                    rss, ds_idx, tl=tl, **bkg_kwargs)
            n_events_list.append(n_bkg)
            events_list.append(bkg_events)

        return (n_events_list, events_list)

    def generate_signal_events(
            self, rss, mean_n_sig, sig_kwargs=None, n_events_list=None,
            events_list=None, tl=None):
        """Generates signal events utilizing the signal generator.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        sig_kwargs : dict | None
            Additional keyword arguments for the `generate_signal_events` method
            of the `sig_generator_cls` class. An usual keyword argument is
            `poisson`.
        n_events_list : list of int | None
            If given, it specifies the number of events of each data set already
            present and the number of signal events will be added.
        events_list : list of DataFieldRecordArray instances | None
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
            The list of the number of events that have been generated for each
            pseudo data set.
        events_list : list of DataFieldRecordArray instances
            The list of DataFieldRecordArray instances containing the pseudo
            signal data events for each data set. An entry is None, if no signal
            events were generated for this particular data set.
        """
        n_datasets = self.n_datasets

        if(not isinstance(rss, RandomStateService)):
            raise TypeError(
                'The rss argument must be an instance of RandomStateService!')

        if(sig_kwargs is None):
            sig_kwargs = dict()

        if(n_events_list is None):
            n_events_list = [0] * n_datasets
        else:
            if(not issequenceof(n_events_list, int)):
                raise TypeError(
                    'The n_events_list argument must be a sequence of '
                    'instances of type int!')
            if(len(n_events_list) != n_datasets):
                raise ValueError(
                    'The n_events_list argument must be a list of int of '
                    'length {:d}! Currently it is of length {:d}.'.format(
                        n_datasets, len(n_events_list)))

        if(events_list is None):
            events_list = [None] * n_datasets
        else:
            if(not issequenceof(
                    events_list, (type(None), DataFieldRecordArray))):
                raise TypeError(
                    'The events_list argument must be a sequence of '
                    'instances of type DataFieldRecordArray!')
            if(len(events_list) != n_datasets):
                raise ValueError(
                    'The events_list argument must be a list of instances of '
                    'type DataFieldRecordArray with a length of {:d}! '
                    'Currently it is of length {:d}.'.format(
                        n_datasets, len(events_list)))

        n_sig = 0

        if(mean_n_sig == 0):
            return (n_sig, n_events_list, events_list)

        # Construct the signal generator if not done yet.
        if(self._sig_generator is None):
            with TaskTimer(tl, 'Constructing signal generator.'):
                self.construct_signal_generator()
        # Generate signal events with the given mean number of signal
        # events.
        sig_kwargs.update(mean=mean_n_sig)
        with TaskTimer(tl, 'Generating signal events.'):
            (n_sig, ds_sig_events_dict) = self._sig_generator.generate_signal_events(
                rss, **sig_kwargs)
        # Inject the signal events to the generated background data.
        for (ds_idx, sig_events) in ds_sig_events_dict.items():
            n_events_list[ds_idx] += len(sig_events)
            if(events_list[ds_idx] is None):
                events_list[ds_idx] = sig_events
            else:
                events_list[ds_idx].append(sig_events)

        return (n_sig, n_events_list, events_list)

    def generate_pseudo_data(
            self, rss, mean_n_bkg_list=None, mean_n_sig=0, bkg_kwargs=None,
            sig_kwargs=None, tl=None):
        """Generates pseudo data with background and possible signal
        events for each data set using the background and signal generation
        methods of the analysis.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
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
            The instance of TimeLord that should be used to time individual
            tasks of this method.

        Returns
        -------
        n_sig : int
            The actual number of injected signal events.
        n_events_list : list of int
            The list of the number of events that have been generated for each
            pseudo data set.
        events_list : list of DataFieldRecordArray instances
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

    def do_trial_with_given_pseudo_data(
            self, rss, mean_n_sig, n_sig, n_events_list, events_list,
            mean_n_sig_0=None,
            minimizer_status_dict=None,
            tl=None):
        """Performs an analysis trial on the given pseudo data.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        mean_n_sig : float
            The mean number of signal events the pseudo data was generated with.
        n_sig : int
            The total number of actual signal events in the pseudo data.
        n_events_list : list of int
            The total number of events for each data set of the pseudo data.
        events_list : list of DataFieldRecordArray instances
            The list of DataFieldRecordArray instances containing the pseudo
            data events for each data sample. The number of events for each
            data sample can be less than the number of events given by
            `n_events_list` if an event selection method was already utilized
            when generating background events.
        mean_n_sig_0 : float | None
            The fixed mean number of signal events for the null-hypothesis,
            when using a ns-profile log-likelihood-ratio function.
            If set to None, this argument is interpreted as 0.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial. It
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
            [<fitparam_name> ... : float ]
                Any additional fit parameters of the LLH function.
        """
        if(mean_n_sig_0 is not None):
            self._llhratio.mean_n_sig_0 = mean_n_sig_0
        else:
            mean_n_sig_0 = 0

        with TaskTimer(tl, 'Initializing trial.'):
            self.initialize_trial(events_list, n_events_list)

        with TaskTimer(tl, 'Maximizing LLH ratio function.'):
            (fitparamset, log_lambda_max, fitparam_values, status) = self.maximize_llhratio(rss, tl=tl)
        if(isinstance(minimizer_status_dict, dict)):
            minimizer_status_dict.update(status)

        with TaskTimer(tl, 'Calculating test statistic.'):
            ts = self.calculate_test_statistic(log_lambda_max, fitparam_values)

        # Create the structured array data type for the result array.
        result_dtype = [
            ('seed', np.int64),
            ('mean_n_sig', np.float64),
            ('n_sig', np.int64),
            ('mean_n_sig_0', np.float64),
            ('ts', np.float64)
        ] + [
            (fitparam_name, np.float64)
                for fitparam_name in fitparamset.fitparam_name_list
        ]
        result = np.empty((1,), dtype=result_dtype)
        result['seed'] = rss.seed
        result['mean_n_sig'] = mean_n_sig
        result['n_sig'] = n_sig
        result['mean_n_sig_0'] = mean_n_sig_0
        result['ts'] = ts
        for (idx, fitparam_name) in enumerate(fitparamset.fitparam_name_list):
            result[fitparam_name] = fitparam_values[idx]

        return result

    def do_trial_with_given_bkg_and_sig_pseudo_data(
            self, rss, mean_n_sig, n_sig, n_bkg_events_list, n_sig_events_list,
            bkg_events_list, sig_events_list,
            mean_n_sig_0=None,
            minimizer_status_dict=None,
            tl=None):
        """Performs an analysis trial on the given background and signal pseudo
        data. This method merges the background and signal pseudo events and
        calls the ``do_trial_with_given_pseudo_data`` method of this class.

        Note
        ----
        This method alters the DataFieldRecordArray instances of the
        bkg_events_list argument!

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
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
        bkg_events_list : list of DataFieldRecordArray instances
            The list of DataFieldRecordArray instances containing the background
            pseudo data events for each data set.
        sig_events_list : list of DataFieldRecordArray instances or None
            The list of DataFieldRecordArray instances containing the signal
            pseudo data events for each data set. If a particular dataset has
            no signal events, the entry for that dataset can be None.
        mean_n_sig_0 : float | None
            The fixed mean number of signal events for the null-hypothesis,
            when using a ns-profile log-likelihood-ratio function.
            If set to None, this argument is interpreted as 0.
        minimizer_status_dict : dict | None
            If a dictionary is provided, it will be updated with the minimizer
            status dictionary.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial.
            See the documentation of the ``do_trial_with_given_pseudo_data``
            method for further information.
        """
        n_events_list = list(
            np.array(n_bkg_events_list) +
            np.array(n_sig_events_list)
        )

        events_list = bkg_events_list

        # Add potential signal events to the background events.
        for ds_idx in range(len(events_list)):
            if(sig_events_list[ds_idx] is not None):
                if(events_list[ds_idx] is None):
                    events_list[ds_idx] = sig_events_list[ds_idx]
                else:
                    events_list[ds_idx].append(sig_events_list[ds_idx])

        return self.do_trial_with_given_pseudo_data(
            rss = rss,
            mean_n_sig = mean_n_sig,
            n_sig = n_sig,
            n_events_list = n_events_list,
            events_list = events_list,
            mean_n_sig_0 = mean_n_sig_0,
            minimizer_status_dict = minimizer_status_dict,
            tl = tl
        )

    def do_trial(
            self, rss, mean_n_bkg_list=None, mean_n_sig=0, mean_n_sig_0=None,
            bkg_kwargs=None, sig_kwargs=None, minimizer_status_dict=None,
            tl=None):
        """Performs an analysis trial by generating a pseudo data sample with
        background events and possible signal events, and performs the LLH
        analysis on that random pseudo data sample.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the background
            generation method needs to obtain this number itself.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        mean_n_sig_0 : float | None
            The fixed mean number of signal events for the null-hypothesis,
            when using a ns-profile log-likelihood-ratio function.
            If set to None, this argument is interpreted as 0.
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
            The instance of TimeLord that should be used to time individual
            tasks.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial. It
            contains the following data fields:

            mean_n_sig : float
                The mean number of signal events.
            n_sig : int
                The actual number of injected signal events.
            mean_n_sig_0 : float
                The fixed mean number of signal events for the null-hypothesis.
            ts : float
                The test-statistic value.
            [<fitparam_name> ... : float ]
                Any additional fit parameters of the LLH function.
        """
        if(mean_n_sig_0 is not None):
            self._llhratio.mean_n_sig_0 = mean_n_sig_0
        else:
            mean_n_sig_0 = 0

        with TaskTimer(tl, 'Generating pseudo data.'):
            (n_sig, n_events_list, events_list) = self.generate_pseudo_data(
                rss=rss, mean_n_bkg_list=mean_n_bkg_list, mean_n_sig=mean_n_sig,
                bkg_kwargs=bkg_kwargs, sig_kwargs=sig_kwargs, tl=tl)

        result = self.do_trial_with_given_pseudo_data(
            rss=rss,
            mean_n_sig=mean_n_sig,
            n_sig=n_sig,
            n_events_list=n_events_list,
            events_list=events_list,
            mean_n_sig_0=mean_n_sig_0,
            minimizer_status_dict=minimizer_status_dict,
            tl=tl
        )

        return result

    def do_trials(
            self, rss, n, mean_n_bkg_list=None, mean_n_sig=0, mean_n_sig_0=None,
            bkg_kwargs=None, sig_kwargs=None, ncpu=None, tl=None, ppbar=None):
        """Executes `do_trial` method `N` times with possible multi-processing.
        One trial performs an analysis trial by generating a pseudo data sample
        with background events and possible signal events, and performs the LLH
        analysis on that random pseudo data sample.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        n : int
            Number of trials to generate using the `do_trial` method.
        mean_n_bkg_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the number of data
            events of each data sample will be used as mean.
        mean_n_sig : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        mean_n_sig_0 : float | None
            The fixed mean number of signal events for the null-hypothesis,
            when using a ns-profile log-likelihood-ratio function.
        bkg_kwargs : dict | None
            Additional keyword arguments for the `generate_events` method of the
            background generation method class. An usual keyword argument is
            `poisson`.
        sig_kwargs : dict | None
            Additional keyword arguments for the `generate_signal_events` method
            of the `SignalGenerator` class. An usual keyword argument is
            `poisson`. If `poisson` is set to True, the actual number of
            generated signal events will be drawn from a Poisson distribution
            with the given mean number of signal events.
            If set to False, the argument ``mean_n_sig`` specifies the actual
            number of generated signal events.
        ncpu : int | None
            The number of CPUs to use, i.e. the number of subprocesses to
            spawn. If set to None, the global setting will be used.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to time individual
            tasks.
        ppbar : instance of ProgressBar | None
            The possible parent ProgressBar instance.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial. It
            contains the following data fields:

            n_sig : int
                The actual number of injected signal events.
            ts : float
                The test-statistic value.
            [<fitparam_name> ... : float ]
                Any additional fit parameters of the LLH function.
        """
        ncpu = get_ncpu(ncpu)
        args_list = [((), {
            'mean_n_bkg_list': mean_n_bkg_list,
            'mean_n_sig': mean_n_sig,
            'mean_n_sig_0': mean_n_sig_0,
            'bkg_kwargs': bkg_kwargs,
            'sig_kwargs': sig_kwargs
            }) for i in range(n)
        ]
        result_list = parallelize(
            self.do_trial, args_list, ncpu, rss=rss, tl=tl, ppbar=ppbar)

        result_dtype = result_list[0].dtype
        result = np.empty(n, dtype=result_dtype)
        result[:] = result_list[:]
        return result


class TimeIntegratedMultiDatasetSingleSourceAnalysis(Analysis):
    """This is an analysis class that implements a time-integrated LLH ratio
    analysis for multiple datasets assuming a single source.

    To run this analysis the following procedure applies:

        1. Add the datasets and their spatial and energy PDF ratio instances
           via the :meth:`.add_dataset` method.
        2. Construct the log-likelihood ratio function via the
           :meth:`construct_llhratio` method.
        3. Initialize a trial via the :meth:`initialize_trial` method.
        4. Fit the global fit parameters to the trial data via the
           :meth:`maximize_llhratio` method.
    """
    def __init__(self, src_hypo_group_manager, src_fitparam_mapper, fitparam_ns,
                 test_statistic, bkg_gen_method=None, sig_generator_cls=None):
        """Creates a new time-integrated point-like source analysis assuming a
        single source.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            efficiency implementation method.
        src_fitparam_mapper : instance of SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        fitparam_ns : FitParameter instance
            The FitParameter instance defining the fit parameter ns.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that will be used to
            generate background events for a new analysis trial. This can be set
            to None, if no background events have to get generated.
        sig_generator_cls : SignalGeneratorBase class | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super().__init__(
            src_hypo_group_manager=src_hypo_group_manager,
            src_fitparam_mapper=src_fitparam_mapper,
            test_statistic=test_statistic,
            bkg_gen_method=bkg_gen_method,
            sig_generator_cls=sig_generator_cls)

        self.fitparam_ns = fitparam_ns

        # Define the member for the list of PDF ratio lists. Each list entry is
        # a list of PDF ratio instances for each data set.
        self._pdfratio_list_list = []

        # Create the FitParameterSet instance holding the fit parameter ns and
        # all the other additional fit parameters. This set is used by the
        # ``maximize_llhratio`` method.
        self._fitparamset = self._src_fitparam_mapper.fitparamset.copy()
        self._fitparamset.add_fitparam(self._fitparam_ns, atfront=True)

    @property
    def fitparam_ns(self):
        """The FitParameter instance for the fit parameter ns.
        """
        return self._fitparam_ns
    @fitparam_ns.setter
    def fitparam_ns(self, fitparam):
        if(not isinstance(fitparam, FitParameter)):
            raise TypeError('The fitparam_ns property must be an instance of FitParameter!')
        self._fitparam_ns = fitparam

    def add_dataset(self, dataset, data, pdfratios, tdm=None,
                    event_selection_method=None):
        """Adds a dataset with its PDF ratio instances to the analysis.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance that should get added.
        data : DatasetData instance
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        pdfratios : PDFRatio instance | sequence of PDFRatio instances
            The PDFRatio instance or the sequence of PDFRatio instances for the
            to-be-added data set.
        tdm : TrialDataManager instance | None
            The TrialDataManager instance that manages the trial data and
            additional data fields for this data set.
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod to use to select only
            signal-like events from the trial data. All other events
            will be treated as pure background events. This reduces the amount
            of log-likelihood-ratio function evaluations. If set to None, all
            events will be evaluated.
        """
        super(TimeIntegratedMultiDatasetSingleSourceAnalysis, self).add_dataset(
            dataset, data, tdm, event_selection_method)

        if(isinstance(pdfratios, PDFRatio)):
            pdfratios = [pdfratios]
        if(not issequenceof(pdfratios, PDFRatio)):
            raise TypeError('The pdfratios argument must be an instance of '
                'PDFRatio or a sequence of PDFRatio!')

        self._pdfratio_list_list.append(list(pdfratios))

    def construct_llhratio(self, minimizer, ppbar=None):
        """Constructs the log-likelihood-ratio (LLH-ratio) function of the
        analysis. This setups all the necessary analysis
        objects like detector signal efficiencies and dataset signal weights,
        constructs the log-likelihood ratio functions for each dataset and the
        final composite llh ratio function.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The instance of Minimizer that should be used to minimize the
            negative of the log-likelihood ratio function.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        llhratio : instance of MultiDatasetTCLLHRatio
            The instance of MultiDatasetTCLLHRatio that implements the
            log-likelihood-ratio function of the analysis.
        """
        # Create the detector signal yield instances for each dataset.
        # Since this is for a single source, we don't have to have individual
        # detector signal yield instances for each source as well.
        detsigyield_list = []
        fluxmodel = self._src_hypo_group_manager.get_fluxmodel_by_src_idx(0)
        detsigyield_implmethod_list = self._src_hypo_group_manager.get_detsigyield_implmethod_list_by_src_idx(0)
        if((len(detsigyield_implmethod_list) != 1) and
           (len(detsigyield_implmethod_list) != self.n_datasets)):
            raise ValueError('The number of detector signal yield '
                'implementation methods is not 1 and does not match the number '
                'of used datasets in the analysis!')
        pbar = ProgressBar(len(self.dataset_list), parent=ppbar).start()
        for (j, (dataset, data)) in enumerate(zip(self.dataset_list,
                                                  self.data_list)):
            if(len(detsigyield_implmethod_list) == 1):
                # Only one detsigyield implementation method was defined, so we
                # use it for all datasets.
                detsigyield_implmethod = detsigyield_implmethod_list[0]
            else:
                detsigyield_implmethod = detsigyield_implmethod_list[j]

            detsigyield = detsigyield_implmethod.construct_detsigyield(
                dataset, data, fluxmodel, data.livetime, ppbar=pbar)
            detsigyield_list.append(detsigyield)
            pbar.increment()
        pbar.finish()

        # For multiple datasets we need a dataset signal weights instance in
        # order to distribute ns over the different datasets.
        dataset_signal_weights = SingleSourceDatasetSignalWeights(
            self._src_hypo_group_manager, self._src_fitparam_mapper,
            detsigyield_list)

        # Create the list of log-likelihood ratio functions, one for each
        # dataset.
        llhratio_list = []
        for j in range(self.n_datasets):
            tdm = self._tdm_list[j]
            pdfratio_list = self._pdfratio_list_list[j]
            llhratio = SingleSourceZeroSigH0SingleDatasetTCLLHRatio(
                minimizer,
                self._src_hypo_group_manager,
                self._src_fitparam_mapper,
                tdm,
                pdfratio_list
            )
            llhratio_list.append(llhratio)

        # Create the final multi-dataset log-likelihood ratio function.
        llhratio = MultiDatasetTCLLHRatio(
            minimizer, dataset_signal_weights, llhratio_list)

        return llhratio

    def change_source(self, source):
        """Changes the source of the analysis to the given source. It makes the
        necessary changes to all the objects of the analysis.

        Parameters
        ----------
        source : SourceModel instance
            The SourceModel instance describing the new source.
        """
        if(not isinstance(source, SourceModel)):
            raise TypeError('The source argument must be an instance of SourceModel')

        if(self._llhratio is None):
            raise RuntimeError('The LLH ratio function has to be constructed, '
                'before the `change_source` method can be called!')

        # Change the source in the SourceHypoGroupManager instance.
        # Because this is a single source analysis, there can only be one source
        # hypothesis group defined.
        self._src_hypo_group_manager.src_hypo_group_list[0].source_list[0] = source

        # Change the source hypo group manager of the EventSelectionMethod
        # instance.
        for event_selection_method in self._event_selection_method_list:
            if(event_selection_method is not None):
                event_selection_method.change_source_hypo_group_manager(
                    self._src_hypo_group_manager)

        # Change the source hypo group manager of the LLH ratio function
        # instance.
        self._llhratio.change_source_hypo_group_manager(self._src_hypo_group_manager)

        # Change the source hypo group manager of the background generator
        # instance.
        if(self._bkg_generator is not None):
            self._bkg_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

        # Change the source hypo group manager of the signal generator instance.
        if(self._sig_generator is not None):
            self._sig_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

    def change_sources(self, sources):
        """Changes the sources of the analysis to the given source list. It
        makes the necessary changes to all the objects of the analysis.

        Parameters
        ----------
        sources : list of SourceModel instances
            The SourceModel instances describing new sources.
        """
        if(isinstance(sources, SourceModel)):
            sources = [sources]
        if(not issequenceof(sources, SourceModel)):
            raise TypeError('The sources argument must be a list of instances '
                            'of SourceModel')

        if(self._llhratio is None):
            raise RuntimeError(
                'The LLH ratio function has to be constructed, '
                'before the `change_source` method can be called!')

        # Change the source in the SourceHypoGroupManager instance.
        # Because this is a single type sources analysis, there can only be one
        # source hypothesis group defined.
        self._src_hypo_group_manager.src_hypo_group_list[0].source_list = sources

        # Change the source hypo group manager of the EventSelectionMethod
        # instance.
        for event_selection_method in self._event_selection_method_list:
            if(event_selection_method is not None):
                event_selection_method.change_source_hypo_group_manager(
                    self._src_hypo_group_manager)

        # Change the source hypo group manager of the LLH ratio function
        # instance.
        self._llhratio.change_source_hypo_group_manager(
            self._src_hypo_group_manager)

        # Change the source hypo group manager of the background generator
        # instance.
        if(self._bkg_generator is not None):
            self._bkg_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

        # Change the source hypo group manager of the signal generator instance.
        if(self._sig_generator is not None):
            self._sig_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

    def initialize_trial(self, events_list, n_events_list=None, tl=None):
        """This method initializes the multi-dataset log-likelihood ratio
        function with a new set of given trial data. This is a low-level method.
        For convenient methods see the `unblind` and `do_trial` methods.

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
        tl : TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        if(n_events_list is None):
            n_events_list = [None] * len(events_list)

        for (idx, (tdm, events, n_events, evt_sel_method)) in enumerate(zip(
                self._tdm_list, events_list, n_events_list,
                self._event_selection_method_list)):

            # Initialize the trial data manager with the given raw events.
            self._tdm_list[idx].initialize_trial(
                self._src_hypo_group_manager, events, n_events, evt_sel_method,
                tl=tl)

        self._llhratio.initialize_for_new_trial(tl=tl)

    def maximize_llhratio(self, rss, tl=None):
        """Maximizes the log-likelihood ratio function, by minimizing its
        negative.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used to draw random
            numbers from. It is used by the minimizer to generate random
            fit parameter initial values.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to time the
            maximization of the LLH ratio function.

        Returns
        -------
        fitparamset : FitParameterSet instance
            The instance of FitParameterSet holding the global fit parameter
            definitions used in the maximization process.
        log_lambda_max : float
            The value of the log-likelihood ratio function at its maximum.
        fitparam_values : (N_fitparam,)-shaped 1D ndarray
            The ndarray holding the global fit parameter values.
            By definition, the first element is the value of the fit parameter
            ns.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        (log_lambda_max, fitparam_values, status) = self._llhratio.maximize(
            rss, self._fitparamset, tl=tl)
        return (self._fitparamset, log_lambda_max, fitparam_values, status)

    def calculate_fluxmodel_scaling_factor(self, mean_ns, fitparam_values):
        """Calculates the factor the source's fluxmodel has to be scaled
        in order to obtain the given mean number of signal events in the
        detector.

        Parameters
        ----------
        mean_ns : float
            The mean number of signal events in the detector for which the
            scaling factor is calculated.
        fitparam_values : (N_fitparams,)-shaped 1D ndarray
            The ndarray holding the fit parameter values that should be used for
            the flux calculation.

        Returns
        -------
        factor : float
            The factor the given fluxmodel needs to be scaled in order to obtain
            the given mean number of signal events in the detector.
        """
        fitparams_arr = self._src_fitparam_mapper.get_fitparams_array(
            fitparam_values)

        # We use the DatasetSignalWeights class instance of this analysis to
        # calculate the detector signal yield for all datasets.
        dataset_signal_weights = self._llhratio.dataset_signal_weights

        # Calculate the detector signal yield, i.e. the mean number of signal
        # events in the detector, for the given reference flux model.
        mean_ns_ref = 0
        detsigyields = dataset_signal_weights.detsigyield_arr[0]
        for detsigyield in detsigyields:
            (Yj, Yj_grads) = detsigyield(
                dataset_signal_weights._src_arr_list[0], fitparams_arr)
            mean_ns_ref += Yj[0]

        factor = mean_ns / mean_ns_ref

        return factor


class TimeIntegratedMultiDatasetMultiSourceAnalysis(
        TimeIntegratedMultiDatasetSingleSourceAnalysis):
    """This is an analysis class that implements a time-integrated LLH ratio
    analysis for multiple datasets assuming multiple sources.

    To run this analysis the following procedure applies:

        1. Add the datasets and their spatial and energy PDF ratio instances
           via the :meth:`.add_dataset` method.
        2. Construct the log-likelihood ratio function via the
           :meth:`construct_llhratio` method.
        3. Initialize a trial via the :meth:`initialize_trial` method.
        4. Fit the global fit parameters to the trial data via the
           :meth:`maximize_llhratio` method.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, fitparam_ns,
            test_statistic, bkg_gen_method=None, sig_generator_cls=None):
        """Creates a new time-integrated point-like source analysis assuming
        multiple sources.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses, their flux model, and their detector signal
            efficiency implementation method.
        src_fitparam_mapper : instance of SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        fitparam_ns : FitParameter instance
            The FitParameter instance defining the fit parameter ns.
        test_statistic : TestStatistic instance
            The TestStatistic instance that defines the test statistic function
            of the analysis.
        bkg_gen_method : instance of BackgroundGenerationMethod | None
            The instance of BackgroundGenerationMethod that will be used to
            generate background events for a new analysis trial. This can be set
            to None, if no background events have to get generated.
        sig_generator_cls : SignalGeneratorBase class | None
            The signal generator class used to create the signal generator
            instance.
            If set to None, the `SignalGenerator` class is used.
        """
        super().__init__(
            src_hypo_group_manager=src_hypo_group_manager,
            src_fitparam_mapper=src_fitparam_mapper,
            fitparam_ns=fitparam_ns,
            test_statistic=test_statistic,
            bkg_gen_method=bkg_gen_method,
            sig_generator_cls=sig_generator_cls)

    def construct_llhratio(self, minimizer, ppbar=None):
        """Constructs the log-likelihood-ratio (LLH-ratio) function of the
        analysis. This setups all the necessary analysis
        objects like detector signal efficiencies and dataset signal weights,
        constructs the log-likelihood ratio functions for each dataset and the
        final composite llh ratio function.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The instance of Minimizer that should be used to minimize the
            negative of the log-likelihood ratio function.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        llhratio : instance of MultiDatasetTCLLHRatio
            The instance of MultiDatasetTCLLHRatio that implements the
            log-likelihood-ratio function of the analysis.
        """
        # Create the detector signal yield instances for each dataset.
        # Multi source analysis has to also support multiple source hypothesis
        # groups.
        # Initialize empty (N_source_hypo_groups, N_datasets)-shaped ndarray.
        detsigyield_array = np.empty(
            (self._src_hypo_group_manager.n_src_hypo_groups,
             len(self.dataset_list)),
            dtype=object
        )

        for (g, shg) in enumerate(self._src_hypo_group_manager._src_hypo_group_list):
            fluxmodel = shg.fluxmodel
            detsigyield_implmethod_list = shg.detsigyield_implmethod_list

            if((len(detsigyield_implmethod_list) != 1) and
               (len(detsigyield_implmethod_list) != self.n_datasets)):
                raise ValueError(
                    'The number of detector signal yield '
                    'implementation methods is not 1 and does not match the number '
                    'of used datasets in the analysis!')
            pbar = ProgressBar(len(self.dataset_list), parent=ppbar).start()
            for (j, (dataset, data)) in enumerate(zip(self.dataset_list,
                                                      self.data_list)):
                if(len(detsigyield_implmethod_list) == 1):
                    # Only one detsigyield implementation method was defined, so we
                    # use it for all datasets.
                    detsigyield_implmethod = detsigyield_implmethod_list[0]
                else:
                    detsigyield_implmethod = detsigyield_implmethod_list[j]

                detsigyield = detsigyield_implmethod.construct_detsigyield(
                    dataset, data, fluxmodel, data.livetime, ppbar=pbar)
                detsigyield_array[g, j] = detsigyield
                pbar.increment()
            pbar.finish()

        # For multiple datasets we need a dataset signal weights instance in
        # order to distribute ns over the different datasets.
        dataset_signal_weights = MultiSourceDatasetSignalWeights(
            self._src_hypo_group_manager, self._src_fitparam_mapper,
            detsigyield_array)

        # Create the list of log-likelihood ratio functions, one for each
        # dataset.
        llhratio_list = []
        for j in range(self.n_datasets):
            tdm = self._tdm_list[j]
            pdfratio_list = self._pdfratio_list_list[j]
            llhratio = MultiSourceZeroSigH0SingleDatasetTCLLHRatio(
                minimizer,
                self._src_hypo_group_manager,
                self._src_fitparam_mapper,
                tdm,
                pdfratio_list,
                detsigyield_array[:, j]
            )
            llhratio_list.append(llhratio)

        # Create the final multi-dataset log-likelihood ratio function.
        llhratio = MultiDatasetTCLLHRatio(
            minimizer, dataset_signal_weights, llhratio_list)

        return llhratio

    def initialize_trial(self, events_list, n_events_list=None, tl=None):
        """This method initializes the multi-dataset log-likelihood ratio
        function with a new set of given trial data. This is a low-level method.
        For convenient methods see the `unblind` and `do_trial` methods.

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
        tl : TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        if(n_events_list is None):
            n_events_list = [None] * len(events_list)

        for (idx, (tdm, events, n_events, evt_sel_method)) in enumerate(zip(
                self._tdm_list, events_list, n_events_list,
                self._event_selection_method_list)):

            # Initialize the trial data manager with the given raw events.
            self._tdm_list[idx].initialize_trial(
                self._src_hypo_group_manager, events, n_events, evt_sel_method,
                store_src_ev_idxs=True, tl=tl)

        self._llhratio.initialize_for_new_trial(tl=tl)
