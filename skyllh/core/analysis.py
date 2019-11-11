# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc
import numpy as np

from skyllh.core.py import (
    classname,
    issequenceof,
    range
)
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
    SingleSourceZeroSigH0SingleDatasetTCLLHRatio
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
from skyllh.core.signal_generator import SignalGenerator
from skyllh.physics.source import SourceModel


class Analysis(object):
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

    __metaclass__ = abc.ABCMeta

    def __init__(self, src_hypo_group_manager, src_fitparam_mapper,
                 test_statistic, bkg_gen_method=None, event_selection_method=None):
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
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod that implements the selection
            of the events, which have potential to be signal. All non-selected
            events will be treated as pure background events. This is for
            runtime optimization only.
            If set to None (default), the AllEventSelectionMethod will be used,
            that selects all events for the analysis.
        """
        # Call the super function to allow for multiple class inheritance.
        super(Analysis, self).__init__()

        if(event_selection_method is None):
            event_selection_method = AllEventSelectionMethod(src_hypo_group_manager)

        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.test_statistic = test_statistic
        self.bkg_gen_method = bkg_gen_method
        self.event_selection_method = event_selection_method

        self._dataset_list = []
        self._data_list = []
        self._tdm_list = []

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
    def event_selection_method(self):
        """The instance of EventSelectionMethod that selects events, which have
        potential to be signal. All non-selected events will be treated as pure
        background events.
        """
        return self._event_selection_method
    @event_selection_method.setter
    def event_selection_method(self, method):
        if(not isinstance(method, EventSelectionMethod)):
            raise TypeError('The event_selection_method property must be an '
                'instance of EventSelectionMethod!')
        self._event_selection_method = method

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

    def add_dataset(self, dataset, data, tdm=None):
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
        """
        if(not isinstance(dataset, Dataset)):
            raise TypeError('The dataset argument must be an instance '
                'of Dataset!')

        if(not isinstance(data, DatasetData)):
            raise TypeError('The data argument must be an instance '
                'of DatasetData!')

        if(tdm is None):
            tdm = TrialDataManager()
        if(not isinstance(tdm, TrialDataManager)):
            raise TypeError('The tdm argument must be None or an instance of '
                'TrialDataManager!')

        self._dataset_list.append(dataset)
        self._data_list.append(data)
        self._tdm_list.append(tdm)

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
        self._sig_generator = SignalGenerator(
            self._src_hypo_group_manager, self._dataset_list, self._data_list)

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
    def maximize_llhratio(self, rss):
        """This method is supposed to maximize the log-likelihood ratio
        function, by calling the ``maximize`` method of the LLHRatio class.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance to draw random numbers from.

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
        if(not isinstance(rss, RandomStateService)):
            raise TypeError('The rss argument must be an instance of '
                'RandomStateService!')

        if(mean_n_bkg_list is None):
            mean_n_bkg_list = [ None ]*self.n_datasets
        if(not issequenceof(mean_n_bkg_list, (type(None), float))):
            raise TypeError('The mean_n_bkg_list argument must be a sequence '
                'of None and/or floats!')

        if(bkg_kwargs is None):
            bkg_kwargs = dict()

        if(sig_kwargs is None):
            sig_kwargs = dict()

        # Construct the background event generator in case it's not constructed
        # yet.
        if(self._bkg_generator is None):
            self.construct_background_generator()

        # Generate pseudo data for each dataset with background and possible
        # signal events.
        n_events_list = []
        events_list = []
        for ds_idx in range(self.n_datasets):
            bkg_kwargs.update(mean=mean_n_bkg_list[ds_idx])
            with TaskTimer(tl, 'Generating background events for data set '
                '{:d}.'.format(ds_idx)):
                (n_bkg, bkg_events) = self._bkg_generator.generate_background_events(
                    rss, ds_idx, tl=tl, **bkg_kwargs)
            n_events_list.append(n_bkg)
            events_list.append(bkg_events)

        n_sig = 0
        if(mean_n_sig > 0):
            # Construct the signal generator if not done yet.
            if(self._sig_generator is None):
                self.construct_signal_generator()
            # Generate signal events with the given mean number of signal
            # events.
            sig_kwargs.update(mean=mean_n_sig)
            with TaskTimer(tl, 'Generating signal events.'):
                (n_sig, ds_sig_events_dict) = self._sig_generator.generate_signal_events(
                    rss, **sig_kwargs)
            # Inject the signal events to the generated background data.
            for (idx, sig_events) in ds_sig_events_dict.items():
                n_events_list[idx] += len(sig_events)
                events_list[idx].append(sig_events)

        return (n_sig, n_events_list, events_list)

    def do_trial(
            self, rss, mean_n_bkg_list=None, mean_n_sig=0, mean_n_sig_0=None,
            bkg_kwargs=None, sig_kwargs=None, tl=None):
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

        with TaskTimer(tl, 'Initializing trial.'):
            self.initialize_trial(events_list, n_events_list)

        with TaskTimer(tl, 'Maximizing LLH ratio function.'):
            (fitparamset, log_lambda_max, fitparam_values, status) = self.maximize_llhratio(rss)

        with TaskTimer(tl, 'Calculating test statistic.'):
            ts = self.calculate_test_statistic(log_lambda_max, fitparam_values)

        # Create the structured array data type for the result array.
        result_dtype = [
            ('mean_n_sig', np.float),
            ('n_sig', np.int),
            ('mean_n_sig_0', np.float),
            ('ts', np.float)
        ] + [
            (fitparam_name, np.float)
                for fitparam_name in fitparamset.fitparam_name_list
        ]
        result = np.empty((1,), dtype=result_dtype)
        result['mean_n_sig'] = mean_n_sig
        result['n_sig'] = n_sig
        result['mean_n_sig_0'] = mean_n_sig_0
        result['ts'] = ts
        for (idx, fitparam_name) in enumerate(fitparamset.fitparam_name_list):
            result[fitparam_name] = fitparam_values[idx]

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
    def __init__(self, src_hypo_group_manager, src_fitparam_mapper,
                 fitparam_ns,
                 test_statistic, bkg_gen_method=None, event_selection_method=None):
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
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod that implements the selection
            of the events, which have potential to be signal. All non-selected
            events will be treated as pure background events. This is for
            runtime optimization only.
            If set to None (default), the AllEventSelectionMethod will be used,
            that selects all events for the analysis.
        """
        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(TimeIntegratedMultiDatasetSingleSourceAnalysis, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper,
            test_statistic, bkg_gen_method, event_selection_method)

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

    def add_dataset(self, dataset, data, pdfratios, tdm=None):
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
        """
        super(TimeIntegratedMultiDatasetSingleSourceAnalysis, self).add_dataset(
            dataset, data, tdm)

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
        self._event_selection_method.change_source_hypo_group_manager(
            self._src_hypo_group_manager)

        # Change the source hypo group manager of the LLH ratio function
        # instance.
        self._llhratio.change_source_hypo_group_manager(self._src_hypo_group_manager)

        # Change the source hypo group manager of the signal generator instance.
        if(self._sig_generator is not None):
            self._sig_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

    def initialize_trial(self, events_list, n_events_list=None):
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
            the number of events is taken from the size of the given events.
        """
        self._llhratio.initialize_for_new_trial(
            events_list, n_events_list, self._event_selection_method)

    def maximize_llhratio(self, rss):
        """Maximizes the log-likelihood ratio function, by minimizing its
        negative.

        Parameters
        ----------
        rss : RandomStateService instance
            The RandomStateService instance that should be used to draw random
            numbers from. It is used by the minimizer to generate random
            fit parameter initial values.

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
            rss, self._fitparamset)
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


class SpatialEnergyTimeIntegratedMultiDatasetSingleSourceAnalysis(
    TimeIntegratedMultiDatasetSingleSourceAnalysis):
    """This is an analysis class that implements a time-integrated LLH ratio
    analysis with separate spatial and energy PDFs for multiple datasets
    assuming a single source.

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
        self, src_hypo_group_manager, src_fitparam_mapper,
        fitparam_ns, test_statistic, bkg_gen_method=None,
        event_selection_method=None):
        """Creates a new time-integrated analysis with separate spatial and
        energy PDFs for multiple datasets assuming a single source.

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
            to `None`, if no background events have to get generated.
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod that implements the selection
            of the events, which have potential to be signal. All non-selected
            events will be treated as pure background events. This is for
            runtime optimization only.
            If set to None (default), the AllEventSelectionMethod will be used,
            that selects all events for the analysis.
        """
        super(SpatialEnergyTimeIntegratedMultiDatasetSingleSourceAnalysis, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper,
            fitparam_ns, test_statistic, bkg_gen_method, event_selection_method)

    def add_dataset(self, dataset, data, spatial_pdfratio, energy_pdfratio,
                    tdm=None):
        """Adds a dataset with its spatial and energy PDF ratio instances to the
        analysis.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance that should get added.
        data : DatasetData instance
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        spatial_pdfratio : SpatialPDF sub-classed PDFRatio instance
            The PDFRatio instance for the spatial PDF.
        energy_pdfratio : EnergyPDF sub-classed PDFRatio instance
            The PDFRatio instance for the energy PDF.
        tdm : TrialDataManager instance | None
            The TrialDataManager instance managing the trial data and additional
            data fields of the data set. If set to None, it means that no
            additional data fields are defined.
        """
        if(not isinstance(spatial_pdfratio, PDFRatio)):
            raise TypeError('The spatial_pdfratio argument must be an instance '
                'of PDFRatio!')
        if(not issubclass(spatial_pdfratio.pdf_type, SpatialPDF)):
            raise TypeError('The PDF type of the PDFRatio instance of argument '
                'spatial_pdfratio must be SpatialPDF!')

        if(not isinstance(energy_pdfratio, PDFRatio)):
            raise TypeError('The energy_pdfratio argument must be an instance '
                'of PDFRatio!')
        if(not issubclass(energy_pdfratio.pdf_type, EnergyPDF)):
            raise TypeError('The PDF type of the PDFRatio instance of argument '
                'energy_pdfratio must be EnergyPDF!')

        pdfratios = [spatial_pdfratio, energy_pdfratio]

        super(SpatialEnergyTimeIntegratedMultiDatasetSingleSourceAnalysis,
            self).add_dataset(dataset, data, pdfratios, tdm)


class SingleMultiDimPDFTimeIntegratedMultiDatasetSingleSourceAnalysis(
    TimeIntegratedMultiDatasetSingleSourceAnalysis):
    """This is an analysis class that implements a time-integrated LLH ratio
    analysis with a single multi-dimensional PDF for multiple datasets
    assuming a single source.

    To run this analysis the following procedure applies:

        1. Add the datasets and their multi-dimensional PDF ratio instance
           via the :meth:`.add_dataset` method.
        2. Construct the log-likelihood ratio function via the
           :meth:`construct_llhratio` method.
        3. Initialize a trial via the :meth:`initialize_trial` method.
        4. Fit the global fit parameters to the trial data via the
           :meth:`maximize_llhratio` method.
    """
    def __init__(
        self, src_hypo_group_manager, src_fitparam_mapper,
        fitparam_ns, test_statistic, bkg_gen_method=None,
        event_selection_method=None):
        """Creates a new time-integrated analysis with a single
        multi-dimensional PDF for multiple datasets assuming a single source.

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
            to `None` if no background events have to get generated.
        event_selection_method : instance of EventSelectionMethod | None
            The instance of EventSelectionMethod that implements the selection
            of the events, which have potential to be signal. All non-selected
            events will be treated as pure background events. This is for
            runtime optimization only.
            If set to None (default), the AllEventSelectionMethod will be used,
            that selects all events for the analysis.
        """
        super(SingleMultiDimPDFTimeIntegratedMultiDatasetSingleSourceAnalysis, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper,
            fitparam_ns, test_statistic, bkg_gen_method, event_selection_method)

    def add_dataset(self, dataset, data, multidim_pdfratio, tdm=None):
        """Adds a dataset with its single multi-dimensional PDF ratio instance
        to the analysis.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance that should get added.
        data : DatasetData instance
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        multidim_pdfratio : PDFRatio instance
            The multi-dimensional PDFRatio instance for the data set.
        tdm : TrialDataManager instance | None
            The TrialDataManager instance managing the trial data and additional
            data fields of the data set. If set to None, it means that no
            additional data fields are defined.
        """
        if(not isinstance(multidim_pdfratio, PDFRatio)):
            raise TypeError('The multidim_pdfratio argument must be an '
                'instance of PDFRatio!')

        pdfratios = [multidim_pdfratio]

        super(SingleMultiDimPDFTimeIntegratedMultiDatasetSingleSourceAnalysis,
            self).add_dataset(dataset, data, pdfratios, tdm)
