# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc
import numpy as np

from skyllh.core.py import issequenceof, range
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.parameters import (
    FitParameter,
    SourceFitParameterMapper,
    SingleSourceFitParameterMapper
)
from skyllh.core.pdf import SpatialPDF, EnergyPDF
from skyllh.core.pdfratio import PDFRatio
from skyllh.core.random import RandomStateService
from skyllh.core.llhratio import (
    SingleSourceDatasetSignalWeights,
    SingleSourceTCLLHRatio,
    MultiDatasetTCLLHRatio
)
from skyllh.core.scrambling import DataScramblingMethod
from skyllh.core.optimize import EventSelectionMethod, AllEventSelectionMethod
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.test_statistic import TestStatistic
from skyllh.core.minimizer import Minimizer
from skyllh.core.multiproc import get_ncpu, parallelize
from skyllh.core.background_generation import BackgroundGenerationMethod
from skyllh.core.background_generator import BackgroundGenerator
from skyllh.core.signal_generator import SignalGenerator
from skyllh.physics.source import SourceModel


class Analysis(object):
    """This is the abstract base class for all analysis classes. It contains
    common properties required by all analyses and defines the overall analysis
    interface howto set-up and run an analysis.

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

    def __init__(self, minimizer, src_hypo_group_manager, src_fitparam_mapper,
                 test_statistic, bkg_gen_method, event_selection_method=None):
        """Constructor of the analysis base class.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of the log-likelihood ratio function.
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
        bkg_gen_method : instance of BackgroundGenerationMethod
            The instance of BackgroundGenerationMethod that should be used to
            generate background events for pseudo data.
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

        self.minimizer = minimizer
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.test_statistic = test_statistic
        self.bkg_gen_method = bkg_gen_method
        self.event_selection_method = event_selection_method

        self._dataset_list = []
        self._data_list = []

        # Predefine the variable for the global fit parameter set, which holds
        # all the global fit parameters.
        self._fitparamset = None

        # Predefine the variable for the log-likelihood ratio function.
        self._llhratio = None

        # Predefine the variable for the background and signal generators.
        self._bkg_generator = None
        self._sig_generator = None

    @property
    def minimizer(self):
        """The Minimizer instance used to minimize the negative of the
        log-likelihood ratio function.
        """
        return self._minimizer
    @minimizer.setter
    def minimizer(self, minimizer):
        if(not isinstance(minimizer, Minimizer)):
            raise TypeError('The minimizer property must be an instance '
                'of Minimizer!')
        self._minimizer = minimizer

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
                'of TestStatistic!')
        self._test_statistic = ts

    @property
    def bkg_gen_method(self):
        """The BackgroundGenerationMethod instance that implements the
        background event generation.
        """
        return self._bkg_gen_method
    @bkg_gen_method.setter
    def bkg_gen_method(self, method):
        if(not isinstance(method, BackgroundGenerationMethod)):
            raise TypeError('The bkg_gen_method property must be an instance '
                'of BackgroundGenerationMethod!')
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
        """(read-only) The log-likelihood ratio function.
        """
        if(self._llhratio is None):
            raise RuntimeError('The log-likelihood ratio function is not '
                'defined yet. Call the construct_analysis method first!')
        return self._llhratio

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

    def add_dataset(self, dataset, data):
        """Adds the given dataset to the list of datasets for this analysis.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance that should get added.
        data : DatasetData instance
            The DatasetData instance holding the original (prepared) data of the
            dataset.
        """
        if(not isinstance(dataset, Dataset)):
            raise TypeError('The dataset argument must be an instance '
                'of Dataset!')
        if(not isinstance(data, DatasetData)):
            raise TypeError('The data argument must be an instance '
                'of DatasetData!')

        self._dataset_list.append(dataset)
        self._data_list.append(data)

    def calculate_test_statistic(self, log_lambda, fitparam_values, *args, **kwargs):
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
        return self._test_statistic.evaluate(self._llhratio, log_lambda, fitparam_values,
                                             *args, **kwargs)

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
    def initialize_trial(self, events_list):
        """This method is supposed to initialize the log-likelihood ratio
        function with a new set of given trial data. This is a low-level method.
        For convenient methods see the `unblind` and `do_trial` methods.

        Parameters
        ----------
        events_list : list of numpy record ndarray
            The list of data events to use for the log-likelihood function
            evaluation. The data arrays for the datasets must be in the same
            order than the added datasets.
        """
        pass

    @abc.abstractmethod
    def maximize_llhratio(self):
        """This method is supposed to maximize the log-likelihood ratio
        function.

        Returns
        -------
        max_log_lambda : float
            The (maximum) value of the log_lambda function for the best fit
            parameter values.
        best_fitparams : dict
            The best fit parameters as a dictionary of fit parameter name and
            value.
        """
        pass

    def unblind(self):
        """Evaluates the unscrambled data, i.e. unblinds the data.

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
        (fitparamset, log_lambda_max, fitparam_values, status) = self.maximize_llhratio()
        TS = self.calculate_test_statistic(log_lambda_max, fitparam_values)

        fitparam_dict = fitparamset.fitparam_values_to_dict(fitparam_values)

        return (TS, fitparam_dict, status)

    def do_trial(self, rss, bkg_mean_list=None, sig_mean=0):
        """Performs an analysis trial by generating a pseudo data sample with
        background events and possible signal events, and performs the LLH
        analysis on that random pseudo data sample.

        Parameters
        ----------
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        bkg_mean_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the number of data
            events of each data sample will be used as mean.
        sig_mean : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial. It
            contains the following data fields:

            n_sig : int
                The actual number of injected signal events.
            TS : float
                The test-statistic value.
            [<fitparam_name> ... : float ]
                Any additional fit parameters of the LLH function.
        """
        if(not isinstance(rss, RandomStateService)):
            raise TypeError('The rss argument must be an instance of '
                'RandomStateService!')

        if(bkg_mean_list is None):
            bkg_mean_list = [ float(len(data.exp)) for data in self._data_list ]
        if(not issequenceof(bkg_mean_list, float)):
            raise TypeError('The bkg_mean_list argument must be a sequence '
                'of floats!')

        # Construct the background event generator in case it's not constructed
        # yet.
        if(self._bkg_generator is None):
            self.construct_background_generator()

        # Generate pseudo data for each dataset with background and possible
        # signal events.
        events_list = []
        for (idx, data) in enumerate(self._data_list):
            events_list.append(
                self._bkg_generator.generate_background_events(rss, idx, bkg_mean_list[idx]))

        if(sig_mean > 0):
            # Generate signal events with the given mean number of signal
            # events.
            # Construct the signal generator if not done yet.
            if(self._sig_generator is None):
                self.construct_signal_generator()
            (n_sig, ds_sig_events_dict) = self._sig_generator.generate_signal_events(
                rss, sig_mean)
            # Inject the signal events to the generated background data.
            for (idx, sig_events) in ds_sig_events_dict:
                field_name_list = list(self._dataset_list[idx].exp_field_names)
                bkg_events = events_list[idx]
                events_list[idx] = np.append(
                    bkg_events,
                    sig_events[field_name_list]
                )

        self.initialize_trial(events_list)

        (fitparamset, log_lambda_max, fitparam_values, status) = self.maximize_llhratio()
        TS = self.calculate_test_statistic(log_lambda_max, fitparam_values)

        # Create the structured array data type for the result array.
        result_dtype = [('n_sig', np.int), ('TS', np.float)] + [
            (fitparam_name, np.float) for fitparam_name in self._fitparamset.fitparam_name_list
        ]
        result = np.empty((1,), dtype=results_dtype)
        result['n_sig'] = n_sig
        result['TS'] = TS
        for (idx, fitparam_name) in enumerate(self._fitparamset.fitparam_name_list):
            result[fitparam_name] = fitparam_values[idx]

        return result

    def do_trials(self, N, rss, bkg_mean_list=None, sig_mean=0, ncpu=None):
        """Executes `do_trial` method `N` times with possible multi-processing.
        One trial performs an analysis trial by generating a pseudo data sample
        with background events and possible signal events, and performs the LLH
        analysis on that random pseudo data sample.

        Parameters
        ----------
        N : int
            Number of trials to generate using the `do_trial` method.
        rss : RandomStateService
            The RandomStateService instance to use for generating random
            numbers.
        bkg_mean_list : list of float | None
            The mean number of background events that should be generated for
            each dataset. If set to None (the default), the number of data
            events of each data sample will be used as mean.
        sig_mean : float
            The mean number of signal events that should be generated for the
            trial. The actual number of generated events will be drawn from a
            Poisson distribution with this given signal mean as mean.
        ncpu : int | None
            The number of CPUs to use, i.e. the number of subprocesses to
            spawn.

        Returns
        -------
        result : structured numpy ndarray
            The structured numpy ndarray holding the result of the trial. It
            contains the following data fields:

            n_sig : int
                The actual number of injected signal events.
            TS : float
                The test-statistic value.
            [<fitparam_name> ... : float ]
                Any additional fit parameters of the LLH function.
        """
        ncpu = get_ncpu(ncpu)
        args_list = [((), {'bkg_mean_list': bkg_mean_list,
            'sig_mean': sig_mean}) for i in range(N)]
        result_list = parallelize(self.do_trial, args_list, ncpu, rss=rss)

        result_dtype = result_list[0].dtype
        result = np.empty(N, dtype=result_dtype)
        result[:] = result_list[:]

        return result


class MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis(Analysis):
    """This analysis class implements a time-integrated analysis with a spatial
    and energy PDF for multiple datasets assuming a single source.

    To run this analysis the following procedure applies:

        1. Add the datasets and their spatial and energy PDF ratio instances
           via the :meth:`.add_dataset` method.
        2. Construct the log-likelihood ratio function via the
           :meth:`construct_llhratio` method.
        3. Initialize a trial via the :meth:`initialize_trial` method.
        4. Fit the global fit parameters to the trial data via the
           :meth:`maximize_llhratio` method.
    """
    def __init__(self, minimizer, src_hypo_group_manager, src_fitparam_mapper,
                 fitparam_ns,
                 test_statistic, bkg_gen_method, event_selection_method=None):
        """Creates a new time-integrated point-like source analysis assuming a
        single source.

        Parameters
        ----------
        minimizer : instance of Minimizer
            The Minimizer instance that should be used to minimize the negative
            of the log-likelihood ratio function.
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
        data_scrambler : instance of DataScrambler
            The instance of DataScrambler that will scramble the data for a new
            analysis trial.
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

        super(MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis, self).__init__(
            minimizer, src_hypo_group_manager, src_fitparam_mapper,
            test_statistic, bkg_gen_method, event_selection_method)

        self.fitparam_ns = fitparam_ns

        self._spatial_pdfratio_list = []
        self._energy_pdfratio_list = []

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

    def add_dataset(self, dataset, data, spatial_pdfratio, energy_pdfratio):
        """Adds a dataset with its spatial and energy PDF ratio instances to the
        analysis.
        """
        super(MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis, self).add_dataset(dataset, data)

        if(not isinstance(spatial_pdfratio, PDFRatio)):
            raise TypeError('The spatial_pdfratio argument must be an instance of PDFRatio!')
        if(not issubclass(spatial_pdfratio.pdf_type, SpatialPDF)):
            raise TypeError('The PDF type of the PDFRatio instance of argument spatial_pdfratio must be SpatialPDF!')
        self._spatial_pdfratio_list.append(spatial_pdfratio)

        if(not isinstance(energy_pdfratio, PDFRatio)):
            raise TypeError('The energy_pdfratio argument must be an instance of PDFRatio!')
        if(not issubclass(energy_pdfratio.pdf_type, EnergyPDF)):
            raise TypeError('The PDF type of the PDFRatio instance of argument energy_pdfratio must be EnergyPDF!')
        self._energy_pdfratio_list.append(energy_pdfratio)

    def construct_llhratio(self):
        """Constructs the analysis. This setups all the necessary analysis
        objects like detector signal efficiencies and dataset signal weights,
        constructs the log-likelihood ratio functions for each dataset and the
        final composite llh ratio function.
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
        for (j, (dataset, data)) in enumerate(zip(self.dataset_list, self.data_list)):
            if(len(detsigyield_implmethod_list) == 1):
                # Only one detsigyield implementation method was defined, so we
                # use it for all datasets.
                detsigyield_implmethod = detsigyield_implmethod_list[0]
            else:
                detsigyield_implmethod = detsigyield_implmethod_list[j]

            detsigyield = detsigyield_implmethod.construct_detsigyield(
                dataset, data, fluxmodel, dataset.livetime)
            detsigyield_list.append(detsigyield)

        # For multiple datasets we need a dataset signal weights instance in
        # order to distribute ns over the different datasets.
        dataset_signal_weights = SingleSourceDatasetSignalWeights(
            self._src_hypo_group_manager, self._src_fitparam_mapper, detsigyield_list)

        # Create the list of log-likelihood ratio functions, one for each
        # dataset.
        llhratio_list = []
        for (j, dataset) in enumerate(self.dataset_list):
            pdfratio_list = [
                self._spatial_pdfratio_list[j],
                self._energy_pdfratio_list[j]
            ]
            llhratio = SingleSourceTCLLHRatio(pdfratio_list, self._src_fitparam_mapper)
            llhratio_list.append(llhratio)

        # Create the final multi-dataset log-likelihood ratio function.
        self._llhratio = MultiDatasetTCLLHRatio(dataset_signal_weights, llhratio_list)

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

        # Change the source hypo group manager of the DatasetSignalWeights
        # instance of the LLH ratio function.
        self._llhratio.dataset_signal_weights.change_source_hypo_group_manager(
            self._src_hypo_group_manager)

        # Change the source hypo group manager of the PDFRatio instances of
        # each single dataset LLH ratio function.
        for llhratio in self._llhratio.llhratio_list:
            for pdfratio in llhratio.pdfratio_list:
                pdfratio.change_source_hypo_group_manager(
                    self._src_hypo_group_manager)

        # Change the source hypo group manager of the signal generator instance.
        if(self._sig_generator is not None):
            self._sig_generator.change_source_hypo_group_manager(
                self._src_hypo_group_manager)

    def initialize_trial(self, events_list):
        """This method initializes the log-likelihood ratio function with a new
        set of given trial data. This is a low-level method. For convenient
        methods see the `unblind` and `do_trial` methods.

        Parameters
        ----------
        events_list : list of numpy record ndarray
            The list of data events to use for the log-likelihood function
            evaluation. The data arrays for the datasets must be in the same
            order than the added datasets.
        """
        for (events, llhratio) in zip(events_list, self._llhratio.llhratio_list):
            n_all_events = len(events)

            # Select events that have potential to be signal. This is for
            # runtime optimization only.
            events = self._event_selection_method.select_events(events)
            n_selected_events = len(events)

            # Initialize the log-likelihood ratio function of the dataset with
            # the selected (scrambled) events.
            n_pure_bkg_events = n_all_events - n_selected_events
            llhratio.initialize_for_new_trial(events, n_pure_bkg_events)

    def maximize_llhratio(self):
        """Maximizes the log-likelihood ratio function, by minimizing its
        negative.

        Returns
        -------
        fitparamset : FitParameterSet instance
            The instance of FitParameterSet holding the global fit parameter
            definitions used in the maximization process.
        log_lambda_max : float
            The value of the log-likelihood ratio function at its maximum.
        fitparam_values : (N_fitparam+1)-shaped 1D ndarray
            The ndarray holding the global fit parameter values.
            By definition, the first element is the value of the fit parameter
            ns.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        # Define the negative llhratio function, that will get minimized.
        def func(fitparam_values):
            (f, grads) = self._llhratio.evaluate(fitparam_values)
            return (-f, -grads)

        (fitparam_values, fmin, status) = self._minimizer.minimize(
            self._fitparamset, func)
        log_lambda_max = -fmin

        return (self._fitparamset, log_lambda_max, fitparam_values, status)

