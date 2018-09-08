# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc

from skylab.core.py import issequenceof
from skylab.core.dataset import Dataset, DatasetData
from skylab.core.parameters import (
    FitParameter,
    SourceFitParameterMapper,
    SingleSourceFitParameterMapper
)
from skylab.core.pdf import SpatialPDF, EnergyPDF
from skylab.core.pdfratio import PDFRatio
from skylab.core.llhratio import (
    SingleSourceDatasetSignalWeights,
    SingleSourceTCLLHRatio,
    MultiDatasetTCLLHRatio
)
from skylab.core.scrambling import DataScramblingMethod
from skylab.core.optimize import EventSelectionMethod, AllEventSelectionMethod
from skylab.core.source_hypothesis import SourceHypoGroupManager
from skylab.core.test_statistic import TestStatistic
from skylab.core.minimizer import Minimizer
from skylab.core.scrambling import DataScrambler

class Analysis(object):
    """This is the abstract base class for all analysis classes. It contains
    common properties required by all analyses and defines the overall analysis
    interface howto set-up and run an analysis.

    To set-up and run an analysis the following procedure applies:

        1. Create an analysis instance.
        2. Add the datasets and their PDF ratio instances via the
           ``add_dataset`` method.
        3. Construct the log-likelihood ratio function via the
           ``construct_llhratio`` method.
        4. Initialize a trial via the ``initialize_trial`` method.
        5. Fit the global fit parameters to the trial data via the
           ``maximize_llhratio`` method.
        6. Calculate the test-statistic value via the
           ``calculate_test_statistic`` method.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, minimizer, src_hypo_group_manager, src_fitparam_mapper,
                 test_statistic, data_scrambler):
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
        data_scrambler : instance of DataScrambler
            The instance of DataScrambler that will scramble the data for a new
            analysis trial.
        """
        # Call the super function to allow for multiple class inheritance.
        super(Analysis, self).__init__()

        self.minimizer = minimizer
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.test_statistic = test_statistic
        self.data_scrambler = data_scrambler

        # Predefine the variable for the log-likelihood ratio function.
        self._llhratio = None

    @property
    def minimizer(self):
        """The Minimizer instance used to minimize the negative of the
        log-likelihood ratio function.
        """
        return self._minimizer
    @minimizer.setter
    def minimizer(self, minimizer):
        if(not isinstance(minimizer, Minimizer)):
            raise TypeError('The minimizer property must be an instance of Minimizer!')
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
            raise TypeError('The src_hypo_group_manager property must be an instance of SourceHypoGroupManager!')
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
            raise TypeError('The src_fitparam_mapper property must be an instance of SourceFitParameterMapper!')
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
            raise TypeError('The test_statistic property must be an instance of TestStatistic!')
        self._test_statistic = ts

    @property
    def data_scrambler(self):
        """The DataScrambler instance that implements the data scrambling.
        """
        return self._data_scrambler
    @data_scrambler.setter
    def data_scrambler(self, scrambler):
        if(not isinstance(scrambler, DataScrambler)):
            raise TypeError('The data_scrambler property must be an instance of DataScrambler!')
        self._data_scrambler = scrambler

    @property
    def llhratio(self):
        """(read-only) The log-likelihood ratio function.
        """
        if(self._llhratio is None):
            raise RuntimeError('The log-likelihood ratio function is not defined yet. Call the construct_analysis method first!')
        return self._llhratio

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

    @abc.abstractmethod
    def initialize_trial(self):
        """This method is supposed to initialize the log-likelihood ratio
        function with a new trial.
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


class IsSingleDatasetAnalysis(object):
    """This is the class classifier class to specify that an analysis is made
    for a single dataset. This class provides the ``dataset`` property.
    """
    def __init__(self):
        super(IsSingleDatasetAnalysis, self).__init__()

        self._dataset = None
        self._data = None

    @property
    def dataset(self):
        """The Dataset instance for the analysis.
        """
        return self._dataset
    @dataset.setter
    def dataset(self, ds):
        if(not isinstance(ds, Dataset)):
            raise TypeError('The dataset property must be an instance of Dataset!')
        self._dataset = ds

    @property
    def data(self):
        """The DatasetData instance holding the original data of the dataset for
        the analysis.
        """
        return self._data
    @data.setter
    def data(self, d):
        if(not isinstance(d, DatasetData)):
            raise TypeError('The data property must be an instance of DatasetData!')
        self._data = d

    @property
    def n_datasets(self):
        """(read-only) The number of used datasets in this analysis.
        """
        if(self._dataset is None):
            return 0
        return 1


class IsMultiDatasetAnalysis(object):
    """This is the class classifier class to specify that an analysis is made
    for multiple datasets. This class provides the ``dataset_list`` and
    ``data_list`` property, and the ``add_dataset`` method.
    """
    def __init__(self):
        super(IsMultiDatasetAnalysis, self).__init__()

        self._dataset_list = []
        self._data_list = []

    @property
    def dataset_list(self):
        """The list of Dataset instances.
        """
        return self._dataset_list
    @dataset_list.setter
    def dataset_list(self, datasets):
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The dataset_list property must be a sequence of Dataset instances!')
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
            raise TypeError('The data_list property must be a sequence of DatasetData instances!')
        self._data_list = list(datas)

    @property
    def n_datasets(self):
        """(read-only) The number of datasets used in this analysis.
        """
        return len(self._dataset_list)

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
            raise TypeError('The dataset argument must be an instance of Dataset!')
        if(not isinstance(data, DatasetData)):
            raise TypeError('The data argument must be an instance of DatasetData!')

        self._dataset_list.append(dataset)
        self._data_list.append(data)


class MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis(Analysis, IsMultiDatasetAnalysis):
    """This analysis class implements a time-integrated analysis with a spatial
    and energy PDF for multiple datasets assuming a single source.

    To run this analysis the following procedure applies:

        1. Add the datasets and their spatial and energy PDF ratio instances
           via the ``add_dataset`` method.
        2. Construct the log-likelihood ratio function via the
           ``construct_llhratio`` method.
        3. Initialize a trial via the ``initialize_trial`` method.
        4. Fit the global fit parameters to the trial data via the
           ``maximize_llhratio`` method.
    """
    def __init__(self, minimizer, src_hypo_group_manager, src_fitparam_mapper,
                 fitparam_ns,
                 test_statistic, data_scrambler, event_selection_method=None):
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
            raise TypeError('The src_fitparam_mapper argument must be an instance of SingleSourceFitParameterMapper!')

        super(MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis, self).__init__(
            minimizer, src_hypo_group_manager, src_fitparam_mapper,
            test_statistic, data_scrambler)

        self.fitparam_ns = fitparam_ns
        self.event_selection_method = event_selection_method

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
            raise TypeError('The event_selection_method property must be an instance of EventSelectionMethod!')
        self._event_selection_method = method

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
        """Constructs the analysis. This loads the dataset data, set-ups all the
        necessary analysis objects like
        detector signal efficiencies and dataset signal weights, constructs the
        log-likelihood ratio functions for each dataset and the final composite
        llh ratio function.
        """
        # Create the detector signal efficiency instances for each dataset.
        # Since this is for a single source, we don't have to have individual
        # detector signal efficiency instances for each source as well.
        detsigeff_list = []
        fluxmodel = self._src_hypo_group_manager.get_fluxmodel_by_src_idx(0)
        detsigeff_implmethod_list = self._src_hypo_group_manager.get_detsigeff_implmethod_list_by_src_idx(0)
        if((len(detsigeff_implmethod_list) != 1) and
           (len(detsigeff_implmethod_list) != self.n_datasets)):
            raise ValueError('The number of detector signal efficiency implementation methods is not 1 and does not match the number of used datasets in the analysis!')
        for (j, (dataset, data)) in enumerate(zip(self.dataset_list, self.data_list)):
            if(len(detsigeff_implmethod_list) == 1):
                # Only one detsigeff implementation method was defined, so we
                # use it for all datasets.
                detsigeff_implmethod = detsigeff_implmethod_list[0]
            else:
                detsigeff_implmethod = detsigeff_implmethod_list[j]

            detsigeff = detsigeff_implmethod.construct_detsigeff(
                dataset, data, fluxmodel, dataset.livetime)
            detsigeff_list.append(detsigeff)

        # For multiple datasets we need a dataset signal weights instance in
        # order to distribute ns over the different datasets.
        dataset_signal_weights = SingleSourceDatasetSignalWeights(
            self._src_hypo_group_manager, self._src_fitparam_mapper, detsigeff_list)

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

    def initialize_trial(self, scramble=True):
        """Initializes the log-likelihood functions of the different datasets
        with data from the datasets. If the scramble argument is set to True,
        the data will get scrambled before.

        Parameters
        ----------
        scramble : bool
            Flag if the data should get scrambled before it is set to the
            log-likelihood ratio functions (default True).
            Note: Depending on the inplace_scrambling setting of the
                  DataScrambler, the scrambling of the data might be inplace,
                  changing the experimental data of the dataset itself!
        """
        for (data, llhratio) in zip(self._data_list, self._llhratio.llhratio_list):
            events = data.exp
            n_all_events = len(events)

            # Scramble the data if requested.
            if(scramble):
                events = self._data_scrambler.scramble_data(events)

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

