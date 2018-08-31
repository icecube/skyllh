# -*- coding: utf-8 -*-

"""The analysis module provides classes for pre-defined analyses.
"""

import abc

from skylab.core.py import issequenceof
from skylab.core.dataset import Dataset
from skylab.core.parameters import SourceFitParameterMapper, SingleSourceFitParameterMapper
from skylab.core.pdf import SpatialPDF, EnergyPDF
from skylab.core.llhratio import SingleSourceDatasetSignalWeights
from skylab.core.scrambling import DataScramblingMethod
from skylab.core.optimize import EventSelectionMethod, AllEventSelectionMethod
from skylab.core.source_hypothesis import SourceHypoGroupManager


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
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, minimizer, src_hypo_group_manager, src_fitparam_mapper):
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
        """
        # Call the super function to allow for multiple class inheritance.
        super(Analysis, self).__init__()

        self.minimizer = minimizer
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper

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
    def llhratio(self):
        """(read-only) The log-likelihood ratio function.
        """
        if(self._llhratio is None):
            raise RuntimeError('The log-likelihood ratio function is not defined yet. Call the construct_analysis method first!')
        return self._llhratio

    @abc.abstractmethod
    def add_dataset(self, dataset, *args, **kwars):
        """This method is supposed to add a dataset and the corresponding PDF
        ratio instances to the analysis.
        """
        pass

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


class IsMultiDatasetAnalysis(object):
    """This is the class classifier class to specify that an analysis is made
    for multiple datasets. This class provides the ``dataset_list`` property
    and the ``add_dataset`` method.
    """
    def __init__(self):
        super(IsMultiDatasetAnalysis, self).__init__()

        self._dataset_list = []

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

    def add_dataset(self, dataset):
        """Adds the given dataset to the list of datasets for this analysis.

        Parameters
        ----------
        dataset : Dataset
            The Dataset instance that should get added.
        """
        if(not isinstance(dataset, Dataset)):
            raise TypeError('The dataset argument must be an instance of Dataset!')
        self._dataset_list.append(dataset)


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
                 data_scrambler, event_selection_method=None):
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
            minimizer, src_hypo_group_manager, src_fitparam_mapper)

        self.data_scrambler = data_scrambler
        self.event_selection_method = event_selection_method

        self._llhratio = None
        self._spatial_pdfratio_list = []
        self._energy_pdfratio_list = []

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

    def add_dataset(self, dataset, spatial_pdfratio, energy_pdfratio):
        """Adds a dataset with its spatial and energy PDF ratio instances to the
        analysis.
        """
        super(MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis, self).add_dataset(dataset)

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
        # Load the data of each dataset.
        for dataset in self.dataset_list:
            dataset.load_and_prepare_data()

        # Create the detector signal efficiency instances for each dataset.
        # Since this is for a single source, we don't have to have individual
        # detector signal efficiency instances for each source as well.
        detsigeff_list = []
        fluxmodel = self._src_hypo_group_manager.get_fluxmodel_by_src_idx(0)
        detsigeff_implmethod = self._src_hypo_group_manager.get_detsigeff_implmethod_by_src_idx(0)
        for dataset in self.dataset_list:
            detsigeff = DetectorSignalEfficiency(
                dataset.data_mc,
                fluxmodel,
                dataset.livetime,
                detsigeff_implmethod)
            detsigeff_list.append(detsigeff)

        # For multiple datasets we need a dataset signal weights instance in
        # order to distribute ns over the different datasets.
        dataset_signal_weights = SingleSourceDatasetSignalWeights(
            self._source, self._src_fitparam_mapper, detsigeff_list)

        # Create the list of log-likelihood ratio functions, one for each
        # dataset.
        llhratio_list = []
        for (j, dataset) in enumerate(self.dataset_list):
            pdfratio_list = [self._spatial_pdfratio_list[j], self._energy_pdfratio_list[j]]
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
        for (dataset, llhratio) in zip(self._dataset_list, self._llhratio.llhratio_list):
            events = dataset.data_exp
            n_all_events = len(events)

            # Scramble the data if requested.
            if(scramble):
                events = self._data_scrambler.scramble_data(events)

            # Select events that have potential to be signal. This is for
            # runtime optimization only.
            events = self._event_selection_method(events)
            n_selected_events = len(events)

            # Initialize the log-likelihood ratio function of the dataset with
            # the selected (scrambled) events.
            n_pure_bkg_events = n_all_events - n_selected_events
            llhratio.initialize_for_new_trial(events, n_pure_bkg_events)

    def maximize_llhratio(self, ns_fitparam):
        """Maximizes the log-likelihood ratio function, by minimizing its
        negative.

        Parameters
        ----------
        ns_fitparam : instance of FitParameter
            The instance of FitParameter for the fit parameter ``ns``.

        Returns
        -------
        fitparam_dict : dict
            The dictionary holding the global fit parameter name and value of
            the log-likelihood ratio function maximum.
        fmax : float
            The value of the log-likelihood ratio function at its maximum.
        status : dict
            The dictionary with status information about the maximization
            process, i.e. from the minimizer.
        """
        # Define the negative llhratio function, that will be minimized.
        def func(fitparam_values):
            (f, grads) = self._llhratio.evaluate(fitparam_values)
            return (-f, -grads)

        # Get the fit parameter set and add the ns fit parameter at the front.
        fitparamset = self._src_fitparam_mapper.fitparamset
        fitparamset.add_fitparam(ns_fitparam, atfront=True)

        (xmin, fmin, status) = self._minimizer.minimize(fitparamset, func)
        fmax = -fmin

        # Convert the fit parameter values into a dictionary with their names.
        fitparam_dict = fitparamset.get_fitparam_dict(xmin)

        return (fitparam_dict, fmax, status)

    def calculate_test_statistic(self, ns, log_lambda):
        """Calculates the test statistic value from the given log_lambda value.

        The test statistic for this analysis is defined as:

            TS = 2 * sgn(ns) * log_lambda

        Parameters
        ----------
        ns : float | array of float
            The best fit ns value.
        log_lambda : float | array of float
            The best fit log_lambda value.
        """
        ns = np.atleast_1d(ns)

        # We need to distinguish between ns=0 and ns!=0, because the np.sign(ns)
        # function returns 0 for ns=0, but we want it to be 1 in such cases.
        sgn_ns = np.where(ns == 0, 1., np.sign(ns))

        TS = 2 * sgn_ns * log_lambda

        return TS
