# -*- coding: utf-8 -*-

"""The analysis module of skylab provides classes for pre-defined analyses.
"""

import abc

from skylab.core.py import issequenceof
from skylab.core.dataset import Dataset
from skylab.core.parameters import SourceFitParameterMapper, SingleSourceFitParameterMapper
from skylab.core.detsigeff import DetSigEffImplMethod
from skylab.core.pdf import SpatialPDF, EnergyPDF
from skylab.core.llhratio import SingleSourceDatasetSignalWeights
from skylab.core.scrambling import DataScramblingMethod
from skylab.core.optimize import EventSelectionMethod, AllEventSelectionMethod
from skylab.physics.source import SourceModel
from skylab.physics.flux import FluxModel


class SourceHypoGroup(object):
    """The source hypothesis group class provides a data container to describe
    a group of sources that share the same flux model and detector signal
    efficiency implementation method.
    """
    def __init__(self, sources, fluxmodel, detsigeff_implmethod):
        """Constructs a new source hypothesis group.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigeff_implmethod : instance of DetSigEffImplMethod
            The instance of a detector signal efficiency implementation method,
            which should be used to create the detector signal efficiency for
            the sources of the group.
        """
        self.source_list = sources
        self.fluxmodel = fluxmodel
        self.detsigeff_implmethod = detsigeff_implmethod

    @property
    def source_list(self):
        """The list of SourceModel instances for which the group is defined.
        """
        return self._source_list
    @source_list.setter
    def source_list(self, sources):
        if(isinstance(sources, SourceModel)):
            sources = [sources]
        if(not issequenceof(sources, SourceModel)):
            raise TypeError('The source_list property must be an instance of SourceModel or a sequence of SourceModel instances!')
        self._source_list = list(sources)

    @property
    def fluxmodel(self):
        """The FluxModel instance that applies to the list of sources of this
        source group.
        """
        return self._fluxmodel
    @fluxmodel.setter
    def fluxmodel(self, fluxmodel):
        if(not isinstance(fluxmodel, FluxModel)):
            raise TypeError('The fluxmodel property must be an instance of FluxModel!')
        self._fluxmodel = fluxmodel

    @property
    def detsigeff_implmethod(self):
        """The instance of DetSigEffImplMethod, the detector signal efficiency
        implementation method, which should be used to create the detector
        signal efficiency for this group of sources.
        """
        return self._detsigeff_implmethod
    @detsigeff_implmethod.setter
    def detsigeff_implmethod(self, method):
        if(not isinstance(method, DetSigEffImplMethod)):
            raise TypeError('The detsigeff_implmethod property must be an instance of DetSigEffImplMethod!')
        self._detsigeff_implmethod = method

    @property
    def N_sources(self):
        """(read-only) The number of sources within this source group.
        """
        return len(self._source_list)


class SourceHypoManager(object):
    """The source hypothesis manager provides the functionality to manage
    several sources with their flux models and corresponding detector signal
    efficiencies in an efficient way for the evaluation of the likelihood
    function.

    Groups of sources can be defined that share the same flux model and hence
    the same detector signal efficiency implementation method.
    """
    def __init__(self):
        self._src_hypo_group_list = list()
        # Define a 2D numpy array of shape (N_sources,2) that maps the source
        # index (0 to N_sources-1) to the index of the group and the source
        # index within the group for fast access.
        self._sidx_to_gidx_gsidx_map_arr = np.empty((0,2), dtype=np.int)

    @property
    def source_list(self):
        """The list of defined SourceModel instances.
        """
        source_list = []
        for group in self._src_hypo_group_list:
            source_list += group.source_list
        return source_list

    @property
    def N_sources(self):
        """(read-only) The total number of sources defined in all source groups.
        """
        return self._sidx_to_gidx_gsidx_map_arr.shape[0]

    @property
    def N_src_groups(self):
        """The number of defined source groups.
        """
        return len(self._src_hypo_group_list)

    def add_source_group(self, sources, fluxmodel, detsigeffmethod):
        """Adds a group of sources to the source hypothesis manager. A group of
        sources share the same flux model and the same detector signal
        efficiency implementation method.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigeffmethod : instance of DetSigEffImplMethod
            The instance of a detector signal efficiency implementation method,
            which should be used to create the detector signal efficiency for
            the sources of the group.
        """
        # Create the source group.
        group = SourceHypoGroup(sources, fluxmodel, detsigeffmethod)

        # Add the group.
        self._src_hypo_group_list.append(group)

        # Extend the source index to (group index, group source index) map
        # array.
        arr = np.empty((group.N_sources,2))
        arr[:,0] = self.N_src_groups-1 # Group index.
        arr[:,1] = np.arange(group.N_sources) # Group source index.
        self._sidx_to_gidx_gsidx_map_arr = np.vstack(
            (self._sidx_to_gidx_gsidx_map_arr, arr))

    def get_fluxmodel_by_src_idx(self, src_idx):
        """Retrieves the FluxModel instance for the source specified by its
        source index.

        Parameters
        ----------
        src_idx : int
            The index of the source, which must be in the range
            [0, N_sources-1].

        Returns
        -------
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the specified source.
        """
        gidx = self._sidx_to_gidx_gsidx_map_arr[src_idx,0]
        return self._src_hypo_group_list[gidx]._fluxmodel

    def get_detsigeff_implmethod_by_src_idx(self, src_idx):
        """Retrieves the DetSigEffImplMethod instance for the source specified
        by its source index.

        Parameters
        ----------
        src_idx : int
            The index of the source, which must be in the range
            [0, N_sources-1].

        Returns
        -------
        detsigeff_implmethod : instance of DetSigEffImplMethod
            The DetSigEffImplMethod instance that applies to the specified
            source.
        """
        gidx = self._sidx_to_gidx_gsidx_map_arr[src_idx,0]
        return self._src_hypo_group_list[gidx]._detsigeff_implmethod


class Analysis(object):
    """This is the abstract base class for all analysis classes. It contains
    common properties required by all analyses and defines the overall analysis
    interface howto set-up and run an analysis.

    The procedure to construct and run an analysis is to create an Analysis
    instance, define the global fit parameters via the source fit parameter
    mapper, construct the analysis via the ``construct_analysis`` method,
    initialize a trial via the ``initialize_trial`` method, and
    then maximize the likelihood function via the ``maximize_llhratio`` method.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, src_fitparam_mapper):
        """Constructor of the analysis base class.

        Parameters
        ----------
        src_fitparam_mapper : instance of SourceFitParameterMapper
            The SourceFitParameterMapper instance managing the global fit
            parameters and their relation to the individual sources.
        """
        super(Analysis, self).__init__()

        self.src_fitparam_mapper = src_fitparam_mapper

        # Predefine the variable for the log-likelihood ratio function.
        self._llhratio = None

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
    def construct_analysis(self):
        """This method is supposed to construct the analysis. This means needs
        to create the log-likelihood ratio function and sets it as the _llhratio
        property.
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


class MultiDatasetAnalysis(Analysis):
    """This is the base class for all analyses that use multiple datasets.
    """
    def __init__(self, src_fitparam_mapper):
        """Constructor of the Analysis base class.

        Parameters
        ----------
        src_fitparam_mapper : instance of SourceFitParameterMapper
            The SourceFitParameterMapper instance managing the global fit
            parameters and their relation to the individual sources.
        datasets : sequence of Dataset
            The sequence of Dataset instances which define the used data. Each
            dataset is treated as an independent set of data, which means for
            each dataset PDF objects need to be created.
        """
        super(MultiDatasetAnalysis, self).__init__(src_fitparam_mapper)

        self._dataset_list = []

    @property
    def dataset_list(self):
        """The list of Dataset instances.
        """
        return self._dataset_list

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


class MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis(MultiDatasetAnalysis):
    """This analysis class implements a time-integrated analysis with a spatial
    and energy PDF for multiple datasets assuming a single source.

    To run this analysis the following procedure applies:

        1. Define the global fit parameters via the SourceFitParameterMapper
           instance accessible via the ``src_fitparam_mapper`` property.
        2. Add the datasets and their spatial and energy PDF ratio instances
           via the ``add_dataset`` method.
        3. Construct the analysis via the ``construct_analysis`` method.

    First the global fit parameters needs to be defined via
    First the datasets have to be added via the ``add_dataset`` method. Then the
    analysis needs to be constructed via the ``construct_analysis`` method.
    """
    def __init__(self, src_hypo_manager, data_scrambler, event_selection_method=None):
        """Creates a new time-integrated point-like source analysis assuming a
        single source.

        Parameters
        ----------
        src_hypo_manager : instance of SourceHypoManager
            The instance of SourceHypoManager, which defines the sources, their
            flux models, and their detector signal efficiency implementation
            methods.
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
        super(MultiDatasetTimeIntegratedSpacialEnergySingleSourceAnalysis, self).__init__(
            SingleSourceFitParameterMapper())

        self.src_hypo_manager = src_hypo_manager
        self.data_scrambler = data_scrambler
        self.event_selection_method = event_selection_method

        self._llhratio = None
        self._spatial_pdfratio_list = []
        self._energy_pdfratio_list = []

    @property
    def src_hypo_manager(self):
        """The SourceHypoManager instance, which defines the sources, their
        flux models, and their detector signal efficiency implementation
        methods.
        """
        return self._src_hypo_manager
    @src_hypo_manager.setter
    def src_hypo_manager(self, manager):
        if(not isinstance(manager, SourceHypoManager)):
            raise TypeError('The src_hypo_manager property must be an instance of SourceHypoManager!')
        self._src_hypo_manager = manager

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

    def construct_analysis(self):
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
        fluxmodel = self._src_hypo_manager.get_fluxmodel_by_src_idx(0)
        detsigeff_implmethod = self._src_hypo_manager.get_detsigeff_implmethod_by_src_idx(0)
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

        self._llhratio = MultiDatasetTCLLHRatio(dataset_signal_weights)

        # Add the log-likelihood functions for each dataset.
        for (j, dataset) in enumerate(self.dataset_list):
            pdfratio_list = [self._spatial_pdfratio_list[j], self._energy_pdfratio_list[j]]
            llhratio = SingleSourceTCLLHRatio(pdfratio_list, self._src_fitparam_mapper)
            self._llhratio.add_llhratio(llhratio)

    def initialize_trial(self, scramble=True):
        """Initializes the log-likelihood functions of the different datasets
        with data from the datasets. If the scramble argument is set to True,
        the data will get scrambled before.

        Parameters
        ----------
        scramble : bool
            Flag if the data should get scrambled before it is set to the
            log-likelihood ratio functions.
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

    def maximize_llhratio(self):
        pass
