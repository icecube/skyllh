# -*- coding: utf-8 -*-

import abc

from skylab.core.dataset import Dataset, DatasetData
from skylab.core.source_hypothesis import SourceHypoGroupManager

class SignalGenerationMethod(object):
    """This is a base class for a source and detector specific signal generation
    method, that calculates the source flux for a given monte-carlo event, which
    is needed to calculate the MC event weights for the signal injector.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Constructs a new signal generation method instance.
        """
        super(SignalGenerationMethod, self).__init__()

    @abc.abstractmethod
    def calc_source_signal_mc_event_flux(self, data_mc, src_hypo_group):
        """This method is supposed to calculate the signal flux of each given
        MC event for each source hypothesis of the given source hypothesis
        group.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The numpy record array holding all the MC events.
        src_hypo_group : SourceHypoGroup instance
            The source hypothesis group, which defines the list of sources, and
            their flux model.

        Returns
        -------
        flux_list : list of 2-element tuples
            The list of 2-element tuples with one tuple for each source. Each
            tuple must be made of two 1D ndarrays of size
            N_selected_signal_events, where the first array contains the global
            MC data event indices and the second array the flux of each selected
            signal event.

        """
        pass

    def signal_event_post_sampling_processing(self, signal_events):
        """This method should be reimplemented by the derived class if there
        is some processing needed after the MC signal events have been sampled
        from the global MC data.

        Parameters
        ----------
        signal_events : numpy record array
            The numpy record array holding the MC signal events in the same
            format as the original MC events.

        Returns
        -------
        signal_events : numpy record array
            The processed signal events. In the default implementation of this
            method this is just the signal_events input array.
        """
        return signal_events


class SignalInjector(object):
    """Abstract base class for a signal injector.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, src_hypo_group_manager):
        """Constructs a new signal injector instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        """
        super(SignalInjector, self).__init__()

        self.src_hypo_group_manager = src_hypo_group_manager

        self._dataset_list = list()
        self._data_list = list()

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance defining the source groups with
        their spectra.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                            'instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    def add_dataset(self, ds, data):
        """Adds the given dataset and its data to the signal injector.
        """
        if(not isinstance(ds, Dataset)):
            raise TypeError('The ds argument must be an instance of Dataset!')
        if(not isinstance(self, DatasetData)):
            raise TypeError('The data argument must be an instance of DatasetData!')
        self._dataset_list.append(ds)
        self._data_list.append(data)

    @abc.abstractmethod
    def construct(self):
        """This method is supposed to construct the signal sampler.
        """
        pass

    @abc.abstractmethod
    def sample(self, mean_signal, poisson=True):
        """This method is supposed to sample monte-carlo events coming from the
        defined sources.
        """


class SingleSourceSignalInjector(SignalInjector):
    """Base class for a single source signal injector class.
    """
    def __init__(self, src_hypo_group_manager):
        """Constructs a new signal injector instance for a single source.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        """
        super(SingleSourceSignalInjector, self).__init__(src_hypo_group_manager)


class MultiSourceSignalInjector(SignalInjector):
    """Base class for a multiple source (stacking) signal injector class.
    """
    def __init__(self, src_hypo_group_manager):
        """Constructs a new signal injector instance for multiple sources.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        """
        super(MultiSourceSignalInjector, self).__init__(src_hypo_group_manager)
