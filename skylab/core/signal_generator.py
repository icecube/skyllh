# -*- coding: utf-8 -*-

import abc

from skylab.core.py import (
    issequence,
    issequenceof,
    float_cast,
    get_smallest_numpy_int_type
)
from skylab.core.dataset import Dataset, DatasetData
from skylab.core.source_hypothesis import SourceHypoGroupManager

class SignalGenerationMethod(object):
    """This is a base class for a source and detector specific signal generation
    method, that calculates the source flux for a given monte-carlo event, which
    is needed to calculate the MC event weights for the signal injector.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, energy_range):
        """Constructs a new signal generation method instance.

        Parameters
        ----------
        energy_range : 2-element tuple of float | None
            The energy range from which to take MC events into account for
            signal event generation.
            If set to None, the entire energy range [0, +inf] is used.
        """
        super(SignalGenerationMethod, self).__init__()

        self.energy_range = energy_range

    @property
    def energy_range(self):
        """The 2-element tuple of floats holding the energy range from which to
        take MC events into account for signal event generation.
        """
        return self._energy_range
    @energy_range.setter
    def energy_range(self, r):
        if(not issequence(r)):
            raise TypeError('The energy_range property must be a sequence!')
        if(len(r) != 2):
            raise ValueError('The energy_range property must be a sequence of '
                '2 elements!')
        r = tuple(
            float_cast(r[0], 'The first element of the energy_range '
                             'sequence must be castable to type float!'),
            float_cast(r[1], 'The second element of the energy_range '
                             'sequence must be castable to type float!')
        )
        self._energy_range = r

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

    def signal_event_post_sampling_processing(self, signal_events, src_hypo_group):
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


class SignalGenerator(object):
    """This is the general signal generator class. It does not depend on the
    detector or source hypothesis, because these dependencies are factored out
    into the signal generation method.
    """
    def __init__(self, src_hypo_group_manager, dataset_list, data_list):
        """Constructs a new signal generator instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        dataset_list : list of Dataset instances
            The list of Dataset instances for which signal events should get
            generated for.
        data_list : list of DatasetData instances
            The list of DatasetData instances holding the actual data of each
            dataset. The order must match the order of ``dataset_list``.
        """
        super(SignalGenerator, self).__init__()

        self.src_hypo_group_manager = src_hypo_group_manager

        self.dataset_list = dataset_list
        self.data_list = data_list

        # Construct an array holding pointer information of signal candidate events
        # pointing into the real MC dataset(s).
        n_datasets = len(self._dataset_list)
        n_sources = self._src_hypo_group_manager.n_sources
        shg_list = self._src_hypo_group_manager.src_hypo_group_list
        mc_ptr_dtype = [
            ('ds_idx', get_smallest_numpy_int_type((0, n_datasets))),
            ('shg_idx', get_smallest_numpy_int_type((0, n_sources))),
            ('shg_src_idx', get_smallest_numpy_int_type(
                [0]+[shg.n_sources for shg in shg_list]))
        ]
        self._mc_ptr = np.empty((0,), dtype=mc_ptr_dtype)


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

    @property
    def dataset_list(self):
        """The list of Dataset instances for which signal events should get
        generated for.
        """
        return self._dataset_list
    @dataset_list.setter
    def dataset_list(self, datasets):
        if(not issequenceof(datasets, Dataset)):
            raise TypeError('The dataset_list property must be a sequence of '
                'Dataset instances!')
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
        if(not issequenceof(datas, DatasetData)):
            raise TypeError('The data_list property must be a sequence of '
                'DatasetData instances!')
        self._data_list = datas

    @abc.abstractmethod
    def sample(self, mean_signal, poisson=True):
        """This method is supposed to sample monte-carlo events coming from the
        defined sources.
        """
        pass


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
