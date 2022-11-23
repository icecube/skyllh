# -*- coding: utf-8 -*-

"""This module contains classes for source hypothesis grouping functionalities.
Same kind sources can be grouped to allow more efficient calculations in the
analysis.
"""

import numpy as np

from skyllh.core.parameters import (
    make_params_hash,
)
from skyllh.core.py import (
    issequenceof,
)
from skyllh.core.detsigyield import (
    DetSigYieldImplMethod,
)
from skyllh.core.signal_generation import (
    SignalGenerationMethod,
)
from skyllh.physics.source import (
    SourceModel,
)
from skyllh.physics.flux_model import (
    FluxModel,
)

class SourceHypoGroup(object):
    """The source hypothesis group class provides a data container to describe
    a group of sources that share the same flux model, detector signal yield,
    and signal generation implementation methods.
    """
    def __init__(
            self, sources, fluxmodel, detsigyield_implmethods,
            sig_gen_method=None, source_weights=None):
        """Constructs a new source hypothesis group.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigyield_implmethods : sequence of DetSigYieldImplMethod instances
            The sequence of detector signal yield implementation method
            instances, which should be used to create the detector signal
            yield for the sources of this group. Each element is the
            detector signal yield implementation method for the particular
            dataset, if several datasets are used. If this list contains only
            one implementation method, it should be used for all datasets.
        sig_gen_method : SignalGenerationMethod instance | None
            The instance of SignalGenerationMethod that implements the signal
            generation for the specific detector and source hypothesis. It can
            be set to None, which means, no signal can be generated. Useful for
            data unblinding and data trial generation, where no signal is
            required.
        source_weights : float | sequence of floats | None
            The sequence of relative source weights, normalized to 1.
        """
        self.source_list = sources
        self.fluxmodel = fluxmodel
        self.detsigyield_implmethod_list = detsigyield_implmethods
        self.sig_gen_method = sig_gen_method
        self.source_weights = source_weights

    @property
    def source_list(self):
        """The list of SourceModel instances for which the group is defined.
        """
        return self._source_list
    @source_list.setter
    def source_list(self, sources):
        if(isinstance(sources, SourceModel)):
            sources = [ sources ]
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
            raise TypeError('The fluxmodel property must be an instance of '
                'FluxModel!')
        self._fluxmodel = fluxmodel

    @property
    def detsigyield_implmethod_list(self):
        """The list of DetSigYieldImplMethod instances, which should be used to
        create the detector signal yield for this group of sources. Each
        element is the detector signal yield implementation method for
        the particular dataset, if several datasets are used. If this list
        contains only one implementation method, it should be used for all
        datasets.
        """
        return self._detsigyield_implmethod_list
    @detsigyield_implmethod_list.setter
    def detsigyield_implmethod_list(self, methods):
        if(isinstance(methods, DetSigYieldImplMethod)):
            methods = [ methods ]
        if(not issequenceof(methods, DetSigYieldImplMethod)):
            raise TypeError('The detsigyield_implmethod_list property must be '
                'a sequence of DetSigYieldImplMethod instances!')
        self._detsigyield_implmethod_list = methods

    @property
    def sig_gen_method(self):
        """The instance of SignalGenerationMethod that implements the signal
        generation for the specific detector and source hypothesis. It can
        be None, which means, no signal can be generated. Useful for
        data unblinding and data trial generation, where no signal is
        required.
        """
        return self._sig_gen_method
    @sig_gen_method.setter
    def sig_gen_method(self, method):
        if(method is not None):
            if(not isinstance(method, SignalGenerationMethod)):
                raise TypeError('The sig_gen_method property must be an '
                    'instance of SignalGenerationMethod!')
        self._sig_gen_method = method

    @property
    def source_weights(self):
        """The 1d array of relative source weights.
        """
        return self._source_weights
    @source_weights.setter
    def source_weights(self, source_weights):
        if(source_weights is None):
            self._source_weights = source_weights
        else:
            if(isinstance(source_weights, (int, float))):
                source_weights = [source_weights]
            if(not issequenceof(source_weights, (int, float))):
                raise TypeError(
                    'The source_weights property must be a sequence of floats!')
            if not(1.0 - 1e-3 <= np.sum(source_weights) <= 1.0 + 1e-3):
                raise ValueError(
                    'The sum of source_weights has to be equal to 1!')
            self._source_weights = np.array(source_weights)

    @property
    def n_sources(self):
        """(read-only) The number of sources within this source hypothesis
        group.
        """
        return len(self._source_list)


class SourceHypoGroupManager(object):
    """The source hypothesis group manager provides the functionality to group
    sources of the same source hypothesis, i.e. spatial model and flux model,
    with an assigned detector signal yield implementation method.

    This helps to evaluate the log-likelihood ratio function in an efficient
    way.
    """
    def __init__(self, src_hypo_groups=None):
        """Creates a new source hypothesis group manager instance.

        Parameters
        ----------
        src_hypo_groups : SourceHypoGroup instance |
                          sequence of SourceHypoGroup instances | None
            The SourceHypoGroup instances to initialize the manager with.
        """
        super(SourceHypoGroupManager, self).__init__()

        self._src_hypo_group_list = list()
        # Define a 2D numpy array of shape (N_sources,2) that maps the source
        # index (0 to N_sources-1) to the index of the group and the source
        # index within the group for fast access.
        self._sidx_to_gidx_gsidx_map_arr = np.empty((0,2), dtype=np.int)

        # Add source hypo groups if specified.
        if(src_hypo_groups is not None):
            if(isinstance(src_hypo_groups, SourceHypoGroup)):
                src_hypo_groups = [ src_hypo_groups ]
            if(not issequenceof(src_hypo_groups, SourceHypoGroup)):
                raise TypeError('The src_hypo_groups argument must be an '
                    'instance of SourceHypoGroup, or a sequence of '
                    'SourceHypoGroup instances!')
            for shg in src_hypo_groups:
                self._src_hypo_group_list.append(shg)
                self._extend_sidx_to_gidx_gsidx_map_arr(shg)

    @property
    def source_list(self):
        """The list of defined SourceModel instances.
        """
        source_list = []
        for group in self._src_hypo_group_list:
            source_list += group.source_list
        return source_list

    @property
    def n_sources(self):
        """(read-only) The total number of sources defined in all source groups.
        """
        return self._sidx_to_gidx_gsidx_map_arr.shape[0]

    @property
    def n_src_hypo_groups(self):
        """The number of defined source hypothesis groups.
        """
        return len(self._src_hypo_group_list)

    @property
    def src_hypo_group_list(self):
        """(read-only) The list of source hypothesis groups, i.e.
        SourceHypoGroup instances.
        """
        return self._src_hypo_group_list

    def _extend_sidx_to_gidx_gsidx_map_arr(self, shg):
        """Extends the source index to (group index, group source index) map
        array by one source hypo group.

        Parameters
        ----------
        shg : SourceHypoGroup instance
            The SourceHypoGroup instance for which the map array should get
            extented.
        """
        arr = np.empty((shg.n_sources,2), dtype=np.int)
        arr[:,0] = self.n_src_hypo_groups-1 # Group index.
        arr[:,1] = np.arange(shg.n_sources) # Group source index.
        self._sidx_to_gidx_gsidx_map_arr = np.vstack(
            (self._sidx_to_gidx_gsidx_map_arr, arr))

    def add_source_hypo_group(
        self, sources, fluxmodel, detsigyield_implmethods, sig_gen_method=None,
        source_weights=None
    ):
        """Adds a source hypothesis group to the source hypothesis group
        manager. A source hypothesis group share sources of the same source
        model with the same flux model and hence the same detector signal
        yield and signal generation implementation methods.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigyield_implmethods : sequence of DetSigYieldImplMethod instances
            The sequence of detector signal yield implementation method
            instances, which should be used to create the detector signal
            yield for the sources of the group. Each element is the
            detector signal yield implementation method for the particular
            dataset, if several datasets are used. If this list contains only
            one implementation method, it should be used for all datasets.
        sig_gen_method : instance of SignalGenerationMethod | None
            The SignalGenerationMethod instance that implements the detector
            and source hypothesis specific signal generation.
            It can be set to None which means no signal can be generated.
        source_weights : float | sequence of floats | None
            The sequence of relative source weights, normalized to 1.
        """
        # Create the source group.
        group = SourceHypoGroup(
            sources=sources,
            fluxmodel=fluxmodel,
            detsigyield_implmethods=detsigyield_implmethods,
            sig_gen_method=sig_gen_method,
            source_weights=source_weights)

        # Add the group.
        self._src_hypo_group_list.append(group)

        # Extend the source index to (group index, group source index) map
        # array.
        self._extend_sidx_to_gidx_gsidx_map_arr(group)

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

    def get_detsigyield_implmethod_list_by_src_idx(self, src_idx):
        """Retrieves the list of DetSigYieldImplMethod instances for the source
        specified by its source index.

        Parameters
        ----------
        src_idx : int
            The index of the source, which must be in the range
            [0, N_sources-1].

        Returns
        -------
        detsigyield_implmethod_list : list of DetSigYieldImplMethod instances
            The list of DetSigYieldImplMethod instances that apply to the
            specified source.
        """
        gidx = self._sidx_to_gidx_gsidx_map_arr[src_idx,0]
        return self._src_hypo_group_list[gidx]._detsigyield_implmethod_list

    def get_fluxmodel_to_source_mapping(self):
        """Returns the list of tuples mapping fluxmodel to the source indices.

        Returns
        -------
        fluxmodel_to_source_mapping : list of (hash, src_index_array) tuples
            The list that maps hash of the source hypothesis fluxmodel to
            the corresponding source indices array in the source hypothesis
            group.
        """
        fluxmodel_to_source_mapping = []
        n_sources_offset = 0
        for shg in self._src_hypo_group_list:
            # Mapping tuple.
            fluxmodel_to_source_mapping.append(
                (
                    make_params_hash({'fluxmodel': shg.fluxmodel}),
                    n_sources_offset + np.arange(shg.n_sources)
                )
            )
            n_sources_offset += shg.n_sources

        return fluxmodel_to_source_mapping
