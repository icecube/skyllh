# -*- coding: utf-8 -*-

"""The source_hypothesis module provides classes to define groups of source
hypotheses. The SourceHypoGroupManager manages the groups of source hypotheses.
"""

import numpy as np

from skyllh.core.parameters import make_params_hash
from skyllh.core.py import issequenceof
from skyllh.core.source_hypo_group import SourceHypoGroup


class SourceHypoGroupManager(object):
    """The source hypothesis group manager provides the functionality to group
    sources of the same source hypothesis, i.e. spatial model and flux model,
    with an assign detector signal efficiency implementation method.

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
        self._sidx_to_gidx_gsidx_map_arr = np.empty((0,2), dtype=np.int64)

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
        arr = np.empty((shg.n_sources,2), dtype=np.int64)
        arr[:,0] = self.n_src_hypo_groups-1 # Group index.
        arr[:,1] = np.arange(shg.n_sources) # Group source index.
        self._sidx_to_gidx_gsidx_map_arr = np.vstack(
            (self._sidx_to_gidx_gsidx_map_arr, arr))

    def add_source_hypo_group(
        self, sources, fluxmodel, detsigyield_implmethods, sig_gen_method=None
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
        """
        # Create the source group.
        group = SourceHypoGroup(sources, fluxmodel, detsigyield_implmethods, sig_gen_method)

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
