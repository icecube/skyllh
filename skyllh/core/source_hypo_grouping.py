# -*- coding: utf-8 -*-

"""This module contains classes for source hypothesis grouping functionalities.
Same kind sources can be grouped to allow more efficient calculations in the
analysis.
"""

import numpy as np

from skyllh.core.display import (
    add_leading_text_line_padding,
    INDENTATION_WIDTH,
)
from skyllh.core.py import (
    classname,
    issequenceof,
)
from skyllh.core.detsigyield import (
    DetSigYieldBuilder,
)
from skyllh.core.flux_model import (
    FluxModel,
)
from skyllh.core.signal_generation import (
    SignalGenerationMethod,
)
from skyllh.core.source_model import (
    SourceModel,
)
from skyllh.core.types import (
    SourceHypoGroup_t,
)


class SourceHypoGroup(
        SourceHypoGroup_t):
    """The source hypothesis group class provides a data container to describe
    a group of sources that share the same flux model, detector signal yield,
    and signal generation methods.
    """

    def __init__(
            self,
            sources,
            fluxmodel,
            detsigyield_builders,
            sig_gen_method=None,
            **kwargs):
        """Constructs a new source hypothesis group.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigyield_builders : sequence of DetSigYieldBuilder instances
            The sequence of detector signal yield builder instances,
            which should be used to create the detector signal
            yield for the sources of this group. Each element is the
            detector signal yield builder for the particular dataset, if
            several datasets are used. If this list contains only one builder,
            it should be used for all datasets.
        sig_gen_method : SignalGenerationMethod instance | None
            The instance of SignalGenerationMethod that implements the signal
            generation for the specific detector and source hypothesis. It can
            be set to None, which means, no signal can be generated. Useful for
            data unblinding and data trial generation, where no signal is
            required.
        """
        self.source_list = sources
        self.fluxmodel = fluxmodel
        self.detsigyield_builder_list = detsigyield_builders
        self.sig_gen_method = sig_gen_method

    @property
    def source_list(self):
        """The list of SourceModel instances for which the group is defined.
        """
        return self._source_list

    @source_list.setter
    def source_list(self, sources):
        if isinstance(sources, SourceModel):
            sources = [sources]
        if not issequenceof(sources, SourceModel):
            raise TypeError(
                'The source_list property must be an instance of SourceModel '
                'or a sequence of SourceModel instances! '
                f'Its current type is {classname(sources)}.')
        self._source_list = list(sources)

    @property
    def fluxmodel(self):
        """The FluxModel instance that applies to the list of sources of this
        source group.
        """
        return self._fluxmodel

    @fluxmodel.setter
    def fluxmodel(self, fluxmodel):
        if not isinstance(fluxmodel, FluxModel):
            raise TypeError(
                'The fluxmodel property must be an instance of FluxModel! '
                f'Its current type is {classname(fluxmodel)}.')
        self._fluxmodel = fluxmodel

    @property
    def detsigyield_builder_list(self):
        """The list of DetSigYieldBuilder instances, which should be used to
        create the detector signal yield for this group of sources. Each
        element is the detector signal yield builder for the particular dataset,
        if several datasets are used. If this list contains only one builder,
        it should be used for all datasets.
        """
        return self._detsigyield_builder_list

    @detsigyield_builder_list.setter
    def detsigyield_builder_list(self, builders):
        if isinstance(builders, DetSigYieldBuilder):
            builders = [builders]
        if not issequenceof(builders, DetSigYieldBuilder):
            raise TypeError(
                'The detsigyield_builder_list property must be a sequence of '
                'DetSigYieldBuilder instances!')
        self._detsigyield_builder_list = builders

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
        if method is not None:
            if not isinstance(method, SignalGenerationMethod):
                raise TypeError(
                    'The sig_gen_method property must be an instance of '
                    'SignalGenerationMethod! '
                    f'Its current type is {classname(method)}.')
        self._sig_gen_method = method

    @property
    def n_sources(self):
        """(read-only) The number of sources within this source hypothesis
        group.
        """
        return len(self._source_list)

    def __str__(self):
        """Pretty string representation of this SourceHypoGroup instance.
        """
        s = f'{classname(self)}:\n'

        s1 = f'sources ({len(self._source_list)}):'
        for (idx, source) in enumerate(self._source_list):
            s1 += '\n'
            s2 = f'{idx}: {source}'
            s1 += add_leading_text_line_padding(INDENTATION_WIDTH, s2)
        s1 += '\n'
        s1 += 'fluxmodel:\n'
        s2 = f'{self._fluxmodel}'
        s1 += add_leading_text_line_padding(INDENTATION_WIDTH, s2)
        s1 += '\n'
        s1 += f'detector signal yield builders ({len(self._detsigyield_builder_list)}):\n'
        s2 = '\n'.join((classname(builder) for builder in self._detsigyield_builder_list))
        s1 += add_leading_text_line_padding(INDENTATION_WIDTH, s2)
        s1 += '\n'
        s1 += 'signal generation method:\n'
        s2 = f'{classname(self._sig_gen_method)}'
        s1 += add_leading_text_line_padding(INDENTATION_WIDTH, s2)

        s += add_leading_text_line_padding(INDENTATION_WIDTH, s1)

        return s

    def get_source_weights(self):
        """Gets the weight from each source of this source hypothesis group.

        Returns
        -------
        weights : numpy ndarray | None
            The (N_sources,)-shaped numpy ndarray holding the theoretical
            weight of each source.
            It is ``None`` if any of the individual source weights is None.
        """
        weights = []
        for src in self._source_list:
            if src.weight is None:
                return None
            weights.append(src.weight)

        return np.array(weights)


class SourceHypoGroupManager(
        object):
    """The source hypothesis group manager provides the functionality to group
    sources of the same source hypothesis, i.e. spatial model and flux model,
    with an assigned detector signal yield implementation method.

    This helps to evaluate the log-likelihood ratio function in an efficient
    way.
    """
    def __init__(
            self,
            src_hypo_groups=None,
            **kwargs):
        """Creates a new source hypothesis group manager instance.

        Parameters
        ----------
        src_hypo_groups : SourceHypoGroup instance |
                          sequence of SourceHypoGroup instances | None
            The SourceHypoGroup instances to initialize the manager with.
        """
        super().__init__(**kwargs)

        self._shg_list = list()
        # Define a 2D numpy array of shape (N_sources,2) that maps the source
        # index (0 to N_sources-1) to the index of the group and the source
        # index within the group for fast access.
        self._sidx_to_gidx_gsidx_map_arr = np.empty((0, 2), dtype=np.int32)

        # Add source hypo groups if specified.
        if src_hypo_groups is not None:
            if isinstance(src_hypo_groups, SourceHypoGroup):
                src_hypo_groups = [src_hypo_groups]
            if not issequenceof(src_hypo_groups, SourceHypoGroup):
                raise TypeError(
                    'The src_hypo_groups argument must be an instance of '
                    'SourceHypoGroup, or a sequence of SourceHypoGroup '
                    'instances!')
            for shg in src_hypo_groups:
                self._shg_list.append(shg)
                self._extend_sidx_to_gidx_gsidx_map_arr(shg)

    @property
    def source_list(self):
        """The list of defined SourceModel instances.
        """
        source_list = []
        for shg in self._shg_list:
            source_list += shg.source_list
        return source_list

    @property
    def n_sources(self):
        """(read-only) The total number of sources defined in all source groups.
        """
        return self._sidx_to_gidx_gsidx_map_arr.shape[0]

    @property
    def n_src_hypo_groups(self):
        """DEPRICATED: Use n_shgs instead.
        The number of defined source hypothesis groups.
        """
        return len(self._shg_list)

    @property
    def n_shgs(self):
        """The number of defined source hypothesis groups.
        """
        return len(self._shg_list)

    @property
    def shg_list(self):
        """(read-only) The list of source hypothesis groups, i.e.
        SourceHypoGroup instances.
        """
        return self._shg_list

    def __str__(self):
        """Pretty string representation of this SourceHypoGroupManager.
        """
        s = f'{classname(self)}\n'

        s1 = 'Source Hypothesis Groups:'
        for (idx, shg) in enumerate(self._shg_list):
            s1 += '\n'
            s1 += add_leading_text_line_padding(INDENTATION_WIDTH, f'{idx}: {shg}')

        s += add_leading_text_line_padding(INDENTATION_WIDTH, s1)

        return s

    def _extend_sidx_to_gidx_gsidx_map_arr(self, shg):
        """Extends the source index to (group index, group source index) map
        array by one source hypo group.

        Parameters
        ----------
        shg : SourceHypoGroup instance
            The SourceHypoGroup instance for which the map array should get
            extented.
        """
        arr = np.empty((shg.n_sources, 2), dtype=np.int32)
        arr[:, 0] = self.n_src_hypo_groups-1  # Group index.
        arr[:, 1] = np.arange(shg.n_sources)  # Group source index.
        self._sidx_to_gidx_gsidx_map_arr = np.vstack(
            (self._sidx_to_gidx_gsidx_map_arr, arr))

    def create_source_hypo_group(
            self,
            sources,
            fluxmodel,
            detsigyield_builders,
            sig_gen_method=None):
        """Creates and adds a source hypothesis group to this source hypothesis
        group manager. A source hypothesis group shares sources of the same
        source model with the same flux model and hence the same detector signal
        yield and signal generation methods.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigyield_builders : sequence of DetSigYieldBuilder instances
            The sequence of detector signal yield builder instances,
            which should be used to create the detector signal
            yield for the sources of this group. Each element is the
            detector signal yield builder for the particular dataset, if
            several datasets are used. If this list contains only one builder,
            it should be used for all datasets.
        sig_gen_method : instance of SignalGenerationMethod | None
            The SignalGenerationMethod instance that implements the detector
            and source hypothesis specific signal generation.
            It can be set to None which means no signal can be generated.
        """
        # Create the source group.
        group = SourceHypoGroup(
            sources=sources,
            fluxmodel=fluxmodel,
            detsigyield_builders=detsigyield_builders,
            sig_gen_method=sig_gen_method)

        # Add the group.
        self._shg_list.append(group)

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
        gidx = self._sidx_to_gidx_gsidx_map_arr[src_idx, 0]
        return self._shg_list[gidx]._fluxmodel

    def get_detsigyield_builder_list_by_src_idx(self, src_idx):
        """Retrieves the list of DetSigYieldBuilder instances for the source
        specified by its source index.

        Parameters
        ----------
        src_idx : int
            The index of the source, which must be in the range
            [0, N_sources-1].

        Returns
        -------
        detsigyield_builder_list : list of DetSigYieldBuilder instances
            The list of DetSigYieldBuilder instances that apply to the
            specified source.
        """
        gidx = self._sidx_to_gidx_gsidx_map_arr[src_idx, 0]
        return self._shg_list[gidx]._detsigyield_builder_list

    def get_src_mask_of_shg(self, shg_idx):
        """Creates a source mask for the sources of the ``shg_idx`` th source
        hypothesis group.

        Parameters
        ----------
        shg_idx : int
            The index of the source hypothesis group.

        Returns
        -------
        src_mask : instance of numpy ndarray
            The (N_sources,)-shaped numpy ndarray of bool holding the mask for
            selecting the sources of the given source hypothesis group.
        """
        return (self._sidx_to_gidx_gsidx_map_arr[:, 0] == shg_idx)

    def get_src_idxs_of_shg(self, shg_idx):
        """Creates an array of indices of sources that belong to the given
        source hypothesis group.

        Parameters
        ----------
        shg_idx : int
            The index of the source hypothesis group.

        Returns
        -------
        src_idxs : instance of numpy ndarray
            The numpy ndarray of int holding the indices of the sources that
            belong to the given source hypothesis group.
        """
        src_idxs = np.arange(self.n_sources)[self.get_src_mask_of_shg(shg_idx)]

        return src_idxs
