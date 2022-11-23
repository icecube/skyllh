# -*- coding: utf-8 -*-

"""This module contains utility classes related to calculate weights.
"""

from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)

class SourceDetectorWeights(object):
    """This class provides the source detector weights, which are the product
    of the source weights with the detector signal yield, denoted with
    a_k(p_s_k) in the mathematics documentation.

    :math::

        a_k({p_{s}}_k) = W_k {Y_s}_k

    """

    @staticmethod
    def create_src_recarray_list(shg_mgr, detsigyield_arr):
        """Creates a list of numpy record ndarrays, one for each source
        hypothesis group suited for evaluating the detector signal yield
        instance of that source hypothesis group.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager defining the source
            hypothesis groups with their sources.
        detsigyield_arr : (N_source_hypo_groups,)-shaped 1D ndarray of
                DetSigYield instances
            The array of DetSigYield instances, one for each source hypothesis
            group.

        Returns
        -------
        src_recarray_list : list of numpy record ndarrays
            The list of the source numpy record ndarrays, one for each source
            hypothesis group, which is needed for evaluating the detector
            signal yield instance.
        """
        src_recarray_list = [
            detsigyield_arr[gidx].source_to_array(shg.source_list)
                for (gidx, shg) in enumerate(shg_mgr.src_hypo_group_list)
        ]

        return src_recarray_list

    @staticmethod
    def create_src_weight_array_list(shg_mgr):
        """Creates a list of numpy 1D ndarrays holding the source weights, one
        for each source hypothesis group.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager defining the source
            hypothesis groups with their sources.

        Returns
        -------
        src_weight_array_list : list of numpy 1D ndarrays
            The list of 1D numpy ndarrays holding the source weights, one for
            each source hypothesis group.
        """
        src_weight_array_list = [
            np.array([src.weight for src in shg.source_list])
                for shg in shg_mgr.src_hypo_group_list
        ]
        return src_weight_array_list

    def __init__(self, shg_mgr, param_model_mapper, detsigyields):
        """Creates a new SourceDetectorWeights instance.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager defining the sources and
            their source hypothesis groups.
        param_model_mapper : ParameterModelMapper instance
            The instance of ParameterModelMapper defining the mapping of the
            global parameters to the individual sources.
        detsigyields : sequence of DetSigYield instances
            The sequence of DetSigYield instances, one instance for each
            source hypothesis group.
        """
        self._set_shg_mgr(shg_mgr=shg_mgr)

        if not isinstance(param_model_mapper, ParameterModelMapper):
            raise TypeError(
                'The param_model_mapper argument must be an instance of '
                'ParameterModelMapper!')
        self._param_model_mapper = param_model_mapper

        if not issequence(detsigyields):
            detsigyields = [detsigyields]
        if not issequenceof(detsigyields, DetSigYield):
            raise TypeError(
                'The detsigyields argument must be a sequence of DetSigYield '
                'instances!')
        self._detsigyield_arr = np.array(detsigyields)
        if len(self._detsigyield_arr) != self._shg_mgr.n_src_hypo_groups:
            raise ValueError('The detsigyields array must have the same number '
                'of source hypothesis groups as the source hypothesis group '
                'manager defines!')

        # Create the list of source record arrays for each source hypothesis
        # group.
        self._src_recarray_list = type(self).create_src_recarray_list(
            shg_mgr=self._shg_mgr,
            detsigyield_arr=self._detsigyield_arr)

        # Create the list of 1D ndarrays holding the source weights for each
        # source hypothesis group.
        self._src_weight_array_list = type(self).create_src_weight_array_list(
            shg_mgr=self._shg_mgr)

    @property
    def shg_mgr(self):
        """(read-only) The SourceHypoGroupManager instance defining the source
        hypothesis groups.
        """
        return self._shg_mgr

    @property
    def param_model_mapper(self):
        """(read-only) The ParameterModelMapper instance mapping the global set
        of parameters to the individual sources.
        """
        return self._param_model_mapper

    @property
    def detsigyield_arr(self):
        """(read-only) The numpy ndarray holding the DetSigYield instances for
        each source hypothesis group.
        """
        return self._detsigyield_arr

    def _set_shg_mgr(self, shg_mgr):
        """Sets the _shg_mgr class attribute and checks for the correct type.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that should be set.
        """
        if not isinstance(shg_mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr argument must be an instance of '
                'SourceHypoGroupManager!')
        self._shg_mgr = shg_mgr

    def change_shg_mgr(self, shg_mgr):
        """Changes the SourceHypoGroupManager instance of this
        SourceDetectorWeights instance. This will also re-create the internal
        source numpy record arrays needed for the detector signal yield
        calculation.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance.
        """
        self._set_shg_mgr(shg_mgr=shg_mgr)
        self._src_recarray_list = type(self).create_src_recarray_list(
            shg_mgr=self._shg_mgr,
            detsigyield_arr=self._detsigyield_arr)
        self._src_weight_array_list = type(self).create_src_weight_array_list(
            shg_mgr=self._shg_mgr)

    def __call__(self, gflp_values):
        """Calculates the source detector weights for each source and their
        derivative w.r.t. each global floating parameter.

        Parameters
        ----------
        gflp_values : (N_gfl_params,)-shaped numpy ndarray
            The ndarray holding the global floating parameter values.

        Returns
        -------
        a : (N_sources,)-shaped numpy ndarray
            The numpy ndarray holding the source detector weight for each
            source.
        a_grads : (N_sources,N_gfl_params)-shaped numpy ndarray
            The derivative of the source detector weight for each source and
            global floating parameter.
        """
        a = np.empty((self.shg_mgr.n_sources,), dtype=np.double)
        a_grads = np.empty(
            (self.shg_mgr.n_sources, len(gflp_values)),
            dtype=np.double)

        sidx = 0
        for (shg, detsigyield, src_recarray, src_weights) in zip(
                self.shg_mgr.src_hypo_group_list,
                self.detsigyield_arr,
                self._src_recarray_list,
                self._src_weight_array_list):

            shg_n_src = shg.n_sources

            src_flp_recarray =\
                self._param_model_mapper.get_source_floating_params_recarray(
                    gflp_values=gflp_values,
                    sources=shg.source_list)
            # TODO: The shape of Yg_grads is not well defined, because it's
            #       not clear on which global floating parameters the detector
            #       signal yield depends.
            (Yg, Yg_grads) = detsigyield(src_recarray, src_flp_recarray)

            a[sidx:sidx+shg_n_src] = src_weights * Yg

            a_grads[sidx:sidx+shg_n_src] = src_weights * Yg_grads

            sidx += shg_n_src
