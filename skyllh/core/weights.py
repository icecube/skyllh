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
    r"""This class provides the source detector weights, which are the product
    of the source weights with the detector signal yield, denoted with
    :math:`a_{j,k}({p_s}_k)` in the math formalism documentation.

    .. math::

        a_{j,k}({p_{s}}_k) = W_k {\mathcal{Y}_s}_{j,k}({p_{s}}_k)

    """

    @staticmethod
    def create_src_recarray_list_list(shg_mgr, detsigyield_arr):
        """Creates a list of numpy record ndarrays, one for each source
        hypothesis group suited for evaluating the detector signal yield
        instance of that source hypothesis group.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager defining the source
            hypothesis groups with their sources.
        detsigyield_arr : ndarray of instance of DetSigYield
            The (N_datasets,N_source_hypo_groups)-shaped 1D ndarray of
            DetSigYield instances, one for each dataset and source hypothesis
            group combination.

        Returns
        -------
        src_recarray_list_list : list of list of numpy record ndarrays
            The (N_datasets,N_source_hypo_groups)-shaped list of list of the
            source numpy record ndarrays, one for each dataset and source
            hypothesis group combination, which is needed for
            evaluating a particular detector signal yield instance.
        """
        n_datasets = detsigyield_arr.shape[0]
        n_shgs = detsigyield_arr.shape[1]
        shg_list = shg_mgr.src_hypo_group_list

        src_recarray_list_list = []
        for ds_idx in range(n_datasets):
            src_recarray_list = []
            for shg_idx in range(n_shgs):
                shg = shg_list[shg_idx]
                src_recarray_list.append(
                    detsigyield_arr[ds_idx][shg_idx].sources_to_recarray(
                        shg.source_list))

            src_recarray_list_list.append(src_recarray_list)

        return src_recarray_list_list

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

    def __init__(self, shg_mgr, pmm, detsigyields):
        """Creates a new SourceDetectorWeights instance.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager defining the sources and
            their source hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper defining the mapping of the
            global parameters to the individual sources.
        detsigyields : sequence of sequence of instance of DetSigYield
            The (N_datasets,N_source_hypo_groups)-shaped sequence of sequence of
            DetSigYield instances, one instance for each combination of dataset
            and source hypothesis group.
        """
        self._set_shg_mgr(shg_mgr=shg_mgr)

        if not isinstance(pmm, ParameterModelMapper):
            raise TypeError(
                'The pmm argument must be an instance of '
                'ParameterModelMapper!')
        self._pmm = pmm

        if not issequence(detsigyields):
            detsigyields = [detsigyields]
        for item in detsigyields:
            if not issequenceof(item, DetSigYield):
                raise TypeError(
                    'The detsigyields argument must be a sequence of sequence '
                    'of DetSigYield instances!')
        self._detsigyield_arr = np.atleast_2d(detsigyields)
        if self._detsigyield_arr.shape[1] != self._shg_mgr.n_src_hypo_groups:
            raise ValueError(
                'The length of the second dimension of the detsigyields array '
                'must be equal to the number of source hypothesis groups which '
                'the source hypothesis group manager defines!')

        # Create the list of list of source record arrays for each combination
        # of dataset and source hypothesis group.
        self._src_recarray_list_list = type(self).create_src_recarray_list(
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
    def pmm(self):
        """(read-only) The ParameterModelMapper instance mapping the global set
        of parameters to the individual sources.
        """
        return self._pmm

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
        self._src_recarray_list_list = type(self).create_src_recarray_list_list(
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
        a : (N_datasets,N_sources)-shaped numpy ndarray
            The numpy ndarray holding the source detector weight for each
            combination of dataset and source.
        a_grads : (N_datasets,N_sources,N_gfl_params)-shaped numpy ndarray
            The derivative of the source detector weight for each combination
            of dataset and source for each global floating parameter.
        """
        n_datasets = self._detsigyield_arr.shape[0]

        a = np.empty(
            (n_datasets, self.shg_mgr.n_sources,),
            dtype=np.double)
        a_grads = np.zeros(
            (n_datasets, self.shg_mgr.n_sources, len(gflp_values)),
            dtype=np.double)

        shg_list = self.shg_mgr.src_hypo_group_list

        # Create a list of src_params_recarray instances, one for each SHG.
        # This will be the same for all datasets.
        src_params_recarray_list = [
            self._pmm.create_src_params_recarray(
                gflp_values=gflp_values,
                sources=shg.source_list)
            for shg in shg_list
        ]

        for ds_idx in range(n_datasets):
            sidx = 0
            for (shg, detsigyield, src_recarray, src_params_recarray,
                 src_weights) in zip(
                    shg_list,
                    self.detsigyield_arr[ds_idx],
                    self._src_recarray_list_list[ds_idx],
                    src_params_recarray_list,
                    self._src_weight_array_list):

                shg_n_src = shg.n_sources

                (Yg, Yg_grads) = detsigyield(
                    src_recarray=src_recarray,
                    src_params_recarray=src_params_recarray)

                shg_src_slice = slice(sidx, sidx+shg_n_src)

                a[ds_idx][shg_src_slice] = src_weights * Yg

                for gpidx in Yg_grads.keys():
                    a_grads[ds_idx,shg_src_slice,gpidx] =\
                        src_weights * Yg_grads[gpidx]

                sidx += shg_n_src

        return (a, a_grads)
