# -*- coding: utf-8 -*-

"""This module contains utility classes related to calculate weights.
"""

from collections import defaultdict
import numpy as np

from skyllh.core.detsigyield import (
    DetSigYield,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.py import (
    issequence,
    issequenceof,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)


class SourceDetectorWeights(object):
    r"""This class provides the source detector weights, which are the product
    of the source weights with the detector signal yield, denoted with
    :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})` in the math formalism documentation.

    .. math::

        a_{j,k}(\vec{p}_{\mathrm{s}_k}) = W_k
            \mathcal{Y}_{\mathrm{s}_{j,k}}(\vec{p}_{\mathrm{s}_k})

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
        self._src_recarray_list_list = type(self).create_src_recarray_list_list(
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
        a_jk : instance of ndarray
            The (N_datasets,N_sources)-shaped numpy ndarray holding the source
            detector weight for each combination of dataset and source.
        a_jk_grads : dict
            The dictionary holding the (N_datasets,N_sources)-shaped numpy
            ndarray with the derivatives w.r.t. the global floating parameter
            the SourceDetectorWeights depend on. The dictionary's key is the
            index of the global floating parameter.
        """
        n_datasets = self._detsigyield_arr.shape[0]

        a_jk = np.empty(
            (n_datasets, self.shg_mgr.n_sources,),
            dtype=np.double)

        a_jk_grads = defaultdict(
            lambda: np.zeros(
                (n_datasets, self.shg_mgr.n_sources),
                dtype=np.double))

        sidx = 0
        for (shg_idx, (shg, src_weights)) in enumerate(zip(
                self.shg_mgr.src_hypo_group_list,
                self._src_weight_array_list)):

            shg_n_src = shg.n_sources

            src_params_recarray = self._pmm.create_src_params_recarray(
                gflp_values=gflp_values,
                sources=shg.source_list)

            for ds_idx in range(n_datasets):
                detsigyield = self._detsigyield_arr[ds_idx,shg_idx]
                src_recarray = self._src_recarray_list_list[ds_idx][shg_idx]

                (Yg, Yg_grads) = detsigyield(
                    src_recarray=src_recarray,
                    src_params_recarray=src_params_recarray)

                shg_src_slice = slice(sidx, sidx+shg_n_src)

                a_jk[ds_idx][shg_src_slice] = src_weights * Yg

                for gpidx in Yg_grads.keys():
                    a_jk_grads[gpidx][ds_idx,shg_src_slice] =\
                        src_weights * Yg_grads[gpidx]

            sidx += shg_n_src

        return (a_jk, a_jk_grads)


class DatasetSignalWeightFactors(object):
    r"""This class calculates the dataset signal weight factors,
    :math:`f_j(\vec{p}_\mathrm{s})`, for each dataset. It utilizes the source
    detector weights :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`, provided by the
    :class:`~SourceDetectorWeights` class.
    """

    def __init__(self, src_det_weights):
        r"""Creates a new DatasetSignalWeightFactors instance.

        Parameters
        ----------
        src_det_weights : instance of SourceDetectorWeights
            The instance of SourceDetectorWeights for calculating the source
            detector weights :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`.
        """
        self.src_det_weights = src_det_weights

    @property
    def src_det_weights(self):
        r"""The instance of SourceDetectorWeights providing the calculation of
        the source detector weights :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`.
        """
        return self._src_det_weights
    @src_det_weights.setter
    def src_det_weights(self, w):
        if not isinstance(w, SourceDetectorWeights):
            raise TypeError(
                'The src_det_weights property must be an instance of '
                'SourceDetectorWeights!')
        self._src_det_weights = w

    def __call__(self, gflp_values):
        r"""Calculates the dataset signal weight factors,
        :math:`f_j(\vec{p}_\mathrm{s})`.

        Parameters
        ----------
        gflp_values : instance of ndarray
            The (N_gfl_params,)-shaped 1D numpy ndarray holding the global
            floating parameter values.

        Returns
        -------
        f_j : instance of ndarray
            The (N_datasets,)-shaped 1D numpy ndarray holding the dataset signal
            weight factor for each dataset.
        f_j_grads : dict
            The dictionary holding the (N_datasets,)-shaped numpy
            ndarray with the derivatives w.r.t. the global floating parameter
            the DatasetSignalWeightFactors depend on. The dictionary's key is
            the index of the global floating parameter.
        """
        (a_jk, a_jk_grads) = self._src_det_weights(gflp_values=gflp_values)

        a_j = np.sum(a_jk, axis=1)
        a = np.sum(a_jk)

        f_j = a_j / a

        # Calculate the derivative of f_j w.r.t. all floating parameters present
        # in the a_jk_grads using the quotient rule of differentation.
        f_j_grads = dict()
        for gpidx in a_jk_grads.keys():
            # a is a scalar.
            # a_j is a (N_datasets)-shaped ndarray.
            # a_jk_grads is a dict of length N_gfl_params with values of
            #    (N_datasets,N_sources)-shaped ndarray.
            # a_j_grads is a (N_datasets,)-shaped ndarray.
            # a_grads is a scalar.
            a_j_grads = np.sum(a_jk_grads[gpidx], axis=1)
            a_grads = np.sum(a_jk_grads[gpidx])
            f_j_grads[gpidx] = (a_j_grads * a - a_j * a_grads) / a**2

        return (f_j, f_j_grads)
