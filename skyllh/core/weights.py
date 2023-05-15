# -*- coding: utf-8 -*-

"""This module contains utility classes related to calculate weights.
"""

import numpy as np

from skyllh.core.services import (
    SrcDetSigYieldWeightsService,
)


class DatasetSignalWeightFactorsService(
        object):
    r"""This class provides a service to calculates the dataset signal weight
    factors, :math:`f_j(\vec{p}_\mathrm{s})`, for each dataset.
    It utilizes the source detector signal yield weights
    :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`, provided by the
    :class:`~SrcDetSigYieldWeightsService` class.
    """

    def __init__(
            self,
            src_detsigyield_weights_service):
        r"""Creates a new DatasetSignalWeightFactors instance.

        Parameters
        ----------
        src_detsigyield_weights_service : instance of SrcDetSigYieldWeightsService
            The instance of SrcDetSigYieldWeightsService providing the source
            detector signal yield weights
            :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`.
        """
        self.src_detsigyield_weights_service = src_detsigyield_weights_service

    @property
    def src_detsigyield_weights_service(self):
        r"""The instance of SrcDetSigYieldWeightsService providing the source
        detector signal yield weights :math:`a_{j,k}(\vec{p}_{\mathrm{s}_k})`.
        """
        return self._src_detsigyield_weights_service

    @src_detsigyield_weights_service.setter
    def src_detsigyield_weights_service(self, service):
        if not isinstance(service, SrcDetSigYieldWeightsService):
            raise TypeError(
                'The src_detsigyield_weights_service property must be an '
                'instance of SrcDetSigYieldWeightsService!')
        self._src_detsigyield_weights_service = service

    @property
    def n_datasets(self):
        """(read-only) The number of datasets.
        """
        return self._src_detsigyield_weights_service.n_datasets

    def calculate(self):
        r"""Calculates the dataset signal weight factors,
        :math:`f_j(\vec{p}_\mathrm{s})`. The result is stored internally as:

            f_j : instance of ndarray
                The (N_datasets,)-shaped 1D numpy ndarray holding the dataset
                signal weight factor for each dataset.
            f_j_grads : dict
                The dictionary holding the (N_datasets,)-shaped numpy
                ndarray with the derivatives w.r.t. the global fit parameter
                the DatasetSignalWeightFactorsService depend on.
                The dictionary's key is the index of the global fit parameter.
        """
        (a_jk, a_jk_grads) = self._src_detsigyield_weights_service.get_weights()

        a_j = np.sum(a_jk, axis=1)
        a = np.sum(a_jk)

        self._f_j = a_j / a

        # Calculate the derivative of f_j w.r.t. all floating parameters present
        # in the a_jk_grads using the quotient rule of differentation.
        self._f_j_grads = dict()
        for gpidx in a_jk_grads.keys():
            # a is a scalar.
            # a_j is a (N_datasets)-shaped ndarray.
            # a_jk_grads is a dict of length N_gfl_params with values of
            #    (N_datasets,N_sources)-shaped ndarray.
            # a_j_grads is a (N_datasets,)-shaped ndarray.
            # a_grads is a scalar.
            a_j_grads = np.sum(a_jk_grads[gpidx], axis=1)
            a_grads = np.sum(a_jk_grads[gpidx])
            self._f_j_grads[gpidx] = (a_j_grads * a - a_j * a_grads) / a**2

    def get_weights(self):
        """Returns the

        Returns
        -------
        f_j : instance of ndarray
            The (N_datasets,)-shaped 1D numpy ndarray holding the dataset
            signal weight factor for each dataset.
        f_j_grads : dict
            The dictionary holding the (N_datasets,)-shaped numpy
            ndarray with the derivatives w.r.t. the global fit parameter
            the DatasetSignalWeightFactorsService depend on.
            The dictionary's key is the index of the global fit parameter.
        """
        return (self._f_j, self._f_j_grads)
