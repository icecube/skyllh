# -*- coding: utf-8 -*-

"""This modules provides classes for calculating dataset signal weights based
on the detector signal yield. This is used split the number of signal events
fit parameter into its contributions of the individual datasets.
"""

import abc

from skyllh.core.detsigyield import (
    DetSigYield,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.py import (
    issequenceof,
)
from skyllh.core.source_hypothesis import (
    SourceHypoGroupManager,
)


class DatasetSignalWeights(object, metaclass=abc.ABCMeta):
    """Abstract base class for a dataset signal weight calculator class.
    """
    def __init__(
            self,
            shg_mgr,
            param_model_mapper,
            detsigyields,
            *args,
            **kwargs):
        """Base class constructor.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        param_model_mapper : ParameterModelMapper instance
            The ParameterModelMapper instance that defines the global set of
            parameters and their mapping to individual models, e.g. sources.
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """
        super().__init__(*args, **kwargs)

        self.shg_mgr = shg_mgr
        self.param_model_mapper = param_model_mapper
        self.detsigyield_arr = detsigyields

        if(self._detsigyield_arr.shape[0] != self._shg_mgr.n_src_hypo_groups):
            raise ValueError(
                'The detsigyields array must have the same number of source '
                'hypothesis groups as the source hypothesis group manager '
                'defines!')

        # Pre-convert the source list of each source hypothesis group into a
        # source array needed for the detector signal yield evaluation.
        # Since all the detector signal yield instances must be of the same
        # kind for each dataset, we can just use the one of the first dataset of
        # each source hypothesis group.
        self._src_arr_list = self._create_src_arr_list(
            self._shg_mgr, self._detsigyield_arr)

    def _create_src_arr_list(self, shg_mgr, detsigyield_arr):
        """Pre-convert the source list of each source hypothesis group into a
        source array needed for the detector signal yield evaluation.
        Since all the detector signal yield instances must be of the same
        kind for each dataset, we can just use the one of the first dataset of
        each source hypothesis group.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the sources.

        detsigyield_arr : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                        DetSigYield instances
            The collection of DetSigYield instances for each dataset and source
            group combination.
        Returns
        -------
        src_arr_list : list of numpy record ndarrays
            The list of the source numpy record ndarrays, one for each source
            hypothesis group, which is needed by the detector signal yield
            instance.
        """
        src_arr_list = []
        for (gidx, src_hypo_group) in enumerate(shg_mgr.src_hypo_group_list):
            src_arr_list.append(
                detsigyield_arr[gidx,0].source_to_array(
                    src_hypo_group.source_list)
            )

        return src_arr_list

    @property
    def shg_mgr(self):
        """The instance of SourceHypoGroupManager, which defines the source
        hypothesis groups.
        """
        return self._shg_mgr
    @shg_mgr.setter
    def shg_mgr(self, mgr):
        if(not isinstance(mgr, SourceHypoGroupManager)):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager!')
        self._shg_mgr = mgr

    @property
    def param_model_mapper(self):
        """The ParameterModelMapper instance defining the global set of
        parameters and their mapping to individual models, e.g. sources.
        """
        return self._param_model_mapper
    @param_model_mapper.setter
    def param_model_mapper(self, mapper):
        if(not isinstance(mapper, ParameterModelMapper)):
            raise TypeError(
                'The param_model_mapper property must be an instance of '
                'ParameterModelMapper!')
        self._param_model_mapper = mapper

    @property
    def detsigyield_arr(self):
        """The 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
        DetSigYield instances.
        """
        return self._detsigyield_arr
    @detsigyield_arr.setter
    def detsigyield_arr(self, detsigyields):
        if(not isinstance(detsigyields, np.ndarray)):
            raise TypeError(
                'The detsigyield_arr property must be an instance of '
                'numpy.ndarray!')
        if(detsigyields.ndim != 2):
            raise ValueError(
                'The detsigyield_arr property must be a numpy.ndarray with 2 '
                'dimensions!')
        if(not issequenceof(detsigyields.flat, DetSigYield)):
            raise TypeError(
                'The detsigyield_arr property must contain DetSigYield '
                'instances, one for each source hypothesis group and dataset '
                'combination!')
        self._detsigyield_arr = detsigyields

    @property
    def n_datasets(self):
        """(read-only) The number of datasets this DatasetSignalWeights instance
        is for.
        """
        return self._detsigyield_arr.shape[1]

    def change_source_hypo_group_manager(self, shg_mgr):
        """Changes the SourceHypoGroupManager instance of this
        DatasetSignalWeights instance. This will also recreate the internal
        source numpy record arrays needed for the detector signal efficiency
        calculation.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance, that should be used for
            this dataset signal weights instance.
        """
        self.shg_mgr = shg_mgr
        self._src_arr_list = self._create_src_arr_list(
            self._shg_mgr, self._detsigyield_arr)

    @abc.abstractmethod
    def __call__(self, fitparam_values):
        """This method is supposed to calculate the dataset signal weights and
        their gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_datasets,)-shaped 1D ndarray
            The dataset signal weight factor for each dataset.
        f_grads : (N_datasets,N_fitparams)-shaped 2D ndarray
            The gradients of the dataset signal weight factors, one for each
            fit parameter.
        """
        pass


class SingleSourceDatasetSignalWeights(DatasetSignalWeights):
    """This class calculates the dataset signal weight factors for each dataset
    assuming a single source.
    """

    def __init__(
            self,
            shg_mgr,
            param_model_mapper,
            detsigyields,
            *args,
            **kwargs):
        """Constructs a new DatasetSignalWeights instance assuming a single
        source.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        param_model_mapper : ParameterModelMapper
            The instance of ParameterModelMapper defining the global set of
            parameters and their mapping to individual models, e.g. sources.
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """
        # Convert sequence into a 2D numpy array.
        detsigyields = np.atleast_2d(detsigyields)

        super().__init__(
            *args,
            shg_mgr=shg_mgr,
            param_model_mapper=param_model_mapper,
            detsigyields=detsigyields,
            **kwargs)

    def __call__(self, fitparam_values):
        """Calculates the dataset signal weight and its fit parameter gradients
        for each dataset.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_datasets,)-shaped 1D ndarray
            The dataset signal weight factor for each dataset.
        f_grads : (N_datasets,N_fitparams)-shaped 2D ndarray | None
            The gradients of the dataset signal weight factors, one for each
            fit parameter. None is returned if there are no fit parameters
            beside ns.
        """
        fitparams_arr =\
            self._param_model_mapper.get_source_floating_params_recarray(
                fitparam_values)

        N_datasets = self.n_datasets

        # Determine how many fit parameters there are (excluding ns).
        N_fitparams = self._param_model_mapper.n_global_floating_params - 1

        Y = np.empty((N_datasets,), dtype=np.float)
        if(N_fitparams > 0):
            Y_grads = np.empty((N_datasets, N_fitparams), dtype=np.float)

        # Loop over the detector signal efficiency instances for the first and
        # only source hypothesis group.
        for (j, detsigyield) in enumerate(self._detsigyield_arr[0]):
            (Yj, Yj_grads) = detsigyield(self._src_arr_list[0], fitparams_arr)
            # Store the detector signal yield and its fit parameter
            # gradients for the first and only source (element 0).
            Y[j] = Yj[0]
            if(N_fitparams > 0):
                if Yj_grads is None:
                    Y_grads[j] = np.zeros_like(Yj[0])
                else:
                    Y_grads[j] = Yj_grads[0]

        # sumj_Y is a scalar.
        sumj_Y = np.sum(Y, axis=0)

        # f is a (N_datasets,)-shaped 1D ndarray.
        f = Y/sumj_Y

        # f_grads is a (N_datasets, N_fitparams)-shaped 2D ndarray.
        if(N_fitparams > 0):
            # sumj_Y_grads is a (N_fitparams,)-shaped 1D array.
            sumj_Y_grads = np.sum(Y_grads, axis=0)
            f_grads = (Y_grads*sumj_Y - Y[...,np.newaxis]*sumj_Y_grads) / sumj_Y**2
        else:
            f_grads = None

        return (f, f_grads)


class MultiSourceDatasetSignalWeights(SingleSourceDatasetSignalWeights):
    """This class calculates the dataset signal weight factors for each dataset
    assuming multiple sources.
    """

    def __init__(
            self,
            shg_mgr,
            param_model_mapper,
            detsigyields,
            *args,
            **kwargs):
        """Constructs a new DatasetSignalWeights instance assuming multiple
        sources.

        Parameters
        ----------
        shg_mgr : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        param_model_mapper : ParameterModelMapper instance
            The instance of ParameterModelMapper defining the global set of
            parameters and their mapping to individual models, e.g. sources.
        detsigyields : 2D (N_source_hypo_groups,N_datasets)-shaped ndarray of
                     DetSigYield instances
            The collection of DetSigYield instances for each
            dataset and source group combination. The detector signal yield
            instances are used to calculate the dataset signal weight factors.
            The order must follow the definition order of the log-likelihood
            ratio functions, i.e. datasets, and the definition order of the
            source hypothesis groups.
        """
        super().__init__(
            *args,
            shg_mgr=shg_mgr,
            param_model_mapper=param_model_mapper,
            detsigyields=detsigyields,
            **kwargs)

    def __call__(self, fitparam_values):
        """Calculates the dataset signal weight and its fit parameter gradients
        for each dataset.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_datasets,)-shaped 1D ndarray
            The dataset signal weight factor for each dataset.
        f_grads : (N_datasets,N_fitparams)-shaped 2D ndarray | None
            The gradients of the dataset signal weight factors, one for each
            fit parameter. None is returned if there are no fit parameters
            beside ns.
        """
        fitparams_arr =\
            self._param_model_mapper.get_source_floating_params_recarray(
                fitparam_values)

        N_datasets = self.n_datasets

        # Determine how many fit parameters there are (excluding ns).
        N_fitparams = self._param_model_mapper.n_global_floating_params - 1

        N_sources = len(self._src_arr_list[0])

        Y = np.empty((N_datasets, N_sources), dtype=np.float)
        if N_fitparams > 0:
            Y_grads = np.empty(
                (N_datasets, N_sources, N_fitparams),
                dtype=np.float)

        # Loop over the detector signal yield instances for each source hypo
        # group and dataset.
        for (k, detsigyield_k) in enumerate(self._detsigyield_arr):
            for (j, detsigyield) in enumerate(detsigyield_k):
                (Yj, Yj_grads) = detsigyield(
                    self._src_arr_list[k], fitparams_arr)
                # Store the detector signal yield and its fit parameter
                # gradients.
                Y[j] = Yj
                if N_fitparams > 0:
                    Y_grads[j] = Yj_grads.T

        sum_Y = np.sum(Y)

        # f is a (N_datasets,)-shaped 1D ndarray.
        f = np.sum(Y, axis=1) / sum_Y

        # f_grads is a (N_datasets, N_fitparams)-shaped 2D ndarray.
        if N_fitparams > 0:
            # sum_Y_grads is a (N_datasets, N_fitparams,)-shaped 2D array.
            sum_Y_grads = np.sum(Y_grads, axis=1)
            f_grads = (sum_Y_grads*sum_Y - (f*sum_Y)[...,np.newaxis]*np.sum(sum_Y_grads, axis=0)) / sum_Y**2
        else:
            f_grads = None

        return (f, f_grads)