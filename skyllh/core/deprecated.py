"""This module contains depricated classes.
"""

class SourceWeights(object, metaclass=abc.ABCMeta):
    """This class is DEPRECATED!
    Use :py:class:`skyllh.core.weights.SourceDetectorWeights` instead!

    Abstract base class for a source weight calculator class.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Constructs a new SourceWeights instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SourceFitParameterMapper
            The SourceFitParameterMapper instance that defines the global fit
            parameters and their mapping to the source fit parameters.
        detsigyields : (N_source_hypo_groups,)-shaped 1D ndarray of DetSigYield
                instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self.src_fitparam_mapper = src_fitparam_mapper
        self.detsigyield_arr = np.atleast_1d(detsigyields)

        if(self._detsigyield_arr.shape[0] != self._src_hypo_group_manager.n_src_hypo_groups):
            raise ValueError('The detsigyields array must have the same number '
                'of source hypothesis groups as the source hypothesis group '
                'manager defines!')

        # Pre-convert the source list of each source hypothesis group into a
        # source array needed for the detector signal yield evaluation.
        # Since all the detector signal yield instances must be of the same
        # kind for each dataset, we can just use the one of the first dataset of
        # each source hypothesis group.
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

    def _create_src_arr_list(self, shg_mgr, detsigyield_arr):
        """Pre-convert the source list of each source hypothesis group into a
        source array needed for the detector signal yield evaluation.
        Since all the detector signal yield instances must be of the same
        kind for each dataset, we can just use the one of the first dataset of
        each source hypothesis group.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The SourceHypoGroupManager instance defining the sources.
        detsigyield_arr : (N_source_hypo_groups,)-shaped 1D ndarray of
                DetSigYield instances
            The collection of DetSigYield instances for each source hypothesis
            group.

        Returns
        -------
        src_arr_list : list of numpy record ndarrays
            The list of the source numpy record ndarrays, one for each source
            hypothesis group, which is needed by the detector signal yield
            instance.
        """
        src_arr_list = []
        for (gidx, shg) in enumerate(shg_mgr.src_hypo_group_list):
            src_arr_list.append(
                detsigyield_arr[gidx].sources_to_recarray(shg.source_list)
            )

        return src_arr_list

    @property
    def src_hypo_group_manager(self):
        """The instance of SourceHypoGroupManager, which defines the source
        hypothesis groups.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                'instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    @property
    def src_fitparam_mapper(self):
        """The SourceFitParameterMapper instance defining the global fit
        parameters and their mapping to the source fit parameters.
        """
        return self._src_fitparam_mapper
    @src_fitparam_mapper.setter
    def src_fitparam_mapper(self, mapper):
        if(not isinstance(mapper, SourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper property must be an '
                'instance of SourceFitParameterMapper!')
        self._src_fitparam_mapper = mapper

    @property
    def detsigyield_arr(self):
        """The (N_source_hypo_groups,)-shaped 1D ndarray of DetSigYield
        instances.
        """
        return self._detsigyield_arr
    @detsigyield_arr.setter
    def detsigyield_arr(self, detsigyields):
        if(not isinstance(detsigyields, np.ndarray)):
            raise TypeError('The detsigyield_arr property must be an instance '
                'of numpy.ndarray!')
        if(detsigyields.ndim != 1):
            raise ValueError('The detsigyield_arr property must be a '
                'numpy.ndarray with 1 dimensions!')
        if(not issequenceof(detsigyields.flat, DetSigYield)):
            raise TypeError('The detsigyield_arr property must contain '
                'DetSigYield instances, one for each source hypothesis group!')
        self._detsigyield_arr = detsigyields

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the SourceHypoGroupManager instance of this
        DatasetSignalWeights instance. This will also recreate the internal
        source numpy record arrays needed for the detector signal efficiency
        calculation.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance, that should be used for
            this dataset signal weights instance.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._src_arr_list = self._create_src_arr_list(
            self._src_hypo_group_manager, self._detsigyield_arr)

    @abc.abstractmethod
    def __call__(self, fitparam_values):
        """This method is supposed to calculate source weights and
        their gradients.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_sources,)-shaped 1D ndarray
            The source weight factor for each source.
        f_grads : (N_sources,)-shaped 1D ndarray | None
            The gradients of the source weight factors. None is returned if
            there are no fit parameters beside ns.
        """
        pass


class MultiPointSourcesRelSourceWeights(SourceWeights):
    """This class is DEPRECATED!
    Use :py:class:`skyllh.core.weights.SourceDetectorWeights` instead!

    This class calculates the relative source weights for a group of point
    sources.
    """
    def __init__(
            self, src_hypo_group_manager, src_fitparam_mapper, detsigyields):
        """Constructs a new MultiPointSourcesRelSourceWeights instance assuming
        multiple sources.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of the SourceHypoGroupManager managing the source
            hypothesis groups.
        src_fitparam_mapper : SingleSourceFitParameterMapper
            The instance of SingleSourceFitParameterMapper defining the global
            fit parameters and their mapping to the source fit parameters.
        detsigyields : (N_source_hypo_groups,)-shaped 1D ndarray of
                DetSigYield instances
            The collection of DetSigYield instances for each source hypothesis
            group.
        """

        if(not isinstance(src_fitparam_mapper, SingleSourceFitParameterMapper)):
            raise TypeError('The src_fitparam_mapper argument must be an '
                'instance of SingleSourceFitParameterMapper!')

        super(MultiPointSourcesRelSourceWeights, self).__init__(
            src_hypo_group_manager, src_fitparam_mapper, detsigyields)

    def __call__(self, fitparam_values):
        """Calculates the source weights and its fit parameter gradients
        for each source.

        Parameters
        ----------
        fitparam_values : (N_fitparams+1,)-shaped 1D numpy ndarray
            The ndarray holding the current values of the fit parameters.
            The first element of that array is, by definition, the number of
            signal events, ns.

        Returns
        -------
        f : (N_sources,)-shaped 1D ndarray
            The source weight factor for each source.
        f_grads : (N_sources,)-shaped 1D ndarray | None
            The gradients of the source weight factors. None is returned if
            there are no fit parameters beside ns.
        """
        fitparams_arr = self._src_fitparam_mapper.get_fitparams_array(fitparam_values[1:])

        N_fitparams = self._src_fitparam_mapper.n_global_fitparams

        Y = []
        Y_grads = []

        # Loop over detector signal efficiency instances for each source
        # hypothesis group in source hypothesis group manager.
        for (g, detsigyield) in enumerate(self._detsigyield_arr):
            (Yg, Yg_grads) = detsigyield(self._src_arr_list[g], fitparams_arr)

            # Store the detector signal yield and its fit parameter
            # gradients for all sources.
            Y.append(Yg)
            if(N_fitparams > 0):
                Y_grads.append(Yg_grads.T)

        Y = np.array(Y)
        sum_Y = np.sum(Y)

        # f is a (N_sources,)-shaped 1D ndarray.
        f = Y / sum_Y

        # Flatten the array so that each relative weight corresponds to specific
        # source.
        f = f.flatten()

        if(N_fitparams > 0):
            Y_grads = np.concatenate(Y_grads)

            # Sum over fit parameter gradients axis.
            # f_grads is a (N_sources,)-shaped 1D ndarray.
            f_grads = np.sum(Y_grads, axis=1) / sum_Y
        else:
            f_grads = None

        return (f, f_grads)
