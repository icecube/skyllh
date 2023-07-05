# -*- coding: utf-8 -*-

import abc

import numpy as np

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Parabola1DGridManifoldInterpolationMethod,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.pdf import (
    PDFSet,
    IsBackgroundPDF,
    IsSignalPDF,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    issequence,
    issequenceof,
)
from skyllh.core.services import (
    SrcDetSigYieldWeightsService,
)
from skyllh.core.timing import (
    TaskTimer,
)


class PDFRatio(
        HasConfig,
        metaclass=abc.ABCMeta,
):
    """Abstract base class for a signal over background PDF ratio class.
    It defines the interface of a signal over background PDF ratio class.
    """

    def __init__(
            self,
            sig_param_names=None,
            bkg_param_names=None,
            **kwargs,
    ):
        """Creates a new PDFRatio instance.

        Parameters
        ----------
        sig_param_names : sequence of str | str | None
            The sequence of signal parameter names this PDFRatio instance is a
            function of.
        bkg_param_names : sequence of str | str | None
            The sequence of background parameter names this PDFRatio instance
            is a function of.
        """
        super().__init__(**kwargs)

        self.sig_param_names = sig_param_names
        self.bkg_param_names = bkg_param_names

    @property
    def n_params(self):
        """(read-only) The number of parameters the PDF ratio depends on.
        This is the sum of signal and background parameters.
        """
        return self.n_sig_params + self.n_bkg_params

    @property
    def param_names(self):
        """(read-only) The list of parameter names this PDF ratio is a
        function of. This is the superset of signal and background parameter
        names.
        """
        return list(
            set(list(self._sig_param_names) + list(self._bkg_param_names)))

    @property
    def n_sig_params(self):
        """(read-only) The number of signal parameters the PDF ratio depends
        on.
        """
        return len(self._sig_param_names)

    @property
    def n_bkg_params(self):
        """(read-only) The number of background parameters the PDF ratio depends
        on.
        """
        return len(self._bkg_param_names)

    @property
    def sig_param_names(self):
        """The list of signal parameter names this PDF ratio is a function of.
        """
        return self._sig_param_names

    @sig_param_names.setter
    def sig_param_names(self, names):
        if names is None:
            names = []
        if not issequence(names):
            names = [names]
        if not issequenceof(names, str):
            raise TypeError(
                'The sig_param_names property must be a sequence of str '
                'instances!')
        self._sig_param_names = names

    @property
    def bkg_param_names(self):
        """The list of background parameter names this PDF ratio is a function
        of.
        """
        return self._bkg_param_names

    @bkg_param_names.setter
    def bkg_param_names(self, names):
        if names is None:
            names = []
        if not issequence(names):
            names = [names]
        if not issequenceof(names, str):
            raise TypeError(
                'The bkg_param_names property must be a sequence of str '
                'instances!')
        self._bkg_param_names = names

    @abc.abstractmethod
    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """Initializes the PDFRatio instance for a new trial. This method can
        be utilized to pre-calculate PDFRatio values that do not depend on any
        fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial data.
        tl : instance of TimeLord
            The optional instance of TimeLord to measure timing information.
        """
        pass

    @abc.abstractmethod
    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None,
    ):
        """Retrieves the PDF ratio value for each given trial data events (and
        sources), given the given set of parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N_values,)-shaped 1d numpy ndarray of float holding the PDF
            ratio value for each trial event and source.
        """
        pass

    @abc.abstractmethod
    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None,
    ):
        """Retrieves the PDF ratio gradient for the global fit parameter
        ``fitparam_id`` for each trial data event and source, given the given
        set of parameters ``src_params_recarray`` for each source.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio gradient values should get calculated.
        src_params_recarray : instance of numpy structured ndarray
            The (N_sources,)-shaped numpy structured ndarray holding the
            parameter names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        fitparam_id : int
            The ID of the global fit parameter for which the gradient should
            get calculated.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        gradient : instance of ndarray | 0
            The (N_values,)-shaped 1d numpy ndarray of float holding the PDF
            ratio gradient value for each source and trial data event.
            If the PDF ratio does not depend on the given global fit parameter,
            0 should be returned.
        """
        pass

    def __mul__(self, other):
        """Implements the mathematical operation ``new = self * other``, where
        ``other`` is an instance of PDFRatio. It creates an instance of
        PDFRatioProduct holding the two PDFRatio instances.
        """
        return PDFRatioProduct(self, other, cfg=self.cfg)


class PDFRatioProduct(
        PDFRatio,
):
    """This is the mathematical product of two PDFRatio instances, which is a
    PDFRatio instance again.
    """
    def __init__(
            self,
            pdfratio1,
            pdfratio2,
            **kwargs,
    ):
        """Creates a new PDFRatioProduct instance representing the product of
        two PDFRatio instances.
        """
        self.pdfratio1 = pdfratio1
        self.pdfratio2 = pdfratio2

        sig_param_names = set(
            list(pdfratio1.sig_param_names) + list(pdfratio2.sig_param_names))
        bkg_param_names = set(
            list(pdfratio1.bkg_param_names) + list(pdfratio2.bkg_param_names))

        super().__init__(
            sig_param_names=sig_param_names,
            bkg_param_names=bkg_param_names,
            **kwargs)

    @property
    def pdfratio1(self):
        """The first PDFRatio instance in the muliplication
        ``pdfratio1 * pdfratio2``.
        """
        return self._pdfratio1

    @pdfratio1.setter
    def pdfratio1(self, pdfratio):
        if not isinstance(pdfratio, PDFRatio):
            raise TypeError(
                'The pdfratio1 property must be an instance of PDFRatio!')
        self._pdfratio1 = pdfratio

    @property
    def pdfratio2(self):
        """The second PDFRatio instance in the muliplication
        ``pdfratio1 * pdfratio2``.
        """
        return self._pdfratio2

    @pdfratio2.setter
    def pdfratio2(self, pdfratio):
        if not isinstance(pdfratio, PDFRatio):
            raise TypeError(
                'The pdfratio2 property must be an instance of PDFRatio!')
        self._pdfratio2 = pdfratio

    def initialize_for_new_trial(
            self,
            **kwargs):
        """Initializes the PDFRatioProduct instance for a new trial.
        It calls the
        :meth:`~skyllh.core.pdfratio.PDFRatio.initialize_for_new_trial` method
        of each of the two :class:`~skyllh.core.pdfratio.PDFRatio` instances.
        """
        self._pdfratio1.initialize_for_new_trial(**kwargs)
        self._pdfratio2.initialize_for_new_trial(**kwargs)

    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None,
    ):
        """Retrieves the PDF ratio product value for each trial data
        event and source, given the given set of parameters for all sources.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        src_params_recarray : instance of numpy record ndarray
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N_values,)-shaped 1d numpy ndarray of float holding the product
            of the PDF ratio values for each trial event and source.
            The PDF ratio product value for each trial event.
        """
        r1 = self._pdfratio1.get_ratio(
            tdm=tdm,
            src_params_recarray=src_params_recarray,
            tl=tl)

        r2 = self._pdfratio2.get_ratio(
            tdm=tdm,
            src_params_recarray=src_params_recarray,
            tl=tl)

        return r1 * r2

    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None,
    ):
        """Retrieves the PDF ratio product gradient for the global fit parameter
        with parameter ID ``fitparam_id`` for each trial data event and source,
        given the set of parameters ``src_params_recarray`` for all sources.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        fitparam_id : int
            The ID of the global fit parameter for which the gradient should
            get calculated.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        gradient : instance of ndarray | 0
            The (N_values,)-shaped 1d numpy ndarray of float holding the PDF
            ratio gradient value for each trial event and source. If none of the
            two PDFRatio instances depend on the given global fit parameter, the
            scalar value ``0`` is returned.
        """
        r1_depends_on_fitparam =\
            ParameterModelMapper.is_global_fitparam_a_local_param(
                fitparam_id=fitparam_id,
                params_recarray=src_params_recarray,
                local_param_names=self._pdfratio1.param_names)

        r2_depends_on_fitparam =\
            ParameterModelMapper.is_global_fitparam_a_local_param(
                fitparam_id=fitparam_id,
                params_recarray=src_params_recarray,
                local_param_names=self._pdfratio2.param_names)

        if r1_depends_on_fitparam:
            r2 = self._pdfratio2.get_ratio(
                tdm=tdm,
                src_params_recarray=src_params_recarray,
                tl=tl)

            r1_grad = self._pdfratio1.get_gradient(
                tdm=tdm,
                src_params_recarray=src_params_recarray,
                fitparam_id=fitparam_id,
                tl=tl)

        if r2_depends_on_fitparam:
            r1 = self._pdfratio1.get_ratio(
                tdm=tdm,
                src_params_recarray=src_params_recarray,
                tl=tl)

            r2_grad = self._pdfratio2.get_gradient(
                tdm=tdm,
                src_params_recarray=src_params_recarray,
                fitparam_id=fitparam_id,
                tl=tl)

        if r1_depends_on_fitparam and r2_depends_on_fitparam:
            gradient = r1 * r2_grad
            gradient += r1_grad * r2
        elif r1_depends_on_fitparam:
            gradient = r1_grad * r2
        elif r2_depends_on_fitparam:
            gradient = r1 * r2_grad
        else:
            gradient = 0

        return gradient


class SourceWeightedPDFRatio(
        PDFRatio):
    r"""This class provides the calculation of a source weighted PDF ratio for
    multiple sources:

    .. math::

        \mathcal{R}_i(\vec{p}_{\mathrm{s}}) = \frac{1}{A(\vec{p}_{\mathrm{s}})}
            \sum_{k=1}^{K} a_k(\vec{p}_{\mathrm{s}_k}) \mathcal{R}_{i,k}
            (\vec{p}_{\mathrm{s}_k})

    """
    def __init__(
            self,
            dataset_idx,
            src_detsigyield_weights_service,
            pdfratio,
            **kwargs):
        """Creates a new SourceWeightedPDFRatio instance.

        Parameters
        ----------
        dataset_idx : int
            The index of the dataset. It is used to access the source detector
            signal yield weight.
        src_detsigyield_weights_service : instance of SrcDetSigYieldWeightsService
            The instance of SrcDetSigYieldWeightsService providing the source
            detector signal yield weights, i.e. the product of the theoretical
            source weight with the detector signal yield.
        pdfratio : instance of PDFRatio
            The instance of PDFRatio providing the PDF ratio values and
            derivatives.
        """
        if not isinstance(pdfratio, PDFRatio):
            raise TypeError(
                'The pdfratio argument must be an instance of PDFRatio! '
                f'Its current type is {classname(pdfratio)}.')
        self._pdfratio = pdfratio

        super().__init__(
            sig_param_names=self._pdfratio.sig_param_names,
            bkg_param_names=self._pdfratio.bkg_param_names,
            **kwargs)

        self._dataset_idx = int_cast(
            dataset_idx,
            'The dataset_idx argument must be castable to type int!')

        if not isinstance(
                src_detsigyield_weights_service,
                SrcDetSigYieldWeightsService):
            raise TypeError(
                'The src_detsigyield_weights_service argument must be an '
                'instance of type SrcDetSigYieldWeightsService! '
                'Its current type is '
                f'{classname(src_detsigyield_weights_service)}.')
        self._src_detsigyield_weights_service = src_detsigyield_weights_service

        self._cache_R_ik = None
        self._cache_R_i = None

    @property
    def dataset_idx(self):
        """(read-only) The index of the dataset for which this
        SourceWeightedPDFRatio instance is made.
        """
        return self._dataset_idx

    @property
    def src_detsigyield_weights_service(self):
        """(read-only) The instance of SrcDetSigYieldWeightsService providing
        the source detector signal yield weights, i.e. the product of the
        theoretical source weight with the detector signal yield.
        """
        return self._src_detsigyield_weights_service

    @property
    def pdfratio(self):
        """(read-only) The PDFRatio instance that is used to calculate the
        source weighted PDF ratio.
        """
        return self._pdfratio

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Initializes the PDFRatio instance for a new trial. It calls the
        :meth:`~skyllh.core.pdfratio.PDFRatio.initialize_for_new_trial` method
        of the :class:`~skyllh.core.pdfratio.PDFRatio` instance.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial data.
        tl : instance of TimeLord
            The optional instance of TimeLord to measure timing information.
        """
        self._pdfratio.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None):
        """Retrieves the PDF ratio value for each given trial data events (and
        sources), given the given set of parameters.

        Note:

            This method uses the source detector signal yield weights service.
            Hence, the
            :meth:`skyllh.core.weights.SrcDetSigYieldWeightsService.calculate`
            method needs to be called prior to calling this method.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        src_params_recarray : instance of numpy record ndarray
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N_selected_events,)-shaped 1d numpy ndarray of float holding
            the PDF ratio value for each selected trial data event.
        """
        (a_jk, a_jk_grads) = self._src_detsigyield_weights_service.get_weights()
        a_k = a_jk[self._dataset_idx]

        n_sources = len(a_k)
        n_sel_events = tdm.n_selected_events

        A = np.sum(a_k)

        R_ik = self._pdfratio.get_ratio(
            tdm=tdm,
            src_params_recarray=src_params_recarray,
            tl=tl)
        # The R_ik ndarray is (N_values,)-shaped.

        R_i = np.zeros((n_sel_events,), dtype=np.double)

        (src_idxs, evt_idxs) = tdm.src_evt_idxs
        for k in range(n_sources):
            src_mask = src_idxs == k
            R_i[evt_idxs[src_mask]] += R_ik[src_mask] * a_k[k]
        R_i /= A

        self._cache_R_ik = R_ik
        self._cache_R_i = R_i

        return R_i

    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None):
        """Retrieves the PDF ratio gradient for the parameter ``fitparam_id``
        for each trial data event, given the given set of parameters
        ``src_params_recarray`` for each source.

        Note:

            This method requires that the get_ratio method has been called prior
            to calling this method.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio gradient values should get calculated.
        src_params_recarray : instance of numpy record ndarray
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        fitparam_id : int
            The ID of the global fit parameter for which the gradient should
            get calculated.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        gradient : instance of ndarray | 0
            The (N_selected_events,)-shaped 1d numpy ndarray of float holding
            the PDF ratio gradient value for each trial data event. If the PDF
            ratio does not depend on the given global fit parameter, 0 will
            be returned.
        """
        (a_jk, a_jk_grads) = self._src_detsigyield_weights_service.get_weights()

        a_k = a_jk[self._dataset_idx]
        A = np.sum(a_k)

        n_sources = a_jk.shape[1]
        n_sel_events = tdm.n_selected_events

        if fitparam_id not in a_jk_grads:
            a_k_grad = 0
            dAdp = 0
        else:
            a_k_grad = a_jk_grads[fitparam_id][self._dataset_idx]
            dAdp = np.sum(a_k_grad)

        R_ik_grad = self._pdfratio.get_gradient(
            tdm=tdm,
            src_params_recarray=src_params_recarray,
            fitparam_id=fitparam_id)
        # R_ik_grad is a (N_values,)-shaped ndarray or 0.

        if (type(a_k_grad) == int) and (a_k_grad == 0) and\
           (type(R_ik_grad) == int) and (R_ik_grad == 0):
            return 0

        R_i_grad = -self._cache_R_i * dAdp

        src_sum_i = np.zeros((n_sel_events,), dtype=np.double)

        (src_idxs, evt_idxs) = tdm.src_evt_idxs
        for k in range(n_sources):
            src_mask = src_idxs == k
            src_evt_idxs = evt_idxs[src_mask]
            if isinstance(a_k_grad, np.ndarray):
                src_sum_i[src_evt_idxs] +=\
                    a_k_grad[k] * self._cache_R_ik[src_mask]
            if isinstance(R_ik_grad, np.ndarray):
                src_sum_i[src_evt_idxs] +=\
                    a_k[k] * R_ik_grad[src_mask]

        R_i_grad += src_sum_i
        R_i_grad /= A

        return R_i_grad


class SigOverBkgPDFRatio(
        PDFRatio):
    """This class implements a generic signal-over-background PDF ratio for a
    signal and a background PDF instance.
    It takes a signal PDF of type *pdf_type* and a background PDF of type
    *pdf_type* and calculates the PDF ratio.
    """
    def __init__(
            self,
            sig_pdf,
            bkg_pdf,
            same_axes=True,
            zero_bkg_ratio_value=1.,
            **kwargs):
        """Creates a new signal-over-background PDF ratio instance.

        Parameters
        ----------
        sig_pdf : class instance derived from `pdf_type`, IsSignalPDF
            The instance of the signal PDF.
        bkg_pdf : class instance derived from `pdf_type`, IsBackgroundPDF
            The instance of the background PDF.
        same_axes : bool
            Flag if the signal and background PDFs are supposed to have the
            same axes. Default is True.
        zero_bkg_ratio_value : float
            The value of the PDF ratio to take when the background PDF value
            is zero. This is to avoid division by zero. Default is 1.
        """
        super().__init__(
            sig_param_names=sig_pdf.param_set.params_name_list,
            bkg_param_names=bkg_pdf.param_set.params_name_list,
            **kwargs)

        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf

        # Check that the PDF axes ranges are the same for the signal and
        # background PDFs.
        if same_axes and not sig_pdf.axes.is_same_as(bkg_pdf.axes):
            raise ValueError(
                'The signal and background PDFs do not have the same axes!')

        self.zero_bkg_ratio_value = zero_bkg_ratio_value

        # Define cache member variables to calculate gradients efficiently.
        self._cache_sig_pd = None
        self._cache_bkg_pd = None
        self._cache_sig_grads = None
        self._cache_bkg_grads = None

    @property
    def sig_pdf(self):
        """The signal PDF object used to create the PDF ratio.
        """
        return self._sig_pdf

    @sig_pdf.setter
    def sig_pdf(self, pdf):
        if not isinstance(pdf, IsSignalPDF):
            raise TypeError(
                'The sig_pdf property must be an instance of IsSignalPDF! '
                f'Its type is "{classname(pdf)}".')
        self._sig_pdf = pdf

    @property
    def bkg_pdf(self):
        """The background PDF object used to create the PDF ratio.
        """
        return self._bkg_pdf

    @bkg_pdf.setter
    def bkg_pdf(self, pdf):
        if not isinstance(pdf, IsBackgroundPDF):
            raise TypeError(
                'The bkg_pdf property must be an instance of IsBackgroundPDF! '
                f'Its type is "{classname(pdf)}".')
        self._bkg_pdf = pdf

    @property
    def zero_bkg_ratio_value(self):
        """The value of the PDF ratio to take when the background PDF value
        is zero. This is to avoid division by zero.
        """
        return self._zero_bkg_ratio_value

    @zero_bkg_ratio_value.setter
    def zero_bkg_ratio_value(self, v):
        v = float_cast(
            v,
            'The zero_bkg_ratio_value must be castable to type float!')
        self._zero_bkg_ratio_value = v

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Initializes the PDFRatio instance for a new trial. It calls the
        :meth:`~skyllh.core.pdf.PDF.assert_is_valid_for_trial_data` of the
        signal and background :class:`~skyllh.core.pdf.PDF` instances.
        """
        self._sig_pdf.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)
        self._sig_pdf.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

        self._bkg_pdf.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)
        self._bkg_pdf.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None):
        """Calculates the PDF ratio for the given trial events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should be calculated.
        src_params_recarray : instance of numpy record ndarray
            The (N_sources,)-shaped numpy record ndarray holding the local
            parameter names and values of the sources.
            See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            ratio for each event and source.
        """
        with TaskTimer(tl, 'Get sig probability densities and grads.'):
            (self._cache_sig_pd, self._cache_sig_grads) = self._sig_pdf.get_pd(
                tdm=tdm,
                params_recarray=src_params_recarray,
                tl=tl)
        with TaskTimer(tl, 'Get bkg probability densities and grads.'):
            (self._cache_bkg_pd, self._cache_bkg_grads) = self._bkg_pdf.get_pd(
                tdm=tdm,
                params_recarray=None,
                tl=tl)

        with TaskTimer(tl, 'Calculate PDF ratios.'):
            # Select only the events, where the background pdf is greater than
            # zero.
            ratios = np.full_like(self._cache_sig_pd, self._zero_bkg_ratio_value)
            m = (self._cache_bkg_pd > 0)
            (m, bkg_pd) = tdm.broadcast_selected_events_arrays_to_values_arrays(
                (m, self._cache_bkg_pd))
            np.divide(
                self._cache_sig_pd,
                bkg_pd,
                where=m,
                out=ratios)

        return ratios

    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None):
        """Retrieves the gradient of the PDF ratio w.r.t. the given parameter.

        Note:

            This method uses cached values from the
            :meth:`~skyllh.core.pdfratio.SigOverBkgPDFRatio.get_ratio` method.
            Hence, that method needs to be called prior to this method.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_models,)-shaped numpy record ndarray holding the parameter
            names and values of the models.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for more information.
            This can be ``None``, if the signal and background PDFs do not
            depend on any parameters.
        fitparam_id : int
            The ID of the global fit parameter for which the gradient should
            get calculated.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        grad : instance of ndarray
            The (N_values,)-shaped 1d numpy ndarray of float holding the PDF
            ratio gradient value for each source and trial event.
        """
        # Create the 1D return array for the gradient.
        grad = np.zeros_like(self._cache_sig_pd, dtype=np.float64)

        # Calculate the gradient for the given parameter.
        # There are four cases:
        #   1) Neither the signal nor the background PDF depend on the
        #      parameter.
        #   2) Only the signal PDF depends on the parameter.
        #   3) Only the background PDF depends on the parameter.
        #   4) Both, the signal and the background PDF depend on the
        #      parameter.
        sig_dep = fitparam_id in self._cache_sig_grads
        bkg_dep = fitparam_id in self._cache_bkg_grads

        if (not sig_dep) and (not bkg_dep):
            # Case 1. Return zeros.
            return grad

        m = self._cache_bkg_pd > 0
        b = self._cache_bkg_pd

        (m, b) = tdm.broadcast_selected_events_arrays_to_values_arrays(
            (m, b))

        if sig_dep and not bkg_dep:
            # Case 2, which should be the most common case.
            grad[m] = self._cache_sig_grads[fitparam_id][m] / b[m]
            return grad

        bgrad = self._cache_bkg_grads[fitparam_id]
        (bgrad,) = tdm.broadcast_selected_events_arrays_to_values_arrays(
            (bgrad,))

        if sig_dep and bkg_dep:
            # Case 4.
            s = self._cache_sig_pd
            sgrad = self._cache_sig_grads[fitparam_id]

            # Make use of quotient rule of differentiation.
            grad[m] = (sgrad[m] * b[m] - bgrad[m] * s[m]) / b[m]**2
            return grad

        # Case 3.
        grad[m] = -self._cache_sig_pd[m] / b[m]**2 * bgrad[m]

        return grad


class SigSetOverBkgPDFRatio(
        PDFRatio):
    """Class for a PDF ratio class that takes a PDFSet as signal PDF and a PDF
    as background PDF.
    The signal PDF depends on signal parameters and an interpolation method
    defines how the PDF ratio gets interpolated between the parameter grid
    values.
    """
    def __init__(
            self,
            sig_pdf_set,
            bkg_pdf,
            interpolmethod_cls=None,
            **kwargs):
        """Constructor called by creating an instance of a class which is
        derived from this PDFRatio class.

        Parameters
        ----------
        sig_pdf_set : instance of PDFSet and instance of IsSignalPDF
            The PDF set, which provides signal PDFs for a set of
            discrete signal parameter values.
        bkg_pdf : instance of PDF and instance of IsBackgroundPDF
            The background PDF instance.
        interpolmethod_cls : class of GridManifoldInterpolationMethod | None
            The class implementing the parameter interpolation method for
            the PDF ratio manifold grid. If set to ``None`` (default), the
            :class:`skyllh.core.interpolate.Parabola1DGridManifoldInterpolationMethod`
            will be used for 1-dimensional parameter manifolds.
        """
        super().__init__(
            sig_param_names=sig_pdf_set.param_grid_set.params_name_list,
            bkg_param_names=bkg_pdf.param_set.params_name_list,
            **kwargs)

        self.sig_pdf_set = sig_pdf_set
        self.bkg_pdf = bkg_pdf

        # Define the default parameter interpolation method. The default
        # depends on the dimensionality of the parameter manifold.
        if interpolmethod_cls is None:
            ndim = self._sig_pdf_set.param_grid_set.ndim
            if ndim == 1:
                interpolmethod_cls = Parabola1DGridManifoldInterpolationMethod
            else:
                raise ValueError(
                    'There is no default parameter manifold grid '
                    f'interpolation method class available for {ndim} '
                    'dimensions!')
        self.interpolmethod_cls = interpolmethod_cls

    @property
    def bkg_pdf(self):
        """The background PDF instance, derived from IsBackgroundPDF.
        """
        return self._bkg_pdf

    @bkg_pdf.setter
    def bkg_pdf(self, pdf):
        if not isinstance(pdf, IsBackgroundPDF):
            raise TypeError(
                'The bkg_pdf property must be an instance derived from '
                'IsBackgroundPDF! '
                f'Its current type is {classname(pdf)}.')
        self._bkg_pdf = pdf

    @property
    def sig_pdf_set(self):
        """The signal PDFSet instance, derived from IsSignalPDF.
        """
        return self._sig_pdf_set

    @sig_pdf_set.setter
    def sig_pdf_set(self, pdfset):
        if not (isinstance(pdfset, PDFSet) and
                isinstance(pdfset, IsSignalPDF)):
            raise TypeError(
                'The sig_pdf_set property must be a class instance which is '
                'derived from PDFSet and IsSignalPDF! '
                f'Its current type is {classname(pdfset)}.')
        self._sig_pdf_set = pdfset

    @property
    def interpolmethod_cls(self):
        """The class derived from GridManifoldInterpolationMethod
        implementing the interpolation of the parameter manifold.
        """
        return self._interpolmethod_cls

    @interpolmethod_cls.setter
    def interpolmethod_cls(self, cls):
        if not issubclass(cls, GridManifoldInterpolationMethod):
            raise TypeError(
                'The interpolmethod_cls property must be a sub-class '
                'of GridManifoldInterpolationMethod! '
                f'Its current type is {classname(cls)}.')
        self._interpolmethod_cls = cls

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Initializes the PDFRatio instance for a new trial. It calls the
        :meth:`~skyllh.core.pdf.PDF.assert_is_valid_for_trial_data` of the
        signal :class:`~skyllh.core.pdf.PDFSet` instance and the background
        :class:`~skyllh.core.pdf.PDF` instance.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord for measuring timing information.
        """
        self._sig_pdf_set.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)
        self._sig_pdf_set.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)

        self._bkg_pdf.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)
        self._bkg_pdf.assert_is_valid_for_trial_data(
            tdm=tdm,
            tl=tl,
            **kwargs)
