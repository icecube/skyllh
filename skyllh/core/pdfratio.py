# -*- coding: utf-8 -*-

import abc
import scipy.interpolate

import numpy as np

from numpy.lib.recfunctions import (
    repack_fields,
)

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Parabola1DGridManifoldInterpolationMethod,
)
from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.pdf import (
    PDFSet,
    IsBackgroundPDF,
    IsSignalPDF,
)
from skyllh.core.pdfratio_fill import (
    MostSignalLikePDFRatioFillMethod,
    PDFRatioFillMethod,
)
from skyllh.core.py import (
    classname,
    float_cast,
    int_cast,
    issequence,
    issequenceof,
    make_dict_hash,
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


class SplinedSingleConditionalEnergySigSetOverBkgPDFRatio(
        SigSetOverBkgPDFRatio,
        IsParallelizable,
):
    """This class implements a splined signal over background PDF ratio for
    enegry PDFs of type SingleConditionalEnergyPDF.
    It takes an instance, which is derived from PDFSet, and which is derived
    from IsSignalPDF, as signal PDF. Furthermore, it takes an instance, which
    is derived from SingleConditionalEnergyPDF and IsBackgroundPDF, as
    background PDF, and creates a spline for the ratio of the signal and
    background PDFs for a grid of different discrete energy signal parameters,
    which are defined by the signal PDF set.
    """
    def __init__(
            self,
            sig_pdf_set,
            bkg_pdf,
            fillmethod=None,
            interpolmethod_cls=None,
            ncpu=None,
            ppbar=None,
            **kwargs,
    ):
        """Creates a new IceCube signal-over-background energy PDF ratio spline
        instance.

        Parameters
        ----------
        sig_pdf_set : instance of PDFSet of SingleConditionalEnergyPDF and IsSignalPDF
            The PDF set, which provides signal energy PDFs for a set of
            discrete signal parameters.
        bkg_pdf : instance of SingleConditionalEnergyPDF and IsBackgroundPDF
            The background energy PDF instance.
        fillmethod : instance of PDFRatioFillMethod | None
            An instance of class derived from PDFRatioFillMethod that implements
            the desired ratio fill method.
            If set to None (default), the default ratio fill method
            MostSignalLikePDFRatioFillMethod will be used.
        interpolmethod_cls : class of GridManifoldInterpolationMethod
            The class implementing the parameter interpolation method for
            the PDF ratio manifold grid.
        ncpu : int | None
            The number of CPUs to use to create the ratio splines for the
            different sets of signal parameters.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Raises
        ------
        ValueError
            If the signal and background PDFs use different binning.
        """
        super().__init__(
            sig_pdf_set=sig_pdf_set,
            bkg_pdf=bkg_pdf,
            interpolmethod_cls=interpolmethod_cls,
            ncpu=ncpu,
            **kwargs)

        # Define the default ratio fill method.
        if fillmethod is None:
            fillmethod = MostSignalLikePDFRatioFillMethod()
        self.fillmethod = fillmethod

        # Ensure same binning of signal and background PDFs.
        for sig_pdf in self._sig_pdf_set.values():
            if not sig_pdf.has_same_binning_as(self._bkg_pdf):
                raise ValueError(
                    'At least one signal PDF does not have the same binning '
                    'as the background PDF!')

        def create_log_ratio_spline(
                sig_pdf_set,
                bkg_pdf,
                fillmethod,
                gridparams,
        ):
            """Creates the signal/background ratio spline for the given signal
            parameters.

            Returns
            -------
            log_ratio_spline : instance of RegularGridInterpolator
                The spline of the logarithmic PDF ratio values.
            """
            # Get the signal PDF for the given signal parameters.
            sig_pdf = sig_pdf_set[gridparams]

            # Create the ratio array with the same shape than the background pdf
            # histogram.
            ratio = np.ones_like(bkg_pdf.hist, dtype=np.float64)

            # Fill the ratio array.
            ratio = fillmethod(
                ratio,
                sig_pdf.hist,
                bkg_pdf.hist,
                sig_pdf.hist_mask_mc_covered,
                sig_pdf.hist_mask_mc_covered_zero_physics,
                bkg_pdf.hist_mask_mc_covered,
                bkg_pdf.hist_mask_mc_covered_zero_physics)

            # Define the grid points for the spline. In general, we use the bin
            # centers of the binning, but for the first and last point of each
            # dimension we use the lower and upper bin edge, respectively, to
            # ensure full coverage of the spline across the binning range.
            points_list = []
            for binning in sig_pdf.binnings:
                points = binning.bincenters
                (points[0], points[-1]) = (
                    binning.lower_edge, binning.upper_edge)
                points_list.append(points)

            # Create the spline for the ratio values.
            log_ratio_spline = scipy.interpolate.RegularGridInterpolator(
                tuple(points_list),
                np.log(ratio),
                method='linear',
                bounds_error=False,
                fill_value=0.)

            return log_ratio_spline

        # Get the list of parameter permutations on the grid for which we
        # need to create PDF ratio splines.
        gridparams_list = self._sig_pdf_set.gridparams_list

        args_list = [
            ((self._sig_pdf_set,
              self._bkg_pdf,
              self._fillmethod,
              gridparams),
             {})
            for gridparams in gridparams_list
        ]

        log_ratio_spline_list = parallelize(
            func=create_log_ratio_spline,
            args_list=args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        # Save all the log_ratio splines in a dictionary.
        self._gridparams_hash_log_ratio_spline_dict = dict()
        for (gridparams, log_ratio_spline) in zip(gridparams_list,
                                                  log_ratio_spline_list):
            gridparams_hash = make_dict_hash(gridparams)
            self._gridparams_hash_log_ratio_spline_dict[gridparams_hash] =\
                log_ratio_spline

        # Save the list of data field names.
        self._data_field_names = [
            binning.name
            for binning in self._bkg_pdf.binnings
        ]

        # Construct the instance for the parameter interpolation method.
        self._interpolmethod = self._interpolmethod_cls(
            func=self._evaluate_splines,
            param_grid_set=sig_pdf_set.param_grid_set)

        # Save the parameter names needed for the interpolation for later usage.
        self._interpol_param_names = \
            self._sig_pdf_set.param_grid_set.params_name_list

        # Create cache variable for the last ratio values and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache = self._create_cache(
            trial_data_state_id=None,
            interpol_params_recarray=None,
            ratio=None,
            grads=None
        )

    @property
    def fillmethod(self):
        """The PDFRatioFillMethod object, which should be used for filling the
        PDF ratio bins.
        """
        return self._fillmethod

    @fillmethod.setter
    def fillmethod(self, obj):
        if not isinstance(obj, PDFRatioFillMethod):
            raise TypeError(
                'The fillmethod property must be an instance of '
                'PDFRatioFillMethod!')
        self._fillmethod = obj

    def _create_cache(
            self,
            trial_data_state_id,
            interpol_params_recarray,
            ratio,
            grads):
        """Creates a cache dictionary holding cache data.

        Parameters
        ----------
        trial_data_state_id : int | None
            The trial data state ID of the TrialDataManager.
        interpol_params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values necessary for the interpolation for all sources.
        ratio : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the PDF ratio values
            for all sources and trial events.
        grads : instance of numpy ndarray
            The (D,N_values)-shaped numpy ndarray holding the gradients for each
            PDF ratio value w.r.t. each interpolation parameter.
        """
        cache = {
            'trial_data_state_id': trial_data_state_id,
            'interpol_params_recarray': interpol_params_recarray,
            'ratio': ratio,
            'grads': grads
        }

        return cache

    def _is_cached(self, trial_data_state_id, interpol_params_recarray):
        """Checks if the ratio and gradients for the given set of interpolation
        parameters are already cached.
        """
        if self._cache['trial_data_state_id'] is None:
            return False

        if self._cache['trial_data_state_id'] != trial_data_state_id:
            return False

        if not np.all(
                self._cache['interpol_params_recarray'] ==
                interpol_params_recarray):
            return False

        return True

    def _get_spline_for_param_values(self, interpol_param_values):
        """Retrieves the spline for a given set of parameter values.

        Parameters
        ----------
        interpol_param_values : instance of numpy ndarray
            The (N_interpol_params,)-shaped numpy ndarray holding the values of
            the interpolation parameters.

        Returns
        -------
        spline : instance of scipy.interpolate.RegularGridInterpolator
            The requested spline instance.
        """
        gridparams = dict(
            zip(self._interpol_param_names, interpol_param_values))
        gridparams_hash = make_dict_hash(gridparams)

        spline = self._gridparams_hash_log_ratio_spline_dict[gridparams_hash]

        return spline

    def _evaluate_splines(
            self,
            tdm,
            eventdata,
            gridparams_recarray,
            n_values):
        """For each set of parameter values given by ``gridparams_recarray``,
        the spline is retrieved and evaluated for the events suitable for that
        source model.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data and the event
            mapping to the sources via the ``src_evt_idx`` property.
        eventdata : instance of numpy ndarray
            The (N_events,V)-shaped numpy ndarray holding the event data, where
            N_events is the number of events, and V the dimensionality of the
            event data.
        gridparams_recarray : instance of numpy structured ndarray
            The numpy structured ndarray of length N_sources with the parameter
            names and values needed for the interpolation on the grid for all
            sources. If the length of this record array is 1, the set of
            parameters will be used for all sources.
        n_values : int
            The size of the output array.

        Returns
        -------
        values : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the values for each set
            of parameter values of the ``gridparams_recarray``. The length of
            the array depends on the ``src_evt_idx`` property of the
            TrialDataManager. In the worst case it is
            ``N_sources * N_selected_events``.
        """
        (src_idxs, evt_idxs) = tdm.src_evt_idxs

        # Check for special case when a single set of parameters are provided.
        if len(gridparams_recarray) == 1:
            # We got a single parameter set. We will use it for all sources.
            spline = self._get_spline_for_param_values(gridparams_recarray[0])

            eventdata = np.take(eventdata, evt_idxs, axis=0)
            values = spline(eventdata)

            return values

        values = np.empty(n_values, dtype=np.float64)

        v_start = 0
        for (sidx, param_values) in enumerate(gridparams_recarray):
            spline = self._get_spline_for_param_values(param_values)

            # Select the eventdata that belongs to the current source.
            m = src_idxs == sidx
            src_eventdata = np.take(eventdata, evt_idxs[m], axis=0)

            n = src_eventdata.shape[0]
            sl = slice(v_start, v_start+n)
            values[sl] = spline(src_eventdata)

            v_start += n

        return values

    def _create_interpol_params_recarray(self, src_params_recarray):
        """Creates the params_recarray needed for the interpolation. It selects
        The interpolation parameters from the ``params_recarray`` argument.
        If all parameters have the same value for all sources, the length will
        be 1.

        Parameters
        ----------
        src_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources holding all local
            parameter names and values.

        Returns
        -------
        interpol_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources or 1 holding only the
            parameters needed for the interpolation.
        """
        interpol_params_recarray = repack_fields(
            src_params_recarray[self._interpol_param_names])

        all_params_are_equal_for_all_sources = True
        for pname in self._interpol_param_names:
            if not np.all(
                    np.isclose(np.diff(interpol_params_recarray[pname]), 0)):
                all_params_are_equal_for_all_sources = False
                break
        if all_params_are_equal_for_all_sources:
            return interpol_params_recarray[:1]

        return interpol_params_recarray

    def _calculate_ratio_and_grads(
            self,
            tdm,
            interpol_params_recarray):
        """Calculates the ratio values and ratio gradients for all the sources
        and trial events given the source parameter values.
        The result is stored in the class member variable ``_cache``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data.
        interpol_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources holding the parameter
            names and values for all sources.
            It must contain only the parameters necessary for the interpolation.
        """
        # Create a 2D event data array holding only the needed event data fields
        # for the PDF ratio spline evaluation.
        eventdata = np.vstack([tdm[fn] for fn in self._data_field_names]).T

        (ratio, grads) = self._interpolmethod(
            tdm=tdm,
            eventdata=eventdata,
            params_recarray=interpol_params_recarray)

        # The interpolation works on the logarithm of the ratio spline, hence
        # we need to transform it using the exp function, and we need to account
        # for the exp function in the gradients.
        ratio = np.exp(ratio)
        grads = ratio * grads

        # Cache the value and the gradients.
        self._cache = self._create_cache(
            trial_data_state_id=tdm.trial_data_state_id,
            interpol_params_recarray=interpol_params_recarray,
            ratio=ratio,
            grads=grads
        )

    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None):
        """Retrieves the PDF ratio values for each given trial event data, given
        the given set of fit parameters. This method is called during the
        likelihood maximization process.
        For computational efficiency reasons, the gradients are calculated as
        well and will be cached.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF ratio values should get calculated.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources. See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            for more information.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratio : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray of float holding the PDF ratio
            value for each source and trial event.
        """
        # Select only the parameters necessary for the interpolation.
        interpol_params_recarray = self._create_interpol_params_recarray(
            src_params_recarray)

        # Check if the ratio values are already cached.
        if self._is_cached(
               trial_data_state_id=tdm.trial_data_state_id,
               interpol_params_recarray=interpol_params_recarray):
            return self._cache['ratio']

        self._calculate_ratio_and_grads(
            tdm=tdm,
            interpol_params_recarray=interpol_params_recarray)

        return self._cache['ratio']

    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None):
        """Retrieves the PDF ratio gradient for the given fit parameter
        ``fitparam_id``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF ratio gradient values should get calculated.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the local
            parameter names and values of all sources. See the
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
        grad : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the gradient values
            for all sources and trial events w.r.t. the given global fit
            parameter.
        """
        # Select only the parameters necessary for the interpolation.
        interpol_params_recarray = self._create_interpol_params_recarray(
            src_params_recarray)

        # Calculate the gradients if necessary.
        if not self._is_cached(
            trial_data_state_id=tdm.trial_data_state_id,
            interpol_params_recarray=interpol_params_recarray
        ):
            self._calculate_ratio_and_grads(
                tdm=tdm,
                interpol_params_recarray=interpol_params_recarray)

        tdm_n_sources = tdm.n_sources

        grad = np.zeros((tdm.get_n_values(),), dtype=np.float64)

        # Loop through the parameters of the signal PDF set and match them with
        # the global fit parameter.
        for (pidx, pname) in enumerate(
                self._sig_pdf_set.param_grid_set.params_name_list):
            if pname not in src_params_recarray.dtype.fields:
                continue
            p_gpidxs = src_params_recarray[f'{pname}:gpidx']
            src_mask = p_gpidxs == (fitparam_id + 1)
            n_sources = np.count_nonzero(src_mask)
            if n_sources == 0:
                continue
            if n_sources == tdm_n_sources:
                # This parameter applies to all sources, hence to all values,
                # and hence it's the only local parameter contributing to the
                # global parameter fitparam_id.
                return self._cache['grads'][pidx]

            # The current parameter does not apply to all sources.
            # Create a values mask that matches a given source mask.
            values_mask = tdm.get_values_mask_for_source_mask(src_mask)
            grad[values_mask] = self._cache['grads'][pidx][values_mask]

        return grad
