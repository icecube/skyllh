# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np

from skyllh.core.py import (
    classname,
    float_cast,
    issequence,
    issequenceof,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
    FitParameter,
)
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Parabola1DGridManifoldInterpolationMethod,
)
from skyllh.core.pdf import (
    PDFSet,
    IsBackgroundPDF,
    IsSignalPDF,
)
from skyllh.core.timing import TaskTimer


class PDFRatio(object, metaclass=abc.ABCMeta):
    """Abstract base class for a signal over background PDF ratio class.
    It defines the interface of a signal over background PDF ratio class.
    """

    def __init__(
            self,
            sig_param_names=None,
            bkg_param_names=None,
            **kwargs):
        """Constructor for a PDF ratio class.

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
        return list(set(self._sig_param_names + self._bkg_param_names))

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

    def initialize_for_new_trial(self, tdm, tl=None):
        """Initializes the PDFRatio instance for a new trial. This method can
        be utilized to pre-calculate PDFRatio values that do not depend on any
        fit parameters.
        """
        pass

    @abc.abstractmethod
    def get_ratio(self, tdm, params_recarray=None, tl=None):
        """Retrieves the PDF ratio value for each given trial data event (and
        source), given the given set of parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        params_recarray : instance of numpy record ndarray | None
            The (N_models,)-shaped numpy record ndarray holding the parameter
            names and values of the models.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for more information.
            This can be ``None``, if the signal and background PDFs do not
            depend on any parameters.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N,)-shaped 1d numpy ndarray of float holding the PDF ratio
            value for each trial event (and source).
        """
        pass

    @abc.abstractmethod
    def get_gradient(self, tdm, params_recarray, fitparam_id, tl=None):
        """Retrieves the PDF ratio gradient for the parameter ``param_name``
        for each given trial event, given the given set of parameters
        ``params``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        params_recarray : instance of numpy record ndarray | None
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
        gradient : instance of ndarray
            The (N_values,)-shaped 1d numpy ndarray of float holding the PDF
            ratio gradient value for each source and trial event.
        """
        pass

    def __mul__(self, other):
        """Implements the mathematical operation ``new = self * other``, where
        ``other`` is an instance of PDFRatio. It creates an instance of
        PDFRatioProduct holding the two PDFRatio instances.
        """
        return PDFRatioProduct(self, other)


class PDFRatioProduct(PDFRatio):
    """This is the mathematical product of two PDFRatio instances, which is a
    PDFRatio instance again.
    """
    def __init__(self, pdfratio1, pdfratio2, **kwargs):
        """Creates a new PDFRatioProduct instance representing the product of
        two PDFRatio instances.
        """
        self.pdfratio1 = pdfratio1
        self.pdfratio2 = pdfratio2

        sig_param_names = set(
            pdfratio1.sig_param_names + pdfratio2.sig_param_names)
        bkg_param_names = set(
            pdfratio1.bkg_param_names + pdfratio2.bkg_param_names)

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

    def get_ratio(self, tdm, params=None, tl=None):
        """Retrieves the PDF ratio product value for each given trial data
        event, given the given set of parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        params : dict | None
            The dictionary with the parameter name-value pairs.
            It can be ``None``, if the PDF ratio does not depend on any
            parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : (N_events,)-shaped 1d numpy ndarray of float
            The PDF ratio product value for each trial event.
        """
        r1 = self._pdfratio1.get_ratio(
            tdm=tdm,
            params=params,
            tl=tl)
        r2 = self._pdfratio2.get_ratio(
            tdm=tdm,
            params=params,
            tl=tl)

        return r1 * r2

    def get_gradient(self, tdm, params, param_name, tl=None):
        """Retrieves the PDF ratio product gradient for the parameter
        ``param_name`` for each given trial event, given the given set of
        parameters ``params``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        params : dict
            The dictionary with the parameter names and values.
        param_name : str
            The name of the parameter for which the gradient should
            get calculated.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        gradient : instance of ndarray
            The (N_events,)-shaped 1d numpy ndarray of float holding the PDF
            ratio gradient value for each trial event.
        """
        (r1, r2) = self.get_ratio(
            tdm=tdm,
            params=params,
            tl=tl)
        r1_grad = self._pdfratio1.get_gradient(
            tdm=tdm,
            params=params,
            param_name=param_name,
            tl=tl)
        r2_grad = self._pdfratio2.get_gradient(
            tdm=tdm,
            params=params,
            param_name=param_name,
            tl=tl)

        return r1*r2_grad + r1_grad*r2


class SingleSourcePDFRatioArrayArithmetic(object):
    """This class is DEPRECATED! Use PDFRatioProduct instead!

    This class provides arithmetic methods for arrays of PDFRatio instances.
    It has methods to calculate the product of the ratio values for a given set
    of PDFRatio objects. This class assumes a single source.

    The rational is that in the calculation of the derivatives of the
    log-likelihood-ratio function for a given fit parameter, the product of the
    PDF ratio values of the PDF ratio objects which do not depend on that fit
    parameter is needed.
    """
    def __init__(self, pdfratios, fitparams, **kwargs):
        """Constructs a PDFRatio array arithmetic object assuming a single
        source.

        Parameters
        ----------
        pdfratios : list of PDFRatio
            The list of PDFRatio instances.
        fitparams : list of FitParameter
            The list of fit parameters. The order must match the fit parameter
            order of the minimizer.
        """
        super().__init__(**kwargs)

        self.pdfratio_list = pdfratios
        self.fitparam_list = fitparams

        # The ``_ratio_values`` member variable will hold a
        # (N_pdfratios,N_events)-shaped array holding the PDF ratio values of
        # each PDF ratio object for each event. It will be created by the
        # ``initialize_for_new_trial`` method.
        self._ratio_values = None

        # Create a mapping of fit parameter index to pdfratio index. We
        # initialize the mapping with -1 first in order to be able to check in
        # the end if all fit parameters found a PDF ratio object.
        self._fitparam_idx_2_pdfratio_idx = np.repeat(
            np.array([-1], dtype=np.int64), len(self._fitparam_list))
        for ((fpidx, fitparam), (pridx, pdfratio)) in itertools.product(
                enumerate(self._fitparam_list), enumerate(self.pdfratio_list)):
            if fitparam.name in pdfratio.fitparam_names:
                self._fitparam_idx_2_pdfratio_idx[fpidx] = pridx
        check_mask = (self._fitparam_idx_2_pdfratio_idx == -1)
        if np.any(check_mask):
            raise KeyError(
                f'{np.sum(check_mask)} fit parameters are not defined in any '
                'of the PDF ratio instances!')

        # Create the list of indices of the PDFRatio instances, which depend on
        # at least one fit parameter.
        self._var_pdfratio_indices = np.unique(
            self._fitparam_idx_2_pdfratio_idx)

    def _precompute_static_pdfratio_values(self, tdm):
        """Pre-compute the PDF ratio values for the PDF ratios that do not
        depend on any fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data for
            which the PDF ratio values should get calculated.
        """
        for (i, pdfratio) in enumerate(self._pdfratio_list):
            if pdfratio.n_fitparams == 0:
                # The PDFRatio does not depend on any fit parameters. So we
                # pre-calculate the PDF ratio values for all the events. Since
                # the get_ratio method of the PDFRatio class might return a 2D
                # (N_sources, N_events)-shaped array, and we assume a single
                # source, we need to reshape the array, which does not involve
                # any data copying.
                self._ratio_values[i] = np.reshape(
                    pdfratio.get_ratio(tdm), (tdm.n_selected_events,))

    @property
    def pdfratio_list(self):
        """The list of PDFRatio objects.
        """
        return self._pdfratio_list

    @pdfratio_list.setter
    def pdfratio_list(self, seq):
        if not issequenceof(seq, PDFRatio):
            raise TypeError(
                'The pdfratio_list property must be a sequence of PDFRatio '
                'instances!')
        self._pdfratio_list = list(seq)

    @property
    def fitparam_list(self):
        """The list of FitParameter instances.
        """
        return self._fitparam_list

    @fitparam_list.setter
    def fitparam_list(self, seq):
        if not issequenceof(seq, FitParameter):
            raise TypeError(
                'The fitparam_list property must be a sequence of FitParameter '
                'instances!')
        self._fitparam_list = list(seq)

    def initialize_for_new_trial(self, tdm):
        """Initializes the PDFRatio array arithmetic for a new trial. For a new
        trial the data events change, hence we need to recompute the PDF ratio
        values of the fit parameter independent PDFRatio instances.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data for
            that this PDFRatioArrayArithmetic instance should get initialized.
        """
        n_events_old = 0
        if self._ratio_values is not None:
            n_events_old = self._ratio_values.shape[1]

        # If the amount of events have changed, we need a new array holding the
        # ratio values.
        if n_events_old != tdm.n_selected_events:
            # Create a (N_pdfratios,N_events)-shaped array to hold the PDF ratio
            # values of each PDF ratio object for each event.
            self._ratio_values = np.empty(
                (len(self._pdfratio_list), tdm.n_selected_events),
                dtype=np.float64)

        self._precompute_static_pdfratio_values(tdm)

    def get_pdfratio(self, idx):
        """Returns the PDFRatio instance that corresponds to the given index.

        Parameters
        ----------
        idx : int
            The index of the PDFRatio.

        Returns
        -------
        pdfratio : PDFRatio
            The PDFRatio instance which corresponds to the given index.
        """
        return self._pdfratio_list[idx]

    def calculate_pdfratio_values(self, tdm, cflp_values, tl=None):
        """Calculates the PDF ratio values for the PDF ratio objects which
        depend on fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data for
            which the PDF ratio values should get calculated.
        cflp_values : dict
            The dictionary with the current floating parameter name-value pairs.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.
        """
        for (i, _pdfratio_i) in enumerate(self._pdfratio_list):
            # Since the get_ratio method of the PDFRatio class might return a 2D
            # (N_sources, N_events)-shaped array, and we assume a single source,
            # we need to reshape the array, which does not involve any data
            # copying.
            self._ratio_values[i] = np.reshape(
                _pdfratio_i.get_ratio(tdm, cflp_values, tl=tl),
                (tdm.n_selected_events,))

    def get_ratio_product(self, excluded_idx=None):
        """Calculates the product of the of the PDF ratio values of each event,
        but excludes the PDF ratio values that correspond to the given excluded
        fit parameter index. This is useful for calculating the derivatives of
        the log-likelihood ratio function.

        Parameters
        ----------
        excluded_idx : int | None
            The index of the PDFRatio instance whose PDF ratio values should get
            excluded from the product. If None, the product over all PDF ratio
            values will be computed.

        Returns
        -------
        product : 1D (N_events,)-shaped ndarray
            The product of the PDF ratio values for each event.
        """
        if excluded_idx is None:
            return np.prod(self._ratio_values, axis=0)

        # Get the index of the PDF ratio object that corresponds to the excluded
        # fit parameter.
        pdfratio_indices = list(range(self._ratio_values.shape[0]))
        pdfratio_indices.pop(excluded_idx)
        return np.prod(self._ratio_values[pdfratio_indices], axis=0)


class SourceWeightedPDFRatio(object):
    r"""This class provides the calculation of a source weighted PDF ratio for
    multiple sources:

    .. math::

        \mathcal{R}_i(\vec{p}_{\mathrm{s}}) = \frac{1}{A(\vec{p}_{\mathrm{s}})}
            \sum_{k=1}^{K} a_k(\vec{p}_{\mathrm{s}_k}) \mathcal{R}_{i,k}
            (\vec{p}_{\mathrm{s}_k})

    """
    def __init__(self, pmm, pdfratio, **kwargs):
        """Creates a new SourceWeightedPDFRatio instance.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the global parameters
            and their mapping to local source parameters.
        pdfratio : instance of PDFRatio
            The instance of PDFRatio providing the PDF ratio values and
            derivatives.
        """
        super().__init__(**kwargs)

        self.pmm = pmm
        self.pdfratio = pdfratio

    @property
    def pmm(self):
        """The ParameterModelMapper which defines the global parameters and
        thier mapping to local source parameters.
        """
        return self._pmm

    @pmm.setter
    def pmm(self, m):
        if not isinstance(m, ParameterModelMapper):
            raise TypeError(
                'The pmm property must be an instance of ParameterModelMapper!')
        self._pmm = m

    @property
    def pdfratio(self):
        """The PDFRatio instance that is used to calculate the source weighted
        PDF ratio.
        """
        return self._pdfratio

    @pdfratio.setter
    def pdfratio(self, r):
        if not isinstance(r, PDFRatio):
            raise TypeError(
                'The pdfratio property must be an instance of PDFRatio!')
        self._pdfratio = r

    def __call__(self, tdm, a_k, a_k_grads, fitparam_values, tl=None):
        """Calculates the source weighted PDF ratio.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which the source weighted PDF ratio values should get calculated.
        a_k : instance of ndarray
            The (N_sources,)-shaped numpy ndarray holding the source detector
            weight for each source.
        a_k_grads : dict
            The dictionary holding the (N_sources,)-shaped ndarray with the
            derivatives of the source detector weights w.r.t. the global fit
            parameter. The key of the dictionary is the index of the global
            fit parameter.
        fitparam_values : instance of ndarray
            The (N_fitparams,)-shaped 1D numpy ndarray holding the global
            fit parameter values.
        tl : instance of TimeLord | None
            The instance of TimeLord that should be used to measure timing
            information.

        Returns
        -------
        R_i : instance of ndarray
            The (N_events,)-shaped numpy ndarray holding the source weighted
            PDF ratio values.
        R_i_grads : dict
            The dictionary holding the (N_events,)-shaped numpy ndarray holding
            the derivatives w.r.t. the global fit parameter, the PDF ratio
            depends on. The dictionary's key is the index of the global fit
            parameter.
        """
        n_sources = len(a_k)

        A = np.sum(a_k)

        R_i_k = np.zeros((tdm.n_selected_events, n_sources), dtype=np.double)

        src_model_idxs = self._pmm.get_src_model_idxs()

        params_list = []
        for (k, sidx) in enumerate(src_model_idxs):
            params = self._pmm.create_model_params_dict(
                gflp_values=fitparam_values,
                model=sidx)
            params_list.append(params)
            R_i_k[:, k] = self._pdfratio.get_ratio(
                tdm=tdm,
                params=params,
                tl=tl)

        R_i = np.sum(a_k[np.newaxis, :] * R_i_k, axis=1) / A

        R_i_grads = dict()
        for gflp_idx in self._pmm.global_paramset.floating_params_idxs:
            a_k_grad = a_k_grads.get(gflp_idx, np.zeros((n_sources,)))
            dAdp = np.sum(a_k_grad)

            R_i_k_grads = np.zeros(
                (tdm.n_selected_events, n_sources), dtype=np.double)
            for (k, sidx) in enumerate(src_model_idxs):
                param_name = self._pmm.get_model_param_name(
                    model_idx=sidx,
                    gp_idx=gflp_idx)
                R_i_k_grads[:, k] = self._pdfratio.get_gradient(
                    tdm=tdm,
                    params=params_list[k],
                    param_name=param_name)

            src_sum = np.sum(
                a_k_grad[np.newaxis, :] * R_i_k +
                a_k[np.newaxis, :] * R_i_k_grads,
                axis=1)

            R_i_grads[gflp_idx] = (-R_i * dAdp + src_sum) / A

        return (R_i, R_i_grads)


class SigOverBkgPDFRatio(PDFRatio):
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
            sig_param_names=sig_pdf.param_set.param_names,
            bkg_param_names=bkg_pdf.param_set.param_names,
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
        self._cache_trial_data_state_id = None
        self._cache_params_recarray = None
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

    def get_ratio(self, tdm, params_recarray=None, tl=None):
        """Calculates the PDF ratio for the given trial events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should be calculated.
        params_recarray : instance of numpy record ndarray | None
            The (N_models,)-shaped numpy record ndarray holding the parameter
            names and values of the models.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for more information.
            This can be ``None``, if the signal and background PDFs do not
            depend on any parameters.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : instance of ndarray
            The (N,)-shaped numpy ndarray holding the probability density ratio
            for each event (and each source).
        """
        with TaskTimer(tl, 'Get sig probability densities and grads.'):
            (sig_pd, self._cache_sig_grads) = self._sig_pdf.get_pd(
                tdm=tdm,
                params_recarray=params_recarray,
                tl=tl)
        with TaskTimer(tl, 'Get bkg probability densities and grads.'):
            (bkg_pd, self._cache_bkg_grads) = self._bkg_pdf.get_pd(
                tdm=tdm,
                params_recarray=params_recarray,
                tl=tl)

        with TaskTimer(tl, 'Calculate PDF ratios.'):
            # Select only the events, where the background pdf is greater than
            # zero.
            ratios = np.full_like(sig_pd, self._zero_bkg_ratio_value)
            m = (bkg_pd > 0)
            ratios[m] = sig_pd[m] / bkg_pd[m]

        # Store the current state of parameter values and trial data, so that
        # the get_gradient method can verify the consistency of the signal and
        # background probabilities and gradients.
        self._cache_trial_data_state_id = tdm.trial_data_state_id
        self._cache_params_recarray = None
        if params_recarray is not None:
            self._cache_params_recarray = np.copy(params_recarray)
        self._cache_sig_pd = sig_pd
        self._cache_bkg_pd = bkg_pd

        return ratios

    def get_gradient(self, tdm, params_recarray, fitparam_id, tl=None):
        """Retrieves the gradient of the PDF ratio w.r.t. the given parameter.
        This method must be called after the ``get_ratio`` method.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        params_recarray : instance of numpy record ndarray | None
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
        if (tdm.trial_data_state_id != self._cache_trial_data_state_id) or\
           (np.all(np.isclose(params_recarray, self._cache_params_recarray))):
            raise RuntimeError(
                'The get_ratio method must be called prior to the get_gradient '
                'method!')

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

        if sig_dep and not bkg_dep:
            # Case 2, which should be the most common case.
            bkg_pd = self._cache_bkg_pd
            m = bkg_pd > 0
            grad[m] = self._cache_sig_grads[fitparam_id][m] / bkg_pd[m]
            return grad
        if (not sig_dep) and (not bkg_dep):
            # Case 1. Return zeros.
            return grad

        if sig_dep and bkg_dep:
            # Case 4.
            m = self._cache_bkg_pd > 0
            s = self._cache_sig_pd[m]
            b = self._cache_bkg_pd[m]
            sgrad = self._cache_sig_grads[fitparam_id][m]
            bgrad = self._cache_bkg_grads[fitparam_id][m]
            # Make use of quotient rule of differentiation.
            grad[m] = (sgrad * b - bgrad * s) / b**2
            return grad

        # Case 3.
        m = self._cache_bkg_pd > 0
        grad[m] = (
            -self._cache_sig_pd[m] / self._cache_bkg_pd[m]**2 *
            self._cache_bkg_grads[fitparam_id][m]
        )
        return grad


class SpatialSigOverBkgPDFRatio(SigOverBkgPDFRatio):
    """This class implements a signal-over-background PDF ratio for spatial
    PDFs. It takes a signal PDF of type SpatialPDF and a background PDF of type
    SpatialPDF and calculates the PDF ratio.
    """
    def __init__(
            self,
            sig_pdf,
            bkg_pdf,
            **kwargs):
        """Creates a new signal-over-background PDF ratio instance for spatial
        PDFs.

        Parameters
        ----------
        sig_pdf : class instance derived from SpatialPDF, IsSignalPDF
            The instance of the spatial signal PDF.
        bkg_pdf : class instance derived from SpatialPDF, IsBackgroundPDF
            The instance of the spatial background PDF.
        """
        super().__init__(
            sig_pdf=sig_pdf,
            bkg_pdf=bkg_pdf,
            **kwargs)

        # Make sure that the PDFs have two dimensions, i.e. RA and Dec.
        if not sig_pdf.ndim == 2:
            raise ValueError(
                'The spatial signal PDF must have two dimensions! '
                f'Currently it has {sig_pdf.ndim}!')


class SigSetOverBkgPDFRatio(PDFRatio):
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
            sig_param_names=sig_pdf_set.param_grid_set.param_names,
            bkg_param_names=bkg_pdf.param_set.param_names,
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
                'IsBackgroundPDF!')
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
                'derived from PDFSet and IsSignalPDF!')
        self._sig_pdf_set = pdfset

    @property
    def interpolmethod(self):
        """The class derived from GridManifoldInterpolationMethod
        implementing the interpolation of the fit parameter manifold.
        """
        return self._interpolmethod

    @interpolmethod.setter
    def interpolmethod(self, cls):
        if not issubclass(cls, GridManifoldInterpolationMethod):
            raise TypeError(
                'The interpolmethod property must be a sub-class '
                'of GridManifoldInterpolationMethod!')
        self._interpolmethod = cls
