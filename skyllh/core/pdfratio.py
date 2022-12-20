# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np

from skyllh.core.py import (
    classname,
    float_cast,
    issequenceof,
    typename
)
from skyllh.core.parameters import (
    FitParameter,
    make_params_hash
)
from skyllh.core.interpolate import (
    GridManifoldInterpolationMethod,
    Parabola1DGridManifoldInterpolationMethod
)
from skyllh.core.pdf import (
    PDF,
    PDFSet,
    IsBackgroundPDF,
    IsSignalPDF,
    SpatialPDF
)
from skyllh.core.timing import TaskTimer


class PDFRatio(object, metaclass=abc.ABCMeta):
    """Abstract base class for a PDF ratio class. It defines the interface
    of a PDF ratio class.
    """

    def __init__(self, pdf_type, *args, **kwargs):
        """Constructor for a PDF ratio class.

        Parameters
        ----------
        pdf_type : type
            The Python type of the PDF object the PDF ratio is made for.
        """
        super(PDFRatio, self).__init__(*args, **kwargs)

        self._pdf_type = pdf_type

    @property
    def n_fitparams(self):
        """(read-only) The number of fit parameters the PDF ratio depends on.
        This is the sum of signal and background fit parameters. At the moment
        only signal fit parameters are supported, so this property is equivalent
        to the n_signal_fitparams property. But this might change in the future.
        """
        return self.n_signal_fitparams

    @property
    def fitparam_names(self):
        """(read-only) The list of fit parameter names this PDF ratio is a
        function of.
        This is the superset of signal and background fit parameter names.
        At the moment only signal fit parameters are supported, so this property
        is equivalent to the signal_fitparam_names property. But this might
        change in the future.
        """
        return self.signal_fitparam_names

    @property
    def n_signal_fitparams(self):
        """(read-only) The number of signal fit parameters the PDF ratio depends
        on.
        """
        return len(self._get_signal_fitparam_names())

    @property
    def signal_fitparam_names(self):
        """(read-only) The list of signal fit parameter names this PDF ratio is
        a function of.
        """
        return self._get_signal_fitparam_names()

    @property
    def pdf_type(self):
        """(read-only) The Python type of the PDF object for which the PDF
        ratio is made for.
        """
        return self._pdf_type

    def _get_signal_fitparam_names(self):
        """This method must be re-implemented by the derived class and needs to
        return the list of signal fit parameter names, this PDF ratio is a
        function of. If it returns an empty list, the PDF ratio is independent
        of any signal fit parameters.

        Returns
        -------
        list of str
            The list of the signal fit parameter names, this PDF ratio is a
            function of. By default this method returns an empty list indicating
            that the PDF ratio depends on no signal parameter.
        """
        return []

    @abc.abstractmethod
    def get_ratio(self, tdm, params=None, tl=None):
        """Retrieves the PDF ratio value for each given trial data event, given
        the given set of fit parameters. This method is called during the
        likelihood maximization process.

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
            The PDF ratio value for each trial event.
        """
        pass

    @abc.abstractmethod
    def get_gradient(self, tdm, params, fitparam_name):
        """Retrieves the PDF ratio gradient for the parameter ``fitparam_name``
        for each given trial event, given the given set of fit parameters.
        This method is called during the likelihood maximization process.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        params : dict
            The dictionary with the parameter names and values.
        fitparam_name : str
            The name of the fit parameter for which the gradient should
            get calculated.

        Returns
        -------
        gradient : (N_events,)-shaped 1d numpy ndarray of float
            The PDF ratio gradient value for each trial event.
        """
        pass


class SingleSourcePDFRatioArrayArithmetic(object):
    """This class provides arithmetic methods for arrays of PDFRatio instances.
    It has methods to calculate the product of the ratio values for a given set
    of PDFRatio objects. This class assumes a single source.

    The rational is that in the calculation of the derivates of the
    log-likelihood-ratio function for a given fit parameter, the product of the
    PDF ratio values of the PDF ratio objects which do not depend on that fit
    parameter is needed.
    """
    def __init__(self, pdfratios, fitparams):
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
        super(SingleSourcePDFRatioArrayArithmetic, self).__init__()

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
            if(fitparam.name in pdfratio.fitparam_names):
                self._fitparam_idx_2_pdfratio_idx[fpidx] = pridx
        check_mask = (self._fitparam_idx_2_pdfratio_idx == -1)
        if(np.any(check_mask)):
            raise KeyError('%d fit parameters are not defined in any of the '
                'PDF ratio instances!'%(np.sum(check_mask)))

        # Create the list of indices of the PDFRatio instances, which depend on
        # at least one fit parameter.
        self._var_pdfratio_indices = np.unique(self._fitparam_idx_2_pdfratio_idx)

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
            if(pdfratio.n_fitparams == 0):
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
        if(not issequenceof(seq, PDFRatio)):
            raise TypeError('The pdfratio_list property must be a sequence of '
                'PDFRatio instances!')
        self._pdfratio_list = list(seq)

    @property
    def fitparam_list(self):
        """The list of FitParameter instances.
        """
        return self._fitparam_list
    @fitparam_list.setter
    def fitparam_list(self, seq):
        if(not issequenceof(seq, FitParameter)):
            raise TypeError('The fitparam_list property must be a sequence of '
                'FitParameter instances!')
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
        if(self._ratio_values is not None):
            n_events_old = self._ratio_values.shape[1]

        # If the amount of events have changed, we need a new array holding the
        # ratio values.
        if(n_events_old != tdm.n_selected_events):
            # Create a (N_pdfratios,N_events)-shaped array to hold the PDF ratio
            # values of each PDF ratio object for each event.
            self._ratio_values = np.empty(
                (len(self._pdfratio_list), tdm.n_selected_events),
                dtype=np.float64)

        self._precompute_static_pdfratio_values(tdm)

    def get_pdfratio(self, idx):
        """Returns the PDFRatio instance that corresponds to the given fit
        parameter index.

        Parameters
        ----------
        fitparam_idx : int
            The index of the fit parameter.

        Returns
        -------
        pdfratio : PDFRatio
            The PDFRatio instance which corresponds to the given fit parameter
            index.
        """
        return self._pdfratio_list[idx]

    def calculate_pdfratio_values(self, tdm, fitparams, tl=None):
        """Calculates the PDF ratio values for the PDF ratio objects which
        depend on fit parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager that holds the trial event data for
            which the PDF ratio values should get calculated.
        fitparams : dict
            The dictionary with the fit parameter name-value pairs.
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
                _pdfratio_i.get_ratio(tdm, fitparams, tl=tl),
                (tdm.n_selected_events,))

    def get_ratio_product(self, excluded_idx=None):
        """Calculates the product of the of the PDF ratio values of each event,
        but excludes the PDF ratio values that correspond to the given excluded
        fit parameter index. This is useful for calculating the derivates of
        the log-likelihood ratio function.

        Parameters
        ----------
        excluded_fitparam_idx : int | None
            The index of the fit parameter whose PDF ratio values should get
            excluded from the product. If None, the product over all PDF ratio
            values will be computed.

        Returns
        -------
        product : 1D (N_events,)-shaped ndarray
            The product of the PDF ratio values for each event.
        """
        if(excluded_idx is None):
            return np.prod(self._ratio_values, axis=0)

        # Get the index of the PDF ratio object that corresponds to the excluded
        # fit parameter.
        #excluded_pdfratio_idx = self._fitparam_idx_2_pdfratio_idx[excluded_fitparam_idx]
        pdfratio_indices = list(range(self._ratio_values.shape[0]))
        pdfratio_indices.pop(excluded_idx)
        return np.prod(self._ratio_values[pdfratio_indices], axis=0)


class PDFRatioFillMethod(object, metaclass=abc.ABCMeta):
    """Abstract base class to implement a PDF ratio fill method. It can happen,
    that there are empty background bins but where signal could possibly be.
    A PDFRatioFillMethod implements what happens in such cases.
    """

    def __init__(self, *args, **kwargs):
        super(PDFRatioFillMethod, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def fill_ratios(self, ratios, sig_prob_h, bkg_prob_h,
                    sig_mask_mc_covered, sig_mask_mc_covered_zero_physics,
                    bkg_mask_mc_covered, bkg_mask_mc_covered_zero_physics):
        """The fill_ratios method is supposed to fill the ratio bins (array)
        with the signal / background division values. For bins (array elements),
        where the division is undefined, e.g. due to zero background, the fill
        method decides how to fill those bins.

        Note: Bins which have neither signal monte-carlo nor background
              monte-carlo coverage, are undefined about their signal-ness or
              background-ness by construction.

        Parameters
        ----------
        ratios : ndarray of float
            The multi-dimensional array for the final ratio bins. The shape is
            the same as the sig_h and bkg_h ndarrays.
        sig_prob_h : ndarray of float
            The multi-dimensional array (histogram) holding the signal
            probabilities.
        bkg_prob_h : ndarray of float
            The multi-dimensional array (histogram) holding the background
            probabilities.
        sig_mask_mc_covered : ndarray of bool
            The mask array indicating which array elements of sig_prob_h have
            monte-carlo coverage.
        sig_mask_mc_covered_zero_physics : ndarray of bool
            The mask array indicating which array elements of sig_prob_h have
            monte-carlo coverage but don't have physics contribution.
        bkg_mask_mc_covered : ndarray of bool
            The mask array indicating which array elements of bkg_prob_h have
            monte-carlo coverage.
            In case of experimental data as background, this mask indicate where
            (experimental data) background is available.
        bkg_mask_mc_covered_zero_physics : ndarray of bool
            The mask array ndicating which array elements of bkg_prob_h have
            monte-carlo coverage but don't have physics contribution.
            In case of experimental data as background, this mask contains only
            False entries.

        Returns
        -------
        ratios : ndarray
            The array holding the final ratio values.
        """
        return ratios

class Skylab2SkylabPDFRatioFillMethod(PDFRatioFillMethod):
    """This PDF ratio fill method implements the exact same fill method as in
    the skylab2 software named "skylab". It exists just for comparsion and
    backward compatibility reasons. In general, it should not be used, because
    it does not distinguish between bins with MC converage and physics model
    contribution, and those with MC coverage and no physics model contribution!
    """
    def __init__(self):
        super(Skylab2SkylabPDFRatioFillMethod, self).__init__()
        self.signallike_percentile = 99.

    def fill_ratios(self, ratio, sig_prob_h, bkg_prob_h,
                    sig_mask_mc_covered, sig_mask_mc_covered_zero_physics,
                    bkg_mask_mc_covered, bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if(np.any(bkg_mask_mc_covered_zero_physics)):
            raise ValueError('Some of the background bins have MC coverage but no physics background prediction. I don\'t know what to do in this case!')

        sig_domain = sig_prob_h > 0
        bkg_domain = bkg_prob_h > 0

        ratio[sig_domain & bkg_domain] = sig_prob_h[sig_domain & bkg_domain] / bkg_prob_h[sig_domain & bkg_domain]

        ratio_value = np.percentile(ratio[ratio > 1.], self.signallike_percentile)
        np.copyto(ratio, ratio_value, where=sig_domain & ~bkg_domain)

        return ratio

class MostSignalLikePDFRatioFillMethod(PDFRatioFillMethod):
    """PDF ratio fill method to set the PDF ratio to the most signal like PDF
    ratio for bins, where there is signal MC coverage but no background (MC)
    coverage.
    """
    def __init__(self, signallike_percentile=99.):
        """Creates the PDF ratio fill method object for filling PDF ratio bins,
        where there is signal MC coverage but no background (MC) coverage
        with the most signal-like ratio value.

        Parameters
        ----------
        signallike_percentile : float in range [0., 100.], default 99.
            The percentile of signal-like ratios, which should be taken as the
            ratio value for ratios with no background probability.
        """
        super(MostSignalLikePDFRatioFillMethod, self).__init__()

        self.signallike_percentile = signallike_percentile

    @property
    def signallike_percentile(self):
        """The percentile of signal-like ratios, which should be taken as the
        ratio value for ratios with no background probability. This percentile
        must be given as a float value in the range [0, 100] inclusively.
        """
        return self._signallike_percentile
    @signallike_percentile.setter
    def signallike_percentile(self, value):
        if(not isinstance(value, float)):
            raise TypeError('The signallike_percentile property must be of type float!')
        if(value < 0. or value > 100.):
            raise ValueError('The value of the signallike_percentile property must be in the range [0, 100]!')
        self._signallike_percentile = value

    def fill_ratios(self, ratio, sig_prob_h, bkg_prob_h,
                    sig_mask_mc_covered, sig_mask_mc_covered_zero_physics,
                    bkg_mask_mc_covered, bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if(np.any(bkg_mask_mc_covered_zero_physics)):
            raise ValueError('Some of the background bins have MC coverage but no physics background prediction. I don\'t know what to do in this case!')

        # Fill the bins where we have signal and background MC coverage.
        mask_sig_and_bkg_mc_covered = sig_mask_mc_covered & bkg_mask_mc_covered
        ratio[mask_sig_and_bkg_mc_covered] = sig_prob_h[mask_sig_and_bkg_mc_covered] / bkg_prob_h[mask_sig_and_bkg_mc_covered]

        # Calculate the ratio value, which should be used for ratio bins, where
        # we have signal MC coverage but no background MC coverage.
        ratio_value = np.percentile(ratio[ratio > 1.], self.signallike_percentile)
        mask_sig_but_notbkg_mc_covered = sig_mask_mc_covered & ~bkg_mask_mc_covered
        np.copyto(ratio, ratio_value, where=mask_sig_but_notbkg_mc_covered)

        return ratio


class MinBackgroundLikePDFRatioFillMethod(PDFRatioFillMethod):
    """PDF ratio fill method to set the PDF ratio to the minimal background like
    value for bins, where there is signal MC coverage but no background (MC)
    coverage.
    """
    def __init__(self):
        """Creates the PDF ratio fill method object for filling PDF ratio bins,
        where there is signal MC coverage but no background (MC) coverage
        with the minimal background-like ratio value.
        """
        super(MinBackgroundLikePDFRatioFillMethod, self).__init__()

    def fill_ratios(self, ratio, sig_prob_h, bkg_prob_h,
                    sig_mask_mc_covered, sig_mask_mc_covered_zero_physics,
                    bkg_mask_mc_covered, bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if(np.any(bkg_mask_mc_covered_zero_physics)):
            raise ValueError('Some of the background bins have MC coverage but no physics background prediction. I don\'t know what to do in this case!')

        # Fill the bins where we have signal and background MC coverage.
        mask_sig_and_bkg_mc_covered = sig_mask_mc_covered & bkg_mask_mc_covered
        ratio[mask_sig_and_bkg_mc_covered] = sig_prob_h[mask_sig_and_bkg_mc_covered] / bkg_prob_h[mask_sig_and_bkg_mc_covered]

        # Calculate the minimal background-like value.
        min_bkg_prob = np.min(bkg_prob_h[bkg_mask_mc_covered])

        # Set the ratio using the minimal background probability where we
        # have signal MC coverage but no background (MC) coverage.
        mask_sig_but_notbkg_mc_covered = sig_mask_mc_covered & ~bkg_mask_mc_covered
        ratio[mask_sig_but_notbkg_mc_covered] = sig_prob_h[mask_sig_but_notbkg_mc_covered] / min_bkg_prob

        return ratio


class SigOverBkgPDFRatio(PDFRatio):
    """This class implements a generic signal-over-background PDF ratio for a
    signal and a background PDF instance.
    It takes a signal PDF of type *pdf_type* and a background PDF of type
    *pdf_type* and calculates the PDF ratio.
    """
    def __init__(self, sig_pdf, bkg_pdf, pdf_type=None, same_axes=True,
        zero_bkg_ratio_value=1., *args, **kwargs):
        """Creates a new signal-over-background PDF ratio instance.

        Parameters
        ----------
        sig_pdf : class instance derived from `pdf_type`, IsSignalPDF
            The instance of the signal PDF.
        bkg_pdf : class instance derived from `pdf_type`, IsBackgroundPDF
            The instance of the background PDF.
        pdf_type : type | None
            The python type of the PDF object for which the PDF ratio is for.
            If set to None, the default class ``PDF`` will be used.
        same_axes : bool
            Flag if the signal and background PDFs are supposed to have the
            same axes. Default is True.
        zero_bkg_ratio_value : float
            The value of the PDF ratio to take when the background PDF value
            is zero. This is to avoid division by zero. Default is 1.
        """
        if(pdf_type is None):
            pdf_type = PDF

        super(SigOverBkgPDFRatio, self).__init__(
            pdf_type=pdf_type, *args, **kwargs)

        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf

        # Check that the PDF axes ranges are the same for the signal and
        # background PDFs.
        if(same_axes and (not sig_pdf.axes.is_same_as(bkg_pdf.axes))):
            raise ValueError('The signal and background PDFs do not have the '
                'same axes.')

        self.zero_bkg_ratio_value = zero_bkg_ratio_value

        # Define cache member variables to calculate gradients efficiently.
        self._cache_trial_data_state_id = None
        self._cache_params_hash = None
        self._cache_sigprob = None
        self._cache_bkgprob = None
        self._cache_siggrads = None
        self._cache_bkggrads = None

    @property
    def sig_pdf(self):
        """The signal PDF object used to create the PDF ratio.
        """
        return self._sig_pdf
    @sig_pdf.setter
    def sig_pdf(self, pdf):
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The sig_pdf property must be an instance of '
                '%s!'%(typename(self.pdf_type)))
        if(not isinstance(pdf, IsSignalPDF)):
            raise TypeError('The sig_pdf property must be an instance of '
                'IsSignalPDF!')
        self._sig_pdf = pdf

    @property
    def bkg_pdf(self):
        """The background PDF object used to create the PDF ratio.
        """
        return self._bkg_pdf
    @bkg_pdf.setter
    def bkg_pdf(self, pdf):
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The bkg_pdf property must be an instance of '
                '%s!'%(typename(self.pdf_type)))
        if(not isinstance(pdf, IsBackgroundPDF)):
            raise TypeError('The bkg_pdf property must be an instance of '
                'IsBackgroundPDF!')
        self._bkg_pdf = pdf

    @property
    def zero_bkg_ratio_value(self):
        """The value of the PDF ratio to take when the background PDF value
        is zero. This is to avoid division by zero.
        """
        return self._zero_bkg_ratio_value
    @zero_bkg_ratio_value.setter
    def zero_bkg_ratio_value(self, v):
        v = float_cast(v, 'The zero_bkg_ratio_value must be castable into a '
            'float!')
        self._zero_bkg_ratio_value = v

    def _get_signal_fitparam_names(self):
        """Returns the list of fit parameter names the signal PDF depends on.
        """
        return self._sig_pdf.param_set.floating_param_name_list

    def get_ratio(self, tdm, params=None, tl=None):
        """Calculates the PDF ratio for the given trial events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should be calculated.
        params : dict | None
            The dictionary holding the parameter names and values for which the
            probability ratio should get calculated.
            This can be ``None``, if the signal and background PDFs do not
            depend on any parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratios : (N_events)-shaped numpy ndarray
            The ndarray holding the probability ratio for each event (and each
            source). The dimensionality of the returned ndarray depends on the
            dimensionality of the probability ndarray returned by the
            ``get_prob`` method of the signal PDF object.
        """
        with TaskTimer(tl, 'Get sig prob.'):
            (sigprob, self._cache_siggrads) = self._sig_pdf.get_prob(
                tdm, params, tl=tl)
        with TaskTimer(tl, 'Get bkg prob.'):
            (bkgprob, self._cache_bkggrads) = self._bkg_pdf.get_prob(
                tdm, params, tl=tl)

        with TaskTimer(tl, 'Calc PDF ratios.'):
            # Select only the events, where background pdf is greater than zero.
            m = (bkgprob > 0)
            ratios = np.full_like(sigprob, self._zero_bkg_ratio_value)
            ratios[m] = sigprob[m] / bkgprob[m]

        # Store the current state of parameter values and trial data, so that
        # the get_gradient method can verify the consistency of the signal and
        # background probabilities and gradients.
        self._cache_trial_data_state_id = tdm.trial_data_state_id
        self._cache_params_hash = make_params_hash(params)
        self._cache_sigprob = sigprob
        self._cache_bkgprob = bkgprob

        return ratios

    def get_gradient(self, tdm, params, fitparam_name):
        """Retrieves the gradient of the PDF ratio w.r.t. the given fit
        parameter. This method must be called after the ``get_ratio`` method.

        Parameters
        ----------
        tdm : TrialDataManager instance
            The instance of TrialDataManager that should be used to get the
            trial data from.
        params : dict
            The dictionary with the parameter names and values.
        fitparam_name : str
            The name of the fit parameter for which the gradient should
            get calculated.

        Returns
        -------
        gradient : (N_events,)-shaped 1d numpy ndarray of float
            The PDF ratio gradient value for each trial event.
        """
        if((tdm.trial_data_state_id != self._cache_trial_data_state_id) or
           (make_params_hash(params) != self._cache_params_hash)):
            raise RuntimeError('The get_ratio method must be called prior to '
                'the get_gradient method!')

        # Create the 1D return array for the gradient.
        grad = np.zeros((tdm.n_selected_events,), dtype=np.float64)

        # Calculate the gradient for the given fit parameter.
        # There are four cases:
        #   1) Neither the signal nor the background PDF depend on the fit
        #      parameter.
        #   2) Only the signal PDF depends on the fit parameter.
        #   3) Only the background PDF depends on the fit parameter.
        #   4) Both, the signal and the background PDF depend on the fit
        #      parameter.
        sig_pdf_param_set = self._sig_pdf.param_set
        bkg_pdf_param_set = self._bkg_pdf.param_set

        sig_dep = sig_pdf_param_set.has_floating_param(fitparam_name)
        bkg_dep = bkg_pdf_param_set.has_floating_param(fitparam_name)

        if(sig_dep and (not bkg_dep)):
            # Case 2, which should be the most common case.
            # Get the signal grad idx for that fit parameter.
            sig_pidx = sig_pdf_param_set.get_floating_pidx(fitparam_name)
            bkgprob = self._cache_bkgprob
            m = bkgprob > 0
            grad[m] = self._cache_siggrads[sig_pidx][m] / bkgprob[m]
            return grad
        if((not sig_dep) and (not bkg_dep)):
            # Case 1. Returns zeros.
            return grad

        if(sig_dep and bkg_dep):
            # Case 4.
            sig_pidx = sig_pdf_param_set.get_floating_pidx(fitparam_name)
            bkg_pidx = bkg_pdf_param_set.get_floating_pidx(fitparam_name)
            m = self._cache_bkgprob > 0
            s = self._cache_sigprob[m]
            b = self._cache_bkgprob[m]
            sgrad = self._cache_siggrads[sig_pidx][m]
            bgrad = self._cache_bkggrads[bkg_pidx][m]
            # Make use of quotient rule of differentiation.
            grad[m] = (sgrad * b - bgrad * s) / b**2
            return grad

        # Case 3.
        bkg_pidx = bkg_pdf_param_set.get_floating_pidx(fitparam_name)
        bkgprob = self._cache_bkgprob
        m = bkgprob > 0
        grad[m] = (-self._cache_sigprob[m] / bkgprob[m]**2 *
            self._cache_bkggrads[bkg_pidx][m])
        return grad


class SpatialSigOverBkgPDFRatio(SigOverBkgPDFRatio):
    """This class implements a signal-over-background PDF ratio for spatial
    PDFs. It takes a signal PDF of type SpatialPDF and a background PDF of type
    SpatialPDF and calculates the PDF ratio.
    """
    def __init__(self, sig_pdf, bkg_pdf, *args, **kwargs):
        """Creates a new signal-over-background PDF ratio instance for spatial
        PDFs.

        Parameters
        ----------
        sig_pdf : class instance derived from SpatialPDF, IsSignalPDF
            The instance of the spatial signal PDF.
        bkg_pdf : class instance derived from SpatialPDF, IsBackgroundPDF
            The instance of the spatial background PDF.
        """
        super(SpatialSigOverBkgPDFRatio, self).__init__(pdf_type=SpatialPDF,
            sig_pdf=sig_pdf, bkg_pdf=bkg_pdf, *args, **kwargs)

        # Make sure that the PDFs have two dimensions, i.e. RA and Dec.
        if(not sig_pdf.ndim == 2):
            raise ValueError('The spatial signal PDF must have two dimensions! '
                'Currently it has %d!'%(sig_pdf.ndim))


class SigSetOverBkgPDFRatio(PDFRatio):
    """Class for a PDF ratio class that takes a PDFSet of PDF type
    *pdf_type* as signal PDF and a PDF of type *pdf_type* as background PDF.
    The signal PDF depends on signal fit parameters and a interpolation method
    defines how the PDF ratio gets interpolated between the fit parameter
    values.
    """
    def __init__(self, pdf_type, signalpdfset, backgroundpdf,
                 interpolmethod=None, *args, **kwargs):
        """Constructor called by creating an instance of a class which is
        derived from this PDFRatio class.

        Parameters
        ----------
        pdf_type : type
            The Python type of the PDF object for which the PDF ratio is for.
        signalpdfset : class instance derived from PDFSet (for PDF type
                       ``pdf_type``), and IsSignalPDF
            The PDF set, which provides signal PDFs for a set of
            discrete signal fit parameters.
        backgroundpdf : class instance derived from ``pdf_type``, and
                        IsBackgroundPDF
            The background PDF instance.
        interpolmethod : class of GridManifoldInterpolationMethod | None
            The class implementing the fit parameter interpolation method for
            the PDF ratio manifold grid.
            If set to None (default), the
            Parabola1DGridManifoldInterpolationMethod will be used for
            1-dimensional parameter manifolds.
        """
        # Call super to allow for multiple class inheritance.
        super(SigSetOverBkgPDFRatio, self).__init__(pdf_type, *args, **kwargs)

        self.signalpdfset = signalpdfset
        self.backgroundpdf = backgroundpdf

        # Define the default fit parameter interpolation method. The default
        # depends on the dimensionality of the fit parameter manifold.
        if(interpolmethod is None):
            ndim = signalpdfset.fitparams_grid_set.ndim
            if(ndim == 1):
                interpolmethod = Parabola1DGridManifoldInterpolationMethod
            else:
                raise ValueError('There is no default fit parameter manifold grid interpolation method available for %d dimensions!'%(ndim))
        self.interpolmethod = interpolmethod

        # Generate the list of signal fit parameter names once here.
        self._cache_signal_fitparam_name_list = self.signal_fitparam_names

    @property
    def backgroundpdf(self):
        """The background PDF object, derived from ``pdf_type`` and
        IsBackgroundPDF.
        """
        return self._bkgpdf
    @backgroundpdf.setter
    def backgroundpdf(self, pdf):
        if(not (isinstance(pdf, self.pdf_type) and isinstance(pdf, IsBackgroundPDF))):
            raise TypeError('The backgroundpdf property must be an object which is derived from %s and IsBackgroundPDF!'%(typename(self.pdf_type)))
        self._bkgpdf = pdf

    @property
    def signalpdfset(self):
        """The signal PDFSet object for ``pdf_type`` PDF objects.
        """
        return self._sigpdfset
    @signalpdfset.setter
    def signalpdfset(self, pdfset):
        if(not (isinstance(pdfset, PDFSet) and isinstance(pdfset, IsSignalPDF) and issubclass(pdfset.pdf_type, self.pdf_type))):
            raise TypeError('The signalpdfset property must be an object which is derived from PDFSet and IsSignalPDF and whose pdf_type property is a subclass of %s!'%(typename(self.pdf_type)))
        self._sigpdfset = pdfset

    @property
    def interpolmethod(self):
        """The class derived from GridManifoldInterpolationMethod
        implementing the interpolation of the fit parameter manifold.
        """
        return self._interpolmethod
    @interpolmethod.setter
    def interpolmethod(self, cls):
        if(not issubclass(cls, GridManifoldInterpolationMethod)):
            raise TypeError('The interpolmethod property must be a sub-class '
                'of GridManifoldInterpolationMethod!')
        self._interpolmethod = cls

    def _get_signal_fitparam_names(self):
        """Returns the list of signal fit parameter names this PDF ratio is a
        function of. The list is taken from the fit parameter grid set of the
        signal PDFSet object. By construction this parameter grid set defines
        the signal fit parameters.
        """
        return self._sigpdfset.fitparams_grid_set.parameter_names

    def convert_signal_fitparam_name_into_index(self, signal_fitparam_name):
        """Converts the given signal fit parameter name into the parameter
        index, i.e. the position of parameter in the signal parameter grid set.

        Parameters
        ----------
        signal_fitparam_name : str
            The name of the signal fit parameter.

        Returns
        -------
        index : int
            The index of the signal fit parameter.
        """
        # If there is only one signal fit parameter, we just return index 0.
        if(len(self._cache_signal_fitparam_name_list) == 1):
            return 0

        # At this point we have to loop through the list and do name
        # comparisons.
        for (index, name) in enumerate(self._cache_signal_fitparam_name_list):
            if(name == signal_fitparam_name):
                return index

        # At this point there is no parameter defined.
        raise KeyError('The PDF ratio "%s" has no signal fit parameter named '
            '"%s"!'%(classname(self), signal_fitparam_name))
