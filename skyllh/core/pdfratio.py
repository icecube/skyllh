# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np

from skyllh.core.py import (
    classname,
    float_cast,
    issequenceof,
    range,
    typename
)
from skyllh.core.parameters import (
    FitParameter,
    FitParameterManifoldGridInterpolationMethod,
    ParabolaFitParameterInterpolationMethod
)
from skyllh.core.pdf import (
    PDFSet,
    IsBackgroundPDF,
    IsSignalPDF,
    SpatialPDF
)


class PDFRatio(object):
    """Abstract base class for a PDF ratio class. It defines the interface
    of a PDF ratio class.
    """
    __metaclass__ = abc.ABCMeta

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

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """This method must be reimplemented by the derived class if necessary.
        It is supposed to change the SourceHypoGroupManager instance of the PDF
        instances, which rely on it. By definition these are the signal PDFs.
        """
        pass

    def initialize_for_new_trial(self, events):
        """This method must be reimplemented by the derived class if necessary.
        It is supposed to tell the PDF ratio class and hence its assigned signal
        and background classes that a new trial is being initialized with the
        given events.
        """
        pass

    @abc.abstractmethod
    def get_ratio(self, events, fitparams=None):
        """Retrieves the PDF ratio value for each given event, given the given
        set of fit parameters. This method is called during the likelihood
        maximization process.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the PDF
            ratio values should get calculated.
        fitparams : dict | None
            The dictionary with the fit parameter name-value pairs.
            It's supposed to be set to None, if the PDF ratio does not depend
            on any fit parameters.

        Returns
        -------
        ratios : (N_events,) or (N_events,N_sources) shaped ndarray
            The PDF ratio value for each given event. If the signal PDF depends
            on the source, a 2D ndarray is returned with the PDF ratio values
            for each event and source.
        """
        pass

    @abc.abstractmethod
    def get_gradient(self, events, fitparams, fitparam_name):
        """Retrieves the PDF ratio gradient for the parameter ``pidx`` for each
        given event, given the given set of fit parameters. This method is
        called during the likelihood maximization process.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the PDF
            ratio gradients should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values.
        fitparam_name : str
            The name of the fit parameter for which the gradient should
            get calculated.

        Returns
        -------
        gradient : 1d ndarray of float
            The PDF ratio gradient value for each given event.
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
        self.pdfratio_list = pdfratios
        self.fitparam_list = fitparams

        # The ``events`` property will be set via the
        # ``initialize_for_new_trial`` method.
        self._events = None

        # The ``_ratio_values`` member variable will hold a
        # (N_pdfratios,N_events)-shaped array holding the PDF ratio values of
        # each PDF ratio object for each event. It will be created by the
        # ``initialize_for_new_trial`` method.
        self._ratio_values = None

        # Create a mapping of fit parameter index to pdfratio index. We
        # initialize the mapping with -1 first in order to be able to check in
        # the end if all fit parameters found a PDF ratio object.
        self._fitparam_idx_2_pdfratio_idx = np.repeat(np.array([-1], dtype=np.int),
                                                      len(self._fitparam_list))
        for ((fpidx, fitparam), (pridx, pdfratio)) in itertools.product(
                enumerate(self._fitparam_list), enumerate(self.pdfratio_list)):
            if(fitparam.name in pdfratio.fitparam_names):
                self._fitparam_idx_2_pdfratio_idx[fpidx] = pridx
        check_mask = (self._fitparam_idx_2_pdfratio_idx == -1)
        if(np.any(check_mask)):
            raise KeyError('%d fit parameters are not defined in any of the PDF ratio instances!'%(np.sum(check_mask)))

        # Create the list of indices of the PDFRatio instances, which depend on
        # at least one fit parameter.
        self._var_pdfratio_indices = np.unique(self._fitparam_idx_2_pdfratio_idx)

    def _precompute_static_pdfratio_values(self):
        """Pre-compute the PDF ratio values for the PDF ratios that do not
        depend on any fit parameters.
        """
        for (i, pdfratio) in enumerate(self._pdfratio_list):
            if(pdfratio.n_fitparams == 0):
                # The PDFRatio does not depend on any fit parameters. So we
                # pre-calculate the PDF ratio values for all the events. Since
                # the get_ratio method of the PDFRatio class might return a 2D
                # (N_sources, N_events)-shaped array, and we assume a single
                # source, we need to reshape the array, which does not involve
                # any data copying.
                self._ratio_values[i] = np.reshape(pdfratio.get_ratio(self._events), (len(self._events),))

    @property
    def pdfratio_list(self):
        """The list of PDFRatio objects.
        """
        return self._pdfratio_list
    @pdfratio_list.setter
    def pdfratio_list(self, seq):
        if(not issequenceof(seq, PDFRatio)):
            raise TypeError('The pdfratio_list property must be a sequence of PDFRatio instances!')
        self._pdfratio_list = list(seq)

    @property
    def fitparam_list(self):
        """The list of FitParameter instances.
        """
        return self._fitparam_list
    @fitparam_list.setter
    def fitparam_list(self, seq):
        if(not issequenceof(seq, FitParameter)):
            raise TypeError('The fitparam_list property must be a sequence of FitParameter instances!')
        self._fitparam_list = list(seq)

    @property
    def events(self):
        """The numpy record array holding the event data.
        """
        return self._events
    @events.setter
    def events(self, arr):
        if(not isinstance(arr, np.ndarray)):
            raise TypeError('The events property must be an instance of numpy.ndarray!')
        self._events = arr

    def initialize_for_new_trial(self, events):
        """Initializes the PDFRatio array arithmetic for a new trial. For a new
        trial the data events change, hence we need to recompute the PDF ratio
        values of the fit parameter independent PDFRatio instances.

        Parameters
        ----------
        events : numpy record array
            The numpy record array holding the new data events of the new trial.
        """
        n_events_old = 0
        if(self._events is not None):
            n_events_old = len(self._events)

        # Set the new events.
        self.events = events

        # If the amount of events have changed, we need a new array holding the
        # ratio values.
        if(n_events_old != len(self._events)):
            # Create a (N_pdfratios,N_events)-shaped array to hold the PDF ratio
            # values of each PDF ratio object for each event.
            self._ratio_values = np.empty((len(self._pdfratio_list),len(self._events)), dtype=np.float)

        self._precompute_static_pdfratio_values()

    def get_pdfratio(self, fitparam_idx):
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
        pdfratio_idx = self._fitparam_idx_2_pdfratio_idx[fitparam_idx]
        return self._pdfratio_list[pdfratio_idx]

    def calculate_pdfratio_values(self, fitparams):
        """Calculates the PDF ratio values for the PDF ratio objects which
        depend on fit parameters.

        Parameters
        ----------
        fitparams : dict
            The dictionary with the fit parameter name-value pairs.
        """
        for i in self._var_pdfratio_indices:
            # Since the get_ratio method of the PDFRatio class might return a 2D
            # (N_sources, N_events)-shaped array, and we assume a single source,
            # we need to reshape the array, which does not involve any data
            # copying.
            self._ratio_values[i] = np.reshape(self._pdfratio_list[i].get_ratio(self._events, fitparams), (len(self._events),))

    def get_ratio_product(self, excluded_fitparam_idx=None):
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
        if(excluded_fitparam_idx is None):
            return np.prod(self._ratio_values, axis=0)

        # Get the index of the PDF ratio object that corresponds to the excluded
        # fit parameter.
        excluded_pdfratio_idx = self._fitparam_idx_2_pdfratio_idx[excluded_fitparam_idx]
        pdfratio_indices = list(range(self._ratio_values.shape[0]))
        pdfratio_indices.pop(excluded_pdfratio_idx)
        return np.prod(self._ratio_values[pdfratio_indices], axis=0)


class PDFRatioFillMethod(object):
    """Abstract base class to implement a PDF ratio fill method. It can happen,
    that there are empty background bins but where signal could possibly be.
    A PDFRatioFillMethod implements what happens in such cases.
    """
    __metaclass__ = abc.ABCMeta

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
    """This class implements a signal-over-background PDF ratio for
    PDFs without any fit parameter dependence.
    It takes a signal PDF of type *pdf_type* and a background PDF of type
    *pdf_type* and calculates the PDF ratio.

    One instance of this class can be used to calculate the PDF ratio
    for several sources. By definition this PDF ratio does not depend on any
    fit parameters. Hence, calling the ``get_gradient`` method will result in
    throwing a RuntimeError exception.
    """
    def __init__(self, pdf_type, signalpdf, backgroundpdf, same_axes=True,
        zero_bkg_ratio_value=1., *args, **kwargs):
        """Creates a new signal-over-background PDF ratio instance.

        Parameters
        ----------
        pdf_type : type
            The python type of the PDF object for which the PDF ratio is for.
        signalpdf : class instance derived from `pdf_type`, IsSignalPDF
            The instance of the signal PDF.
        backgroundpdf : class instance derived from `pdf_type`, IsBackgroundPDF
            The instance of the background PDF.
        same_axes : bool
            Flag if the signal and background PDFs are supposed to have the
            same axes. Default is True.
        zero_bkg_ratio_value : float
            The value of the PDF ratio to take when the background PDF value
            is zero. This is to avoid division by zero. Default is 1.
        """
        super(SigOverBkgPDFRatio, self).__init__(pdf_type, *args, **kwargs)

        self.signalpdf = signalpdf
        self.backgroundpdf = backgroundpdf

        # Check that the PDF axes ranges are the same for the signal and
        # background PDFs.
        if(same_axes and (not signalpdf.axes.is_same_as(backgroundpdf.axes))):
            raise ValueError('The signal and background PDFs do not have the '
                'same axes.')

        self.zero_bkg_ratio_value = zero_bkg_ratio_value

    @property
    def signalpdf(self):
        """The signal spatial PDF object used to create the PDF ratio.
        """
        return self._signalpdf
    @signalpdf.setter
    def signalpdf(self, pdf):
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The signalpdf property must be an instance of %s!'%(typename(self.pdf_type)))
        if(not isinstance(pdf, IsSignalPDF)):
            raise TypeError('The signalpdf property must be an instance of IsSignalPDF!')
        self._signalpdf = pdf

    @property
    def backgroundpdf(self):
        """The background spatial PDF object used to create the PDF ratio.
        """
        return self._backgroundpdf
    @backgroundpdf.setter
    def backgroundpdf(self, pdf):
        if(not isinstance(pdf, self.pdf_type)):
            raise TypeError('The backgroundpdf property must be an instance of %s!'%(typename(self.pdf_type)))
        if(not isinstance(pdf, IsBackgroundPDF)):
            raise TypeError('The backgroundpdf property must be an instance of IsBackgroundPDF!')
        self._backgroundpdf = pdf

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

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Calls the ``change_source_hypo_group_manager`` method of the signal
        and background PDF.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance.
        """
        self._signalpdf.change_source_hypo_group_manager(src_hypo_group_manager)
        self._backgroundpdf.change_source_hypo_group_manager(src_hypo_group_manager)

    def initialize_for_new_trial(self, events):
        """Notifies the signal and background PDFs that a new data trial is
        being initialized with the given events. This calls the
        `initialize_for_new_trial` method of the signal and background PDF class
        instance.
        """
        self._signalpdf.initialize_for_new_trial(events)
        self._backgroundpdf.initialize_for_new_trial(events)

    def get_ratio(self, events, fitparams=None):
        """Calculates the PDF ratio for the given events.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the PDF
            ratio values should get calculated.
        fitparams : None
            Unused interface argument.

        Returns
        -------
        ratios : (N_events) or (N_sources,N_events) shaped ndarray
            The ndarray holding the probability ratio for each event (and each
            source). The dimensionality of the returned ndarray depends on the
            dimensionality of the probability ndarray returned by the
            ``get_prob`` method of signal PDF object.
        """
        sigprob = self._signalpdf.get_prob(events)
        bkgprob = self._backgroundpdf.get_prob(events)

        # Select only the events, where background pdf is greater than zero.
        m = (bkgprob > 0)
        minv = np.invert(m)

        ratios = np.empty_like(events, dtype=np.float)
        ratios[m] = sigprob[m] / bkgprob[m]
        ratios[minv] = self._zero_bkg_ratio_value

        return ratios

    def get_gradient(self, events, fitparams, fitparam_name):
        """Calling this method results in throwing a RuntimeError exception,
        because this PDF ratio class handles only spatial PDFs without any fit
        parameters.
        """
        raise RuntimeError('The SigOverBkgPDFRatio handles only PDFs with no '
            'fit parameters! So calling get_gradient is meaningless!')


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
        interpolmethod : class of FitParameterManifoldGridInterpolationMethod | None
            The class implementing the fit parameter interpolation method for
            the PDF ratio manifold grid.
            If set to None (default), the ParabolaFitParameterInterpolationMethod
            will be used for 1-dimensional fit parameter manifolds.
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
                interpolmethod = ParabolaFitParameterInterpolationMethod
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
        """The class derived from FitParameterManifoldGridInterpolationMethod
        implementing the interpolation of the fit parameter manifold.
        """
        return self._interpolmethod
    @interpolmethod.setter
    def interpolmethod(self, cls):
        if(not issubclass(cls, FitParameterManifoldGridInterpolationMethod)):
            raise TypeError('The interpolmethod property must be a sub-class of FitParameterManifoldGridInterpolationMethod!')
        self._interpolmethod = cls

    def _get_signal_fitparam_names(self):
        """Returns the list of signal fit parameter names this PDF ratio is a
        function of. The list is taken from the fit parameter grid set of the
        signal PDFSet object. By construction this parameter grid set defines
        the signal fit parameters.
        """
        return self._sigpdfset.fitparams_grid_set.parameter_names

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Calls the change_source_hypo_group_manager method of the signal
        PDFSet and background PDF instances.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance.
        """
        self._sigpdfset.change_source_hypo_group_manager(src_hypo_group_manager)
        self._bkgpdf.change_source_hypo_group_manager(src_hypo_group_manager)

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
        raise KeyError('The PDF ratio "%s" has no signal fit parameter named "%s"!'%(classname(self), signal_fitparam_name))


class SpatialSigOverBkgPDFRatio(SigOverBkgPDFRatio):
    """This class implements a spatial signal-over-background PDF ratio for
    PDFs without any fit parameter dependence.
    It takes a signal PDF of type SpatialPDF and a background PDF of type
    SpatialPDF and calculates the PDF ratio.
    """
    def __init__(self, signalpdf, backgroundpdf, *args, **kwargs):
        """Creates a new signal-over-background PDF ratio instance for spatial
        PDFs.

        Parameters
        ----------
        signalpdf : class instance derived from SpatialPDF, IsSignalPDF
            The instance of the spatial signal PDF.
        backgroundpdf : class instance derived from SpatialPDF, IsBackgroundPDF
            The instance of the spatial background PDF.
        """
        super(SpatialSigOverBkgPDFRatio, self).__init__(pdf_type=SpatialPDF,
            signalpdf=signalpdf, backgroundpdf=backgroundpdf, *args, **kwargs)

        # Make sure that the PDFs have two dimensions, i.e. RA and Dec.
        if(not signalpdf.ndim == 2):
            raise ValueError('The spatial signal PDF must have two dimensions! Currently it has %d!'%(signalpdf.ndim))
