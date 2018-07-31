# -*- coding: utf-8 -*-

import abc
import numpy as np

from skylab.core.py import typename
from skylab.core.parameters import FitParameterManifoldGridInterpolationMethod, ParabolaFitParameterInterpolationMethod
from skylab.core.pdf import SpatialPDF, PDFSet, IsSignalPDF, IsBackgroundPDF

class IsPDFRatio(object):
    """Abstract classifier class for a PDF ratio class. It defines the interface
    of a PDF ratio class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(IsPDFRatio, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def get_ratio(self, events, fitparams):
        """Retrieves the PDF ratio value for each given event, given the given
        set of fit parameters. This method is called during the likelihood
        maximization process.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the PDF
            ratio values should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values.

        Returns
        -------
        ratios : (N_events,) or (N_events,N_sources) shaped ndarray
            The PDF ratio value for each given event. If the signal PDF depends
            on the source, a 2D ndarray is returned with the PDF ratio values
            for each event and source.
        """
        pass

    @abc.abstractmethod
    def get_gradient(self, events, fitparams, pidx):
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
        pidx : int
            The index of the fit parameter for which the gradient should
            get calculated.

        Returns
        -------
        gradient : 1d ndarray of float
            The PDF ratio gradient value for each given event.
        """
        pass

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


class SigOverBkgPDFRatio(IsPDFRatio):
    """This class implements a signal-over-background PDF ratio for
    PDFs without any fit parameter dependence.
    It takes a signal PDF of type *pdf_type* and a background PDF of type
    *pdf_type* and calculates the PDF ratio.

    One instance of this class can be used to calculate the PDF ratio
    for several sources. By definition this PDF ratio does not depend on any
    fit parameters. Hence, calling the ``get_gradient`` method will result in
    throwing a RuntimeError exception.
    """
    def __init__(self, pdf_type, signalpdf, backgroundpdf, *args, **kwargs):
        """Creates a new signal-over-background PDF ratio instance.

        Parameters
        ----------
        pdf_type : type
            The python type of the PDF object for which the PDF ratio is for.
        signalpdf : class instance derived from `pdf_type`, IsSignalPDF
            The instance of the signal PDF.
        backgroundpdf : class instance derived from `pdf_type`, IsBackgroundPDF
            The instance of the background PDF.
        """
        super(SigOverBkgPDFRatio, self).__init__(*args, **kwargs)

        self._pdf_type = pdf_type

        self.signalpdf = signalpdf
        self.backgroundpdf = backgroundpdf

        # Check that the PDF axes ranges are the same for the signal and
        # background PDFs.
        if(not signalpdf.axes.is_same_as(backgroundpdf.axes)):
            raise ValueError('The signal and background PDFs do not have the same axes.')

    @property
    def pdf_type(self):
        """(read-only) The python type of the PDF object for which the PDF
        ratio is for.
        """
        return self._pdf_type

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

        ratios = sigprob / bkgprob

        return ratios

    def get_gradient(self, events, fitparams, pidx):
        """Calling this method results in throwing a RuntimeError exception,
        because this PDF ratio class handles only spatial PDFs without any fit
        parameters.
        """
        raise RuntimeError('The BasicSpatialSigOverBkgPDFRatio handles only PDFs with no fit parameters! So calling get_gradient is meaningless!')


class SigSetOverBkgPDFRatio(IsPDFRatio):
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
            The python type of the PDF object for which the PDF ratio is for.
        signalpdfset : class instance derived from PDFSet (for PDF type
                       ``pdf_type``), and IsSignalPDF
            The PDF set, which provides signal PDFs for a set of
            discrete signal fit parameters.
        backgroundpdf : class instance derived from ``pdf_type``, and
                        IsBackgroundPDF
        interpolmethod : class of FitParameterManifoldGridInterpolationMethod | None
            The class implementing the fit parameter interpolation method for
            the PDF ratio manifold grid.
            If set to None (default), the ParabolaFitParameterInterpolationMethod
            will be used for 1-dimensional fit parameter manifolds.
        """
        # Call super to allow for multiple class inheritance.
        super(SigSetOverBkgPDFRatio, self).__init__(*args, **kwargs)

        self._pdf_type = pdf_type

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

    @property
    def pdf_type(self):
        """(read-only) The python type of the PDF object for which the PDF
        ratio is for.
        """
        return self._pdf_type

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
