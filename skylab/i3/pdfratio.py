# -*- coding: utf-8 -*-

import abc

from skylab.core.multiproc import IsParallelizable, parallelize
from skylab.core.pdf import PDFRatio, PDFRatioFillMethod, PDFSet, IsSignalPDF, IsBackgroundPDF
from skylab.i3.pdf import I3EnergyPDF


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


class SignalOverBackgroundI3EnergyPDFRatio(PDFRatio, IsParallelizable):
    """This class implements the signal over background PDF ratio for
    I3EnergyPDF enegry PDFs. It takes an object, which is derived from PDFSet
    for I3EnergyPDF PDF types, and which is derived from IsSignalPDF, as signal
    PDF. Furthermore, it takes an object, which is derived from I3EnergyPDF and
    IsBackgroundPDF, as background PDF, and creates a spline for the ratio of
    the signal and background PDFs for a grid of different discrete energy
    signal parameters, which are defined by the signal PDF set.
    """
    def __init__(self, signalpdfset, backgroundpdf, fillmethod, ncpu=None):
        """Creates a new IceCube signal-over-background energy PDF ratio object.

        Parameters
        ----------
        signalpdfset : class derived from PDFSet (for PDF type I3EnergyPDF),
                                          IsSignalPDF,
                                          UsesBinning
            The PDF set, which provides signal energy PDFs for a set of
            discrete signal parameters.
        backgroundpdf : class derived from I3EnergyPDF,
                                           IsBackgroundPDF
            The background energy PDF object.
        fillmethod : PDFRatioFillMethod
            An instance of class derived from PDFRatioFillMethod that implements
            the desired ratio fill method.
        ncpu : int | None
            The number of CPUs to use to create the ratio splines for the
            different sets of signal parameters.

        Errors
        ------
        ValueError
            If the signal and background PDFs use different binning.
        """
        super(SignalOverBackgroundI3EnergyPDFRatio, self).__init__(fillmethod=fillmethod, ncpu=ncpu)
        self.signalpdfset = signalpdfset
        self.backgroundpdf = backgroundpdf

        # Ensure same binning of signal and background PDFs.
        if(not signalpdfset.has_same_binning_as(backgroundpdf)):
            raise ValueError('The signal PDF has not the same binning as the background PDF!')

        def create_log_ratio_spline(sigpdfset, bkgpdf, fillmethod, params):
            """Creates the signal/background ratio spline for the given signal
            parameters.

            Returns
            -------
            log_ratio_spline : RegularGridInterpolator
                The spline of the logarithmic PDF ratio values.
            """
            # Get the signal PDF for the given signal parameters.
            sigpdf = sigpdfset.get_pdf(params)

            # Create the ratio array with the same shape than the background pdf
            # histogram.
            ratio = np.ones_like(bkgpdf.hist, dtype=np.float)

            # Fill the ratio array.
            ratio = fillmethod.fill_ratios(ratio,
                sigpdf.hist, bkgpdf.hist,
                sigpdf.hist_mask_mc_covered, sigpdf.hist_mask_mc_covered_zero_physics,
                bkgpdf.hist_mask_mc_covered, bkgpdf.hist_mask_mc_covered_zero_physics)

            # Define the grid points for the spline. In general, we use the bin
            # centers of the binning, but for the first and last point of each
            # dimension we use the lower and upper bin edge, respectively, to
            # ensure full coverage of the spline across the binning range.
            points_list = []
            for binning in sigpdf.binnings:
                points = binning.bincenters
                (points[0], points[-1]) = (binning.lower_edge, binning.upper_edge)
                points_list.append(points)

            # Create the spline for the ratio values.
            log_ratio_spline = scipy.interpolate.RegularGridInterpolator(
                tuple(points_list),
                np.log(ratio),
                method='linear',
                bounds_error=True)

            return log_ratio_spline

        # Get the list of parameter permutations for which we need to create
        # PDF ratio arrays.
        params_list = self.signalpdfset.params_list

        args_list = [ ((signalpdfset, backgroundpdf, fillmethod, params),{})
                     for params in params_list ]

        log_ratio_spline_list = parallelize(create_log_ratio_spline, args_list, self.ncpu)

        # Save all the log_ratio splines in a dictionary.
        self._params_hash_log_ratio_spline_dict = dict()
        for (params, log_ratio_spline) in zip(params_list, log_ratio_spline_list):
            params_hash = make_params_hash(params)
            self._params_hash_log_ratio_spline_dict[params_hash] = log_ratio_spline

    @property
    def backgroundpdf(self):
        """The background PDF object, derived from I3EnergyPDF and
        IsBackgroundPDF.
        """
        return self._bkgpdf
    @backgroundpdf.setter
    def backgroundpdf(self, pdf):
        if(not (isinstance(pdf, I3EnergyPDF) and isinstance(pdf, IsBackgroundPDF))):
            raise TypeError('The backgroundpdf property must be an object which is derived from I3EnergyPDF and IsBackgroundPDF!')
        self._bkgpdf = pdf

    @property
    def signalpdfset(self):
        """The signal PDFSet object for I3EnergyPDF.
        """
        return self._sigpdfset
    @signalpdfset.setter
    def signalpdfset(self, pdfset):
        if(not (isinstance(pdfset, PDFSet) and isinstance(pdfset, IsSignalPDF) and issubclass(pdfset.pdf_type, I3EnergyPDF))):
            raise TypeError('The signalpdfset property must be an object which is derived from PDFSet and IsSignalPDF and whose pdf_type property is a subclass of I3EnergyPDF!')
        self._sigpdfset = pdfset

