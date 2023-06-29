# -*- coding: utf-8 -*-

"""This module defines the interface and provides particular implementations
of PDF ratio fill methods. For a binned PDF ratio it could happen, that some
bins don't have background information but signal information is available.
Hence, a ratio cannot be computed for those bins. The PDF ratio fill method
specifies how those bins should get filled.
"""

import abc
import numpy as np

from skyllh.core.py import (
    float_cast,
)


class PDFRatioFillMethod(object, metaclass=abc.ABCMeta):
    """Abstract base class to implement a PDF ratio fill method. It can happen,
    that there are empty background bins but where signal could possibly be.
    A PDFRatioFillMethod implements what happens in such cases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def __call__(
            self,
            ratios,
            sig_pd_h,
            bkg_pd_h,
            sig_mask_mc_covered,
            sig_mask_mc_covered_zero_physics,
            bkg_mask_mc_covered,
            bkg_mask_mc_covered_zero_physics):
        """The __call__ method is supposed to fill the ratio bins (array)
        with the signal / background ratio values. For bins (array elements),
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
        sig_pd_h : ndarray of float
            The multi-dimensional array (histogram) holding the signal
            probability densities.
        bkg_pd_h : ndarray of float
            The multi-dimensional array (histogram) holding the background
            probability densities.
        sig_mask_mc_covered : ndarray of bool
            The mask array indicating which array elements of sig_pd_h have
            monte-carlo coverage.
        sig_mask_mc_covered_zero_physics : ndarray of bool
            The mask array indicating which array elements of sig_pd_h have
            monte-carlo coverage but don't have physics contribution.
        bkg_mask_mc_covered : ndarray of bool
            The mask array indicating which array elements of bkg_pd_h have
            monte-carlo coverage.
            In case of experimental data as background, this mask indicate where
            (experimental data) background is available.
        bkg_mask_mc_covered_zero_physics : ndarray of bool
            The mask array ndicating which array elements of bkg_pd_h have
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.signallike_percentile = 99.

    @property
    def signallike_percentile(self):
        """The percentile of signal-like ratios, which should be taken as the
        ratio value for ratios with no background probability. This percentile
        must be given as a float value in the range [0, 100] inclusively.
        """
        return self._signallike_percentile

    @signallike_percentile.setter
    def signallike_percentile(self, value):
        value = float_cast(
            value,
            'The value for the signallike_percentile property must be castable '
            'to type float!')
        if (value < 0) or (value > 100):
            raise ValueError(
                f'The value "{value}" of the signallike_percentile property '
                'must be in the range [0, 100]!')
        self._signallike_percentile = value

    def __call__(
            self,
            ratio,
            sig_pd_h,
            bkg_pd_h,
            sig_mask_mc_covered,
            sig_mask_mc_covered_zero_physics,
            bkg_mask_mc_covered,
            bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array ``ratio``.
        For more information see the documentation of
        :meth:`skyllh.core.pdfratio_fill.PDFRatioFillMethod.__call__`.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if np.any(bkg_mask_mc_covered_zero_physics):
            raise ValueError(
                'Some of the background bins have MC coverage but no physics '
                'background prediction. I don\'t know what to do in this case!')

        sig_domain = sig_pd_h > 0
        bkg_domain = bkg_pd_h > 0

        sig_bkg_domain = sig_domain & bkg_domain

        ratio[sig_bkg_domain] = (
            sig_pd_h[sig_bkg_domain] / bkg_pd_h[sig_bkg_domain]
        )

        ratio_value = np.percentile(
            ratio[ratio > 1.], self._signallike_percentile)
        np.copyto(ratio, ratio_value, where=sig_domain & ~bkg_domain)

        return ratio


class MostSignalLikePDFRatioFillMethod(PDFRatioFillMethod):
    """PDF ratio fill method to set the PDF ratio to the most signal like PDF
    ratio for bins, where there is signal but no background coverage.
    """
    def __init__(self, signallike_percentile=99., **kwargs):
        """Creates the PDF ratio fill method object for filling PDF ratio bins,
        where there is signal MC coverage but no background (MC) coverage
        with the most signal-like ratio value.

        Parameters
        ----------
        signallike_percentile : float in range [0., 100.], default 99.
            The percentile of signal-like ratios, which should be taken as the
            ratio value for ratios with no background probability.
        """
        super().__init__(**kwargs)

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
        value = float_cast(
            value,
            'The value for the signallike_percentile property must be castable '
            'to type float!')
        if (value < 0) or (value > 100):
            raise ValueError(
                f'The value "{value}" of the signallike_percentile property '
                'must be in the range [0, 100]!')
        self._signallike_percentile = value

    def __call__(
            self,
            ratio,
            sig_pd_h,
            bkg_pd_h,
            sig_mask_mc_covered,
            sig_mask_mc_covered_zero_physics,
            bkg_mask_mc_covered,
            bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array ``ratio``.
        For more information see the documentation of
        :meth:`skyllh.core.pdfratio_fill.PDFRatioFillMethod.__call__`.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if np.any(bkg_mask_mc_covered_zero_physics):
            raise ValueError(
                'Some of the background bins have MC coverage but no physics '
                'background prediction. I don\'t know what to do in this case!')

        # Fill the bins where we have signal and background MC coverage.
        mask_sig_and_bkg_mc_covered = sig_mask_mc_covered & bkg_mask_mc_covered
        ratio[mask_sig_and_bkg_mc_covered] = (
            sig_pd_h[mask_sig_and_bkg_mc_covered] /
            bkg_pd_h[mask_sig_and_bkg_mc_covered]
        )

        # Calculate the ratio value, which should be used for ratio bins, where
        # we have signal MC coverage but no background MC coverage.
        ratio_value = np.percentile(
            ratio[ratio > 1.], self._signallike_percentile)
        mask_sig_but_notbkg_mc_covered = (
            sig_mask_mc_covered & ~bkg_mask_mc_covered
        )
        np.copyto(ratio, ratio_value, where=mask_sig_but_notbkg_mc_covered)

        return ratio


class MinBackgroundLikePDFRatioFillMethod(PDFRatioFillMethod):
    """PDF ratio fill method to set the PDF ratio to the minimal background like
    value for bins, where there is signal but no background coverage.
    """
    def __init__(self, **kwargs):
        """Creates the PDF ratio fill method object for filling PDF ratio bins,
        where there is signal MC coverage but no background (MC) coverage
        with the minimal background-like ratio value.
        """
        super().__init__(**kwargs)

    def __call__(
            self,
            ratio,
            sig_pd_h,
            bkg_pd_h,
            sig_mask_mc_covered,
            sig_mask_mc_covered_zero_physics,
            bkg_mask_mc_covered,
            bkg_mask_mc_covered_zero_physics):
        """Fills the ratio array ``ratio``.
        For more information see the documentation of
        :meth:`skyllh.core.pdfratio_fill.PDFRatioFillMethod.__call__`.
        """
        # Check if we have predicted background for the entire background MC
        # range.
        if np.any(bkg_mask_mc_covered_zero_physics):
            raise ValueError(
                'Some of the background bins have MC coverage but no physics '
                'background prediction. I don\'t know what to do in this case!')

        # Fill the bins where we have signal and background MC coverage.
        mask_sig_and_bkg_mc_covered = sig_mask_mc_covered & bkg_mask_mc_covered
        ratio[mask_sig_and_bkg_mc_covered] = (
            sig_pd_h[mask_sig_and_bkg_mc_covered] /
            bkg_pd_h[mask_sig_and_bkg_mc_covered]
        )

        # Calculate the minimal background-like value.
        min_bkg_prob = np.min(bkg_pd_h[bkg_mask_mc_covered])

        # Set the ratio using the minimal background probability where we
        # have signal MC coverage but no background (MC) coverage.
        mask_sig_but_notbkg_mc_covered = (
            sig_mask_mc_covered & ~bkg_mask_mc_covered
        )
        ratio[mask_sig_but_notbkg_mc_covered] =\
            sig_pd_h[mask_sig_but_notbkg_mc_covered] / min_bkg_prob

        return ratio
