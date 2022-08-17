# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.binning import UsesBinning
from skyllh.core.pdf import (
    EnergyPDF,
    IsBackgroundPDF,
    PDFAxis
)
from skyllh.core.py import issequenceof
from skyllh.core.storage import DataFieldRecordArray
from skyllh.core.timing import TaskTimer
from skyllh.core.smoothing import (
    UNSMOOTH_AXIS,
    SmoothingFilter,
    HistSmoothingMethod,
    NoHistSmoothingMethod,
    NeighboringBinHistSmoothingMethod
)
from skyllh.core.timing import TaskTimer

from scipy.stats import gaussian_kde


class PDEnergyPDF(EnergyPDF, UsesBinning):
    """This is the base class for IceCube specific energy PDF models.
    IceCube energy PDFs depend soley on the energy and the
    zenith angle, and hence, on the declination of the event.

    The IceCube energy PDF is modeled as a 1d histogram in energy,
    but for different sin(declination) bins, hence, stored as a 2d histogram.
    """

    _KDE_BW_NORTH = 0.4
    _KDE_BW_SOUTH = 0.32

    def __init__(self, data_logE, data_sinDec, data_mcweight, data_physicsweight,
                 logE_binning, sinDec_binning, smoothing_filter, kde_smoothing=False):
        """Creates a new IceCube energy PDF object.

        Parameters
        ----------
        data_logE : 1d ndarray
            The array holding the log10(E) values of the events.
        data_sinDec : 1d ndarray
            The array holding the sin(dec) values of the events.
        data_mcweight : 1d ndarray
            The array holding the monte-carlo weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        data_physicsweight : 1d ndarray
            The array holding the physics weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        logE_binning : BinningDefinition
            The binning definition for the log(E) axis.
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination) axis.
        smoothing_filter : SmoothingFilter instance | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        kde_smoothing : bool
            Apply a kde smoothing to the enrgy pdf for each sine of the
            muon declination.
            Default: False.
        """
        super(PDEnergyPDF, self).__init__()

        # self.logger = logging.getLogger(__name__)

        # Define the PDF axes.
        self.add_axis(PDFAxis(name='log_energy',
                              vmin=logE_binning.lower_edge,
                              vmax=logE_binning.upper_edge))
        self.add_axis(PDFAxis(name='sin_dec',
                              vmin=sinDec_binning.lower_edge,
                              vmax=sinDec_binning.upper_edge))

        self.add_binning(logE_binning, 'log_energy')
        self.add_binning(sinDec_binning, 'sin_dec')

        # Create the smoothing method instance tailored to the energy PDF.
        # We will smooth only the first axis (logE).
        if((smoothing_filter is not None) and
           (not isinstance(smoothing_filter, SmoothingFilter))):
            raise TypeError(
                'The smoothing_filter argument must be None or an instance of SmoothingFilter!')
        if(smoothing_filter is None):
            self.hist_smoothing_method = NoHistSmoothingMethod()
        else:
            self.hist_smoothing_method = NeighboringBinHistSmoothingMethod(
                (smoothing_filter.axis_kernel_array, UNSMOOTH_AXIS))

        # We have to figure out, which histogram bins are zero due to no
        # monte-carlo coverage, and which due to zero physics model
        # contribution.

        # Create a 2D histogram with only the MC events to determine the MC
        # coverage.
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE, data_sinDec,
            bins=[
                logE_binning.binedges, sinDec_binning.binedges],
            range=[
                logE_binning.range, sinDec_binning.range],
            normed=False)
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered = h > 0

        # Select the events which have MC coverage but zero physics
        # contribution, i.e. the physics model predicts zero contribution.
        mask = data_physicsweight == 0.

        # Create a 2D histogram with only the MC events that have zero physics
        # contribution. Note: By construction the zero physics contribution bins
        # are a subset of the MC covered bins.
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE[mask], data_sinDec[mask],
            bins=[
                logE_binning.binedges, sinDec_binning.binedges],
            range=[
                logE_binning.range, sinDec_binning.range],
            normed=False)
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered_zero_physics = h > 0

        # Create a 2D histogram with only the data which has physics
        # contribution. We will do the normalization along the logE
        # axis manually.
        data_weights = data_mcweight[~mask] * data_physicsweight[~mask]
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE[~mask], data_sinDec[~mask],
            bins=[
                logE_binning.binedges, sinDec_binning.binedges],
            weights=data_weights,
            range=[
                logE_binning.range, sinDec_binning.range],
            normed=False)

        # If a bandwidth is passed, apply a KDE-based smoothing with the given
        # bw parameter as bandwidth for the fit.
        # Warning: right now this implies an additional dependency on an
        # external package for KDE analysis.
        if kde_smoothing:
            if not isinstance(kde_smoothing, bool):
                raise ValueError(
                    "The bandwidth parameter must be True or False!")
            kde_pdf = np.empty(
                (len(sinDec_binning.bincenters),), dtype=object)
            data_logE_mask = data_logE[~mask]
            data_sinDec_mask = data_sinDec[~mask]
            for i in range(len(sinDec_binning.bincenters)):
                sindec_mask = np.logical_and(
                    data_sinDec_mask >= sinDec_binning.binedges[i],
                    data_sinDec_mask < sinDec_binning.binedges[i+1]
                )
                this_energy = data_logE_mask[sindec_mask]
                if sinDec_binning.binedges[i] >= 0:
                    kde_pdf[i] = gaussian_kde(
                        this_energy, bw_method=self._KDE_BW_NORTH)
                else:
                    kde_pdf[i] = gaussian_kde(
                        this_energy, bw_method=self._KDE_BW_SOUTH)
            h = np.vstack(
                [kde_pdf[i].evaluate(logE_binning.bincenters)
                 for i in range(len(sinDec_binning.bincenters))]).T

        # Calculate the normalization for each logE bin. Hence we need to sum
        # over the logE bins (axis 0) for each sin(dec) bin and need to divide
        # by the logE bin widths along the sin(dec) bins. The result array norm
        # is a 2D array of the same shape as h.
        norms = np.sum(h, axis=(0,))[np.newaxis, ...] * \
            np.diff(logE_binning.binedges)[..., np.newaxis]
        h /= norms
        h = self._hist_smoothing_method.smooth(h)

        self._hist_logE_sinDec = h

    @ property
    def hist_smoothing_method(self):
        """The HistSmoothingMethod instance defining the smoothing filter of the
        energy PDF histogram.
        """
        return self._hist_smoothing_method

    @ hist_smoothing_method.setter
    def hist_smoothing_method(self, method):
        if(not isinstance(method, HistSmoothingMethod)):
            raise TypeError(
                'The hist_smoothing_method property must be an instance of HistSmoothingMethod!')
        self._hist_smoothing_method = method

    @ property
    def hist(self):
        """(read-only) The 2D logE-sinDec histogram array.
        """
        return self._hist_logE_sinDec

    @ property
    def hist_mask_mc_covered(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage.
        """
        return self._hist_mask_mc_covered

    @ property
    def hist_mask_mc_covered_zero_physics(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage but zero physics
        contribution.
        """
        return self._hist_mask_mc_covered_zero_physics

    @ property
    def hist_mask_mc_covered_with_physics(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage and has physics
        contribution.
        """
        return self._hist_mask_mc_covered & ~self._hist_mask_mc_covered_zero_physics

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this energy PDF is valid for all the given experimental
        data.
        It checks if all the data is within the logE and sin(dec) binning range.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:

            - 'log_energy' : float
                The logarithm of the energy value of the data event.
            - 'dec' : float
                The declination of the data event.

        Raises
        ------
        ValueError
            If some of the data is outside the logE or sin(dec) binning range.
        """
        logE_binning = self.get_binning('log_energy')
        sinDec_binning = self.get_binning('sin_dec')

        exp_logE = data_exp['log_energy']
        exp_sinDec = np.sin(data_exp['dec'])

        # Check if all the data is within the binning range.
        # if(logE_binning.any_data_out_of_binning_range(exp_logE)):
        # self.logger.warning('Some data is outside the logE range (%.3f, %.3f)', logE_binning.lower_edge, logE_binning.upper_edge)
        # if(sinDec_binning.any_data_out_of_binning_range(exp_sinDec)):
        # self.logger.warning('Some data is outside the sin(dec) range (%.3f, %.3f)', sinDec_binning.lower_edge, sinDec_binning.upper_edge)

    def get_prob(self, tdm, fitparams=None, tl=None):
        """Calculates the energy probability (in logE) of each event.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            - 'log_energy' : float
                The logarithm of the energy value of the event.
            - 'sin_dec' : float
                The sin(declination) value of the event.

        fitparams : None
            Unused interface parameter.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        prob : 1D (N_events,) shaped ndarray
            The array with the energy probability for each event.
        """
        get_data = tdm.get_data

        logE_binning = self.get_binning('log_energy')
        sinDec_binning = self.get_binning('sin_dec')

        logE_idx = np.digitize(
            get_data('log_energy'), logE_binning.binedges) - 1
        sinDec_idx = np.digitize(
            get_data('sin_dec'), sinDec_binning.binedges) - 1

        with TaskTimer(tl, 'Evaluating logE-sinDec histogram.'):
            prob = self._hist_logE_sinDec[(logE_idx, sinDec_idx)]

        return prob


class PDDataBackgroundI3EnergyPDF(PDEnergyPDF, IsBackgroundPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    experimental data. This class is derived from I3EnergyPDF.
    """

    def __init__(self, data_exp, logE_binning, sinDec_binning,
                 smoothing_filter=None, kde_smoothing=False):
        """Constructs a new IceCube energy background PDF from experimental
        data.

        Parameters
        ----------
        data_exp : instance of DataFieldRecordArray
            The array holding the experimental data. The following data fields
            must exist:

            - 'log_energy' : float
                The logarithm of the reconstructed energy value of the data
                event.
            - 'sin_dec' : float
                The sine of the reconstructed declination of the data event.

        logE_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        smoothing_filter : SmoothingFilter instance | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        """
        if(not isinstance(data_exp, DataFieldRecordArray)):
            raise TypeError('The data_exp argument must be an instance of '
                            'DataFieldRecordArray!')

        data_logE = data_exp['log_energy']
        data_sinDec = data_exp['sin_dec']
        # For experimental data, the MC and physics weight are unity.
        data_mcweight = np.ones((len(data_exp),))
        data_physicsweight = data_mcweight

        # Create the PDF using the base class.
        super(PDDataBackgroundI3EnergyPDF, self).__init__(
            data_logE, data_sinDec, data_mcweight, data_physicsweight,
            logE_binning, sinDec_binning, smoothing_filter, kde_smoothing
        )
        # Check if this PDF is valid for all the given experimental data.
        self.assert_is_valid_for_exp_data(data_exp)
