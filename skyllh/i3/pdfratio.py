# -*- coding: utf-8 -*-

import abc
import numpy as np
import scipy.interpolate

from skyllh.core.parameters import make_params_hash
from skyllh.core.multiproc import IsParallelizable, parallelize
from skyllh.core.pdfratio import SigSetOverBkgPDFRatio, PDFRatioFillMethod, MostSignalLikePDFRatioFillMethod

from skyllh.i3.pdf import I3EnergyPDF


class I3EnergySigSetOverBkgPDFRatioSpline(SigSetOverBkgPDFRatio, IsParallelizable):
    """This class implements a signal over background PDF ratio spline for
    I3EnergyPDF enegry PDFs. It takes an object, which is derived from PDFSet
    for I3EnergyPDF PDF types, and which is derived from IsSignalPDF, as signal
    PDF. Furthermore, it takes an object, which is derived from I3EnergyPDF and
    IsBackgroundPDF, as background PDF, and creates a spline for the ratio of
    the signal and background PDFs for a grid of different discrete energy
    signal fit parameters, which are defined by the signal PDF set.
    """
    def __init__(
            self, signalpdfset, backgroundpdf,
            fillmethod=None, interpolmethod=None, ncpu=None, ppbar=None):
        """Creates a new IceCube signal-over-background energy PDF ratio object.

        Parameters
        ----------
        signalpdfset : class instance derived from PDFSet (for PDF type
                       I3EnergyPDF), IsSignalPDF, and UsesBinning
            The PDF set, which provides signal energy PDFs for a set of
            discrete signal parameters.
        backgroundpdf : class instance derived from I3EnergyPDF, and
                        IsBackgroundPDF
            The background energy PDF object.
        fillmethod : instance of PDFRatioFillMethod | None
            An instance of class derived from PDFRatioFillMethod that implements
            the desired ratio fill method.
            If set to None (default), the default ratio fill method
            MostSignalLikePDFRatioFillMethod will be used.
        interpolmethod : class of GridManifoldInterpolationMethod
            The class implementing the fit parameter interpolation method for
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
        super(I3EnergySigSetOverBkgPDFRatioSpline, self).__init__(
            pdf_type=I3EnergyPDF,
            signalpdfset=signalpdfset, backgroundpdf=backgroundpdf,
            interpolmethod=interpolmethod,
            ncpu=ncpu)

        # Define the default ratio fill method.
        if(fillmethod is None):
            fillmethod = MostSignalLikePDFRatioFillMethod()
        self.fillmethod = fillmethod

        # Ensure same binning of signal and background PDFs.
        for (sigpdf_hash, sigpdf) in self.signalpdfset.items():
            if(not sigpdf.has_same_binning_as(self.backgroundpdf)):
                raise ValueError('At least one signal PDF does not have the same binning as the background PDF!')

        def create_log_ratio_spline(sigpdfset, bkgpdf, fillmethod, gridfitparams):
            """Creates the signal/background ratio spline for the given signal
            parameters.

            Returns
            -------
            log_ratio_spline : RegularGridInterpolator
                The spline of the logarithmic PDF ratio values.
            """
            # Get the signal PDF for the given signal parameters.
            sigpdf = sigpdfset.get_pdf(gridfitparams)

            # Create the ratio array with the same shape than the background pdf
            # histogram.
            ratio = np.ones_like(bkgpdf.hist, dtype=np.float64)

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
                bounds_error=False,
                fill_value=0.)

            return log_ratio_spline

        # Get the list of fit parameter permutations on the grid for which we
        # need to create PDF ratio arrays.
        gridfitparams_list = self.signalpdfset.gridfitparams_list

        args_list = [ ((signalpdfset, backgroundpdf, self.fillmethod, gridfitparams),{})
                     for gridfitparams in gridfitparams_list ]

        log_ratio_spline_list = parallelize(
            create_log_ratio_spline, args_list, self.ncpu, ppbar=ppbar)

        # Save all the log_ratio splines in a dictionary.
        self._gridfitparams_hash_log_ratio_spline_dict = dict()
        for (gridfitparams, log_ratio_spline) in zip(gridfitparams_list, log_ratio_spline_list):
            gridfitparams_hash = make_params_hash(gridfitparams)
            self._gridfitparams_hash_log_ratio_spline_dict[gridfitparams_hash] = log_ratio_spline

        # Save the list of data field names.
        self._data_field_names = [ binning.name
                                  for binning in self.backgroundpdf.binnings ]

        # Construct the instance for the fit parameter interpolation method.
        self._interpolmethod_instance = self.interpolmethod(self._get_spline_value, signalpdfset.fitparams_grid_set)

        # Create cache variables for the last ratio value and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache_fitparams_hash = None
        self._cache_ratio = None
        self._cache_gradients = None

    @property
    def fillmethod(self):
        """The PDFRatioFillMethod object, which should be used for filling the
        PDF ratio bins.
        """
        return self._fillmethod
    @fillmethod.setter
    def fillmethod(self, obj):
        if(not isinstance(obj, PDFRatioFillMethod)):
            raise TypeError('The fillmethod property must be an instance of PDFRatioFillMethod!')
        self._fillmethod = obj

    def _get_spline_value(self, tdm, gridfitparams, eventdata):
        """Selects the spline object for the given fit parameter grid point and
        evaluates the spline for all the given events.
        """
        # Get the spline object for the given fit parameter grid values.
        gridfitparams_hash = make_params_hash(gridfitparams)
        spline = self._gridfitparams_hash_log_ratio_spline_dict[gridfitparams_hash]

        # Evaluate the spline.
        value = spline(eventdata)

        return value

    def _is_cached(self, tdm, fitparams_hash):
        """Checks if the ratio and gradients for the given set of fit parameters
        are already cached.
        """
        if((self._cache_fitparams_hash == fitparams_hash) and
           (len(self._cache_ratio) == tdm.n_selected_events)
          ):
            return True
        return False

    def _calculate_ratio_and_gradients(self, tdm, fitparams, fitparams_hash):
        """Calculates the ratio values and ratio gradients for all the events
        given the fit parameters. It caches the results.
        """
        get_data = tdm.get_data

        # Create a 2D event data array holding only the needed event data fields
        # for the PDF ratio spline evaluation.
        eventdata = np.vstack([get_data(fn) for fn in self._data_field_names]).T

        (ratio, gradients) = self._interpolmethod_instance.get_value_and_gradients(
            tdm, eventdata, fitparams)
        # The interpolation works on the logarithm of the ratio spline, hence
        # we need to transform it using the exp function, and we need to account
        # for the exp function in the gradients.
        ratio = np.exp(ratio)
        gradients = ratio * gradients

        # Cache the value and the gradients.
        self._cache_fitparams_hash = fitparams_hash
        self._cache_ratio = ratio
        self._cache_gradients = gradients

    def get_ratio(self, tdm, fitparams, tl=None):
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
        fitparams : dict
            The dictionary with the fit parameter values.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratio : 1d ndarray of float
            The PDF ratio value for each given event.
        """
        fitparams_hash = make_params_hash(fitparams)

        # Check if the ratio value is already cached.
        if(self._is_cached(tdm, fitparams_hash)):
            return self._cache_ratio

        self._calculate_ratio_and_gradients(tdm, fitparams, fitparams_hash)

        return self._cache_ratio

    def get_gradient(self, tdm, fitparams, fitparam_name):
        """Retrieves the PDF ratio gradient for the pidx'th fit parameter.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF ratio gradient values should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values.
        fitparam_name : str
            The name of the fit parameter for which the gradient should get
            calculated.
        """
        fitparams_hash = make_params_hash(fitparams)

        # Convert the fit parameter name into the local fit parameter index.
        pidx = self.convert_signal_fitparam_name_into_index(fitparam_name)

        # Check if the gradients have been calculated already.
        if(self._is_cached(tdm, fitparams_hash)):
            return self._cache_gradients[pidx]

        # The gradients have not been calculated yet.
        self._calculate_ratio_and_gradients(tdm, fitparams, fitparams_hash)

        return self._cache_gradients[pidx]
