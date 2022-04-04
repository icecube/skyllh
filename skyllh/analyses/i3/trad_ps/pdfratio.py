# -*- coding: utf-8 -*-

from scipy.interpolate import RegularGridInterpolator

from skyllh.core.pdf import EnergyPDF
from skyllh.core.pdfratio import (
    SigSetOverBkgPDFRatio,
    PDFRatioFillMethod,
    MostSignalLikePDFRatioFillMethod
)
from skyllh.core.multiproc import IsParallelizable

from skyllh.analyses.i3.trad_ps.signalpdf import PublicDataSignalI3EnergyPDFSet

class PublicDataI3EnergySigSetOverBkgPDFRatioSpline(
        SigSetOverBkgPDFRatio,
        IsParallelizable):
    """This class implements a signal over background PDF ratio spline for a
    signal PDF that is derived from PublicDataSignalI3EnergyPDFSet and a
    background PDF that is derived from I3EnergyPDF. It creates a spline for the
    ratio of the signal and background PDFs for a grid of different discrete
    energy signal fit parameters, which are defined by the signal PDF set.
    """
    def __init__(
            self, signalpdfset, backgroundpdf,
            fillmethod=None, interpolmethod=None, ncpu=None, ppbar=None):
        """Creates a new IceCube signal-over-background energy PDF ratio object
        specialized for the public data.

        Paramerers
        ----------
        signalpdfset : class instance derived from PDFSet (for PDF type
                       EnergyPDF), IsSignalPDF, and UsesBinning
            The PDF set, which provides signal energy PDFs for a set of
            discrete signal parameters.
        backgroundpdf : class instance derived from EnergyPDF, and
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
        """
        super().__init__(
            pdf_type=EnergyPDF,
            signalpdfset=signalpdfset, backgroundpdf=backgroundpdf,
            interpolmethod=interpolmethod,
            ncpu=ncpu)

        # Define the default ratio fill method.
        if(fillmethod is None):
            fillmethod = MostSignalLikePDFRatioFillMethod()
        self.fillmethod = fillmethod

        def create_log_ratio_spline(
                sigpdfset, bkgpdf, fillmethod, gridfitparams, src_dec_idx):
            """Creates the signal/background ratio 2d spline for the given
            signal parameters.

            Returns
            -------
            log_ratio_spline : RegularGridInterpolator
                The spline of the logarithmic PDF ratio values in the
                (log10(E_reco),sin(dec_reco)) space.
            """
            # Get the signal PDF for the given signal parameters.
            sigpdf = sigpdfset.get_pdf(gridfitparams)

            bkg_log_e_bincenters = bkgpdf.get_binning('log_energy').bincenters
            sigpdf_hist = sigpdf.calc_prob_for_true_dec_idx(
                src_dec_idx, bkg_log_e_bincenters)
            # Transform the (log10(E_reco),)-shaped 1d array into the
            # (log10(E_reco),sin(dec_reco))-shaped 2d array.
            sigpdf_hist = np.repeat(
                [sigpdf_hist], bkgpdf.hist.shape[1], axis=0).T

            sig_mask_mc_covered = np.ones_like(sigpdf_hist, dtype=np.bool)
            bkg_mask_mc_covered = np.ones_like(bkgpdf.hist, dtype=np.bool)
            sig_mask_mc_covered_zero_physics = sigpdf_hist == 0
            bkg_mask_mc_covered_zero_physics = bkgpdf.hist == 0

            # Create the ratio array with the same shape than the background pdf
            # histogram.
            ratio = np.ones_like(bkgpdf.hist, dtype=np.float)

            # Fill the ratio array.
            ratio = fillmethod.fill_ratios(ratio,
                sigpdf_hist, bkgpdf.hist,
                sig_mask_mc_covered,
                sig_mask_mc_covered_zero_physics,
                bkg_mask_mc_covered,
                bkg_mask_mc_covered_zero_physics)

            # Define the grid points for the spline. In general, we use the bin
            # centers of the binning, but for the first and last point of each
            # dimension we use the lower and upper bin edge, respectively, to
            # ensure full coverage of the spline across the binning range.
            points_list = []
            for binning in bkgpdf.binnings:
                points = binning.bincenters
                (points[0], points[-1]) = (binning.lower_edge, binning.upper_edge)
                points_list.append(points)

            # Create the spline for the ratio values.
            log_ratio_spline = RegularGridInterpolator(
                tuple(points_list),
                np.log(ratio),
                method='linear',
                bounds_error=False,
                fill_value=0.)

            return log_ratio_spline

        # Get the list of fit parameter permutations on the grid for which we
        # need to create PDF ratio arrays.
        gridfitparams_list = signalpdfset.gridfitparams_list

        self._gridfitparams_hash_log_ratio_spline_dict_list = []
        for src_dec_idx in range(sigpdfset.true_dec_binning.nbins):
            args_list = [
                (
                    (
                        signalpdfset,
                        backgroundpdf,
                        fillmethod,
                        gridfitparams,
                        src_dec_idx
                    ),
                    {}
                )
                for gridfitparams in gridfitparams_list
            ]

            log_ratio_spline_list = parallelize(
                create_log_ratio_spline, args_list, self.ncpu, ppbar=ppbar)

            # Save all the log_ratio splines in a dictionary.
            gridfitparams_hash_log_ratio_spline_dict = dict()
            for (gridfitparams, log_ratio_spline) in zip(
                gridfitparams_list, log_ratio_spline_list):
                gridfitparams_hash = make_params_hash(gridfitparams)
                gridfitparams_hash_log_ratio_spline_dict[
                    gridfitparams_hash] = log_ratio_spline
            self._gridfitparams_hash_log_ratio_spline_dict_list.append(
                gridfitparams_hash_log_ratio_spline_dict)

        # Save the list of data field names.
        self._data_field_names = [
            binning.name
                for binning in self.backgroundpdf.binnings
        ]

        # Construct the instance for the fit parameter interpolation method.
        self._interpolmethod_instance = self.interpolmethod(
            self._get_spline_value,
            signalpdfset.fitparams_grid_set)

        # Create cache variables for the last ratio value and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache_src_dec_idx = None
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
            raise TypeError('The fillmethod property must be an instance of '
                'PDFRatioFillMethod!')
        self._fillmethod = obj

    def _get_spline_value(self, tdm, gridfitparams, eventdata):
        """Selects the spline object for the given fit parameter grid point and
        evaluates the spline for all the given events.
        """
        if(self._cache_src_dec_idx is None):
            raise RuntimeError('There was no source declination bin index '
                'pre-calculated!')

        # Get the spline object for the given fit parameter grid values.
        gridfitparams_hash = make_params_hash(gridfitparams)
        spline = self._gridfitparams_hash_log_ratio_spline_dict_list\
            [self._cache_src_dec_idx][gridfitparams_hash]

        # Evaluate the spline.
        value = spline(eventdata)

        return value

    def _get_src_dec_idx_from_source_array(self, src_array):
        """Determines the source declination index given the source array from
        the trial data manager. For now only a single source is supported!
        """
        if(len(src_array) != 1):
            raise NotImplementedError(
                'The PDFRatio class "{}" is only implemneted for a single '
                'source! But {} sources were defined!'.format(
                    self.__class__.name, len(src_array)))
        src_dec = get_data('src_array')['dec'][0]
        true_dec_binning = self.signalpdfset.true_dec_binning
        src_dec_idx = np.digitize(src_dec, true_dec_binning.binedges)

        return src_dec_idx

    def _is_cached(self, tdm, src_dec_idx, fitparams_hash):
        """Checks if the ratio and gradients for the given set of fit parameters
        are already cached.
        """
        if((self._cache_src_dec_idx == src_dec_idx) and
           (self._cache_fitparams_hash == fitparams_hash) and
           (len(self._cache_ratio) == tdm.n_selected_events)
          ):
            return True
        return False

    def _calculate_ratio_and_gradients(
            self, tdm, src_dec_idx, fitparams, fitparams_hash):
        """Calculates the ratio values and ratio gradients for all the events
        given the fit parameters. It caches the results.
        """
        get_data = tdm.get_data

        # The _get_spline_value method needs the cache source dec index for the
        # current evaluation of the PDF ratio.
        self._cache_src_dec_idx = src_dec_idx

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

        # Determine the source declination bin index.
        src_array = get_data('src_array')
        src_dec_idx = self._get_src_dec_idx_from_source_array(src_array)

        # Check if the ratio value is already cached.
        if(self._is_cached(tdm, src_dec_idx, fitparams_hash)):
            return self._cache_ratio

        self._calculate_ratio_and_gradients(
            tdm, src_dec_idx, fitparams, fitparams_hash)

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

        # Determine the source declination bin index.
        src_array = get_data('src_array')
        src_dec_idx = self._get_src_dec_idx_from_source_array(src_array)

        # Check if the gradients have been calculated already.
        if(self._is_cached(tdm, src_dec_idx, fitparams_hash)):
            return self._cache_gradients[pidx]

        # The gradients have not been calculated yet.
        self._calculate_ratio_and_gradients(
            tdm, src_dec_idx, fitparams, fitparams_hash)

        return self._cache_gradients[pidx]
