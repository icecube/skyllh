# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate

from skyllh.core.py import (
    make_dict_hash,
)
from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize,
)
from skyllh.core.pdfratio import (
    MostSignalLikePDFRatioFillMethod,
    PDFRatioFillMethod,
    SigSetOverBkgPDFRatio,
)


class I3EnergySigSetOverBkgPDFRatioSpline(
        SigSetOverBkgPDFRatio,
        IsParallelizable):
    """This class implements a signal over background PDF ratio spline for
    enegry PDFs of type I3EnergyPDF.
    It takes an instance, which is derived from PDFSet, and which is derived
    from IsSignalPDF, as signal PDF. Furthermore, it takes an instance, which
    is derived from I3EnergyPDF and IsBackgroundPDF, as background PDF, and
    creates a spline for the ratio of the signal and background PDFs for a grid
    of different discrete energy signal parameters, which are defined by the
    signal PDF set.
    """
    def __init__(
            self,
            sig_pdf_set,
            bkg_pdf,
            fillmethod=None,
            interpolmethod_cls=None,
            ncpu=None,
            ppbar=None):
        """Creates a new IceCube signal-over-background energy PDF ratio spline
        instance.

        Parameters
        ----------
        sig_pdf_set : class instance derived from PDFSet (for PDF type
                       I3EnergyPDF), IsSignalPDF, and UsesBinning
            The PDF set, which provides signal energy PDFs for a set of
            discrete signal parameters.
        bkg_pdf : class instance derived from I3EnergyPDF, and
                        IsBackgroundPDF
            The background energy PDF object.
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
            ncpu=ncpu)

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
                gridparams):
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
            ratio = fillmethod.fill_ratios(
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
            self._sig_pdf_set.param_grid_set.param_names

        # Create cache variable for the last ratio values and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache = self._create_cache(
            trial_data_state_id=None,
            params_recarray=None,
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

    def _create_cache(self, trial_data_state_id, params_recarray, ratio, grads):
        """Creates a cache dictionary holding cache data.
        """
        params = dict()
        if params_recarray is not None:
            for pname in self.param_names:
                params[pname] = np.copy(params_recarray[pname])

        cache = {
            'trial_data_state_id': trial_data_state_id,
            'params': params,
            'ratio': ratio,
            'grads': grads
        }

        return cache

    def _is_cached(self, tdm, params_recarray):
        """Checks if the ratio and gradients for the given set of parameters
        are already cached.
        """
        if self._cache['trial_data_state_id'] != tdm.trial_data_state_id:
            return False

        for (pname, pvals) in self._cache['params'].items():
            if not np.all(np.isclose(pvals, params_recarray[pname])):
                return False

        return True

    def _evaluate_splines(
            self,
            tdm,
            eventdata,
            gridparams_recarray,
            src_idxs,
            n_values,
            ret_gridparams_recarray):
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
        gridparams_recarray : instance of numpy record ndarray
            The numpy record ndarray with the parameter names and values on the
            grid for all sources.
        src_idxs : instance of numpy ndarray
            The (N_sources,)-shaped ndarray holding the indices of the sources
            for which the splines should get evaluated.
        n_values : int
            The size of the output array.
        ret_gridparams_recarray : bool
            Switch if the gridparams_recarray should be returned, where the
            source parameter values are broadcasted to each value.

        Returns
        -------
        values : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the values for each set
            of parameter values of the ``gridparams_recarray``. The length of
            the array depends on the ``src_evt_idx`` property of the
            TrialDataManager. In the worst case it is N_sources * N_events.
        gridparams_recarray_output : instance of numpy record array
            If the ``ret_gridparams_recarray`` argument is set to ``True``,
            this is the numpy record array of length N with the input grid
            parameters from ``gridparams_recarray`` broadcasted to each value.
        """
        if len(gridparams_recarray) != len(src_idxs):
            raise ValueError(
                'The length of the gridparams_recarray argument '
                f'({len(len(gridparams_recarray))}) must be equal to the '
                f'length of the src_idxs argument ({len(src_idxs)})!')

        if tdm.src_evt_idx is not None:
            (_src_idxs, _evt_idxs) = tdm.src_evt_idxs

        values = np.empty(n_values, dtype=np.float64)

        if ret_gridparams_recarray:
            gridparams_recarray_output = np.empty(
                (n_values,),
                dtype=gridparams_recarray.dtype)

        v_start = 0
        for (sidx, p_values) in zip(src_idxs, gridparams_recarray):
            gridparams = dict(zip(self._interpol_param_names, p_values))
            gridparams_hash = make_dict_hash(gridparams)
            spline = self._gridparams_hash_log_ratio_spline_dict[
                gridparams_hash]

            # Select the eventdata that belongs to the current source.
            src_eventdata = eventdata
            if tdm.src_evt_idx is not None:
                m = _src_idxs == sidx
                src_eventdata = np.take(eventdata, _evt_idxs[m], axis=0)

            n = src_eventdata.shape[0]
            sl = slice(v_start, v_start+n)
            values[sl] = spline(src_eventdata)

            if ret_gridparams_recarray:
                src_gridparams_recarray = np.array(
                    [tuple(p_values)],
                    dtype=gridparams_recarray_output.dtype)
                gridparams_recarray_output[sl] = np.tile(
                    src_gridparams_recarray, n)

            v_start += n

        if ret_gridparams_recarray:
            return (values, gridparams_recarray_output)

        return values

    def _calculate_ratio_and_gradients(self, tdm, params_recarray):
        """Calculates the ratio values and ratio gradients for all the trial
        events and sources given the source parameter values.
        It caches the results.
        """
        # Create a 2D event data array holding only the needed event data fields
        # for the PDF ratio spline evaluation.
        eventdata = np.vstack([tdm[fn] for fn in self._data_field_names]).T

        (ratio, grads) = self._interpolmethod(
            tdm=tdm,
            eventdata=eventdata,
            params_recarray=params_recarray)

        # The interpolation works on the logarithm of the ratio spline, hence
        # we need to transform it using the exp function, and we need to account
        # for the exp function in the gradients.
        ratio = np.exp(ratio)
        grads = ratio * grads

        # Cache the value and the gradients.
        self._cache = self._create_cache(
            trial_data_state_id=tdm.trial_data_state_id,
            params_recarray=params_recarray,
            ratio=ratio,
            grads=grads
        )

    def get_ratio(self, tdm, params_recarray=None, tl=None):
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
        params_recarray : instance of numpy record ndarray | None
            The (N_models,)-shaped numpy record ndarray holding the parameter
            names and values of the models.
            See :meth:`skyllh.core.pdf.PDF.get_pd` for more information.
            This can be ``None``, if the signal and background PDFs do not
            depend on any parameters.
        tl : TimeLord instance | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratio : 1d ndarray of float
            The PDF ratio value for each given event.
        """
        # Check if the ratio values are already cached.
        if self._is_cached(tdm=tdm, params_recarray=params_recarray):
            return self._cache_ratio

        self._calculate_ratio_and_gradients(
            tdm=tdm,
            params_recarray=params_recarray)

        return self._cache_ratio

    def get_gradient(self, tdm, params_recarray, fitparam_id, tl=None):
        """Retrieves the PDF ratio gradient for the given fit parameter.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF ratio gradient values should get calculated.
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
            The (N_values,)-shaped numpy ndarray holding the gradient values
            for all sources and trial events w.r.t. the given global fit
            parameter.
        """
        # Calculate the gradients if necessary.
        if not self._is_cached(tdm=tdm, params_recarray=params_recarray):
            self._calculate_ratio_and_gradients(
                tdm=tdm, params_recarray=params_recarray)

        tdm_n_sources = tdm.n_sources

        grad = np.zeros((tdm.get_n_values(),), dtype=np.float64)

        # Loop through the parameters of the signal PDF set and match them with
        # the global fit parameter.
        for (pidx, pname) in enumerate(self._sig_pdf_set.param_names):
            if pname not in params_recarray.dtype.fields:
                continue
            p_gpidxs = params_recarray[f'{pname}:gpidx']
            src_mask = p_gpidxs == (fitparam_id + 1)
            n_sources = np.count_nonzero(src_mask)
            if n_sources == 0:
                continue
            if n_sources == tdm_n_sources:
                # This parameter applies to all sources, hence to all values,
                # and hence it's the only local parameter contributing to the
                # global parameter fitparam_id.
                return self._cache_grads[pidx]

            # The current parameter does not apply to all sources.
            # Create a values mask that matches a given source mask.
            values_mask = tdm.get_values_mask_for_source_mask(src_mask)
            grad[values_mask] = self._cache_grads[pidx][values_mask]

        return grad
