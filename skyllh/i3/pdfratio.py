# -*- coding: utf-8 -*-

import scipy.interpolate

import numpy as np

from numpy.lib.recfunctions import (
    repack_fields,
)

from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize,
)
from skyllh.core.pdfratio import (
    SigSetOverBkgPDFRatio,
)
from skyllh.core.pdfratio_fill import (
    MostSignalLikePDFRatioFillMethod,
    PDFRatioFillMethod,
)
from skyllh.core.py import (
    make_dict_hash,
)


class SplinedI3EnergySigSetOverBkgPDFRatio(
        SigSetOverBkgPDFRatio,
        IsParallelizable):
    """This class implements a splined signal over background PDF ratio for
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
            ppbar=None,
            **kwargs,
    ):
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
            ncpu=ncpu,
            **kwargs)

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
            ratio = fillmethod(
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
            self._sig_pdf_set.param_grid_set.params_name_list

        # Create cache variable for the last ratio values and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache = self._create_cache(
            trial_data_state_id=None,
            interpol_params_recarray=None,
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

    def _create_cache(
            self,
            trial_data_state_id,
            interpol_params_recarray,
            ratio,
            grads):
        """Creates a cache dictionary holding cache data.

        Parameters
        ----------
        trial_data_state_id : int | None
            The trial data state ID of the TrialDataManager.
        interpol_params_recarray : instance of numpy record ndarray | None
            The numpy record ndarray of length N_sources holding the parameter
            names and values necessary for the interpolation for all sources.
        ratio : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the PDF ratio values
            for all sources and trial events.
        grads : instance of numpy ndarray
            The (D,N_values)-shaped numpy ndarray holding the gradients for each
            PDF ratio value w.r.t. each interpolation parameter.
        """
        cache = {
            'trial_data_state_id': trial_data_state_id,
            'interpol_params_recarray': interpol_params_recarray,
            'ratio': ratio,
            'grads': grads
        }

        return cache

    def _is_cached(self, trial_data_state_id, interpol_params_recarray):
        """Checks if the ratio and gradients for the given set of interpolation
        parameters are already cached.
        """
        if self._cache['trial_data_state_id'] is None:
            return False

        if self._cache['trial_data_state_id'] != trial_data_state_id:
            return False

        if not np.all(
                self._cache['interpol_params_recarray'] ==
                interpol_params_recarray):
            return False

        return True

    def _get_spline_for_param_values(self, interpol_param_values):
        """Retrieves the spline for a given set of parameter values.

        Parameters
        ----------
        interpol_param_values : instance of numpy ndarray
            The (N_interpol_params,)-shaped numpy ndarray holding the values of
            the interpolation parameters.

        Returns
        -------
        spline : instance of scipy.interpolate.RegularGridInterpolator
            The requested spline instance.
        """
        gridparams = dict(
            zip(self._interpol_param_names, interpol_param_values))
        gridparams_hash = make_dict_hash(gridparams)

        spline = self._gridparams_hash_log_ratio_spline_dict[gridparams_hash]

        return spline

    def _evaluate_splines(
            self,
            tdm,
            eventdata,
            gridparams_recarray,
            n_values):
        """For each set of parameter values given by ``gridparams_recarray``,
        the spline is retrieved and evaluated for the events suitable for that
        source model.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data and the event
            mapping to the sources via the ``src_evt_idx`` property.
        eventdata : instance of numpy ndarray
            The (V,N_events)-shaped numpy ndarray holding the event data, where
            N_events is the number of events, and V the dimensionality of the
            event data.
        gridparams_recarray : instance of numpy structured ndarray
            The numpy structured ndarray of length N_sources with the parameter
            names and values needed for the interpolation on the grid for all
            sources. If the length of this record array is 1, the set of
            parameters will be used for all sources.
        n_values : int
            The size of the output array.

        Returns
        -------
        values : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the values for each set
            of parameter values of the ``gridparams_recarray``. The length of
            the array depends on the ``src_evt_idx`` property of the
            TrialDataManager. In the worst case it is
            ``N_sources * N_selected_events``.
        """
        (src_idxs, evt_idxs) = tdm.src_evt_idxs

        # Check for special case when a single set of parameters are provided.
        if len(gridparams_recarray) == 1:
            # We got a single parameter set. We will use it for all sources.
            spline = self._get_spline_for_param_values(gridparams_recarray[0])

            eventdata = np.take(eventdata, evt_idxs, axis=1)
            values = spline(eventdata.T)

            return values

        values = np.empty(n_values, dtype=np.float64)

        v_start = 0
        for (sidx, param_values) in enumerate(gridparams_recarray):
            spline = self._get_spline_for_param_values(param_values)

            # Select the eventdata that belongs to the current source.
            m = src_idxs == sidx
            src_eventdata = np.take(eventdata, evt_idxs[m], axis=1)

            n = src_eventdata.shape[1]
            sl = slice(v_start, v_start+n)
            values[sl] = spline(src_eventdata.T)

            v_start += n

        return values

    def _create_interpol_params_recarray(self, src_params_recarray):
        """Creates the params_recarray needed for the interpolation. It selects
        The interpolation parameters from the ``params_recarray`` argument.
        If all parameters have the same value for all sources, the length will
        be 1.

        Parameters
        ----------
        src_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources holding all local
            parameter names and values.

        Returns
        -------
        interpol_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources or 1 holding only the
            parameters needed for the interpolation.
        """
        interpol_params_recarray = repack_fields(
            src_params_recarray[self._interpol_param_names])

        all_params_are_equal_for_all_sources = True
        for pname in self._interpol_param_names:
            if not np.all(
                    np.isclose(np.diff(interpol_params_recarray[pname]), 0)):
                all_params_are_equal_for_all_sources = False
                break
        if all_params_are_equal_for_all_sources:
            return interpol_params_recarray[:1]

        return interpol_params_recarray

    def _calculate_ratio_and_grads(
            self,
            tdm,
            interpol_params_recarray):
        """Calculates the ratio values and ratio gradients for all the sources
        and trial events given the source parameter values.
        The result is stored in the class member variable ``_cache``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data.
        interpol_params_recarray : instance of numpy record ndarray
            The numpy record ndarray of length N_sources holding the parameter
            names and values for all sources.
            It must contain only the parameters necessary for the interpolation.
        """
        # Create a 2D event data array holding only the needed event data fields
        # for the PDF ratio spline evaluation.
        eventdata = np.vstack([tdm[fn] for fn in self._data_field_names])

        (ratio, grads) = self._interpolmethod(
            tdm=tdm,
            eventdata=eventdata,
            params_recarray=interpol_params_recarray)

        # The interpolation works on the logarithm of the ratio spline, hence
        # we need to transform it using the exp function, and we need to account
        # for the exp function in the gradients.
        ratio = np.exp(ratio)
        grads = ratio * grads

        # Cache the value and the gradients.
        self._cache = self._create_cache(
            trial_data_state_id=tdm.trial_data_state_id,
            interpol_params_recarray=interpol_params_recarray,
            ratio=ratio,
            grads=grads
        )

    def get_ratio(
            self,
            tdm,
            src_params_recarray,
            tl=None):
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
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the parameter
            names and values of the sources. See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            for more information.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        ratio : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray of float holding the PDF ratio
            value for each source and trial event.
        """
        # Select only the parameters necessary for the interpolation.
        interpol_params_recarray = self._create_interpol_params_recarray(
            src_params_recarray)

        # Check if the ratio values are already cached.
        if self._is_cached(
               trial_data_state_id=tdm.trial_data_state_id,
               interpol_params_recarray=interpol_params_recarray):
            return self._cache['ratio']

        self._calculate_ratio_and_grads(
            tdm=tdm,
            interpol_params_recarray=interpol_params_recarray)

        return self._cache['ratio']

    def get_gradient(
            self,
            tdm,
            src_params_recarray,
            fitparam_id,
            tl=None):
        """Retrieves the PDF ratio gradient for the given fit parameter
        ``fitparam_id``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial event data for which
            the PDF ratio gradient values should get calculated.
        src_params_recarray : instance of numpy record ndarray | None
            The (N_sources,)-shaped numpy record ndarray holding the local
            parameter names and values of all sources. See the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information.
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
        # Select only the parameters necessary for the interpolation.
        interpol_params_recarray = self._create_interpol_params_recarray(
            src_params_recarray)

        # Calculate the gradients if necessary.
        if not self._is_cached(
            trial_data_state_id=tdm.trial_data_state_id,
            interpol_params_recarray=interpol_params_recarray
        ):
            self._calculate_ratio_and_grads(
                tdm=tdm,
                interpol_params_recarray=interpol_params_recarray)

        tdm_n_sources = tdm.n_sources

        grad = np.zeros((tdm.get_n_values(),), dtype=np.float64)

        # Loop through the parameters of the signal PDF set and match them with
        # the global fit parameter.
        for (pidx, pname) in enumerate(
                self._sig_pdf_set.param_grid_set.params_name_list):
            if pname not in src_params_recarray.dtype.fields:
                continue
            p_gpidxs = src_params_recarray[f'{pname}:gpidx']
            src_mask = p_gpidxs == (fitparam_id + 1)
            n_sources = np.count_nonzero(src_mask)
            if n_sources == 0:
                continue
            if n_sources == tdm_n_sources:
                # This parameter applies to all sources, hence to all values,
                # and hence it's the only local parameter contributing to the
                # global parameter fitparam_id.
                return self._cache['grads'][pidx]

            # The current parameter does not apply to all sources.
            # Create a values mask that matches a given source mask.
            values_mask = tdm.get_values_mask_for_source_mask(src_mask)
            grad[values_mask] = self._cache['grads'][pidx][values_mask]

        return grad
