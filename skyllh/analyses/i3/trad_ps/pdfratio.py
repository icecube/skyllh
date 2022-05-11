# -*- coding: utf-8 -*-

import sys

import numpy as np

from skyllh.core.parameters import make_params_hash
from skyllh.core.pdf import PDF
from skyllh.core.pdfratio import SigSetOverBkgPDFRatio


class PDPDFRatio(SigSetOverBkgPDFRatio):
    def __init__(self, sig_pdf_set, bkg_pdf, **kwargs):
        """Creates a PDFRatio instance for the public data.
        It takes a signal PDF set for different discrete gamma values.

        Parameters
        ----------
        sig_pdf_set :
        """
        super().__init__(
            pdf_type=PDF,
            signalpdfset=sig_pdf_set,
            backgroundpdf=bkg_pdf,
            **kwargs)

        # Construct the instance for the fit parameter interpolation method.
        self._interpolmethod_instance = self.interpolmethod(
            self._get_ratio_values, sig_pdf_set.fitparams_grid_set)

        """
        # Get the requires field names from the background and signal pdf.
        self._data_field_name_list = []
        for axis in sig_pdf_set.axes:
            field_name_list.append(axis.name)
        for axis in bkg_pdf.axes:
            if axis.name not in field_name_list:
                field_name_list.append(axis.name)
        """

        # Create cache variables for the last ratio value and gradients in order
        # to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache_fitparams_hash = None
        self._cache_ratio = None
        self._cache_gradients = None

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
        fitparam_names = self.signalpdfset.fitparams_grid_set.parameter_names
        return fitparam_names

    def _is_cached(self, tdm, fitparams_hash):
        """Checks if the ratio and gradients for the given set of fit parameters
        are already cached.
        """
        if((self._cache_fitparams_hash == fitparams_hash) and
           (len(self._cache_ratio) == tdm.n_selected_events)
          ):
            return True
        return False

    def _get_ratio_values(self, tdm, gridfitparams, eventdata):
        """Select the signal PDF for the given fit parameter grid point and
        evaluates the S/B ratio for all the given events.
        """
        sig_prob = self.signalpdfset.get_prob(tdm, gridfitparams)
        if isinstance(sig_prob, tuple):
            (sig_prob, _) = sig_prob

        bkg_prob = self.backgroundpdf.get_prob(tdm)
        if isinstance(bkg_prob, tuple):
            (bkg_prob, _) = bkg_prob

        if np.any(np.invert(bkg_prob > 0)):
            raise ValueError(
                'For at least one event no background probability can be '
                'calculated! Check your background PDF!')

        ratio = sig_prob / bkg_prob

        return ratio

    def _calculate_ratio_and_gradients(self, tdm, fitparams, fitparams_hash):
        """Calculates the ratio values and ratio gradients for all the events
        given the fit parameters using the interpolation method for the fit
        parameter. It caches the results.
        """
        (ratio, gradients) =\
            self._interpolmethod_instance.get_value_and_gradients(
                tdm, eventdata=None, params=fitparams)

        # Cache the value and the gradients.
        self._cache_fitparams_hash = fitparams_hash
        self._cache_ratio = ratio
        self._cache_gradients = gradients

    def get_ratio(self, tdm, fitparams=None, tl=None):
        """Calculates the PDF ratio values for all the events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data events for
            which the PDF ratio values should get calculated.
        fitparams : dict | None
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
        fitparams_hash = make_params_hash(fitparams)

        # Check if the ratio value is already cached.
        #if(self._is_cached(tdm, fitparams_hash)):
        #    return self._cache_ratio

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
        #if(self._is_cached(tdm, fitparams_hash)):
        #    return self._cache_gradients[pidx]

        # The gradients have not been calculated yet.
        self._calculate_ratio_and_gradients(tdm, fitparams, fitparams_hash)

        return self._cache_gradients[pidx]
