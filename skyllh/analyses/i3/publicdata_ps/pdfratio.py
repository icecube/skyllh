# -*- coding: utf-8 -*-
# Authors:
#   Dr. Martin Wolf <mail@martin-wolf.org>

import numpy as np

from skyllh.core.py import module_classname
from skyllh.core.debugging import get_logger
from skyllh.core.parameters import make_params_hash
from skyllh.core.pdf import PDF
from skyllh.core.pdfratio import SigSetOverBkgPDFRatio


class PDPDFRatio(SigSetOverBkgPDFRatio):
    def __init__(self, sig_pdf_set, bkg_pdf, cap_ratio=False, **kwargs):
        """Creates a PDFRatio instance for the public data.
        It takes a signal PDF set for different discrete gamma values.

        Parameters
        ----------
        sig_pdf_set : instance of PDSignalEnergyPDFSet
            The PDSignalEnergyPDFSet instance holding the set of signal energy
            PDFs.
        bkg_pdf : instance of PDDataBackgroundI3EnergyPDF
            The PDDataBackgroundI3EnergyPDF instance holding the background
            energy PDF.
        cap_ratio : bool
            Switch whether the S/B PDF ratio should get capped where no
            background is available. Default is False.
        """
        self._logger = get_logger(module_classname(self))

        super().__init__(
            pdf_type=PDF,
            signalpdfset=sig_pdf_set,
            backgroundpdf=bkg_pdf,
            **kwargs)

        # Construct the instance for the fit parameter interpolation method.
        self._interpolmethod_instance = self.interpolmethod(
            self._get_ratio_values, sig_pdf_set.fitparams_grid_set)

        self.cap_ratio = cap_ratio
        if self.cap_ratio:
            self._logger.info('The energy PDF ratio will be capped!')

            # Calculate the ratio value for the phase space where no background
            # is available. We will take the p_sig percentile of the signal
            # like phase space.
            ratio_perc = 99

            # Get the log10 reco energy values where the background pdf has
            # non-zero values.
            n_logE = bkg_pdf.get_binning('log_energy').nbins
            n_sinDec = bkg_pdf.get_binning('sin_dec').nbins
            bd = bkg_pdf._hist_logE_sinDec > 0
            log10_e_bc = bkg_pdf.get_binning('log_energy').bincenters
            self.ratio_fill_value_dict = dict()
            for sig_pdf_key in sig_pdf_set.pdf_keys:
                sigpdf = sig_pdf_set[sig_pdf_key]
                sigvals = sigpdf.get_pd_by_log10_reco_e(log10_e_bc)
                sigvals = np.broadcast_to(sigvals, (n_sinDec, n_logE)).T
                r = sigvals[bd] / bkg_pdf._hist_logE_sinDec[bd]
                # Remove possible inf values.
                r = r[np.invert(np.isinf(r))]
                val = np.percentile(r[r > 1.], ratio_perc)
                self.ratio_fill_value_dict[sig_pdf_key] = val
                self._logger.info(
                    f'The cap value for the energy PDF ratio key {sig_pdf_key} '
                    f'is {val}.')

        # Create cache variables for the last ratio value and gradients in
        # order to avoid the recalculation of the ratio value when the
        # ``get_gradient`` method is called (usually after the ``get_ratio``
        # method was called).
        self._cache_fitparams_hash = None
        self._cache_ratio = None
        self._cache_gradients = None

    @property
    def cap_ratio(self):
        """Boolean switch whether to cap the ratio where no background
        information is available (True) or use the smallest possible floating
        point number greater than zero as background pdf value (False).
        """
        return self._cap_ratio
    @cap_ratio.setter
    def cap_ratio(self, b):
        self._cap_ratio = b

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
        sig_pdf_key = self.signalpdfset.make_pdf_key(gridfitparams)

        sig_prob = self.signalpdfset.get_pdf(sig_pdf_key).get_prob(tdm)
        if isinstance(sig_prob, tuple):
            (sig_prob, _) = sig_prob

        bkg_prob = self.backgroundpdf.get_prob(tdm)
        if isinstance(bkg_prob, tuple):
            (bkg_prob, _) = bkg_prob

        if len(sig_prob) != len(bkg_prob):
            raise ValueError(
                f'The number of signal ({len(sig_prob)}) and background '
                f'({len(bkg_prob)}) probability values is not equal!')

        m_nonzero_bkg = bkg_prob > 0
        m_zero_bkg = np.invert(m_nonzero_bkg)
        if np.any(m_zero_bkg):
            ev_idxs = np.where(m_zero_bkg)[0]
            self._logger.debug(
                f'For {len(ev_idxs)} events the background probability is '
                f'zero. The event indices of these events are: {ev_idxs}')

        ratio = np.empty((len(sig_prob),), dtype=np.double)
        ratio[m_nonzero_bkg] = sig_prob[m_nonzero_bkg] / bkg_prob[m_nonzero_bkg]

        if self._cap_ratio:
            ratio[m_zero_bkg] = self.ratio_fill_value_dict[sig_pdf_key]
        else:
            ratio[m_zero_bkg] = (sig_prob[m_zero_bkg] /
                                 np.finfo(np.double).resolution)

        # Check for positive inf values in the ratio and set the ratio to a
        # finite number. Here we choose the maximum value of float32 to keep
        # room for additional computational operations.
        m_inf = np.isposinf(ratio)
        ratio[m_inf] = np.finfo(np.float32).max

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
        if self._is_cached(tdm, fitparams_hash):
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
        if self._is_cached(tdm, fitparams_hash):
            return self._cache_gradients[pidx]

        # The gradients have not been calculated yet.
        self._calculate_ratio_and_gradients(tdm, fitparams, fitparams_hash)

        return self._cache_gradients[pidx]
