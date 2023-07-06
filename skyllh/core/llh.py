# -*- coding: utf-8 -*-

"""The llh module provides classes for implementing a log-likelihood function.
In general these should be detector independent, because they implement the pure
math of the log-likelihood function.
"""

import abc

import numpy as np

from skyllh.core.config import (
    HasConfig,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.minimizer import (
    Minimizer,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)


logger = get_logger(__name__)


class LLH(
    HasConfig,
    metaclass=abc.ABCMeta,
):
    """Abstract base class for a log-likelihood (LLH) function.
    """

    def __init__(
            self,
            pmm,
            shg_mgr,
            minimizer=None,
            **kwargs
    ):
        """Creates a new LLH function instance.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global parameters to local parameters of individual models.
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups.
        minimizer : instance of Minimizer | None
            The optional instance of Minimizer that should be used to minimize
            the negative of this log-likelihood function.
        """
        super().__init__(
            **kwargs)

        self.pmm = pmm
        self.shg_mgr = shg_mgr
        self.minimizer = minimizer

    @property
    def pmm(self):
        """The instance of ParameterModelMapper providing the mapping of the
        global parameters to local parameters of the individual models.
        """
        return self._pmm

    @pmm.setter
    def pmm(self, mapper):
        if not isinstance(mapper, ParameterModelMapper):
            raise TypeError(
                'The pmm property must be an instance of ParameterModelMapper! '
                f'Its current type is {classname(mapper)}.')
        self._pmm = mapper

    @property
    def shg_mgr(self):
        """The instance of SourceHypoGroupManager that defines the source
        hypothesis groups.
        """
        return self._shg_mgr

    @shg_mgr.setter
    def shg_mgr(self, mgr):
        if not isinstance(mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr property must be an instance of '
                'SourceHypoGroupManager! '
                f'Its current type is {classname(mgr)}.')
        self._shg_mgr = mgr

    @property
    def minimizer(self):
        """The instance of Minimizer used to minimize the negative of the
        log-likelihood function. This can be ``None``, if no minimizer was
        specified.
        """
        return self._minimizer

    @minimizer.setter
    def minimizer(self, minimizer):
        if minimizer is not None:
            if not isinstance(minimizer, Minimizer):
                raise TypeError(
                    'The minimizer property must be an instance of Minimizer! '
                    f'Its current type is {classname(minimizer)}.')
        self._minimizer = minimizer

    def change_shg_mgr(
            self,
            shg_mgr,
    ):
        """Changes the source hypothesis group manager of this LL function.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The new instance of SourceHypoGroupManager.
        """
        self.shg_mgr = shg_mgr

    @abc.abstractmethod
    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """This method will be called after new trial data has been initialized
        to the trial data manager. Derived classes can make use of this call
        hook to perform LLH function specific trial initialization.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the new trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for timing measurements.
        """
        pass

    @abc.abstractmethod
    def evaluate(
            self,
            tdm,
            fitparam_values,
            src_params_recarray=None,
            tl=None,
    ):
        """This method evaluates the LLH function for the given set of
        fit parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        fitparam_values : instance of numpy.ndarray
            The (N_fitparams,)-shaped numpy.ndarray holding the current
            values of the global fit parameters.
        src_params_recarray : instance of numpy.ndarray | None
            The structured instance of numpy.ndarray of length N_sources holding
            the parameter names and values of all sources. If set to ``None`` it
            will be created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing.

        Returns
        -------
        log_lh : float
            The value of the LLH function.
        grads : instance of numpy.ndarray
            The (N_fitparams,)-shaped numpy.ndarray holding the gradient
            value for each global fit parameter.
        """
        pass


class EvtProbLLH(
    LLH,
):
    r"""This class provides the following log-likelihood function:

    .. math::

        \log(L(\vec{p},\gamma|\vec{x})) = \sum_{i=1}^{N}\log
        \left[ p_{i} \mathcal{S}(x_{i}|\gamma) + (1-p_{i}) \mathcal{B}(x_{i}) \right],

    where the vector :math:`\vec{p}` of length :math:`N` contains a probability
    for each event, :math:`\gamma` is an additional parameter, and the vector
    :math:`\vec{x}` of length :math:`N` contains the data values of each event.
    The probability density functions :math:`\mathcal{S}` and
    :math:`\mathcal{B}` are the signal and background PDFs, respectively.

    The gradient :math:`\partial \log(L)/\partial p_i` is given as

    .. math::

        \frac{\partial \log(L)}{\partial p_i} = \frac
        {\mathcal{S}(x_{i}|\gamma) - \mathcal{B}(x_{i})}
        {p_i\mathcal{S}(x_{i}|\gamma) + (1-p_i)\mathcal{B}(x_{i})}

    The gradient :math:`\partial \log(L)/\partial \gamma` is given as

    .. math::

        \frac{\partial \log(L)}{\partial \gamma} = \sum_{i=1}^{N} \frac
        {p_i \frac{\partial \mathcal{S}(x_{i}|\gamma)}{\partial \gamma}}
        {p_i\mathcal{S}(x_{i}|\gamma) + (1-p_i)\mathcal{B}(x_{i})}

    .. warning::

        This LH function works only for single sources and with no applied event
        selection!

    """

    def __init__(
            self,
            pmm,
            shg_mgr,
            sig_pdf,
            bkg_pdf,
            evtp_name_fmt='p{i:d}',
            minimizer=None,
            **kwargs,
    ):
        """Creates a new instance of EvtProbLLH.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper providing the mapping of
            global parameters to local parameters of individual models.
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups.
        sig_pdf : instance of PDF and IsSignalPDF
            The signal PDF, which must be an instance of PDF and
            IsSignalPDF.
        bkg_pdf : instance of PDF and IsBackgroundPDF
            The background PDF, which must be an instance of PDF and
            IsBackgroundPDF.
        evtp_name_fmt : str
            The format of the name of the event probability parameters. The
            varibale ``i`` will be the parameter's event index.
        minimizer : instance of Minimizer | None
            The optional instance of Minimizer that should be used to minimize
            the negative of this log-likelihood function.
        """
        super().__init__(
            pmm=pmm,
            shg_mgr=shg_mgr,
            minimizer=minimizer,
            **kwargs)

        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf
        self.evtp_name_fmt = evtp_name_fmt

    @property
    def evtp_name_fmt(self):
        return self._evtp_name_fmt

    @evtp_name_fmt.setter
    def evtp_name_fmt(self, fmt):
        if not isinstance(fmt, str):
            raise TypeError(
                'The evtp_name_fmt property must be an instance of str!')
        self._evtp_name_fmt = fmt

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs,
    ):
        """Initializes this LH function for a new trial by calling the
        :meth:`~skyllh.core.pdf.PDF.initialize_for_new_trial` methods of the
        signal and background PDFs.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the new trial data.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure timing information.
        **kwargs
            Additional keyword arguments are passed to the
            :meth:`~skyllh.core.pdf.PDF.initialize_for_new_trial` methods of the
            signal and background PDFs.
        """
        self.sig_pdf.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)

        self.bkg_pdf.initialize_for_new_trial(
            tdm=tdm,
            tl=tl,
            **kwargs)

    def evaluate(
            self,
            tdm,
            fitparam_values,
            src_params_recarray=None,
            tl=None,
    ):
        """
        Evaluates this LLH function for the given trial data and set of fit
        parameter values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data.
        fitparam_values : instance of numpy.ndarray
            The (N_fitparams,)-shaped numpy.ndarray holding the current
            values of the global fit parameters.
        src_params_recarray : instance of numpy.ndarray | None
            The structured instance of numpy.ndarray of length N_sources holding
            the parameter names and values of all sources. If set to ``None`` it
            will be created automatically from the ``fitparam_values`` array.
            See the documentation of the
            :meth:`skyllh.core.parameters.ParameterModelMapper.create_src_params_recarray`
            method for more information about this array.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to use for measuring timing.

        Returns
        -------
        log_lh : float
            The value of the LLH function.
        grads : instance of numpy.ndarray
            The (N_fitparams,)-shaped numpy.ndarray holding the gradient
            value for each global fit parameter.
        """
        if tdm.n_sources != 1:
            raise ValueError(
                f'The LH function {classname(self)} is defined only for a '
                f'single source! Currently there are {tdm.n_sources} sources!')

        if src_params_recarray is None:
            src_params_recarray = self._pmm.create_src_params_recarray(
                gflp_values=fitparam_values)

        (sig_pd, sig_grads) = self.sig_pdf.get_pd(
            tdm=tdm,
            params_recarray=src_params_recarray,
            tl=tl)

        (bkg_pd, bkg_grads) = self.bkg_pdf.get_pd(
            tdm=tdm,
            params_recarray=None,
            tl=tl)

        N = tdm.n_events

        p_idx0 = self._pmm.get_gflp_idx(self.evtp_name_fmt.format(i=0))
        p = fitparam_values[p_idx0:p_idx0+N]

        log_lh = np.sum(np.log(p*sig_pd + (1 - p)*bkg_pd))

        grads = np.zeros_like(fitparam_values)

        denum = (p*sig_pd + (1-p)*bkg_pd)

        grads[p_idx0:p_idx0+N] = (sig_pd - bkg_pd) / denum

        gflp_idxs_S = np.array(sig_grads.keys())
        gflp_idxs_B = np.array(bkg_grads.keys())

        # Loop over the fit parameters which only the signal PDF depends on.
        gflp_idxs_S_not_B = gflp_idxs_S[
            np.invert(np.isin(gflp_idxs_S, gflp_idxs_B))]
        for gflp_idx in gflp_idxs_S_not_B:
            grads[gflp_idx] = np.sum(p * sig_grads[gflp_idx] / denum)

        return (log_lh, grads)
