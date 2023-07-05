# -*- coding: utf-8 -*-

import numpy as np

from scipy import integrate

from skyllh.analyses.i3.publicdata_ps.aeff import (
    PDAeff,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    FctSpline1D,
)
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix,
)
from skyllh.core.binning import (
    get_bincenters_from_binedges,
)
from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core.flux_model import (
    FactorizedFluxModel,
)
from skyllh.core.multiproc import (
    IsParallelizable,
    parallelize,
)
from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet,
)
from skyllh.core.pdf import (
    IsSignalPDF,
    PDF,
    PDFAxis,
    PDFSet,
)
from skyllh.core.py import (
    classname,
    module_classname,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.i3.dataset import (
    I3Dataset,
)


class PDSignalEnergyPDF(
        PDF,
        IsSignalPDF,
):
    """This class provides a signal energy PDF for a spectrial index value.
    """
    def __init__(
            self,
            f_e_spl,
            **kwargs,
    ):
        """Creates a new signal energy PDF instance for a particular spectral
        index value.

        Parameters
        ----------
        f_e_spl : instance of FctSpline1D
            The instance of FctSpline1D representing the spline of the energy
            PDF.
        """
        super().__init__(
            pmm=None,
            **kwargs)

        if not isinstance(f_e_spl, FctSpline1D):
            raise TypeError(
                'The f_e_spl argument must be an instance of FctSpline1D! '
                f'Its current type is {classname(f_e_spl)}!')

        self.f_e_spl = f_e_spl

        self.log10_reco_e_lower_binedges = self.f_e_spl.x_binedges[:-1]
        self.log10_reco_e_upper_binedges = self.f_e_spl.x_binedges[1:]

        self.log10_reco_e_min = self.log10_reco_e_lower_binedges[0]
        self.log10_reco_e_max = self.log10_reco_e_upper_binedges[-1]

        # Add the PDF axes.
        self.add_axis(
            PDFAxis(
                name='log_energy',
                vmin=self.log10_reco_e_min,
                vmax=self.log10_reco_e_max))

        # Check integrity.
        integral = integrate.quad(
            self.f_e_spl.evaluate,
            self.log10_reco_e_min,
            self.log10_reco_e_max,
            limit=200,
            full_output=1
        )[0] / self.f_e_spl.norm
        if not np.isclose(integral, 1):
            raise ValueError(
                'The integral over log10_reco_e of the energy term must be '
                f'unity! But it is {integral}!')

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None):
        pass

    def get_pd_by_log10_reco_e(
            self,
            log10_reco_e,
            tl=None):
        """Calculates the probability density for the given log10(E_reco/GeV)
        values using the spline representation of the PDF.

        Parameters
        ----------
        log10_reco_e : instance of ndarray
            The (n_log10_reco_e,)-shaped numpy ndarray holding the
            log10(E_reco/GeV) values for which the energy PDF should get
            evaluated.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (n_log10_reco_e,)-shaped numpy ndarray with the probability
            density for each energy value.
        """
        # Select energy values that actually have a signal energy PDF.
        # All other energy values will get zero signal probability density.
        m = (
            (log10_reco_e >= self.log10_reco_e_min) &
            (log10_reco_e < self.log10_reco_e_max)
        )

        with TaskTimer(tl, 'Evaluate PDSignalEnergyPDF'):
            pd = np.zeros((len(log10_reco_e),), dtype=np.double)
            pd[m] = self.f_e_spl(log10_reco_e[m]) / self.f_e_spl.norm

        return pd

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the probability density for all given trial data events
        and sources.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events for
            which the probability density should be looked up.
            The following data fields must be present:

            log_energy : float
                The base-10 logarithm of the reconstructed energy.

        params_recarray : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the probability density
            for each trial data event and source.
        grads : dict
            The dictionary holding the gradient values for each global fit
            parameter. By definition this PDF does not depend on any fit
            parameters, hence, this is an empty dictionary.
        """
        evt_idxs = tdm.src_evt_idxs[1]

        log10_reco_e = np.take(tdm['log_energy'], evt_idxs)

        pd = self.get_pd_by_log10_reco_e(
            log10_reco_e=log10_reco_e,
            tl=tl)

        grads = dict()

        return (pd, grads)


class PDSignalEnergyPDFSet(
        PDFSet,
        IsSignalPDF,
        PDF,
        IsParallelizable,
):
    """This class provides a signal energy PDF set using the public data.
    It creates a set of PDSignalEnergyPDF instances, one for each spectral
    index value on a grid.
    """
    def __init__(
            self,
            ds,
            src_dec,
            fluxmodel,
            param_grid_set,
            ncpu=None,
            ppbar=None,
            **kwargs,
    ):
        """Creates a new PDSignalEnergyPDFSet instance for the public data.

        Parameters
        ----------
        ds : instance of Dataset
            The instance of Dataset that defines the dataset of the public data.
        src_dec : float
            The declination of the source in radians.
        fluxmodel : instance of FactorizedFluxModel
            The instance of FactorizedFluxModel that defines the source's flux
            model.
        param_grid_set : instance of ParameterGrid | instance of ParameterGridSet
            The parameter grid set defining the grids of the parameters this
            energy PDF set depends on.
        ncpu : int | None
            The number of CPUs to utilize. Global setting will take place if
            not specified, i.e. set to None.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar for the optional parent progress bar.
        """
        self._logger = get_logger(module_classname(self))

        # Check for the correct types of the arguments.
        if not isinstance(ds, I3Dataset):
            raise TypeError(
                'The ds argument must be an instance of I3Dataset!')

        if not isinstance(fluxmodel, FactorizedFluxModel):
            raise TypeError(
                'The fluxmodel argument must be an instance of '
                'FactorizedFluxModel! '
                f'Its current type is {classname(fluxmodel)}')

        if (not isinstance(param_grid_set, ParameterGrid)) and\
           (not isinstance(param_grid_set, ParameterGridSet)):
            raise TypeError(
                'The param_grid_set argument must be an instance of type '
                'ParameterGrid or ParameterGridSet! '
                f'Its current type is {classname(param_grid_set)}!')

        # Extend the param_grid_set to allow for parameter interpolation
        # values at the grid edges.
        param_grid_set = param_grid_set.copy()
        param_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            param_grid_set=param_grid_set,
            ncpu=ncpu,
            **kwargs)

        # Load the smearing matrix.
        sm = PDSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('smearing_datafile')))

        # Select the slice of the smearing matrix corresponding to the
        # source declination band.
        true_dec_idx = sm.get_true_dec_idx(src_dec)
        sm_pdf = sm.pdf[:, true_dec_idx]

        # Only look at true neutrino energies for which a recostructed
        # muon energy distribution exists in the smearing matrix.
        (min_log_true_e,
         max_log_true_e) = sm.get_true_log_e_range_with_valid_log_e_pdfs(
            true_dec_idx)
        log_true_e_mask = np.logical_and(
            sm.log10_true_enu_binedges >= min_log_true_e,
            sm.log10_true_enu_binedges <= max_log_true_e)
        true_enu_binedges = np.power(
            10, sm.log10_true_enu_binedges[log_true_e_mask])
        true_enu_binedges_lower = true_enu_binedges[:-1]
        true_enu_binedges_upper = true_enu_binedges[1:]
        valid_true_e_idxs = [
            sm.get_log10_true_e_idx(0.5 * (he + le))
            for (he, le) in zip(
                sm.log10_true_enu_binedges[log_true_e_mask][1:],
                sm.log10_true_enu_binedges[log_true_e_mask][:-1])
        ]

        xvals_binedges = ds.get_binning_definition('log_energy').binedges
        xvals = get_bincenters_from_binedges(xvals_binedges)

        # Calculate the neutrino enegry bin widths in GeV.
        d_enu = np.diff(true_enu_binedges)
        self._logger.debug(
            'dE_nu = {}'.format(d_enu)
        )

        # Load the effective area.
        aeff = PDAeff(
            pathfilenames=ds.get_abs_pathfilename_list(
                ds.get_aux_data_definition('eff_area_datafile')))

        # Calculate the probability to detect a neutrino of energy
        # E_nu given a neutrino declination: p(E_nu|dec).
        det_prob = aeff.get_detection_prob_for_decnu(
            decnu=src_dec,
            enu_min=true_enu_binedges[:-1],
            enu_max=true_enu_binedges[1:],
            enu_range_min=true_enu_binedges[0],
            enu_range_max=true_enu_binedges[-1]
        )

        self._logger.debug('det_prob = {}, sum = {}'.format(
            det_prob, np.sum(det_prob)))

        if not np.isclose(np.sum(det_prob), 1):
            self._logger.warn(
                'The sum of the detection probabilities is not unity! It is '
                '{}.'.format(np.sum(det_prob)))

        psi_edges_bw = sm.psi_upper_edges - sm.psi_lower_edges
        ang_err_bw = sm.ang_err_upper_edges - sm.ang_err_lower_edges

        # Create the energy pdf for different gamma values.
        def create_energy_pdf(sm_pdf, fluxmodel, gridparams):
            """Creates an energy pdf for a specific gamma value.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            my_fluxmodel = fluxmodel.copy(newparams=gridparams)

            self._logger.debug(
                f'Generate signal energy PDF for parameters {gridparams} in '
                f'{len(valid_true_e_idxs)} E_nu bins.')

            # Calculate the flux probability p(E_nu|gamma).
            flux_prob = (
                my_fluxmodel.energy_profile.get_integral(
                    true_enu_binedges_lower,
                    true_enu_binedges_upper
                ) /
                my_fluxmodel.energy_profile.get_integral(
                    true_enu_binedges[0],
                    true_enu_binedges[-1]
                )
            )
            if not np.isclose(np.sum(flux_prob), 1):
                self._logger.warn(
                    'The sum of the flux probabilities is not unity! It is '
                    '{}.'.format(np.sum(flux_prob)))

            self._logger.debug(
                'flux_prob = {}, sum = {}'.format(
                    flux_prob, np.sum(flux_prob))
            )

            p = flux_prob * det_prob

            true_e_prob = p / np.sum(p)

            self._logger.debug(
                'true_e_prob = {}'.format(
                    true_e_prob))

            def create_reco_e_pdf_for_true_e(idx, true_e_idx):
                """This functions creates a spline for the reco energy
                distribution given a true neutrino engery.
                """
                # Create the energy PDF f_e = P(log10_E_reco|dec) =
                # \int dPsi dang_err P(E_reco,Psi,ang_err).
                f_e = np.sum(
                    sm_pdf[true_e_idx] *
                    psi_edges_bw[true_e_idx, true_dec_idx, :, :, np.newaxis] *
                    ang_err_bw[true_e_idx, true_dec_idx, :, :, :],
                    axis=(-1, -2)
                )

                # Build the spline for this P(E_reco|E_nu). Weigh the pdf
                # with the true neutrino energy probability (flux prob).
                log10_reco_e_binedges = sm.log10_reco_e_binedges[
                    true_e_idx, true_dec_idx]

                p = f_e * true_e_prob[idx]

                spline = FctSpline1D(p, log10_reco_e_binedges)

                return spline(xvals)

            # Integrate over the true neutrino energy and spline the output.
            sum_pdf = np.sum(
                [
                    create_reco_e_pdf_for_true_e(i, true_e_idx)
                    for (i, true_e_idx) in enumerate(valid_true_e_idxs)
                ],
                axis=0)

            spline = FctSpline1D(sum_pdf, xvals_binedges, norm=True)

            pdf = PDSignalEnergyPDF(spline, cfg=self._cfg)

            return pdf

        args_list = [
            ((sm_pdf, fluxmodel, gridparams), {})
            for gridparams in self.gridparams_list
        ]

        pdf_list = parallelize(
            create_energy_pdf,
            args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        del sm_pdf

        # Save all the energy PDF objects in the PDFSet PDF registry with
        # the hash of the individual parameters as key.
        for (gridparams, pdf) in zip(self.gridparams_list, pdf_list):
            self.add_pdf(pdf, gridparams)
