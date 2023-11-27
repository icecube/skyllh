# -*- coding: utf-8 -*-

from skyllh.core.signalpdf import (
    SignalSingleConditionalEnergyPDFSet,
)


class SignalI3EnergyPDFSet(
        SignalSingleConditionalEnergyPDFSet,
):
    """This is the signal energy PDF for IceCube. It creates a set of
    SingleConditionalEnergyPDF objects for a discrete set of energy signal
    parameters. Energy signal parameters influence the source's flux model.
    """
    def __init__(
            self,
            cfg,
            data_mc,
            log10_energy_binning,
            sin_dec_binning,
            fluxmodel,
            param_grid_set,
            smoothing_filter=None,
            ncpu=None,
            ppbar=None,
            **kwargs,
    ):
        """Creates a new IceCube energy signal PDF for a given flux model and
        a set of parameter grids for the flux model.
        It creates a set of SingleConditionalEnergyPDF objects for each signal
        parameter value permutation and stores it in an internal dictionary,
        where the hash of the parameters dictionary is the key.

        Parameters
        ----------
        cfg : instance of Config
            The instance of Config holding the local configuration.
        data_mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monte-carlo data.
            The following data fields must exist:

            true_energy : float
                The true energy value of the data event.
            ``log10_energy_binning.name`` : float
                The base10-logarithm of the reconstructed energy value of the
                data event.
            ``sin_dec_binning.name`` : float
                The declination of the data event.
            mcweight : float
                The monte-carlo weight value of the data events in unit
                GeV cm^2 sr.

        log10_energy_binning : instance of BinningDefinition
            The binning definition for the reconstructed energy binning in
            log10(E_reco). The name of this binning definition defines the field
            name in the MC and trial data.
        sin_dec_binning : instance of BinningDefinition
            The binning definition for the binning in sin(declination).
            The name of this binning definition defines the field name in the MC
            and trial data.
        fluxmodel : instance of FluxModel
            The flux model to use to create the signal energy PDF.
        param_grid_set : instance of ParameterGridSet |
                         instance of ParameterGrid
            The set of parameter grids. A ParameterGrid instance for each
            energy parameter, for which an SingleConditionalEnergyPDF object
            needs to be created.
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If ``None``, no smoothing will be applied.
        ncpu : int | None
            The number of CPUs to use to create the different
            SingleConditionalEnergyPDF instances for the different parameter
            grid values. If set to ``None``, the configured default number of
            CPUs will be used.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.
        """
        super().__init__(
            cfg=cfg,
            data_mc=data_mc,
            log10_energy_binning=log10_energy_binning,
            cond_param_binning=sin_dec_binning,
            flux_model=fluxmodel,
            param_grid_set=param_grid_set,
            smoothing_filter=smoothing_filter,
            ncpu=ncpu,
            ppbar=ppbar,
            **kwargs
        )
