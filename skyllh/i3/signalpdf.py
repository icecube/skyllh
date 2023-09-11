# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.flux_model import (
    FluxModel,
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
    PDF,
    PDFSet,
    IsSignalPDF,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.smoothing import (
    SmoothingFilter,
)
from skyllh.i3.pdf import (
    I3EnergyPDF,
)


class SignalI3EnergyPDFSet(
        PDFSet,
        IsSignalPDF,
        PDF,
        IsParallelizable,
):
    """This is the signal energy PDF for IceCube. It creates a set of
    I3EnergyPDF objects for a discrete set of energy signal parameters. Energy
    signal parameters influence the source's flux model.
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
        It creates a set of I3EnergyPDF objects for each signal parameter value
        permutation and stores it in an internal dictionary, where the hash of
        the parameters dictionary is the key.

        Parameters
        ----------
        cfg : instance of Config
            The instance of Config holding the local configuration.
        data_mc : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the monte-carlo data.
            The following data fields must exist:

            true_energy : float
                The true energy value of the data event.
            log_energy : float
                The base10-logarithm of the reconstructed energy value of the
                data event.
            sin_dec : float
                The declination of the data event.
            mcweight : float
                The monte-carlo weight value of the data events in unit
                GeV cm^2 sr.

        log10_energy_binning : instance of BinningDefinition
            The binning definition for the reconstructed energy binning in
            log10(E).
        sin_dec_binning : instance of BinningDefinition
            The binning definition for the binning in sin(declination).
        fluxmodel : instance of FluxModel
            The flux model to use to create the signal energy PDF.
        param_grid_set : instance of ParameterGridSet |
                         instance of ParameterGrid
            The set of parameter grids. A ParameterGrid instance for each
            energy parameter, for which an I3EnergyPDF object needs to be
            created.
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If ``None``, no smoothing will be applied.
        ncpu : int | None
            The number of CPUs to use to create the different I3EnergyPDF
            instances for the different parameter grid values.
            If set to ``None``, the configured default number of CPUs will be
            used.
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.
        """
        if isinstance(param_grid_set, ParameterGrid):
            param_grid_set = ParameterGridSet([param_grid_set])
        if not isinstance(param_grid_set, ParameterGridSet):
            raise TypeError(
                'The param_grid_set argument must be an instance of '
                'ParameterGrid or ParameterGridSet! But its type is '
                f'{classname(param_grid_set)}!')

        # We need to extend the parameter grids on the lower and upper end
        # by one bin to allow for the calculation of the interpolation. But we
        # will do this on a copy of the object.
        param_grid_set = param_grid_set.copy()
        param_grid_set.add_extra_lower_and_upper_bin()

        super().__init__(
            cfg=cfg,
            param_grid_set=param_grid_set,
            ncpu=ncpu,
            **kwargs)

        if not isinstance(log10_energy_binning, BinningDefinition):
            raise TypeError(
                'The log10_energy_binning argument must be an instance of '
                'BinningDefinition! '
                f'Its type is {classname(log10_energy_binning)}!')
        if not isinstance(sin_dec_binning, BinningDefinition):
            raise TypeError(
                'The sin_dec_binning argument must be an instance '
                'of BinningDefinition! '
                f'Its type is {classname(sin_dec_binning)}!')
        if not isinstance(fluxmodel, FluxModel):
            raise TypeError(
                'The fluxmodel argument must be an instance of FluxModel! '
                f'Its type is {classname(fluxmodel)}!')
        if (smoothing_filter is not None) and\
           (not isinstance(smoothing_filter, SmoothingFilter)):
            raise TypeError(
                'The smoothing_filter argument must be None or '
                'an instance of SmoothingFilter! '
                f'Its type is {classname(smoothing_filter)}!')

        # Create I3EnergyPDF objects for all permutations of the parameter
        # grid values.
        def create_I3EnergyPDF(
                cfg,
                data_log10_energy,
                data_sin_dec,
                data_mcweight,
                data_true_energy,
                log10_energy_binning,
                sin_dec_binning,
                smoothing_filter,
                fluxmodel,
                flux_unit_conv_factor,
                gridparams,
        ):
            """Creates an I3EnergyPDF object for the given flux model and flux
            parameters.

            Parameters
            ----------
            cfg : instance of Config
                The instance of Config holding the local configuration.
            data_log10_energy : 1d ndarray
                The base-10 logarithm of the reconstructed energy value of the
                data events.
            data_sin_dec : 1d ndarray
                The sin(dec) value of the the data events.
            data_mcweight : 1d ndarray
                The monte-carlo weight value of the data events.
            data_true_energy : 1d ndarray
                The true energy value of the data events.
            log10_energy_binning : instance of BinningDefinition
                The binning definition for the binning in log10(E).
            sin_dec_binning : instance of BinningDefinition
                The binning definition for the sin(declination).
            smoothing_filter : instance of SmoothingFilter | None
                The smoothing filter to use for smoothing the energy histogram.
                If ``None``, no smoothing will be applied.
            fluxmodel : instance of FluxModel
                The flux model to use to create the signal event weights.
            flux_unit_conv_factor : float
                The factor to convert the flux unit into the internal flux unit.
            gridparams : dict
                The dictionary holding the specific signal flux parameters.

            Returns
            -------
            i3energypdf : instance of I3EnergyPDF
                The created I3EnergyPDF instance for the given flux model and
                flux parameters.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            myfluxmodel = fluxmodel.copy(newparams=gridparams)

            # Calculate the signal energy weight of the event. Note, that
            # because we create a normalized PDF, we can ignore all constants.
            # So we don't have to convert the flux unit into the internally used
            # flux unit.
            data_physicsweight = np.squeeze(myfluxmodel(E=data_true_energy))
            data_physicsweight *= flux_unit_conv_factor

            i3energypdf = I3EnergyPDF(
                cfg=cfg,
                pmm=None,
                data_log10_energy=data_log10_energy,
                data_sin_dec=data_sin_dec,
                data_mcweight=data_mcweight,
                data_physicsweight=data_physicsweight,
                log10_energy_binning=log10_energy_binning,
                sin_dec_binning=sin_dec_binning,
                smoothing_filter=smoothing_filter)

            return i3energypdf

        data_log10_energy = data_mc['log_energy']
        data_sin_dec = data_mc['sin_dec']
        data_mcweight = data_mc['mcweight']
        data_true_energy = data_mc['true_energy']

        flux_unit_conv_factor =\
            fluxmodel.to_internal_flux_unit()

        args_list = [
            (
                (cfg,
                 data_log10_energy,
                 data_sin_dec,
                 data_mcweight,
                 data_true_energy,
                 log10_energy_binning,
                 sin_dec_binning,
                 smoothing_filter,
                 fluxmodel,
                 flux_unit_conv_factor,
                 gridparams),
                {}
            )
            for gridparams in self.gridparams_list
        ]

        i3energypdf_list = parallelize(
            func=create_I3EnergyPDF,
            args_list=args_list,
            ncpu=self.ncpu,
            ppbar=ppbar)

        # Save all the I3EnergyPDF instances in the PDFSet registry with
        # the hash of the individual parameters as key.
        for (gridparams, i3energypdf) in zip(self.gridparams_list,
                                             i3energypdf_list):
            self.add_pdf(i3energypdf, gridparams)
