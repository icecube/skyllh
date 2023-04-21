# -*- coding: utf-8 -*-

import numpy as np

import scipy.interpolate

from skyllh.core import (
    multiproc,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.i3.detsigyield import (
    SingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
    SingleParamFluxPointLikeSourceI3DetSigYield,
)
from skyllh.analyses.i3.publicdata_ps.aeff import (
    load_effective_area_array,
)


class PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
        SingleParamFluxPointLikeSourceI3DetSigYieldBuilder):
    """This detector signal yield builder class constructs a
    detector signal yield instance for a variable flux model of a single
    parameter, assuming a point-like source.
    It constructs a two-dimensional spline function in sin(dec) and the
    parameter, using a :class:`scipy.interpolate.RectBivariateSpline`.
    Hence, the detector signal yield can vary with the declination and the
    parameter of the flux model.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends solely on the zenith angle, and hence on the
    declination, of the source.

    It takes the effective area for the detector signal yield from the auxilary
    detector effective area data file given by the public data.
    """

    def __init__(
            self,
            param_grid,
            spline_order_sinDec=2,
            spline_order_param=2,
            ncpu=None,
            **kwargs):
        """Creates a new IceCube detector signal yield builder instance for
        a flux model with a single parameter.
        It requires the effective area from the public data, and a parameter
        grid to compute the parameter dependency of the detector signal yield.

        Parameters
        ----------
        param_grid : instance of ParameterGrid
            The instance of ParameterGrid which defines the grid of parameter
            values.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(dec) axis.
            The default is 2.
        spline_order_param : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the parameter axis.
            The default is 2.
        ncpu : int | None
            The number of CPUs to utilize. If set to ``None``, global setting
            will take place.
        """
        super().__init__(
            param_grid=param_grid,
            sin_dec_binning=None,
            spline_order_sinDec=spline_order_sinDec,
            spline_order_param=spline_order_param,
            ncpu=ncpu,
            **kwargs)

    def construct_detsigyield(
            self,
            dataset,
            data,
            fluxmodel,
            livetime,
            ppbar=None):
        """Constructs a detector signal yield 2-dimensional log spline
        function for the given flux model with varying parameter values.

        Parameters
        ----------
        dataset : instance of Dataset
            The Dataset instance holding the sin(dec) binning definition.
        data : instance of DatasetData
            The instance of DatasetData holding the monte-carlo event data.
            This implementation loads the effective area from the provided
            public data and hence does not need monte-carlo data.
        fluxmodel : instance of FluxModel
            The flux model instance. Must be an instance of PowerLawFlux.
        livetime : float | Livetime instance
            The live-time in days or an instance of Livetime to use for the
            detector signal yield.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : instance of SingleParamFluxPointLikeSourceI3DetSigYield
            The DetSigYield instance for a point-like source with a flux model
            of a single parameter.
        """
        self.assert_types_of_construct_detsigyield_arguments(
            dataset=dataset,
            data=data,
            fluxmodel=fluxmodel,
            livetime=livetime,
            ppbar=ppbar
        )

        # Get integrated live-time in days.
        livetime_days = Livetime.get_integrated_livetime(livetime)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = fluxmodel.get_conversion_factor_to_internal_flux_unit()

        # Load the effective area data from the public dataset.
        aeff_fnames = dataset.get_abs_pathfilename_list(
            dataset.get_aux_data_definition('eff_area_datafile'))
        (
            aeff_arr,
            sin_true_dec_binedges_lower,
            sin_true_dec_binedges_upper,
            log_true_e_binedges_lower,
            log_true_e_binedges_upper
        ) = load_effective_area_array(aeff_fnames)

        # Calculate the detector signal yield in sin_dec vs gamma.
        def _create_hist(
                energy_bin_edges_lower,
                energy_bin_edges_upper,
                aeff,
                fluxmodel):
            """Creates a histogram of the detector signal yield for the given
            sin(dec) binning.

            Parameters
            ----------
            energy_bin_edges_lower : 1d ndarray
                The array holding the lower bin edges in E_nu/GeV.
            energy_bin_edges_upper : 1d ndarray
                The array holding the upper bin edges in E_nu/GeV.
            aeff : (n_bins_sin_dec, n_bins_log_energy)-shaped 2d ndarray
                The effective area binned data array.

            Returns
            -------
            h : instance of ndarray
                The (n_bins_sin_dec,)-shaped 1d numpy ndarray containing the
                detector signal yield values for the different sin_dec bins and
                the given flux model.
            """
            h_phi = fluxmodel.get_integral(
                energy_bin_edges_lower, energy_bin_edges_upper)

            # Sum over the enegry bins for each sin_dec row.
            h = np.sum(aeff*h_phi, axis=1)

            return h

        energy_bin_edges_lower = np.power(10, log_true_e_binedges_lower)
        energy_bin_edges_upper = np.power(10, log_true_e_binedges_upper)

        # Make a copy of the parameter grid and extend the grid by one bin on
        # each side.
        param_grid = self.param_grid.copy()
        param_grid.add_extra_lower_and_upper_bin()

        # Construct the arguments for the hist function to be used in the
        # multiproc.parallelize function.
        args_list = [
            (
                (
                    energy_bin_edges_lower,
                    energy_bin_edges_upper,
                    aeff_arr,
                    fluxmodel.copy({param_grid.name: param_val})
                ),
                {}
            )
            for param_val in param_grid.grid
        ]
        h = np.vstack(
            multiproc.parallelize(
                _create_hist, args_list, self.ncpu, ppbar=ppbar)).T
        h *= toGeVcm2s * livetime_days * 86400.

        # Create a 2d spline in log of the detector signal yield.
        sin_dec_bincenters = 0.5*(
            sin_true_dec_binedges_lower + sin_true_dec_binedges_upper)
        log_spl_sinDec_param = scipy.interpolate.RectBivariateSpline(
            sin_dec_bincenters,
            param_grid.grid,
            np.log(h),
            kx=self.spline_order_sinDec,
            ky=self.spline_order_param,
            s=0)

        # Construct the detector signal yield instance with the created spline.
        sin_dec_binedges = np.concatenate(
            (sin_true_dec_binedges_lower, [sin_true_dec_binedges_upper[-1]]))
        sin_dec_binning = BinningDefinition('sin_dec', sin_dec_binedges)

        detsigyield = SingleParamFluxPointLikeSourceI3DetSigYield(
            param_name=param_grid.name,
            dataset=dataset,
            fluxmodel=fluxmodel,
            livetime=livetime,
            sin_dec_binning=sin_dec_binning,
            log_spl_sinDec_param=log_spl_sinDec_param)

        return detsigyield
