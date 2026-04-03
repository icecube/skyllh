import numpy as np
import scipy.interpolate
from astropy import units

from skyllh.analyses.i3.publicdata_ps.aeff import (
    load_effective_area_array,
)
from skyllh.core import (
    multiproc,
)
from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.flux_model import (
    FactorizedFluxModel,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.parameters import ParameterGrid
from skyllh.core.progressbar import ProgressBar
from skyllh.core.py import (
    classname,
    issequence,
)
from skyllh.core.source_hypo_grouping import SourceHypoGroup
from skyllh.i3.detsigyield import (
    SingleParamFluxPointLikeSourceI3DetSigYield,
    SingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)


class PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(
    SingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
):
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

    It takes the effective area for the detector signal yield from the auxiliary
    detector effective area data file given by the public data.
    """

    def __init__(
        self,
        param_grid: ParameterGrid,
        spline_order_sinDec: int = 2,
        spline_order_param: int = 2,
        ncpu: int | None = None,
        **kwargs,
    ):
        """Creates a new IceCube detector signal yield builder instance for
        a flux model with a single parameter.
        It requires the effective area from the public data, and a parameter
        grid to compute the parameter dependency of the detector signal yield.

        Parameters
        ----------
        param_grid
            The instance of ParameterGrid which defines the grid of parameter
            values.
        spline_order_sinDec
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(dec) axis.
        spline_order_param
            The order of the spline function for the logarithmic values of the
            detector signal yield along the parameter axis.
        ncpu
            The number of CPUs to utilize. If set to ``None``, global setting
            will take place.
        """
        super().__init__(
            param_grid=param_grid,
            sin_dec_binning=None,
            spline_order_sinDec=spline_order_sinDec,
            spline_order_param=spline_order_param,
            ncpu=ncpu,
            **kwargs,
        )

    def assert_types_of_construct_detsigyield_arguments(self, dataset, data, shgs, ppbar, **kwargs):
        """Checks the correct types of the arguments for the
        ``construct_detsigyield`` method.
        """
        super().assert_types_of_construct_detsigyield_arguments(dataset, data, shgs, ppbar, **kwargs)

        if not issequence(shgs):
            shgs = [shgs]
        for shg in shgs:
            if not isinstance(shg.fluxmodel, FactorizedFluxModel):
                raise TypeError(
                    'The fluxmodel of the source hypothesis group must be an '
                    'instance of FactorizedFluxModel! '
                    f'Its current type is {classname(shg.fluxmodel)}!'
                )

    def construct_detsigyield(
        self, dataset: Dataset, data: DatasetData, shg: SourceHypoGroup, ppbar: ProgressBar | None = None
    ) -> SingleParamFluxPointLikeSourceI3DetSigYield:
        """Constructs a detector signal yield 2-dimensional log spline
        function for the given flux model with varying parameter values.

        Parameters
        ----------
        dataset
            The Dataset instance holding the sin(dec) binning definition.
        data
            The instance of DatasetData holding the monte-carlo event data.
            This implementation loads the effective area from the provided
            public data and hence does not need monte-carlo data.
        shg
            The instance of SourceHypoGroup (i.e. sources and flux model) for
            which the detector signal yield should get constructed.
        ppbar
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield
            The DetSigYield instance for a point-like source with a flux model
            of a single parameter.
        """
        self.assert_types_of_construct_detsigyield_arguments(
            dataset=dataset,
            data=data,
            shgs=shg,
            ppbar=ppbar,
        )

        # Get integrated live-time in days.
        assert data.livetime is not None
        livetime_days = Livetime.get_integrated_livetime(data.livetime)

        to_internal_time_unit_factor = self._cfg.to_internal_time_unit(time_unit=units.day)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        to_internal_flux_unit_factor = shg.fluxmodel.to_internal_flux_unit()

        # Load the effective area data from the public dataset.
        aeff_fnames = dataset.get_abs_pathfilename_list(dataset.get_aux_data_definition('eff_area_datafile'))
        (
            aeff_arr,
            sin_true_dec_binedges_lower,
            sin_true_dec_binedges_upper,
            log_true_e_binedges_lower,
            log_true_e_binedges_upper,
        ) = load_effective_area_array(aeff_fnames)

        # Calculate the detector signal yield in sin_dec vs gamma.
        def _create_hist(
            energy_bin_edges_lower: np.ndarray,
            energy_bin_edges_upper: np.ndarray,
            aeff: np.ndarray,
            fluxmodel: FactorizedFluxModel,
            to_internal_flux_unit_factor: float,
        ) -> np.ndarray:
            """Creates a histogram of the detector signal yield for the given
            sin(dec) binning.

            Parameters
            ----------
            energy_bin_edges_lower
                The array holding the lower bin edges in E_nu/GeV.
            energy_bin_edges_upper
                The array holding the upper bin edges in E_nu/GeV.
            aeff
                The (n_bins_sin_dec, n_bins_log_energy)-shaped 2d ndarray holding the effective area binned data array.
            fluxmodel
                The flux model for which the detector signal yield should get calculated.
            to_internal_flux_unit_factor
                The factor to convert the flux model unit into the internal flux unit.

            Returns
            -------
            h
                The (n_bins_sin_dec,)-shaped 1d numpy ndarray containing the
                detector signal yield values for the different sin_dec bins and
                the given flux model.
            """
            h_phi = fluxmodel.energy_profile.get_integral(E1=energy_bin_edges_lower, E2=energy_bin_edges_upper)
            h_phi *= to_internal_flux_unit_factor

            # Sum over the energy bins for each sin_dec row.
            h = np.sum(aeff * h_phi, axis=1)

            # Make sure `h` is greater than 0 everywhere.
            min_h = np.min(h[h > 0])
            return np.where(h == 0, min_h * 1e-10, h)

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
                    shg.fluxmodel.copy({param_grid.name: param_val}),
                    to_internal_flux_unit_factor,
                ),
                {},
            )
            for param_val in param_grid.grid
        ]
        h = np.vstack(multiproc.parallelize(_create_hist, args_list, self.ncpu, ppbar=ppbar)).T
        h *= livetime_days * to_internal_time_unit_factor

        # Create a 2d spline in log of the detector signal yield.
        sin_dec_bincenters = 0.5 * (sin_true_dec_binedges_lower + sin_true_dec_binedges_upper)
        log_spl_sinDec_param = scipy.interpolate.RectBivariateSpline(
            sin_dec_bincenters, param_grid.grid, np.log(h), kx=self.spline_order_sinDec, ky=self.spline_order_param, s=0
        )

        # Construct the detector signal yield instance with the created spline.
        sin_dec_binedges = np.concatenate((sin_true_dec_binedges_lower, [sin_true_dec_binedges_upper[-1]]))
        sin_dec_binning = BinningDefinition('sin_dec', sin_dec_binedges)

        detsigyield = SingleParamFluxPointLikeSourceI3DetSigYield(
            param_name=param_grid.name,
            dataset=dataset,
            fluxmodel=shg.fluxmodel,
            livetime=data.livetime,
            sin_dec_binning=sin_dec_binning,
            log_spl_sinDec_param=log_spl_sinDec_param,
        )

        return detsigyield
