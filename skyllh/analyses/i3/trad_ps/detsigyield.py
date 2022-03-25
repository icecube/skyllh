# -*- coding: utf-8 -*-

import numpy as np

import scipy.interpolate

from skyllh.core import multiproc
from skyllh.core.binning import BinningDefinition
from skyllh.core.dataset import (
    Dataset,
    DatasetData
)
from skyllh.core.livetime import Livetime
from skyllh.core.parameters import ParameterGrid
from skyllh.core.detsigyield import (
    get_integrated_livetime_in_days
)
from skyllh.core.storage import (
    create_FileLoader
)
from skyllh.physics.flux import (
    PowerLawFlux,
    get_conversion_factor_to_internal_flux_unit
)
from skyllh.i3.detsigyield import (
    PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod,
    PowerLawFluxPointLikeSourceI3DetSigYield
)


class PublicDataPowerLawFluxPointLikeSourceI3DetSigYieldImplMethod(
        PowerLawFluxPointLikeSourceI3DetSigYieldImplMethod,
        multiproc.IsParallelizable):
    """Thus detector signal yield constructor class constructs a
    detector signal yield instance for a varibale power law flux model, which
    has the spectral index gama as fit parameter, assuming a point-like source.
    It constructs a two-dimensional spline function in sin(dec) and gamma, using
    a :class:`scipy.interpolate.RectBivariateSpline`. Hence, the detector signal
    yield can vary with the declination and the spectral index, gamma, of the
    source.

    This detector signal yield implementation method works with a
    PowerLawFlux flux model.

    It is tailored to the IceCube detector at the South Pole, where the
    effective area depends soley on the zenith angle, and hence on the
    declination, of the source.

    It takes the effective area for the detector signal yield from the auxilary
    detector effective area data file given by the public data.
    """
    def __init__(
            self, gamma_grid, spline_order_sinDec=2, spline_order_gamma=2,
            ncpu=None):
        """Creates a new IceCube detector signal yield constructor instance for
        a power law flux model. It requires the effective area from the public
        data, and a gamma parameter grid to compute the gamma dependency of the
        detector signal yield.

        Parameters
        ----------
        gamma_grid : ParameterGrid instance
            The ParameterGrid instance which defines the grid of gamma values.
        spline_order_sinDec : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the sin(dec) axis.
            The default is 2.
        spline_order_gamma : int
            The order of the spline function for the logarithmic values of the
            detector signal yield along the gamma axis.
            The default is 2.
        ncpu : int | None
            The number of CPUs to utilize. Global setting will take place if
            not specified, i.e. set to None.
        """
        super().__init__(
            gamma_grid=gamma_grid,
            sin_dec_binning=None,
            spline_order_sinDec=spline_order_sinDec,
            spline_order_gamma=spline_order_gamma,
            ncpu=ncpu)

    def construct_detsigyield(
            self, dataset, data, fluxmodel, livetime, ppbar=None):
        """Constructs a detector signal yield 2-dimensional log spline
        function for the given power law flux model with varying gamma values.

        Parameters
        ----------
        dataset : Dataset instance
            The Dataset instance holding the sin(dec) binning definition.
        data : DatasetData instance
            The DatasetData instance holding the monte-carlo event data.
            This implementation loads the effective area from the provided
            public data and hence does not need monte-carlo data.
        fluxmodel : FluxModel
            The flux model instance. Must be an instance of PowerLawFlux.
        livetime : float | Livetime instance
            The live-time in days or an instance of Livetime to use for the
            detector signal yield.
        ppbar : ProgressBar instance | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield : PowerLawFluxPointLikeSourceI3DetSigYield instance
            The DetSigYield instance for a point-like source with a power law
            flux with variable gamma parameter.
        """
        # Check for the correct data types of the input arguments.
        if(not isinstance(dataset, Dataset)):
            raise TypeError('The dataset argument must be an instance of '
                'Dataset!')
        if(not isinstance(data, DatasetData)):
            raise TypeError('The data argument must be an instance of '
                'DatasetData!')
        if(not self.supports_fluxmodel(fluxmodel)):
            raise TypeError('The DetSigYieldImplMethod "%s" does not support '
                'the flux model "%s"!'%(
                    self.__class__.__name__,
                    fluxmodel.__class__.__name__))
        if((not isinstance(livetime, float)) and
           (not isinstance(livetime, Livetime))):
            raise TypeError('The livetime argument must be an instance of '
                'float or Livetime!')

        # Get integrated live-time in days.
        livetime_days = get_integrated_livetime_in_days(livetime)

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)

        # Load the effective area data from the public dataset.
        aeff_fnames = dataset.get_abs_pathfilename_list(
            dataset.get_aux_data_definition('eff_area_datafile'))
        floader = create_FileLoader(aeff_fnames)
        aeff_data = floader.load_data()
        aeff_data.rename_fields(
            {
                'log10(E_nu/GeV)_min': 'log_true_energy_nu_min',
                'log10(E_nu/GeV)_max': 'log_true_energy_nu_max',
                'Dec_nu_min[deg]':     'true_sin_dec_nu_min',
                'Dec_nu_max[deg]':     'true_sin_dec_nu_max',
                'A_Eff[cm^2]':         'a_eff'
            },
            must_exist=True)
        # Convert the true neutrino declination from degrees to radians and into
        # sin values.
        aeff_data['true_sin_dec_nu_min'] = np.sin(np.deg2rad(
            aeff_data['true_sin_dec_nu_min']))
        aeff_data['true_sin_dec_nu_max'] = np.sin(np.deg2rad(
            aeff_data['true_sin_dec_nu_max']))

        # Determine the binning for energy and declination.
        log_energy_bin_edges_lower = np.unique(
            aeff_data['log_true_energy_nu_min'])
        log_energy_bin_edges_upper = np.unique(
            aeff_data['log_true_energy_nu_max'])

        sin_dec_bin_edges_lower = np.unique(aeff_data['true_sin_dec_nu_min'])
        sin_dec_bin_edges_upper = np.unique(aeff_data['true_sin_dec_nu_max'])

        if(len(log_energy_bin_edges_lower) != len(log_energy_bin_edges_upper)):
            raise ValueError('Cannot extract the log10(E/GeV) binning of the '
                'effective area for dataset "{}". The number of lower and '
                'upper bin edges is not equal!'.format(dataset.name))
        if(len(sin_dec_bin_edges_lower) != len(sin_dec_bin_edges_upper)):
            raise ValueError('Cannot extract the Dec_nu binning of the '
                'effective area for dataset "{}". The number of lower and '
                'upper bin edges is not equal!'.format(dataset.name))

        n_bins_log_energy = len(log_energy_bin_edges_lower)
        n_bins_sin_dec = len(sin_dec_bin_edges_lower)

        # Construct the 2d array for the effective area.
        aeff_arr = np.zeros((n_bins_sin_dec, n_bins_log_energy), dtype=np.float)

        sin_dec_idx = np.digitize(
            0.5*(aeff_data['true_sin_dec_nu_min'] +
                 aeff_data['true_sin_dec_nu_max']),
            sin_dec_bin_edges_lower) - 1
        log_e_idx = np.digitize(
            0.5*(aeff_data['log_true_energy_nu_min'] +
                 aeff_data['log_true_energy_nu_max']),
            log_energy_bin_edges_lower) - 1

        aeff_arr[sin_dec_idx,log_e_idx] = aeff_data['a_eff']

        # Calculate the detector signal yield in sin_dec vs gamma.
        def hist(
                energy_bin_edges_lower, energy_bin_edges_upper,
                aeff, fluxmodel):
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
            h : (n_bins_sin_dec,)-shaped 1d ndarray
                The numpy array containing the detector signal yield values for
                the different sin_dec bins and the given flux model.
            """
            # Create histogram for the number of neutrinos with each energy
            # bin.
            h_phi = fluxmodel.get_integral(
                energy_bin_edges_lower, energy_bin_edges_upper)

            # Sum over the enegry bins for each sin_dec row.
            h = np.sum(aeff*h_phi, axis=1)

            return h

        energy_bin_edges_lower = np.power(10, log_energy_bin_edges_lower)
        energy_bin_edges_upper = np.power(10, log_energy_bin_edges_upper)

        # Make a copy of the gamma grid and extend the grid by one bin on each
        # side.
        gamma_grid = self._gamma_grid.copy()
        gamma_grid.add_extra_lower_and_upper_bin()

        # Construct the arguments for the hist function to be used in the
        # multiproc.parallelize function.
        args_list = [
            ((energy_bin_edges_lower,
              energy_bin_edges_upper,
              aeff_arr,
              fluxmodel.copy({'gamma':gamma})), {})
            for gamma in gamma_grid.grid
        ]
        h = np.vstack(
            multiproc.parallelize(
                hist, args_list, self.ncpu, ppbar=ppbar)).T
        h *= toGeVcm2s * livetime_days * 86400.

        # Create a 2d spline in log of the detector signal yield.
        sin_dec_bincenters = 0.5*(
            sin_dec_bin_edges_lower + sin_dec_bin_edges_upper)
        log_spl_sinDec_gamma = scipy.interpolate.RectBivariateSpline(
            sin_dec_bincenters, gamma_grid.grid, np.log(h),
            kx = self.spline_order_sinDec, ky = self.spline_order_gamma, s = 0)

        # Construct the detector signal yield instance with the created spline.
        sin_dec_binedges = np.concatenate(
            (sin_dec_bin_edges_lower, [sin_dec_bin_edges_upper[-1]]))
        sin_dec_binning = BinningDefinition('sin_dec', sin_dec_binedges)
        detsigyield = PowerLawFluxPointLikeSourceI3DetSigYield(
            self, dataset, fluxmodel, livetime, sin_dec_binning, log_spl_sinDec_gamma)

        return detsigyield
