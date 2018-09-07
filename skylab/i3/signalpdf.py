# -*- coding: utf-8 -*-

import numpy as np

from skylab.core.binning import BinningDefinition
from skylab.core.multiproc import IsParallelizable, parallelize
from skylab.core.pdf import PDFSet, IsSignalPDF
from skylab.physics.flux import FluxModel

from skylab.i3.pdf import I3EnergyPDF

class SignalI3EnergyPDFSet(PDFSet, IsSignalPDF, IsParallelizable):
    """This is the signal energy PDF for IceCube. It creates a set of
    I3EnergyPDF objects for a discrete set of energy signal parameters. Energy
    signal parameters are the parameters that influence the source flux model.
    """
    def __init__(self, data_mc, logE_binning, sinDec_binning, fluxmodel,
                 fitparams_grid_set, ncpu=None):
        """Creates a new IceCube energy signal PDF for a given flux model and
        a set of fit parameter grids for the flux model.
        It creates a set of I3EnergyPDF objects for each signal parameter value
        permutation and stores it inside the ``_params_hash_I3EnergyPDF_dict``
        dictionary, where the hash of the fit parameters dictionary is the key.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The array holding the monte-carlo data. The following data fields
            must exist:
            'true_energy' : float
                The true energy value of the data event.
            'log_energy' : float
                The logarithm of the reconstructed energy value of the data
                event.
            'dec' : float
                The declination of the data event.
            'mcweight' : float
                The monte-carlo weight value of the data events in unit
                GeV cm^2 sr.
        logE_binning : BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : BinningDefinition
            The binning definition for the sin(declination).
        fluxmodel : FluxModel
            The flux model to use to create the signal energy PDF.
        fitparams_grid_set : ParameterGridSet
            The set of fit parameter grids. A ParameterGrid object for each
            energy fit parameter, for which an I3EnergyPDF object needs to be
            created.
        ncpu : int | None (default)
            The number of CPUs to use to create the different I3EnergyPDF
            objects for the different fit parameter grid values.
        """
        # We need to extend the fit parameter grids on the lower and upper end
        # by one bin to allow for the calculation of the interpolation. But we
        # will do this on a copy of the object.
        fitparams_grid_set = fitparams_grid_set.copy()
        fitparams_grid_set.add_extra_lower_and_upper_bin()
        print fitparams_grid_set.grid

        super(SignalI3EnergyPDFSet, self).__init__(pdf_type=I3EnergyPDF,
            fitparams_grid_set=fitparams_grid_set, ncpu=ncpu)

        if(not isinstance(logE_binning, BinningDefinition)):
            raise TypeError('The logE_binning argument must be an instance of BinningDefinition!')
        if(not isinstance(sinDec_binning, BinningDefinition)):
            raise TypeError('The sinDec_binning argument must be an instance of BinningDefinition!')
        if(not isinstance(fluxmodel, FluxModel)):
            raise TypeError('The fluxmodel argument must be an instance of FluxModel!')

        # Create I3EnergyPDF objects for all permutations of the fit parameter
        # grid values.
        def create_I3EnergyPDF(data_logE, data_sinDec, data_mcweight, data_true_energy,
                               logE_binning, sinDec_binning, fluxmodel, gridfitparams):
            """Creates an I3EnergyPDF object for the given flux model and flux
            parameters.

            Parameters
            ----------
            data_logE : 1d ndarray
                The logarithm of the reconstructed energy value of the data
                events.
            data_sinDec : 1d ndarray
                The sin(dec) value of the the data events.
            data_mcweight : 1d ndarray
                The monte-carlo weight value of the data events.
            data_true_energy : 1d ndarray
                The true energy value of the data events.
            logE_binning : BinningDefinition
                The binning definition for the binning in log10(E).
            sinDec_binning : BinningDefinition
                The binning definition for the sin(declination).
            fluxmodel : FluxModel
                The flux model to use to create the signal event weights.
            gridfitparams : dict
                The dictionary holding the specific signal flux parameters.

            Returns
            -------
            i3energypdf : I3EnergyPDF
                The created I3EnergyPDF object for the given flux model and flux
                parameters.
            """
            # Create a copy of the FluxModel with the given flux parameters.
            # The copy is needed to not interfer with other CPU processes.
            myfluxmodel = fluxmodel.copy(newprop=gridfitparams)

            # Calculate the signal energy weight of the event. Note, that
            # because we create a normalized PDF, we can ignore all constants.
            # So we don't have to convert the flux unit into the internally used
            # flux unit.
            data_physicsweight = myfluxmodel(data_true_energy)

            i3energypdf = I3EnergyPDF(data_logE, data_sinDec, data_mcweight, data_physicsweight,
                                      logE_binning, sinDec_binning)

            return i3energypdf

        data_logE = data_mc['log_energy']
        data_sinDec = np.sin(data_mc['dec'])
        data_mcweight = data_mc['mcweight']
        data_true_energy = data_mc['true_energy']

        args_list = [ ((data_logE, data_sinDec, data_mcweight, data_true_energy,
                        logE_binning, sinDec_binning, fluxmodel, gridfitparams), {})
                     for gridfitparams in self.gridfitparams_list ]

        i3energypdf_list = parallelize(create_I3EnergyPDF, args_list, self.ncpu)

        # Save all the I3EnergyPDF objects in the IsSignalPDF PDF registry with
        # the hash of the individual parameters as key.
        for (gridfitparams, i3energypdf) in zip(self.gridfitparams_list, i3energypdf_list):
            self.add_pdf(i3energypdf, gridfitparams)

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this signal energy PDF is valid for all the given
        experimental data.
        It checks if all the data is within the logE and sin(dec) binning range.

        Parameters
        ----------
        data_exp : numpy record ndarray
            The array holding the experimental data. The following data fields
            must exist:
            'log_energy' : float
                The logarithm of the energy value of the data event.
            'dec' : float
                The declination of the data event.

        Errors
        ------
        ValueError
            If some of the data is outside the logE or sin(dec) binning range.
        """
        # Since we use the same binning for all the I3EnergyPDF objects, we
        # can just use an arbitrary object to verify the data.
        self.get_pdf(self.pdf_keys[0]).assert_is_valid_for_exp_data(data_exp)

    def get_prob(self, events, gridfitparams):
        """Calculates the signal energy probability (in logE) of each event for
        a given set of signal fit parameters on a grid.

        Parameters
        ----------
        events : numpy record ndarray
            The array holding the event data. The following data fields must
            exist:
            'log_energy' : float
                The logarithm of the energy value of the event.
            'sinDec' : float
                The sin(declination) value of the event.
        gridfitparams : dict
            The dictionary holding the signal parameter values for which the
            signal energy probability should be calculated. Note, that the
            parameter values must match a set of parameter grid values for which
            an I3EnergyPDF object has been created at construction time of this
            I3SignalEnergyPDF object. There is no interpolation method defined
            at this point to allow for arbitrary parameter values!

        Returns
        -------
        prob : 1d ndarray
            The array with the signal energy probability for each event.

        Errors
        ------
        KeyError
            If no energy PDF can be found for the given signal parameter values.
        """
        i3energypdf = self.get_pdf(gridfitparams)

        prob = i3energypdf.get_prob(events)
        return prob
