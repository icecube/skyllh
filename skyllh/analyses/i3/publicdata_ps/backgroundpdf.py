# -*- coding: utf-8 -*-

import numpy as np
import pkgutil
import importlib.resources as impresources

from skyllh.analyses.i3.publicdata_ps.detsigyield import (
    PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)
from scipy.stats import (
    gaussian_kde,
)
from skyllh.core.binning import (
    BinningDefinition,
    UsesBinning,
)
from skyllh.core.flux_model import (
    PowerLawEnergyFluxProfile,
    SteadyPointlikeFFM,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterModelMapper,
)
from skyllh.analyses.i3.publicdata_ps.signalpdf import (
    PDSignalEnergyPDFSetMultiSource,
)
from skyllh.core.pdf import (
    EnergyPDF,
    IsBackgroundPDF,
    PDFAxis,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.smoothing import (
    UNSMOOTH_AXIS,
    SmoothingFilter,
    HistSmoothingMethod,
    NoHistSmoothingMethod,
    NeighboringBinHistSmoothingMethod,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.timing import (
    TaskTimer,
)


class PDBackgroundI3EnergyPDF(
        EnergyPDF,
        IsBackgroundPDF,
        UsesBinning):
    """
    This is the base class for an IceCube specific energy background PDF for
    the public data.

    IceCube energy PDFs depend solely on the energy and the zenith angle, and
    hence, on the declination of the event.

    The IceCube energy PDF is modeled as a 1d histogram in energy,
    but for different sin(declination) bins, hence, stored as a 2d histogram.
    """

    _KDE_BW_NORTH = 0.4
    _KDE_BW_SOUTH = 0.32
    _ALPHA = 2.0

    def __init__(
            self,
            ds,
            data_logE,
            data_sinDec,
            data_mcweight,
            data_physicsweight,
            logE_binning,
            sinDec_binning,
            smoothing_filter,
            kde_smoothing=False,
            numerical_stabilizer=None,
            astro_diff_bkg_file='default',
            **kwargs):
        """Creates a new IceCube energy PDF object for the public data.

        Parameters
        ----------
        ds : instance of Dataset | None
            The instance of Dataset that defines the dataset of the public data.
        data_logE : instance of ndarray
            The 1d ndarray holding the log10(E) values of the events.
        data_sinDec : instance of ndarray
            The 1d ndarray holding the sin(dec) values of the events.
        data_mcweight : instance of ndarray
            The 1d ndarray holding the monte-carlo weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        data_physicsweight : instance of ndarray
            The 1d ndarray holding the physics weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        logE_binning : instance of BinningDefinition
            The binning definition for the log10(E) axis.
        sinDec_binning : instance of BinningDefinition
            The binning definition for the sin(declination) axis.
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        kde_smoothing : bool
            Apply a kde smoothing to the energy pdf for each bin in sin(dec).
            This is useful for signal injections, because it ensures that the
            background is not zero when injecting high energy events.
            Default: False.
        numerical_stabilizer : {None,'numerical','physical'}
            It artificially sets the pdf values in the region where it is not
            originally defined in order to avoid artificially low values.
        astro_diff_bkg_file : None | str
            The file of the 2D histogram with the number of expected muons
            from the astrophisical diffuse background. 'Default' loads the 
            precomputed values for that dataset. See 
            analysis/i3/publicdata_ps/astro_bkg_flux folder for more details.
        """

        super().__init__(
            pmm=None,
            **kwargs)

        # Define the PDF axes.
        self.add_axis(
            PDFAxis(
                name='log_energy',
                vmin=logE_binning.lower_edge,
                vmax=logE_binning.upper_edge))
        self.add_axis(
            PDFAxis(
                name='sin_dec',
                vmin=sinDec_binning.lower_edge,
                vmax=sinDec_binning.upper_edge))

        self.add_binning(logE_binning, 'log_energy')
        self.add_binning(sinDec_binning, 'sin_dec')

        # Create the smoothing method instance tailored to the energy PDF.
        # We will smooth only the first axis (logE).
        if (smoothing_filter is not None) and\
           (not isinstance(smoothing_filter, SmoothingFilter)):
            raise TypeError(
                'The smoothing_filter argument must be None or an instance of '
                'SmoothingFilter! '
                f'Its current type is {classname(smoothing_filter)}!')
        if smoothing_filter is None:
            self.hist_smoothing_method = NoHistSmoothingMethod()
        else:
            self.hist_smoothing_method = NeighboringBinHistSmoothingMethod(
                (smoothing_filter.axis_kernel_array, UNSMOOTH_AXIS))

        if not isinstance(kde_smoothing, bool):
            raise ValueError(
                'The kde_smoothing argument must be an instance of bool! '
                f'Its current type is {classname(kde_smoothing)}!')

        # We have to figure out, which histogram bins are zero due to no
        # monte-carlo coverage, and which due to zero physics model
        # contribution.

        # Create a 2D histogram with only the MC events to determine the MC
        # coverage.
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE,
            data_sinDec,
            bins=[
                logE_binning.binedges,
                sinDec_binning.binedges
            ],
            range=[
                logE_binning.range,
                sinDec_binning.range
            ],
            density=False)
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered = h > 0

        # Select the events which have MC coverage but zero physics
        # contribution, i.e. the physics model predicts zero contribution.
        mask = data_physicsweight == 0.

        # Create a 2D histogram with only the MC events that have zero physics
        # contribution. Note: By construction the zero physics contribution bins
        # are a subset of the MC covered bins.
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE[mask],
            data_sinDec[mask],
            bins=[
                logE_binning.binedges,
                sinDec_binning.binedges
            ],
            range=[
                logE_binning.range,
                sinDec_binning.range
            ],
            density=False)
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered_zero_physics = h > 0

        # Create a 2D histogram with only the data which has physics
        # contribution. We will do the normalization along the logE
        # axis manually.
        data_weights = data_mcweight[~mask] * data_physicsweight[~mask]
        (h, bins_logE, bins_sinDec) = np.histogram2d(
            data_logE[~mask],
            data_sinDec[~mask],
            bins=[
                logE_binning.binedges,
                sinDec_binning.binedges
            ],
            weights=data_weights,
            range=[
                logE_binning.range,
                sinDec_binning.range
            ],
            density=False)
        
        self._hist_logE_sinDec_total = np.copy(h)

        if numerical_stabilizer == 'physical':
            # Add the expected events for background in the bins where it 
            # would have been zero.
            zero_bkg__mask = (h==0)
            if astro_diff_bkg_file == 'default':
                datafile_name = f"astro_diff_mu_signal_2D-{ds.name}.npy"
                datafile_name = impresources.path("skyllh.analyses.i3."
                    "publicdata_ps.astro_bkg_flux",
                    f"astro_diff_mu_signal_2D-{ds.name}.npy")
                hist_astro_diff_bkg = np.load(datafile_name)
            else:
                hist_astro_diff_bkg = np.load(astro_diff_bkg_file)
            h[zero_bkg__mask] += hist_astro_diff_bkg[zero_bkg__mask]
        
        h = self._hist_smoothing_method.smooth(h)
        self.compute_hist_energy_max_min_idx(h,sinDec_binning)
    
        if kde_smoothing:
            # If a bandwidth is passed, apply a KDE-based smoothing with the
            # given bandwidth parameter as bandwidth for the fit.
            kde_pdf_list = []
            data_logE_masked = data_logE[~mask]
            data_sinDec_masked = data_sinDec[~mask]

            for (sindec_lower, sindec_upper) in zip(
                    sinDec_binning.binedges[:-1],
                    sinDec_binning.binedges[1:]):

                sindec_mask = np.logical_and(
                    data_sinDec_masked >= sindec_lower,
                    data_sinDec_masked < sindec_upper
                )
                this_energy = data_logE_masked[sindec_mask]
                if sindec_lower >= 0:
                    kde = gaussian_kde(
                        this_energy,
                        bw_method=self._KDE_BW_NORTH)
                else:
                    kde = gaussian_kde(
                        this_energy,
                        bw_method=self._KDE_BW_SOUTH)
                kde_pdf_list.append(kde.evaluate(logE_binning.bincenters))

            h = np.vstack(kde_pdf_list).T
            h = self._hist_smoothing_method.smooth(h)

        if numerical_stabilizer == 'numerical':
            h = self.artificial_stabilizer(h,logE_binning,
                        self.hist_energy_minima_idx,self.hist_energy_maxima_idx,
                        kde_smoothing=kde_smoothing)

        # Calculate the normalization for each logE bin. Hence we need to sum
        # over the logE bins (axis 0) for each sin(dec) bin and need to divide
        # by the logE bin widths along the sin(dec) bins. The result array norm
        # is a 2D array of the same shape as h.
        norms = np.sum(h, axis=(0,))[np.newaxis, ...] * \
            np.diff(logE_binning.binedges)[..., np.newaxis]
        h /= norms

        self._hist_logE_sinDec = h

    @property
    def hist_smoothing_method(self):
        """The instance of HistSmoothingMethod defining the smoothing filter of
        the energy PDF histogram.
        """
        return self._hist_smoothing_method

    @hist_smoothing_method.setter
    def hist_smoothing_method(self, method):
        if not isinstance(method, HistSmoothingMethod):
            raise TypeError(
                'The hist_smoothing_method property must be an instance of '
                'HistSmoothingMethod! '
                f'Its current type is {classname(method)}!')
        self._hist_smoothing_method = method

    @property
    def hist(self):
        """(read-only) The 2D logE-sinDec histogram array.
        """
        return self._hist_logE_sinDec
    
    @property
    def hist_total_counts(self):
        """(read-only) The 2D logE-sinDec histogram array. Non normalized
        """
        return self._hist_logE_sinDec_total

    @property
    def hist_mask_mc_covered(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage.
        """
        return self._hist_mask_mc_covered

    @property
    def hist_mask_mc_covered_zero_physics(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage but zero physics
        contribution.
        """
        return self._hist_mask_mc_covered_zero_physics

    @property
    def hist_mask_mc_covered_with_physics(self):
        """(read-only) The boolean ndarray holding the mask of the 2D histogram
        bins for which there is monte-carlo coverage and has physics
        contribution.
        """
        return (
            self._hist_mask_mc_covered &
            ~self._hist_mask_mc_covered_zero_physics)
    
    @property
    def hist_energy_minima_idx(self):
        """(read-only) The indices of the lowest energy bin for each sin dec 
        bin for which the original histogram is non-zero. 
        Used for the numerical stabilizer.
        """
        return self._hist_energy_minima_idx
    
    @property
    def hist_energy_maxima_idx(self):
        """(read-only) The indices of the highest energy bin for each sin dec 
        bin for which the original histogram is non-zero.
        Used for the numerical stabilizer.
        """
        return self._hist_energy_maxima_idx
    
    def compute_hist_energy_max_min_idx(self,h,sinDec_binning):
        '''It computes the indices of the highest and lowest energy bin for 
        each sin dec bin for which the original histogram is non-zero.
        '''
        hist_energy_minima_idx = np.zeros_like(sinDec_binning.bincenters,int)
        hist_energy_maxima_idx = np.zeros_like(sinDec_binning.bincenters,int)
        for idx_sin_dec in range(h.shape[1]):
            energy_hist = h[:,idx_sin_dec]
            idx_nonzero_bins = np.nonzero(energy_hist)[0]
            idx_nonzero_low = idx_nonzero_bins[0]
            idx_nonzero_high = idx_nonzero_bins[-1]

            hist_energy_minima_idx[idx_sin_dec] = idx_nonzero_low
            hist_energy_maxima_idx[idx_sin_dec] = idx_nonzero_high
        
        self._hist_energy_minima_idx = hist_energy_minima_idx
        self._hist_energy_maxima_idx = hist_energy_maxima_idx

        pass

    def artificial_stabilizer(self,h,logE_binning,
                              hist_energy_minima_idx,hist_energy_maxima_idx,
                              kde_smoothing):
        '''Auxiliary function to perform an artificial numerical 
        stabilization.It is implemented for the option 
        numerical_stabilizer='numerical'.

            It extends the energy range in which the pdf is defined
        by imposing an exponential decay (in log10 E scale) regulated
        by the parameter _ALPHA=2.0 by default. This avoids 
        artificially large values when injecting events in energy
        regions where there was no bkg data available.
        '''
        for idx_sin_dec in range(h.shape[1]):
            energy_hist = h[:,idx_sin_dec]
            idx_nonzero_low = hist_energy_minima_idx[idx_sin_dec] 
            idx_nonzero_high = hist_energy_maxima_idx[idx_sin_dec] 

            if kde_smoothing:
                if idx_nonzero_low > 0:
                    idx_nonzero_low -= 1
                if idx_nonzero_high < (len(logE_binning.bincenters)-1):
                    idx_nonzero_high += 1
            energy_hist_low = energy_hist[idx_nonzero_low]
            energy_hist_high = energy_hist[idx_nonzero_high]

            logE_low = logE_binning.bincenters[idx_nonzero_low]
            logE_high = logE_binning.bincenters[idx_nonzero_high]
        
            energy_hist[:idx_nonzero_low] = energy_hist_low *\
                self.extend_energy_hist(logE_binning.bincenters[:idx_nonzero_low],
                                logE_low,
                                self._ALPHA)
            
            energy_hist[idx_nonzero_high+1:] = energy_hist_high *\
                self.extend_energy_hist(logE_binning.bincenters[idx_nonzero_high+1:],
                                logE_high,
                                self._ALPHA)
        return h 
    
    def extend_energy_hist(self,log_E,log_E_0,alpha):
        '''Auxiliary function to ensure numerical stability option
        when extending the bkg pdf outside the region with data.
        This version performs an exponential decrease controlled
        by the parameter alpha, which expreses how many orders
        of magnitude the pdf decreases per decade in energy.'''

        return np.power(10,-alpha * np.abs(log_E - log_E_0))

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Pre-compute the probability densitiy values of the trial data,
        which has to be done only once for a particular trial data.
        """

        logE_binning = self.get_binning('log_energy')
        sinDec_binning = self.get_binning('sin_dec')

        logE_idx = np.digitize(
            tdm['log_energy'], logE_binning.binedges) - 1
        sinDec_idx = np.digitize(
            tdm['sin_dec'], sinDec_binning.binedges) - 1

        with TaskTimer(tl, 'Evaluating logE-sinDec histogram.'):
            self._pd = self._hist_logE_sinDec[(logE_idx, sinDec_idx)]

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None):
        """Checks if this energy PDF covers the entire value range of the trail
        data events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events.
            The following data fields need to exist:

                log_energy : float
                    The base-10 logarithm of the reconstructed energy value.
                sin_dec : float
                    The sine of the declination value of the event.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure timing information.

        Raises
        ------
        ValueError
            If parts of the trial data is outside the value range of this
            PDF.
        """
        log10emu = tdm['log_energy']
        log10emu_axis = self.axes['log_energy']
        if np.min(log10emu) < log10emu_axis.vmin:
            raise ValueError(
                f'The minimum log10emu value {np.min(log10emu):g} of the trial '
                'data is lower than the minimum value of the PDF '
                f'{log10emu_axis.vmin:g}!')
        if np.max(log10emu) > log10emu_axis.vmax:
            raise ValueError(
                f'The maximum log10emu value {np.max(log10emu):g} of the trial '
                'data is larger than the maximum value of the PDF '
                f'{log10emu_axis.vmax}:g!')

        sindecmu = tdm['sin_dec']
        sindecmu_axis = self.axes['sin_dec']
        if np.min(sindecmu) < sindecmu_axis.vmin:
            raise ValueError(
                f'The minimum sindecmu value {np.min(sindecmu):g} of the trial '
                'data is lower than the minimum value of the PDF '
                f'{sindecmu_axis.vmin:g}!')
        if np.max(sindecmu) > sindecmu_axis.vmax:
            raise ValueError(
                f'The maximum sindecmu value {np.max(sindecmu):g} of the trial '
                'data is larger than the maximum value of the PDF '
                f'{sindecmu_axis.vmax:g}!')

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Calculates the energy probability density (in 1/log10(E/GeV)) of each
        trial data event.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the data events for which the
            probability should be calculated for. The following data fields must
            exist:

            log_energy : float
                The base-10 logarithm of the energy value of the event.
            sin_dec : float
                The sin(declination) value of the event.

        params_recarray : None
            Unused interface parameter.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of ndarray
            The (N_selected_events,)-shaped numpy ndarray holding the energy
            probability density value for each trial data event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter. By definition this PDF does not
            depend on any fit parameter, hence, this dictionary is empty.
        """
        grads = dict()

        return (self._pd, grads)


class PDDataBackgroundI3EnergyPDF(
        PDBackgroundI3EnergyPDF):
    """This is the IceCube energy background PDF, which gets constructed from
    the experimental data of the public data.
    """

    def __init__(
            self,
            data_exp,
            logE_binning,
            sinDec_binning,
            smoothing_filter=None,
            kde_smoothing=False,
            numerical_stabilizer=None,
            astro_diff_bkg_file='default',
            ds=None,
            **kwargs):
        """Constructs a new IceCube energy background PDF from experimental
        data.

        Parameters
        ----------
        ds : instance of Dataset
            The instance of Dataset that defines the dataset of the public data.
        data_exp : instance of DataFieldRecordArray
            The array holding the experimental data. The following data fields
            must exist:

            log_energy : float
                The base-10 logarithm of the reconstructed energy value of the
                data event.
            sin_dec : float
                The sine of the reconstructed declination of the data event.

        logE_binning : instance of BinningDefinition
            The binning definition for the binning in log10(E).
        sinDec_binning : instance of BinningDefinition
            The binning definition for the sin(declination).
        smoothing_filter : instance of SmoothingFilter | None
            The smoothing filter to use for smoothing the energy histogram.
            If None, no smoothing will be applied.
        kde_smoothing : bool
            Apply a kde smoothing to the energy pdf for each bin in sin(dec).
            This is useful for signal injections, because it ensures that the
            background is not zero when injecting high energy events.
            Default: False.
        numerical_stabilizer : {None,'numerical','physical'}
            It artificially sets the pdf values in the region where it is not
            originally defined in order to avoid artificially low values.
        astro_diff_bkg_file : None | str
            The file of the 2D histogram with the number of expected muons
            from the astrophisical diffuse background. 'Default' loads the 
            precomputed values for that dataset. See 
            analysis/i3/publicdata_ps/astro_bkg_flux folder for more details.
        ds : instance of Dataset | None
            The instance of Dataset that defines the dataset of the public data.
        """
        if not isinstance(data_exp, DataFieldRecordArray):
            raise TypeError(
                'The data_exp argument must be an instance of '
                'DataFieldRecordArray! '
                f'Its current type is {classname(data_exp)}!')

        data_logE = data_exp['log_energy']
        data_sinDec = data_exp['sin_dec']
        # For experimental data, the MC and physics weight are unity.
        data_mcweight = np.ones((len(data_exp),))
        data_physicsweight = data_mcweight

        # Create the energy PDF using the base class.
        super().__init__(
            ds=ds,
            data_logE=data_logE,
            data_sinDec=data_sinDec,
            data_mcweight=data_mcweight,
            data_physicsweight=data_physicsweight,
            logE_binning=logE_binning,
            sinDec_binning=sinDec_binning,
            smoothing_filter=smoothing_filter,
            kde_smoothing=kde_smoothing,
            numerical_stabilizer=numerical_stabilizer,
            astro_diff_bkg_file=astro_diff_bkg_file,
            **kwargs)


class PDMCBackgroundI3EnergyPDF(
        EnergyPDF,
        IsBackgroundPDF,
        UsesBinning):
    """This class provides a background energy PDF constructed from the public
    data and a monte-carlo background flux model.
    """

    def __init__(
            self,
            pdf_log10emu_sindecmu,
            log10emu_binning,
            sindecmu_binning,
            **kwargs):
        """Constructs a new background energy PDF with the given PDF data and
        binning.

        Parameters
        ----------
        pdf_log10emu_sindecmu : instance of numpy ndarray
            The (n_log10emu, n_sindecmu)-shaped 2D numpy ndarray holding the
            PDF values in unit 1/log10(E_mu/GeV).
            A copy of this data will be created and held within this class
            instance.
        log10emu_binning : instance of BinningDefinition
            The binning definition for the binning in log10(E_mu/GeV).
        sindecmu_binning : instance of BinningDefinition
            The binning definition for the binning in sin(dec_mu).
        """
        if not isinstance(pdf_log10emu_sindecmu, np.ndarray):
            raise TypeError(
                'The pdf_log10emu_sindecmu argument must be an instance of '
                'numpy.ndarray!')
        if not isinstance(sindecmu_binning, BinningDefinition):
            raise TypeError(
                'The sindecmu_binning argument must be an instance of '
                'BinningDefinition!')
        if not isinstance(log10emu_binning, BinningDefinition):
            raise TypeError(
                'The log10emu_binning argument must be an instance of '
                'BinningDefinition!')

        super().__init__(
            pmm=None,
            **kwargs)

        self.add_axis(PDFAxis(
            log10emu_binning.name,
            log10emu_binning.lower_edge,
            log10emu_binning.upper_edge,
        ))

        self.add_axis(PDFAxis(
            sindecmu_binning.name,
            sindecmu_binning.lower_edge,
            sindecmu_binning.upper_edge,
        ))

        self._hist_logE_sinDec = np.copy(pdf_log10emu_sindecmu)
        self.add_binning(log10emu_binning, name='log_energy')
        self.add_binning(sindecmu_binning, name='sin_dec')

    def assert_is_valid_for_trial_data(
            self,
            tdm,
            tl=None):
        """Checks if this energy PDF covers the entire value range of the trail
        data events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events.
            The following data fields need to exist:

                log_energy : float
                    The base-10 logarithm of the reconstructed energy value.
                sin_dec : float
                    The sine of the declination value of the event.
        tl : instance of TimeLord | None
            The optional instance of TimeLord to measure timing information.

        Raises
        ------
        ValueError
            If parts of the trial data is outside the value range of this
            PDF.
        """
        log10emu = tdm['log_energy']
        log10emu_axis = self.get_axis(0)
        if np.min(log10emu) < log10emu_axis.vmin:
            raise ValueError(
                f'The minimum log10emu value {np.min(log10emu):g} of the trial '
                'data is lower than the minimum value of the PDF '
                f'{log10emu_axis.vmin:g}!')
        if np.max(log10emu) > log10emu_axis.vmax:
            raise ValueError(
                f'The maximum log10emu value {np.max(log10emu):g} of the trial '
                'data is larger than the maximum value of the PDF '
                f'{log10emu_axis.vmax}:g!')

        sindecmu = tdm['sin_dec']
        sindecmu_axis = self.get_axis(1)
        if np.min(sindecmu) < sindecmu_axis.vmin:
            raise ValueError(
                f'The minimum sindecmu value {np.min(sindecmu):g} of the trial '
                'data is lower than the minimum value of the PDF '
                f'{sindecmu_axis.vmin:g}!')
        if np.max(sindecmu) > sindecmu_axis.vmax:
            raise ValueError(
                f'The maximum sindecmu value {np.max(sindecmu):g} of the trial '
                'data is larger than the maximum value of the PDF '
                f'{sindecmu_axis.vmax:g}!')

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """Gets the probability density for the given trial data events.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial data events.
            The following data fields need to exist:

                log_energy : float
                    The base-10 logarithm of the reconstructed energy value.
                sin_dec : float
                    The sine of the declination value of the event.

        params_recarray : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of ndarray
            The (N_selected_events,)-shaped numpy ndarray holding the
            probability density value for each event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter. By definition this PDF does not
            depend on any fit parameter, hence, this dictionary is empty.
        """
        log10emu = tdm['log_energy']
        sindecmu = tdm['sin_dec']

        log10emu_idxs = np.digitize(
            log10emu, self.get_binning('log_energy').binedges) - 1
        sindecmu_idxs = np.digitize(
            sindecmu, self.get_binning('sin_dec').binedges) - 1

        with TaskTimer(tl, 'Evaluating sindecmu-log10emu PDF.'):
            pd = self._hist_logE_sinDec[(log10emu_idxs, sindecmu_idxs)]

        grads = dict()

        return (pd, grads)
