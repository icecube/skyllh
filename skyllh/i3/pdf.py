import numpy as np

from skyllh.core.binning import BinningDefinition, UsesBinning
from skyllh.core.logging import (
    get_logger,
)
from skyllh.core.parameters import ParameterModelMapper
from skyllh.core.pdf import (
    EnergyPDF,
    PDFAxis,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.smoothing import (
    UNSMOOTH_AXIS,
    HistSmoothingMethod,
    NeighboringBinHistSmoothingMethod,
    NoHistSmoothingMethod,
    SmoothingFilter,
)
from skyllh.core.timing import TaskTimer, TimeLord
from skyllh.core.trialdata import TrialDataManager

logger = get_logger(__name__)


class I3EnergyPDF(
    EnergyPDF,
    UsesBinning,
):
    """This is the base class for all IceCube specific energy PDF models.
    IceCube energy PDFs depend soley on the energy and the
    zenith angle, and hence, on the declination of the event.

    The IceCube energy PDF is modeled as a 1d histogram in energy,
    but for different sin(declination) bins, hence, stored as a 2d histogram.
    """

    def __init__(
        self,
        pmm: ParameterModelMapper | None,
        data_log10_energy: np.ndarray,
        data_sin_dec: np.ndarray,
        data_mcweight: np.ndarray,
        data_physicsweight: np.ndarray,
        log10_energy_binning: BinningDefinition,
        sin_dec_binning: BinningDefinition,
        smoothing_filter: SmoothingFilter | None,
        **kwargs,
    ):
        """Creates a new IceCube energy PDF object.

        Parameters
        ----------
        pmm
            The instance of ParameterModelMapper defining the global parameters
            and their mapping to local model/source parameters.
            It can be ``None``, if the PDF does not depend on any parameters.
        data_log10_energy
            The array holding the log10(E) values of the events.
        data_sin_dec
            The array holding the sin(dec) values of the events.
        data_mcweight
            The array holding the monte-carlo weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        data_physicsweight
            The array holding the physics weights of the events.
            The final data weight will be the product of data_mcweight and
            data_physicsweight.
        log10_energy_binning
            The binning definition for the log10(E) axis.
        sin_dec_binning
            The binning definition for the sin(declination) axis.
        smoothing_filter
            The smoothing filter to use for smoothing the energy histogram.
            If ``None``, no smoothing will be applied.
        """
        super().__init__(pmm=pmm, **kwargs)

        # Define the PDF axes.
        self.add_axis(
            PDFAxis(name='log_energy', vmin=log10_energy_binning.lower_edge, vmax=log10_energy_binning.upper_edge)
        )
        self.add_axis(PDFAxis(name='sin_dec', vmin=sin_dec_binning.lower_edge, vmax=sin_dec_binning.upper_edge))

        self.add_binning(log10_energy_binning, 'log_energy')
        self.add_binning(sin_dec_binning, 'sin_dec')

        # Create the smoothing method instance tailored to the energy PDF.
        # We will smooth only the first axis (log10(E)).
        if (smoothing_filter is not None) and (not isinstance(smoothing_filter, SmoothingFilter)):
            raise TypeError(
                'The smoothing_filter argument must be None or an instance of '
                f'SmoothingFilter! It is of type {classname(smoothing_filter)}'
            )
        if smoothing_filter is None:
            self.hist_smoothing_method = NoHistSmoothingMethod()
        else:
            self.hist_smoothing_method = NeighboringBinHistSmoothingMethod(
                (smoothing_filter.axis_kernel_array, UNSMOOTH_AXIS)
            )

        # We have to figure out, which histogram bins are zero due to no
        # monte-carlo coverage, and which due to zero physics model
        # contribution.

        # Create a 2D histogram with only the MC events to determine the MC
        # coverage.
        (h, _, _) = np.histogram2d(
            data_log10_energy,
            data_sin_dec,
            bins=[log10_energy_binning.binedges, sin_dec_binning.binedges],
            range=[log10_energy_binning.range, sin_dec_binning.range],
            density=False,
        )
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered = h > 0

        # Select the events which have MC coverage but zero physics
        # contribution, i.e. the physics model predicts zero contribution.
        mask = data_physicsweight == 0.0

        # Create a 2D histogram with only the MC events that have zero physics
        # contribution. Note: By construction the zero physics contribution bins
        # are a subset of the MC covered bins.
        (h, _, _) = np.histogram2d(
            data_log10_energy[mask],
            data_sin_dec[mask],
            bins=[log10_energy_binning.binedges, sin_dec_binning.binedges],
            range=[log10_energy_binning.range, sin_dec_binning.range],
            density=False,
        )
        h = self._hist_smoothing_method.smooth(h)
        self._hist_mask_mc_covered_zero_physics = h > 0

        # Create a 2D histogram with only the data which has physics
        # contribution. We will do the normalization along the logE
        # axis manually.
        data_weights = data_mcweight[~mask] * data_physicsweight[~mask]
        (h, _, _) = np.histogram2d(
            data_log10_energy[~mask],
            data_sin_dec[~mask],
            bins=[log10_energy_binning.binedges, sin_dec_binning.binedges],
            weights=data_weights,
            range=[log10_energy_binning.range, sin_dec_binning.range],
            density=False,
        )

        # Calculate the normalization for each logE bin. Hence we need to sum
        # over the logE bins (axis 0) for each sin(dec) bin and need to divide
        # by the logE bin widths along the sin(dec) bins. The result array norm
        # is a 2D array of the same shape as h.
        norms = np.sum(h, axis=(0,))[np.newaxis, ...] * np.diff(log10_energy_binning.binedges)[..., np.newaxis]
        h /= norms
        h = self._hist_smoothing_method.smooth(h)

        self._hist_log10_energy_sin_dec = h

    @property
    def hist_smoothing_method(self):
        """The HistSmoothingMethod instance defining the smoothing filter of the
        energy PDF histogram.
        """
        return self._hist_smoothing_method

    @hist_smoothing_method.setter
    def hist_smoothing_method(self, method):
        if not isinstance(method, HistSmoothingMethod):
            raise TypeError(
                'The hist_smoothing_method property must be an instance of '
                f'HistSmoothingMethod! It is of type {classname(method)}'
            )
        self._hist_smoothing_method = method

    @property
    def hist(self):
        """(read-only) The 2D logE-sinDec histogram array."""
        return self._hist_log10_energy_sin_dec

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
        mask = self._hist_mask_mc_covered & ~self._hist_mask_mc_covered_zero_physics
        return mask

    def assert_is_valid_for_trial_data(self, tdm: TrialDataManager, tl: TimeLord | None = None, **kwargs):
        """Checks if this energy PDF is valid for all the given trial events.
        It checks if all the data is within the log10(E) and sin(dec) binning
        range.

        Parameters
        ----------
        tdm
            The instance of TrialDataManager holding the trial data events.
            The following data fields must exist:

            log_energy
                The base-10 logarithm of the energy value of the data event.
            dec
                The declination of the data event.

        tl
            The optional instance of TimeLord for measuring timing information.

        Raises
        ------
        ValueError
            If some of the data is outside the log10(E) or sin(dec) binning
            range.
        """
        log10_energy_binning = self.get_binning('log_energy')
        sin_dec_binning = self.get_binning('sin_dec')

        data_log10_energy = tdm['log_energy']
        data_sin_dec = np.sin(tdm['dec'])

        if log10_energy_binning.any_data_out_of_range(data_log10_energy):
            oor_data = log10_energy_binning.get_out_of_range_data(data_log10_energy)
            raise ValueError(
                'Some data is outside the log10(E) range '
                f'({log10_energy_binning.lower_edge:.3f},'
                f' {log10_energy_binning.upper_edge:.3f})! '
                f'The following data values are out of range: {oor_data}'
            )

        if sin_dec_binning.any_data_out_of_range(data_sin_dec):
            oor_data = sin_dec_binning.get_out_of_range_data(data_sin_dec)
            raise ValueError(
                'Some data is outside the sin(dec) range '
                f'({sin_dec_binning.lower_edge:.3f},'
                f' {sin_dec_binning.upper_edge:.3f})! '
                f'The following data values are out of range: {oor_data}'
            )

    def get_pd(  # type: ignore[override]
        self, tdm: TrialDataManager, params_recarray: None = None, tl: TimeLord | None = None
    ) -> tuple[np.ndarray, dict]:
        """Calculates the energy probability density of each event.

        Parameters
        ----------
        tdm
            The TrialDataManager instance holding the data events for which the
            probability density should be calculated.
            The following data fields must exist:

            log_energy
                The base-10 logarithm of the energy value of the event.
            sin_dec
                The sin(declination) value of the event.

        params_recarray
            Unused interface parameter.
        tl
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd
            The 1D (N_events,)-shaped numpy ndarray with the energy probability
            density for each event.
        grads
            The dictionary holding the gradients of the probability density
            w.r.t. each fit parameter. The key of the dictionary is the id
            of the global fit parameter. Because this energy PDF does not depend
            on any fit parameters, an empty dictionary is returned.
        """
        log10_energy_binning = self.get_binning('log_energy')
        sin_dec_binning = self.get_binning('sin_dec')

        log10_energy_idx = np.digitize(tdm['log_energy'], log10_energy_binning.binedges) - 1
        sin_dec_idx = np.digitize(tdm['sin_dec'], sin_dec_binning.binedges) - 1

        with TaskTimer(tl, 'Evaluating log10_energy-sin_dec histogram.'):
            pd = self._hist_log10_energy_sin_dec[(log10_energy_idx, sin_dec_idx)]

        return (pd, dict())
