from collections.abc import Sequence

import numpy as np

from skyllh.core.binning import (
    get_bin_indices_from_binedges,
    get_bincenters_from_binedges,
)
from skyllh.core.storage import (
    create_FileLoader,
)


def load_effective_area_array(
    pathfilenames: str | Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the (nbins_decnu, nbins_log10enu)-shaped 2D effective
    area array from the given data file.

    Parameters
    ----------
    pathfilenames
        The file name(s) of the data file(s).

    Returns
    -------
    aeff_decnu_log10enu
        The ndarray holding the effective area for each
        (dec_nu,log10(E_nu/GeV)) bin.
    decnu_binedges_lower
        The ndarray holding the lower bin edges of the dec_nu axis.
    decnu_binedges_upper
        The ndarray holding the upper bin edges of the dec_nu axis.
    log10_enu_binedges_lower
        The ndarray holding the lower bin edges of the log10(E_nu/GeV) axis.
    log10_enu_binedges_upper
        The ndarray holding the upper bin edges of the log10(E_nu/GeV) axis.
    """
    loader = create_FileLoader(pathfilenames=pathfilenames)
    data = loader.load_data()
    renaming_dict = {
        'log10(E_nu/GeV)_min': 'log10_enu_min',
        'log10(E_nu/GeV)_max': 'log10_enu_max',
        'Dec_nu_min[deg]': 'decnu_min',
        'Dec_nu_max[deg]': 'decnu_max',
        'A_Eff[cm^2]': 'a_eff',
    }
    data.rename_fields(renaming_dict, must_exist=True)

    # Convert the true neutrino declination from degrees to radians.
    data['decnu_min'] = np.deg2rad(data['decnu_min'])
    data['decnu_max'] = np.deg2rad(data['decnu_max'])

    # Determine the binning for energy and declination.
    log10_enu_binedges_lower = np.unique(data['log10_enu_min'])
    log10_enu_binedges_upper = np.unique(data['log10_enu_max'])
    decnu_binedges_lower = np.unique(data['decnu_min'])
    decnu_binedges_upper = np.unique(data['decnu_max'])

    if len(log10_enu_binedges_lower) != len(log10_enu_binedges_upper):
        raise ValueError(
            'Cannot extract the log10(E/GeV) binning of the '
            f'effective area from data file "{loader.pathfilename_list!s}". The number of lower and upper '
            'bin edges is not equal!'
        )
    if len(decnu_binedges_lower) != len(decnu_binedges_upper):
        raise ValueError(
            'Cannot extract the dec_nu binning of the effective '
            f'area from data file "{loader.pathfilename_list!s}". The number of lower and upper bin edges '
            'is not equal!'
        )

    nbins_log10_enu = len(log10_enu_binedges_lower)
    nbins_decnu = len(decnu_binedges_lower)

    # Construct the 2d array for the effective area.
    aeff_decnu_log10enu = np.zeros((nbins_decnu, nbins_log10_enu), dtype=np.double)

    decnu_binedges = np.append(decnu_binedges_lower, decnu_binedges_upper[-1])
    log10_enu_binedges = np.append(log10_enu_binedges_lower, log10_enu_binedges_upper[-1])

    decnu_idx = get_bin_indices_from_binedges(decnu_binedges, 0.5 * (data['decnu_min'] + data['decnu_max']))
    log10enu_idx = get_bin_indices_from_binedges(
        log10_enu_binedges, 0.5 * (data['log10_enu_min'] + data['log10_enu_max'])
    )

    aeff_decnu_log10enu[decnu_idx, log10enu_idx] = data['a_eff']

    return (
        aeff_decnu_log10enu,
        decnu_binedges_lower,
        decnu_binedges_upper,
        log10_enu_binedges_lower,
        log10_enu_binedges_upper,
    )


class PDAeff:
    """This class provides a representation of the effective area provided by
    the public data.
    """

    def __init__(
        self,
        pathfilenames: str | list[str],
        src_dec: float | None = None,
        min_log10enu: float | None = None,
        max_log10enu: float | None = None,
        **kwargs,
    ):
        """Creates an effective area instance by loading the effective area
        data from the given file.

        Parameters
        ----------
        pathfilenames
            The path file name(s) of the effective area data file(s) which should
            be used for this public data effective area instance.
        src_dec
            The source declination in radians for which detection probabilities
            should get pre-calculated using the ``get_detection_prob_for_decnu``
            method.
        min_log10enu
            The minimum log10(E_nu/GeV) value that should be used for
            calculating the detection probability.
            If None, the lowest available neutrino energy bin edge of the
            effective area is used.
        max_log10enu
            The maximum log10(E_nu/GeV) value that should be used for
            calculating the detection probability.
            If None, the highest available neutrino energy bin edge of the
            effective area is used.
        """
        super().__init__(**kwargs)

        (
            self._aeff_decnu_log10enu,
            self._decnu_binedges_lower,
            self._decnu_binedges_upper,
            self._log10_enu_binedges_lower,
            self._log10_enu_binedges_upper,
        ) = load_effective_area_array(pathfilenames)

        # Note: self._aeff_decnu_log10enu is numpy 2D ndarray of shape
        # (nbins_decnu, nbins_log10enu).

        # Cut the energies where all effective areas are zero.
        m = np.sum(self._aeff_decnu_log10enu, axis=0) > 0
        self._aeff_decnu_log10enu = self._aeff_decnu_log10enu[:, m]
        self._log10_enu_binedges_lower = self._log10_enu_binedges_lower[m]
        self._log10_enu_binedges_upper = self._log10_enu_binedges_upper[m]

        self._decnu_binedges = np.concatenate((self._decnu_binedges_lower, self._decnu_binedges_upper[-1:]))
        self._log10_enu_binedges = np.concatenate((self._log10_enu_binedges_lower, self._log10_enu_binedges_upper[-1:]))

        # Pre-calculate detection probabilities for a certain neutrino
        # declination if requested.
        if src_dec is not None:
            lowest = float(self._log10_enu_binedges_lower[0])
            highest = float(self._log10_enu_binedges_upper[-1])
            range_min = lowest if min_log10enu is None else max(lowest, min_log10enu)
            range_max = highest if max_log10enu is None else min(highest, max_log10enu)

            m = self.get_log10_enu_bin_mask(range_min, range_max)
            low_bin_edges = self._log10_enu_binedges_lower[m]
            high_bin_edges = self._log10_enu_binedges_upper[m]

            # Get the detection probability P(E_nu | sin(dec)) per whole bin,
            # also for the bins the energy range cuts.
            self.det_prob = self.get_detection_prob_for_decnu(
                src_dec, 10**low_bin_edges, 10**high_bin_edges, 10 ** low_bin_edges[0], 10 ** high_bin_edges[-1]
            )

    def get_log10_enu_bin_mask(self, min_log10enu: float, max_log10enu: float) -> np.ndarray:
        """Returns the mask of the energy bins that overlap the range
        [``min_log10enu``, ``max_log10enu``], including bins the range only
        partly covers.
        """
        return (self._log10_enu_binedges_upper > min_log10enu) & (self._log10_enu_binedges_lower < max_log10enu)

    @property
    def decnu_binedges(self):
        """(read-only) The bin edges of the neutrino declination axis in
        radians.
        """
        return self._decnu_binedges

    @property
    def sin_decnu_binedges(self):
        """(read-only) The sin of the bin edges of the neutrino declination
        in radians.
        """
        return np.sin(self._decnu_binedges)

    @property
    def decnu_bincenters(self):
        """(read-only) The bin center values of the neutrino declination axis in
        radians.
        """
        return get_bincenters_from_binedges(self._decnu_binedges)

    @property
    def n_decnu_bins(self) -> int:
        """(read-only) The number of bins of the neutrino declination axis."""
        return len(self._decnu_binedges) - 1

    @property
    def log10_enu_binedges(self):
        """(read-only) The bin edges of the log10(E_nu/GeV) neutrino energy
        axis.
        """
        return self._log10_enu_binedges

    @property
    def log10_enu_binedges_lower(self):
        """(read-only) The lower binedges of the log10(E_nu/GeV) neutrino energy
        axis.
        """
        return self._log10_enu_binedges_lower

    @property
    def log10_enu_binedges_upper(self):
        """(read-only) The upper binedges of the log10(E_nu/GeV) neutrino energy
        axis.
        """
        return self._log10_enu_binedges_upper

    @property
    def log10_enu_bincenters(self):
        """(read-only) The bin center values of the log10(E_nu/GeV) neutrino
        energy axis.
        """
        return get_bincenters_from_binedges(self._log10_enu_binedges)

    @property
    def n_log10_enu_bins(self) -> int:
        """(read-only) The number of bins of the log10 neutrino energy axis."""
        return len(self._log10_enu_binedges) - 1

    @property
    def aeff_decnu_log10enu(self):
        """(read-only) The effective area in cm^2 as (n_decnu,n_log10enu)-shaped
        2D numpy ndarray.
        """
        return self._aeff_decnu_log10enu

    def get_decnu_idx(self, decnu: float) -> int:
        """Returns the index of the true neutrino declination bin for the given declination.

        Parameters
        ----------
        decnu
            The true neutrino declination.

        Returns
        -------
        decnu_idx
            The index of the true neutrino declination bin.

        Raises
        ------
        ValueError
            If the given declination is outside the declination binning of the
            effective area.
        """
        if (decnu < self._decnu_binedges[0]) or (decnu > self._decnu_binedges[-1]):
            raise ValueError(f'The declination {decnu} radians is not supported by the effective area!')

        decnu_idx = get_bin_indices_from_binedges(self._decnu_binedges, decnu)

        return int(decnu_idx)

    def get_aeff_for_decnu(self, decnu: float) -> np.ndarray:
        """Retrieves the effective area as function of log10_enu.

        Parameters
        ----------
        decnu
            The true neutrino declination.

        Returns
        -------
        aeff
            The effective area in cm^2 for the given true neutrino declination
            as a function of log10 true neutrino energy.
        """
        aeff = self._aeff_decnu_log10enu[self.get_decnu_idx(decnu)]

        return aeff

    def get_detection_prob_for_decnu(
        self,
        decnu: float,
        enu_min: float | np.ndarray,
        enu_max: float | np.ndarray,
        enu_range_min: float,
        enu_range_max: float,
    ) -> np.ndarray:
        """Calculates the detection probability for given true neutrino energy
        ranges for a given neutrino declination.

        Parameters
        ----------
        decnu
            The neutrino declination in radians.
        enu_min
            The minimum energy in GeV.
        enu_max
            The maximum energy in GeV.
        enu_range_min
            The minimum energy in GeV of the entire energy range.
        enu_range_max
            The maximum energy in GeV of the entire energy range.

        Returns
        -------
        det_prob
            The neutrino energy detection probabilities for the given true
            enegry ranges.
        """
        # The effective area is a histogram: constant within each energy bin.
        # Integrating it as such keeps the probability of a whole bin equal to
        # its share of the effective area, as the flux weights assume, and
        # includes the part of a bin that a range ends in.
        enu_binedges = np.power(10, self.log10_enu_binedges)
        daeff_dE = self.get_aeff_for_decnu(decnu) / np.diff(enu_binedges)

        def integral(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
            overlap = np.clip(
                np.minimum(e2[:, np.newaxis], enu_binedges[np.newaxis, 1:])
                - np.maximum(e1[:, np.newaxis], enu_binedges[np.newaxis, :-1]),
                0,
                None,
            )
            return np.sum(overlap * daeff_dE[np.newaxis, :], axis=1)

        norm = integral(np.atleast_1d(enu_range_min), np.atleast_1d(enu_range_max))[0]

        return integral(np.atleast_1d(enu_min), np.atleast_1d(enu_max)) / norm
