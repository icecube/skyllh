# -*- coding: utf-8 -*-

import numpy as np

from scipy import (
    integrate,
    interpolate,
)

from skyllh.analyses.i3.publicdata_ps.utils import (
    FctSpline2D,
)
from skyllh.core.binning import (
    get_bin_indices_from_lower_and_upper_binedges,
    get_bincenters_from_binedges,
)
from skyllh.core.storage import (
    create_FileLoader,
)


def load_effective_area_array(pathfilenames):
    """Loads the (nbins_decnu, nbins_log10enu)-shaped 2D effective
    area array from the given data file.

    Parameters
    ----------
    pathfilename : str | list of str
        The file name of the data file.

    Returns
    -------
    aeff_decnu_log10enu : (nbins_decnu, nbins_log10enu)-shaped 2D ndarray
        The ndarray holding the effective area for each
        (dec_nu,log10(E_nu/GeV)) bin.
    decnu_binedges_lower : (nbins_decnu,)-shaped ndarray
        The ndarray holding the lower bin edges of the dec_nu axis.
    decnu_binedges_upper : (nbins_decnu,)-shaped ndarray
        The ndarray holding the upper bin edges of the dec_nu axis.
    log10_enu_binedges_lower : (nbins_log10enu,)-shaped ndarray
        The ndarray holding the lower bin edges of the log10(E_nu/GeV) axis.
    log10_enu_binedges_upper : (nbins_log10enu,)-shaped ndarray
        The ndarray holding the upper bin edges of the log10(E_nu/GeV) axis.
    """
    loader = create_FileLoader(pathfilenames=pathfilenames)
    data = loader.load_data()
    renaming_dict = {
        'log10(E_nu/GeV)_min': 'log10_enu_min',
        'log10(E_nu/GeV)_max': 'log10_enu_max',
        'Dec_nu_min[deg]':     'decnu_min',
        'Dec_nu_max[deg]':     'decnu_max',
        'A_Eff[cm^2]':         'a_eff'
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
            'effective area from data file "{}". The number of lower and upper '
            'bin edges is not equal!'.format(str(loader.pathfilename_list)))
    if len(decnu_binedges_lower) != len(decnu_binedges_upper):
        raise ValueError(
            'Cannot extract the dec_nu binning of the effective '
            'area from data file "{}". The number of lower and upper bin edges '
            'is not equal!'.format(str(loader.pathfilename_list)))

    nbins_log10_enu = len(log10_enu_binedges_lower)
    nbins_decnu = len(decnu_binedges_lower)

    # Construct the 2d array for the effective area.
    aeff_decnu_log10enu = np.zeros(
        (nbins_decnu, nbins_log10_enu), dtype=np.double)

    decnu_idx = np.digitize(
        0.5*(data['decnu_min'] +
             data['decnu_max']),
        decnu_binedges_lower) - 1
    log10enu_idx = np.digitize(
        0.5*(data['log10_enu_min'] +
             data['log10_enu_max']),
        log10_enu_binedges_lower) - 1

    aeff_decnu_log10enu[decnu_idx, log10enu_idx] = data['a_eff']

    return (
        aeff_decnu_log10enu,
        decnu_binedges_lower,
        decnu_binedges_upper,
        log10_enu_binedges_lower,
        log10_enu_binedges_upper
    )


class PDAeff(object):
    """This class provides a representation of the effective area provided by
    the public data.
    """
    def __init__(
            self, pathfilenames, src_dec=None,
            min_log10enu=None, max_log10enu=None,
            **kwargs):
        """Creates an effective area instance by loading the effective area
        data from the given file.

        Parameters
        ----------
        pathfilenames : str | list of str
            The path file names of the effective area data file(s) which should
            be used for this public data effective area instance.
        src_dec : float | None
            The source declination in radians for which detection probabilities
            should get pre-calculated using the ``get_detection_prob_for_decnu``
            method.
        min_log10enu : float | None
            The minimum log10(E_nu/GeV) value that should be used for
            calculating the detection probability.
            If None, the lowest available neutrino energy bin edge of the
            effective area is used.
        max_log10enu : float | None
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
            self._log10_enu_binedges_upper
        ) = load_effective_area_array(pathfilenames)

        # Note: self._aeff_decnu_log10enu is numpy 2D ndarray of shape
        # (nbins_decnu, nbins_log10enu).

        # Cut the energies where all effective areas are zero.
        m = np.sum(self._aeff_decnu_log10enu, axis=0) > 0
        self._aeff_decnu_log10enu = self._aeff_decnu_log10enu[:, m]
        self._log10_enu_binedges_lower = self._log10_enu_binedges_lower[m]
        self._log10_enu_binedges_upper = self._log10_enu_binedges_upper[m]

        self._decnu_binedges = np.concatenate(
            (self._decnu_binedges_lower,
             self._decnu_binedges_upper[-1:])
        )
        self._log10_enu_binedges = np.concatenate(
            (self._log10_enu_binedges_lower,
             self._log10_enu_binedges_upper[-1:])
        )

        # Pre-calculate detection probabilities for a certain neutrino
        # declination if requested.
        if src_dec is not None:
            if min_log10enu is None:
                min_log10enu = self._log10_enu_binedges_lower[0]
            else:
                min_log10enu = max(
                    self._log10_enu_binedges_lower[0],
                    min_log10enu)

            if max_log10enu is None:
                max_log10enu = self._log10_enu_binedges_upper[-1]
            else:
                max_log10enu = min(
                    self._log10_enu_binedges_upper[-1],
                    max_log10enu)

            m = (
                (self.log10_enu_bincenters >= min_log10enu) &
                (self.log10_enu_bincenters < max_log10enu)
            )
            low_bin_edges = self._log10_enu_binedges_lower[m]
            high_bin_edges = self._log10_enu_binedges_upper[m]

            # Get the detection probability P(E_nu | sin(dec)) per bin.
            self.det_prob = self.get_detection_prob_for_decnu(
                src_dec,
                10**low_bin_edges,
                10**high_bin_edges,
                10**low_bin_edges[0],
                10**high_bin_edges[-1]
            )

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
    def n_decnu_bins(self):
        """(read-only) The number of bins of the neutrino declination axis.
        """
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
    def n_log10_enu_bins(self):
        """(read-only) The number of bins of the log10 neutrino energy axis.
        """
        return len(self._log10_enu_binedges) - 1

    @property
    def aeff_decnu_log10enu(self):
        """(read-only) The effective area in cm^2 as (n_decnu,n_log10enu)-shaped
        2D numpy ndarray.
        """
        return self._aeff_decnu_log10enu

    def create_sin_decnu_log10_enu_spline(self):
        """DEPRECATED!
        Creates a FctSpline2D object representing a 2D spline of the
        effective area in sin(dec_nu)-log10(E_nu/GeV)-space.

        Returns
        -------
        spl : FctSpline2D instance
            The FctSpline2D instance representing a spline in the
            sin(dec_nu)-log10(E_nu/GeV)-space.
        """
        spl = FctSpline2D(
            self._aeff_decnu_log10enu,
            self.sin_decnu_binedges,
            self.log10_enu_binedges
        )
        return spl

    def get_aeff_for_decnu(self, decnu):
        """Retrieves the effective area as function of log10_enu.

        Parameters
        ----------
        decnu : float
            The true neutrino declination.

        Returns
        -------
        aeff : (n,)-shaped numpy ndarray
            The effective area in cm^2 for the given true neutrino declination
            as a function of log10 true neutrino energy.
        """
        decnu_idx = np.digitize(decnu, self._decnu_binedges) - 1

        aeff = self._aeff_decnu_log10enu[decnu_idx]

        return aeff

    def get_detection_prob_for_decnu(
            self, decnu, enu_min, enu_max, enu_range_min, enu_range_max):
        """Calculates the detection probability for given true neutrino energy
        ranges for a given neutrino declination.

        Parameters
        ----------
        decnu : float
            The neutrino declination in radians.
        enu_min : float | ndarray of float
            The minimum energy in GeV.
        enu_max : float | ndarray of float
            The maximum energy in GeV.
        enu_range_min : float
            The minimum energy in GeV of the entire energy range.
        enu_range_max : float
            The maximum energy in GeV of the entire energy range.

        Returns
        -------
        det_prob : ndarray of float
            The neutrino energy detection probabilities for the given true
            enegry ranges.
        """
        enu_binedges = np.power(10, self.log10_enu_binedges)

        # Get the bin indices for the lower and upper energy range values.
        (lidx,) = get_bin_indices_from_lower_and_upper_binedges(
            enu_binedges[:-1],
            enu_binedges[1:],
            np.array([enu_range_min])
        )
        if enu_range_max >= enu_binedges[-1]:
            uidx = len(enu_binedges)-1
        else:
            (uidx,) = get_bin_indices_from_lower_and_upper_binedges(
                enu_binedges[:-1],
                enu_binedges[1:],
                np.array([enu_range_max])
            )
            # Note: The get_bin_indices_from_lower_and_upper_binedges function
            #       is based on the lower edges. So by definition the upper bin
            #       index is one too large.
            # uidx -= 1

        aeff = self.get_aeff_for_decnu(decnu)
        aeff = aeff[lidx:uidx]
        enu_binedges = enu_binedges[lidx:uidx+1]

        dE = np.diff(enu_binedges)

        daeff_dE = aeff / dE

        # Create a spline representation that spans the entire enu range.
        x = np.empty((len(enu_binedges)+1,), dtype=np.double)
        x[0] = enu_binedges[0]
        x[1:-1] = get_bincenters_from_binedges(enu_binedges)
        x[-1] = enu_binedges[-1]

        y = np.empty((len(enu_binedges)+1,), dtype=np.double)
        y[0] = daeff_dE[0]
        y[1:-1] = daeff_dE
        y[-1] = daeff_dE[-1]

        spl = interpolate.splrep(
            x,
            y,
            k=1,
            s=0
        )

        def _eval_spl_func(x):
            return interpolate.splev(x, spl, der=0, ext=1)

        norm = integrate.quad(
            _eval_spl_func,
            enu_range_min,
            enu_range_max,
            limit=200,
            full_output=1
        )[0]

        enu_min = np.atleast_1d(enu_min)
        enu_max = np.atleast_1d(enu_max)

        det_prob = np.empty((len(enu_min),), dtype=np.double)
        for i in range(len(enu_min)):
            integral = integrate.quad(
                _eval_spl_func,
                enu_min[i],
                enu_max[i],
                limit=200,
                full_output=1
            )[0]

            det_prob[i] = integral / norm

        return det_prob
