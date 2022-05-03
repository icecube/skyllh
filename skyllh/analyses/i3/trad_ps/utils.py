# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.binning import (
    get_bincenters_from_binedges
)
from skyllh.core.storage import create_FileLoader


def load_effective_area_array(pathfilenames):
    """Loads the (nbins_sin_true_dec, nbins_log_true_e)-shaped 2D effective
    area array from the given data file.

    Parameters
    ----------
    pathfilename : str | list of str
        The file name of the data file.

    Returns
    -------
    arr : (nbins_sin_true_dec, nbins_log_true_e)-shaped 2D ndarray
        The ndarray holding the effective area for each
        sin(dec_true),log(e_true) bin.
    sin_true_dec_binedges_lower : (nbins_sin_true_dec,)-shaped ndarray
        The ndarray holding the lower bin edges of the sin(dec_true) axis.
    sin_true_dec_binedges_upper : (nbins_sin_true_dec,)-shaped ndarray
        The ndarray holding the upper bin edges of the sin(dec_true) axis.
    log_true_e_binedges_lower : (nbins_log_true_e,)-shaped ndarray
        The ndarray holding the lower bin edges of the log(E_true) axis.
    log_true_e_binedges_upper : (nbins_log_true_e,)-shaped ndarray
        The ndarray holding the upper bin edges of the log(E_true) axis.
    """
    loader = create_FileLoader(pathfilenames=pathfilenames)
    data = loader.load_data()
    renaming_dict = {
        'log10(E_nu/GeV)_min': 'log_true_e_min',
        'log10(E_nu/GeV)_max': 'log_true_e_max',
        'Dec_nu_min[deg]':     'sin_true_dec_min',
        'Dec_nu_max[deg]':     'sin_true_dec_max',
        'A_Eff[cm^2]':         'a_eff'
    }
    data.rename_fields(renaming_dict, must_exist=True)

    # Convert the true neutrino declination from degrees to radians and into
    # sin values.
    data['sin_true_dec_min'] = np.sin(np.deg2rad(
        data['sin_true_dec_min']))
    data['sin_true_dec_max'] = np.sin(np.deg2rad(
        data['sin_true_dec_max']))

    # Determine the binning for energy and declination.
    log_true_e_binedges_lower = np.unique(
        data['log_true_e_min'])
    log_true_e_binedges_upper = np.unique(
        data['log_true_e_max'])
    sin_true_dec_binedges_lower = np.unique(
        data['sin_true_dec_min'])
    sin_true_dec_binedges_upper = np.unique(
        data['sin_true_dec_max'])

    if(len(log_true_e_binedges_lower) != len(log_true_e_binedges_upper)):
        raise ValueError('Cannot extract the log10(E/GeV) binning of the '
            'effective area from data file "{}". The number of lower and upper '
            'bin edges is not equal!'.format(str(loader.pathfilename_list)))
    if(len(sin_true_dec_binedges_lower) != len(sin_true_dec_binedges_upper)):
        raise ValueError('Cannot extract the Dec_nu binning of the effective '
            'area from data file "{}". The number of lower and upper bin edges '
            'is not equal!'.format(str(loader.pathfilename_list)))

    nbins_log_true_e = len(log_true_e_binedges_lower)
    nbins_sin_true_dec = len(sin_true_dec_binedges_lower)

    # Construct the 2d array for the effective area.
    arr = np.zeros((nbins_sin_true_dec, nbins_log_true_e), dtype=np.double)

    sin_true_dec_idx = np.digitize(
        0.5*(data['sin_true_dec_min'] +
             data['sin_true_dec_max']),
            sin_true_dec_binedges_lower) - 1
    log_true_e_idx = np.digitize(
        0.5*(data['log_true_e_min'] +
             data['log_true_e_max']),
        log_true_e_binedges_lower) - 1

    arr[sin_true_dec_idx, log_true_e_idx] = data['a_eff']

    return (
        arr,
        sin_true_dec_binedges_lower,
        sin_true_dec_binedges_upper,
        log_true_e_binedges_lower,
        log_true_e_binedges_upper
    )


def load_smearing_histogram(pathfilenames):
    """Loads the 5D smearing histogram from the given data file.

    Parameters
    ----------
    pathfilenames : str | list of str
        The file name of the data file.

    Returns
    -------
    histogram : 5d ndarray
        The 5d histogram array holding the probability values of the smearing
        matrix.
        The axes are (true_e, true_dec, reco_e, psi, ang_err).
    true_e_bin_edges : 1d ndarray
        The ndarray holding the bin edges of the true energy axis.
    true_dec_bin_edges : 1d ndarray
        The ndarray holding the bin edges of the true declination axis in
        radians.
    reco_e_lower_edges : 3d ndarray
        The 3d ndarray holding the lower bin edges of the reco energy axis.
        For each pair of true_e and true_dec different reco energy bin edges
        are provided.
        The shape is (n_true_e, n_true_dec, n_reco_e).
    reco_e_upper_edges : 3d ndarray
        The 3d ndarray holding the upper bin edges of the reco energy axis.
        For each pair of true_e and true_dec different reco energy bin edges
        are provided.
        The shape is (n_true_e, n_true_dec, n_reco_e).
    psi_lower_edges : 4d ndarray
        The 4d ndarray holding the lower bin edges of the psi axis in radians.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psi).
    psi_upper_edges : 4d ndarray
        The 4d ndarray holding the upper bin edges of the psi axis in radians.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psi).
    ang_err_lower_edges : 5d ndarray
        The 5d ndarray holding the lower bin edges of the angular error axis
        in radians.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psi, n_ang_err).
    ang_err_upper_edges : 5d ndarray
        The 5d ndarray holding the upper bin edges of the angular error axis
        in radians.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psi, n_ang_err).
    """
    # Load the smearing data from the public dataset.
    loader = create_FileLoader(pathfilenames=pathfilenames)
    data = loader.load_data()
    # Rename the data fields.
    renaming_dict = {
        'log10(E_nu/GeV)_min': 'true_e_min',
        'log10(E_nu/GeV)_max': 'true_e_max',
        'Dec_nu_min[deg]':     'true_dec_min',
        'Dec_nu_max[deg]':     'true_dec_max',
        'log10(E/GeV)_min':    'e_min',
        'log10(E/GeV)_max':    'e_max',
        'PSF_min[deg]':        'psi_min',
        'PSF_max[deg]':        'psi_max',
        'AngErr_min[deg]':     'ang_err_min',
        'AngErr_max[deg]':     'ang_err_max',
        'Fractional_Counts':   'norm_counts'
    }
    data.rename_fields(renaming_dict)

    def _get_nbins_from_edges(lower_edges, upper_edges):
        """Helper function to extract the number of bins from the data's
        bin edges.
        """
        n = 0
        # Select only valid rows.
        mask = upper_edges - lower_edges > 0
        data = lower_edges[mask]
        # Go through the valid rows and search for the number of increasing
        # bin edge values.
        v0 = None
        for v in data:
            if(v0 is not None and v < v0):
                # Reached the end of the edges block.
                break
            if(v0 is None or v > v0):
                v0 = v
                n += 1
        return n

    true_e_bin_edges = np.union1d(
        data['true_e_min'], data['true_e_max'])
    true_dec_bin_edges = np.union1d(
        data['true_dec_min'], data['true_dec_max'])

    n_true_e = len(true_e_bin_edges) - 1
    n_true_dec = len(true_dec_bin_edges) - 1

    n_reco_e = _get_nbins_from_edges(
        data['e_min'], data['e_max'])
    n_psi = _get_nbins_from_edges(
        data['psi_min'], data['psi_max'])
    n_ang_err = _get_nbins_from_edges(
        data['ang_err_min'], data['ang_err_max'])

    # Get reco energy bin_edges as a 3d array.
    idxs = np.array(
        range(len(data))
    ) % (n_psi * n_ang_err) == 0

    reco_e_lower_edges = np.reshape(
        data['e_min'][idxs],
        (n_true_e, n_true_dec, n_reco_e)
    )
    reco_e_upper_edges = np.reshape(
        data['e_max'][idxs],
        (n_true_e, n_true_dec, n_reco_e)
    )

    # Get psi bin_edges as a 4d array.
    idxs = np.array(
        range(len(data))
    ) % n_ang_err == 0

    psi_lower_edges = np.reshape(
        data['psi_min'][idxs],
        (n_true_e, n_true_dec, n_reco_e, n_psi)
    )
    psi_upper_edges = np.reshape(
        data['psi_max'][idxs],
        (n_true_e, n_true_dec, n_reco_e, n_psi)
    )

    # Get angular error bin_edges as a 5d array.
    ang_err_lower_edges = np.reshape(
        data['ang_err_min'],
        (n_true_e, n_true_dec, n_reco_e, n_psi, n_ang_err)
    )
    ang_err_upper_edges = np.reshape(
        data['ang_err_max'],
        (n_true_e, n_true_dec, n_reco_e, n_psi, n_ang_err)
    )

    # Create 5D histogram for the probabilities.
    histogram = np.reshape(
        data['norm_counts'],
        (
            n_true_e,
            n_true_dec,
            n_reco_e,
            n_psi,
            n_ang_err
        )
    )

    # Convert degrees into radians.
    true_dec_bin_edges = np.radians(true_dec_bin_edges)
    psi_lower_edges = np.radians(psi_lower_edges)
    psi_upper_edges = np.radians(psi_upper_edges)
    ang_err_lower_edges = np.radians(ang_err_lower_edges)
    ang_err_upper_edges = np.radians(ang_err_upper_edges)

    return (
        histogram,
        true_e_bin_edges,
        true_dec_bin_edges,
        reco_e_lower_edges,
        reco_e_upper_edges,
        psi_lower_edges,
        psi_upper_edges,
        ang_err_lower_edges,
        ang_err_upper_edges
    )

def psi_to_dec_and_ra(rss, src_dec, src_ra, psi):
    """Generates random declinations and right-ascension coordinates for the
    given source location and opening angle `psi`.

    Parameters
    ----------
    rss : instance of RandomStateService
        The instance of RandomStateService to use for drawing random numbers.
    src_dec : float
        The declination of the source in radians.
    src_ra : float
        The right-ascension of the source in radians.
    psi : 1d ndarray of float
        The opening-angle values in radians.

    Returns
    -------
    dec : 1d ndarray of float
        The declination values.
    ra : 1d ndarray of float
        The right-ascension values.
    """

    psi = np.atleast_1d(psi)

    # Transform everything in radians and convert the source declination
    # to source zenith angle
    a = psi
    b = np.pi/2 - src_dec
    c = src_ra

    # Random rotation angle for the 2D circle
    t = rss.random.uniform(0, 2*np.pi, size=len(psi))

    # Parametrize the circle
    x = (
        (np.sin(a)*np.cos(b)*np.cos(c)) * np.cos(t) + \
        (np.sin(a)*np.sin(c)) * np.sin(t) - \
        (np.cos(a)*np.sin(b)*np.cos(c))
    )
    y = (
        -(np.sin(a)*np.cos(b)*np.sin(c)) * np.cos(t) + \
        (np.sin(a)*np.cos(c)) * np.sin(t) + \
        (np.cos(a)*np.sin(b)*np.sin(c))
    )
    z = (
        (np.sin(a)*np.sin(b)) * np.cos(t) + \
        (np.cos(a)*np.cos(b))
    )

    # Convert back to right-ascension and declination.
    # This is to distinguish between diametrically opposite directions.
    zen = np.arccos(z)
    azi = np.arctan2(y,x)

    dec = np.pi/2 - zen
    ra = np.pi - azi

    return (dec, ra)

def create_unionized_smearing_matrix_array(sm, src_dec):
    """Creates a unionized smearing matrix array which covers the entire
    observable space by keeping all original bins.

    Parameters
    ----------
    sm : PublicDataSmearingMatrix instance
        The PublicDataSmearingMatrix instance that holds the smearing matrix
        data.
    src_dec : float
        The source declination in radians.

    Returns
    -------
    result : dict
        The result dictionary with the following fields:
            union_arr : (nbins_true_e,
                         nbins_reco_e,
                         nbins_psi,
                         nbins_ang_err)-shaped 4D numpy ndarray
                The 4D ndarray holding the smearing matrix values.
            log10_true_e_bin_edges : 1D numpy ndarray
                The unionized bin edges of the log10 true energy axis.
            log10_reco_e_binedges : 1D numpy ndarray
                The unionized bin edges of the log10 reco energy axis.
            psi_binedges : 1D numpy ndarray
                The unionized bin edges of psi axis.
            ang_err_binedges : 1D numpy ndarray
                The unionized bin edges of the angular error axis.
    """
    true_dec_idx = sm.get_true_dec_idx(src_dec)

    true_e_bincenters = get_bincenters_from_binedges(
        sm.true_e_bin_edges)
    nbins_true_e = len(sm.true_e_bin_edges) - 1

    # Determine the unionized bin edges along all dimensions.
    reco_e_edges = np.unique(np.concatenate((
        sm.reco_e_lower_edges[:,true_dec_idx,...].flatten(),
        sm.reco_e_upper_edges[:,true_dec_idx,...].flatten()
    )))
    reco_e_bincenters = get_bincenters_from_binedges(reco_e_edges)
    nbins_reco_e = len(reco_e_edges) - 1

    psi_edges = np.unique(np.concatenate((
        sm.psi_lower_edges[:,true_dec_idx,...].flatten(),
        sm.psi_upper_edges[:,true_dec_idx,...].flatten()
    )))
    psi_bincenters = get_bincenters_from_binedges(psi_edges)
    nbins_psi = len(psi_edges) - 1

    ang_err_edges = np.unique(np.concatenate((
        sm.ang_err_lower_edges[:,true_dec_idx,...].flatten(),
        sm.ang_err_upper_edges[:,true_dec_idx,...].flatten()
    )))
    ang_err_bincenters = get_bincenters_from_binedges(ang_err_edges)
    nbins_ang_err = len(ang_err_edges) - 1

    # Create the unionized pdf array, which contains an axis for the
    # true energy bins.
    union_arr = np.zeros(
        (nbins_true_e, nbins_reco_e, nbins_psi, nbins_ang_err),
        dtype=np.double)
    # Fill the 4D array.
    for (true_e_idx, true_e) in enumerate(true_e_bincenters):
        for (e_idx, e) in enumerate(reco_e_bincenters):
            # Get the bin index of reco_e in the smearing matrix.
            sm_e_idx = sm.get_reco_e_idx(
                true_e_idx, true_dec_idx, e)
            if sm_e_idx is None:
                continue
            for (p_idx, p) in enumerate(psi_bincenters):
                # Get the bin index of psi in the smearing matrix.
                sm_p_idx = sm.get_psi_idx(
                    true_e_idx, true_dec_idx, sm_e_idx, p)
                if sm_p_idx is None:
                    continue
                for (a_idx, a) in enumerate(ang_err_bincenters):
                    # Get the bin index of the angular error in the
                    # smearing matrix.
                    sm_a_idx = sm.get_ang_err_idx(
                        true_e_idx, true_dec_idx, sm_e_idx, sm_p_idx, a)
                    if sm_a_idx is None:
                        continue

                    # Get the bin volume of the smearing matrix's bin.
                    idx = (
                        true_e_idx, true_dec_idx, sm_e_idx)
                    reco_e_bw = (
                        sm.reco_e_upper_edges[idx] -
                        sm.reco_e_lower_edges[idx]
                    )
                    idx = (
                        true_e_idx, true_dec_idx, sm_e_idx, sm_p_idx)
                    psi_bw = 2 * np.pi * (
                        np.cos(sm.psi_lower_edges[idx]) -
                        np.cos(sm.psi_upper_edges[idx])
                    )
                    idx = (
                        true_e_idx, true_dec_idx, sm_e_idx, sm_p_idx, sm_a_idx)
                    ang_err_bw = 2 * np.pi * (
                        np.cos(sm.ang_err_lower_edges[idx]) -
                        np.cos(sm.ang_err_upper_edges[idx])
                    )
                    bin_volume = reco_e_bw * psi_bw * ang_err_bw

                    union_arr[
                        true_e_idx,
                        e_idx,
                        p_idx,
                        a_idx
                    ] = sm.histogram[
                        true_e_idx,
                        true_dec_idx,
                        sm_e_idx,
                        sm_p_idx,
                        sm_a_idx
                    ] / bin_volume

    result = dict({
        'union_arr': union_arr,
        'log10_true_e_binedges': sm.true_e_bin_edges,
        'log10_reco_e_binedges': reco_e_edges,
        'psi_binedges': psi_edges,
        'ang_err_binedges': ang_err_edges
    })

    return result


class PublicDataAeff(object):
    """This class is a helper class for dealing with the effective area
    provided by the public data.
    """
    def __init__(
            self, pathfilenames, **kwargs):
        """Creates an effective area instance by loading the effective area
        data from the given file.
        """
        super().__init__(**kwargs)

        (
            self.aeff_arr,
            self.sin_true_dec_binedges_lower,
            self.sin_true_dec_binedges_upper,
            self.log_true_e_binedges_lower,
            self.log_true_e_binedges_upper
        ) = load_effective_area_array(pathfilenames)

        self.sin_true_dec_binedges = np.concatenate(
            (self.sin_true_dec_binedges_lower,
             self.sin_true_dec_binedges_upper[-1:])
        )
        self.log_true_e_binedges = np.concatenate(
            (self.log_true_e_binedges_lower,
             self.log_true_e_binedges_upper[-1:])
        )

    @property
    def log_true_e_bincenters(self):
        """The bin center values of the log true energy axis.
        """
        bincenters = 0.5 * (
            self.log_true_e_binedges[:-1] + self.log_true_e_binedges[1:]
        )

        return bincenters

    def get_aeff_for_sin_true_dec(self, sin_true_dec):
        """Retrieves the effective area as function of log_true_e.

        Parameters
        ----------
        sin_true_dec : float
            The sin of the true declination.

        Returns
        -------
        aeff : (n,)-shaped numpy ndarray
            The effective area for the given true declination as a function of
            log true energy.
        """
        sin_true_dec_idx = np.digitize(
            sin_true_dec, self.sin_true_dec_binedges) - 1

        aeff = self.aeff_arr[sin_true_dec_idx]

        return aeff

    def get_detection_pd_for_sin_true_dec(self, sin_true_dec, true_e):
        """Calculates the detection probability density p(E_nu|sin_dec) in
        unit GeV^-1 for the given true energy values.

        Parameters
        ----------
        sin_true_dec : float
            The sin of the true declination.
        true_e : (n,)-shaped 1d numpy ndarray of float
            The values of the true energy in GeV for which the probability
            density value should get calculated.

        Returns
        -------
        det_pd : (n,)-shaped 1d numpy ndarray of float
            The detection probability density values for the given true energy
            value.
        """
        aeff = self.get_aeff_for_sin_true_dec(sin_true_dec)

        dE = np.power(10, np.diff(self.log_true_e_binedges))

        det_pdf = aeff / np.sum(aeff) / dE

        det_pd = np.interp(
            true_e,
            np.power(10, self.log_true_e_bincenters),
            det_pdf)

        return det_pd

    def get_aeff_integral_for_sin_true_dec(
            self, sin_true_dec, log_true_e_min, log_true_e_max):
        """Calculates the integral of the effective area using the trapezoid
        method.

        Returns
        -------
        integral : float
            The integral in unit cm^2 GeV.
        """
        aeff = self.get_aeff_for_sin_true_dec(sin_true_dec)

        integral = (
            (np.power(10, log_true_e_max) -
             np.power(10, log_true_e_min)) *
            0.5 *
            (np.interp(log_true_e_min, self.log_true_e_bincenters, aeff) +
             np.interp(log_true_e_max, self.log_true_e_bincenters, aeff))
        )

        return integral

    def get_aeff(self, sin_true_dec, log_true_e):
        """Retrieves the effective area for the given sin(dec_true) and
        log(E_true) value pairs.

        Parameters
        ----------
        sin_true_dec : (n,)-shaped 1D ndarray
            The sin(dec_true) values.
        log_true_e : (n,)-shaped 1D ndarray
            The log(E_true) values.

        Returns
        -------
        aeff : (n,)-shaped 1D ndarray
            The 1D ndarray holding the effective area values for each value
            pair. For value pairs outside the effective area data zero is
            returned.
        """
        valid = (
            (sin_true_dec >= self.sin_true_dec_binedges[0]) &
            (sin_true_dec <= self.sin_true_dec_binedges[-1]) &
            (log_true_e >= self.log_true_e_binedges[0]) &
            (log_true_e <= self.log_true_e_binedges[-1])
        )
        sin_true_dec_idxs = np.digitize(
            sin_true_dec[valid], self.sin_true_dec_binedges) - 1
        log_true_e_idxs = np.digitize(
            log_true_e[valid], self.log_true_e_binedges) - 1

        aeff = np.zeros((len(valid),), dtype=np.double)
        aeff[valid] = self.aeff_arr[sin_true_dec_idxs,log_true_e_idxs]

        return aeff


class PublicDataSmearingMatrix(object):
    """This class is a helper class for dealing with the smearing matrix
    provided by the public data.
    """
    def __init__(
            self, pathfilenames, **kwargs):
        """Creates a smearing matrix instance by loading the smearing matrix
        from the given file.
        """
        super().__init__(**kwargs)

        (
            self.histogram,
            self._true_e_bin_edges,
            self._true_dec_bin_edges,
            self.reco_e_lower_edges,
            self.reco_e_upper_edges,
            self.psi_lower_edges,
            self.psi_upper_edges,
            self.ang_err_lower_edges,
            self.ang_err_upper_edges
        ) = load_smearing_histogram(pathfilenames)

    @property
    def true_e_bin_edges(self):
        """(read-only) The (n_true_e+1,)-shaped 1D numpy ndarray holding the
        bin edges of the true energy.
        """
        return self._true_e_bin_edges

    @property
    def true_e_bin_centers(self):
        """(read-only) The (n_true_e,)-shaped 1D numpy ndarray holding the bin
        center values of the true energy.
        """
        return 0.5*(self._true_e_bin_edges[:-1] +
                    self._true_e_bin_edges[1:])

    @property
    def true_dec_bin_edges(self):
        """(read-only) The (n_true_dec+1,)-shaped 1D numpy ndarray holding the
        bin edges of the true declination.
        """
        return self._true_dec_bin_edges

    @property
    def true_dec_bin_centers(self):
        """(read-only) The (n_true_dec,)-shaped 1D ndarray holding the bin
        center values of the true declination.
        """
        return 0.5*(self._true_dec_bin_edges[:-1] +
                    self._true_dec_bin_edges[1:])

    def get_true_dec_idx(self, true_dec):
        """Returns the true declination index for the given true declination
        value.

        Parameters
        ----------
        dec : float
            The declination value in radians.

        Returns
        -------
        true_dec_idx : int
            The index of the declination bin for the given declination value.
        """
        if (true_dec < self.true_dec_bin_edges[0]) or\
           (true_dec > self.true_dec_bin_edges[-1]):
            raise ValueError('The declination {} degrees is not supported by '
                'the smearing matrix!'.format(true_dec))

        true_dec_idx = np.digitize(true_dec, self.true_dec_bin_edges) - 1

        return true_dec_idx

    def get_reco_e_idx(self, true_e_idx, true_dec_idx, reco_e):
        """Returns the bin index for the given reco energy value given the
        given true energy and true declination bin indices.

        Parameters
        ----------
        true_e_idx : int
            The index of the true energy bin.
        true_dec_idx : int
            The index of the true declination bin.
        reco_e : float
            The reco energy value for which the bin index should get returned.

        Returns
        -------
        reco_e_idx : int | None
            The index of the reco energy bin the given reco energy value falls
            into. It returns None if the value is out of range.
        """
        lower_edges = self.reco_e_lower_edges[true_e_idx,true_dec_idx]
        upper_edges = self.reco_e_upper_edges[true_e_idx,true_dec_idx]

        m = (lower_edges <= reco_e) & (upper_edges > reco_e)
        idxs = np.nonzero(m)[0]
        if(len(idxs) == 0):
            return None

        reco_e_idx = idxs[0]

        return reco_e_idx

    def get_psi_idx(self, true_e_idx, true_dec_idx, reco_e_idx, psi):
        """Returns the bin index for the given psi value given the
        true energy, true declination and reco energy bin indices.

        Parameters
        ----------
        true_e_idx : int
            The index of the true energy bin.
        true_dec_idx : int
            The index of the true declination bin.
        reco_e_idx : int
            The index of the reco energy bin.
        psi : float
            The psi value in radians for which the bin index should get
            returned.

        Returns
        -------
        psi_idx : int | None
            The index of the psi bin the given psi value falls into.
            It returns None if the value is out of range.
        """
        lower_edges = self.psi_lower_edges[true_e_idx,true_dec_idx,reco_e_idx]
        upper_edges = self.psi_upper_edges[true_e_idx,true_dec_idx,reco_e_idx]

        m = (lower_edges <= psi) & (upper_edges > psi)
        idxs = np.nonzero(m)[0]
        if(len(idxs) == 0):
            return None

        psi_idx = idxs[0]

        return psi_idx

    def get_ang_err_idx(
            self, true_e_idx, true_dec_idx, reco_e_idx, psi_idx, ang_err):
        """Returns the bin index for the given angular error value given the
        true energy, true declination, reco energy, and psi bin indices.

        Parameters
        ----------
        true_e_idx : int
            The index of the true energy bin.
        true_dec_idx : int
            The index of the true declination bin.
        reco_e_idx : int
            The index of the reco energy bin.
        psi_idx : int
            The index of the psi bin.
        ang_err : float
            The angular error value in radians for which the bin index should
            get returned.

        Returns
        -------
        ang_err_idx : int | None
            The index of the angular error bin the given angular error value
            falls into. It returns None if the value is out of range.
        """
        lower_edges = self.ang_err_lower_edges[
            true_e_idx,true_dec_idx,reco_e_idx,psi_idx]
        upper_edges = self.ang_err_upper_edges[
            true_e_idx,true_dec_idx,reco_e_idx,psi_idx]

        m = (lower_edges <= ang_err) & (upper_edges > ang_err)
        idxs = np.nonzero(m)[0]
        if(len(idxs) == 0):
            return None

        ang_err_idx = idxs[0]

        return ang_err_idx

    def get_true_log_e_range_with_valid_log_e_pdfs(self, dec_idx):
        """Determines the true log energy range for which log_e PDFs are
        available for the given declination bin.

        Parameters
        ----------
        dec_idx : int
            The declination bin index.

        Returns
        -------
        min_log_true_e : float
            The minimum true log energy value.
        max_log_true_e : float
            The maximum true log energy value.
        """
        m = np.sum(
            (self.reco_e_upper_edges[:,dec_idx] -
             self.reco_e_lower_edges[:,dec_idx] > 0),
            axis=1) != 0
        min_log_true_e = np.min(self.true_e_bin_edges[:-1][m])
        max_log_true_e = np.max(self.true_e_bin_edges[1:][m])

        return (min_log_true_e, max_log_true_e)

    def get_log_e_pdf(
            self, log_true_e_idx, dec_idx):
        """Retrieves the log_e PDF from the given true energy bin index and
        source bin index.
        Returns (None, None, None, None) if any of the bin indices are less then
        zero, or if the sum of all pdf bins is zero.

        Parameters
        ----------
        log_true_e_idx : int
            The index of the true energy bin.
        dec_idx : int
            The index of the declination bin.

        Returns
        -------
        pdf : 1d ndarray
            The log_e pdf values.
        lower_bin_edges : 1d ndarray
            The lower bin edges of the energy pdf histogram.
        upper_bin_edges : 1d ndarray
            The upper bin edges of the energy pdf histogram.
        bin_widths : 1d ndarray
            The bin widths of the energy pdf histogram.
        """
        if log_true_e_idx < 0 or dec_idx < 0:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, dec_idx]
        pdf = np.sum(pdf, axis=(-2, -1))

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the reco energy bin edges and widths.
        lower_bin_edges = self.reco_e_lower_edges[
            log_true_e_idx, dec_idx
        ]
        upper_bin_edges = self.reco_e_upper_edges[
            log_true_e_idx, dec_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf /= np.sum(pdf) * bin_widths

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def get_psi_pdf(
            self, log_true_e_idx, dec_idx, log_e_idx):
        """Retrieves the psi PDF from the given true energy bin index, the
        source bin index, and the log_e bin index.
        Returns (None, None, None, None) if any of the bin indices are less then
        zero, or if the sum of all pdf bins is zero.

        Parameters
        ----------
        log_true_e_idx : int
            The index of the true energy bin.
        dec_idx : int
            The index of the declination bin.
        log_e_idx : int
            The index of the log_e bin.

        Returns
        -------
        pdf : 1d ndarray
            The psi pdf values.
        lower_bin_edges : 1d ndarray
            The lower bin edges of the psi pdf histogram.
        upper_bin_edges : 1d ndarray
            The upper bin edges of the psi pdf histogram.
        bin_widths : 1d ndarray
            The bin widths of the psi pdf histogram.
        """
        if log_true_e_idx < 0 or dec_idx < 0 or log_e_idx < 0:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, dec_idx, log_e_idx]
        pdf = np.sum(pdf, axis=-1)

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the PSI bin edges and widths.
        lower_bin_edges = self.psi_lower_edges[
            log_true_e_idx, dec_idx, log_e_idx
        ]
        upper_bin_edges = self.psi_upper_edges[
            log_true_e_idx, dec_idx, log_e_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf /= np.sum(pdf) * bin_widths

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def get_ang_err_pdf(
            self, log_true_e_idx, dec_idx, log_e_idx, psi_idx):
        """Retrieves the angular error PDF from the given true energy bin index,
        the source bin index, the log_e bin index, and the psi bin index.
        Returns (None, None, None, None) if any of the bin indices are less then
        zero, or if the sum of all pdf bins is zero.

        Parameters
        ----------
        log_true_e_idx : int
            The index of the true energy bin.
        dec_idx : int
            The index of the declination bin.
        log_e_idx : int
            The index of the log_e bin.
        psi_idx : int
            The index of the psi bin.

        Returns
        -------
        pdf : 1d ndarray
            The ang_err pdf values.
        lower_bin_edges : 1d ndarray
            The lower bin edges of the ang_err pdf histogram.
        upper_bin_edges : 1d ndarray
            The upper bin edges of the ang_err pdf histogram.
        bin_widths : 1d ndarray
            The bin widths of the ang_err pdf histogram.
        """
        if log_true_e_idx < 0 or dec_idx < 0 or log_e_idx < 0 or psi_idx < 0:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, dec_idx, log_e_idx, psi_idx]

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the ang_err bin edges and widths.
        lower_bin_edges = self.ang_err_lower_edges[
            log_true_e_idx, dec_idx, log_e_idx, psi_idx
        ]
        upper_bin_edges = self.ang_err_upper_edges[
            log_true_e_idx, dec_idx, log_e_idx, psi_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Some bins might not be defined, i.e. have zero bin widths.
        valid = bin_widths > 0

        pdf = pdf[valid]
        lower_bin_edges = lower_bin_edges[valid]
        upper_bin_edges = upper_bin_edges[valid]
        bin_widths = bin_widths[valid]

        # Normalize the PDF.
        pdf = pdf / (np.sum(pdf) * bin_widths)

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def sample_log_e(
            self, rss, dec_idx, log_true_e_idxs):
        """Samples log energy values for the given source declination and true
        energy bins.

        Parameters
        ----------
        rss : instance of RandomStateService
            The RandomStateService which should be used for drawing random
            numbers from.
        dec_idx : int
            The index of the source declination bin.
        log_true_e_idxs : 1d ndarray of int
            The bin indices of the true energy bins.

        Returns
        -------
        log_e_idx : 1d ndarray of int
            The bin indices of the log_e pdf corresponding to the sampled
            log_e values.
        log_e : 1d ndarray of float
            The sampled log_e values.
        """
        n_evt = len(log_true_e_idxs)
        log_e_idx = np.empty((n_evt,), dtype=np.int_)
        log_e = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            b_size = np.count_nonzero(m)
            (
                pdf,
                low_bin_edges,
                up_bin_edges,
                bin_widths
            ) = self.get_log_e_pdf(
                b_log_true_e_idx,
                dec_idx)

            if pdf is None:
                log_e_idx[m] = -1
                log_e[m] = np.nan
                continue

            b_log_e_idx = rss.random.choice(
                np.arange(len(pdf)),
                p=(pdf * bin_widths),
                size=b_size)
            b_log_e = rss.random.uniform(
                low_bin_edges[b_log_e_idx],
                up_bin_edges[b_log_e_idx],
                size=b_size)

            log_e_idx[m] = b_log_e_idx
            log_e[m] = b_log_e

        return (log_e_idx, log_e)

    def sample_psi(
            self, rss, dec_idx, log_true_e_idxs, log_e_idxs):
        """Samples psi values for the given source declination, true
        energy bins, and log_e bins.

        Parameters
        ----------
        rss : instance of RandomStateService
            The RandomStateService which should be used for drawing random
            numbers from.
        dec_idx : int
            The index of the source declination bin.
        log_true_e_idxs : 1d ndarray of int
            The bin indices of the true energy bins.
        log_e_idxs : 1d ndarray of int
            The bin indices of the log_e bins.

        Returns
        -------
        psi_idx : 1d ndarray of int
            The bin indices of the psi pdf corresponding to the sampled psi
            values.
        psi : 1d ndarray of float
            The sampled psi values in radians.
        """
        if(len(log_true_e_idxs) != len(log_e_idxs)):
            raise ValueError(
                'The lengths of log_true_e_idxs and log_e_idxs must be equal!')

        n_evt = len(log_true_e_idxs)
        psi_idx = np.empty((n_evt,), dtype=np.int_)
        psi = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            bb_unique_log_e_idxs = np.unique(log_e_idxs[m])
            for bb_log_e_idx in bb_unique_log_e_idxs:
                mm = m & (log_e_idxs == bb_log_e_idx)
                bb_size = np.count_nonzero(mm)
                (
                    pdf,
                    low_bin_edges,
                    up_bin_edges,
                    bin_widths
                ) = self.get_psi_pdf(
                    b_log_true_e_idx,
                    dec_idx,
                    bb_log_e_idx)

                if pdf is None:
                    psi_idx[mm] = -1
                    psi[mm] = np.nan
                    continue

                bb_psi_idx = rss.random.choice(
                    np.arange(len(pdf)),
                    p=(pdf * bin_widths),
                    size=bb_size)
                bb_psi = rss.random.uniform(
                    low_bin_edges[bb_psi_idx],
                    up_bin_edges[bb_psi_idx],
                    size=bb_size)

                psi_idx[mm] = bb_psi_idx
                psi[mm] = bb_psi

        return (psi_idx, psi)

    def sample_ang_err(
            self, rss, dec_idx, log_true_e_idxs, log_e_idxs, psi_idxs):
        """Samples ang_err values for the given source declination, true
        energy bins, log_e bins, and psi bins.

        Parameters
        ----------
        rss : instance of RandomStateService
            The RandomStateService which should be used for drawing random
            numbers from.
        dec_idx : int
            The index of the source declination bin.
        log_true_e_idxs : 1d ndarray of int
            The bin indices of the true energy bins.
        log_e_idxs : 1d ndarray of int
            The bin indices of the log_e bins.
        psi_idxs : 1d ndarray of int
            The bin indices of the psi bins.

        Returns
        -------
        ang_err_idx : 1d ndarray of int
            The bin indices of the angular error pdf corresponding to the
            sampled angular error values.
        ang_err : 1d ndarray of float
            The sampled angular error values in radians.
        """
        if (len(log_true_e_idxs) != len(log_e_idxs)) and\
           (len(log_e_idxs) != len(psi_idxs)):
            raise ValueError(
                'The lengths of log_true_e_idxs, log_e_idxs, and psi_idxs must '
                'be equal!')

        n_evt = len(log_true_e_idxs)
        ang_err_idx = np.empty((n_evt,), dtype=np.int_)
        ang_err = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            bb_unique_log_e_idxs = np.unique(log_e_idxs[m])
            for bb_log_e_idx in bb_unique_log_e_idxs:
                mm = m & (log_e_idxs == bb_log_e_idx)
                bbb_unique_psi_idxs = np.unique(psi_idxs[mm])
                for bbb_psi_idx in bbb_unique_psi_idxs:
                    mmm = mm & (psi_idxs == bbb_psi_idx)
                    bbb_size = np.count_nonzero(mmm)
                    (
                        pdf,
                        low_bin_edges,
                        up_bin_edges,
                        bin_widths
                    ) = self.get_ang_err_pdf(
                        b_log_true_e_idx,
                        dec_idx,
                        bb_log_e_idx,
                        bbb_psi_idx)

                    if pdf is None:
                        ang_err_idx[mmm] = -1
                        ang_err[mmm] = np.nan
                        continue

                    bbb_ang_err_idx = rss.random.choice(
                        np.arange(len(pdf)),
                        p=(pdf * bin_widths),
                        size=bbb_size)
                    bbb_ang_err = rss.random.uniform(
                        low_bin_edges[bbb_ang_err_idx],
                        up_bin_edges[bbb_ang_err_idx],
                        size=bbb_size)

                    ang_err_idx[mmm] = bbb_ang_err_idx
                    ang_err[mmm] = bbb_ang_err

        return (ang_err_idx, ang_err)
