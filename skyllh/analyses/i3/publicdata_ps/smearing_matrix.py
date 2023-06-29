# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.storage import create_FileLoader


def load_smearing_histogram(
        pathfilenames):
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
        mask = (upper_edges - lower_edges) > 0
        data = lower_edges[mask]
        # Go through the valid rows and search for the number of increasing
        # bin edge values.
        v0 = None
        for v in data:
            if (v0 is not None) and (v < v0):
                # Reached the end of the edges block.
                break
            if (v0 is None) or (v > v0):
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


class PDSmearingMatrix(
        object):
    """This class is a helper class for dealing with the smearing matrix
    provided by the public data.
    """
    def __init__(
            self,
            pathfilenames,
            **kwargs):
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

        self.n_psi_bins = self.histogram.shape[3]
        self.n_ang_err_bins = self.histogram.shape[4]

        # Create bin edges array for log10_reco_e.
        s = np.array(self.reco_e_lower_edges.shape)
        s[-1] += 1
        self.log10_reco_e_binedges = np.empty(s, dtype=np.double)
        self.log10_reco_e_binedges[..., :-1] = self.reco_e_lower_edges
        self.log10_reco_e_binedges[..., -1] = self.reco_e_upper_edges[..., -1]

        # Create bin edges array for psi.
        s = np.array(self.psi_lower_edges.shape)
        s[-1] += 1
        self.psi_binedges = np.empty(s, dtype=np.double)
        self.psi_binedges[..., :-1] = self.psi_lower_edges
        self.psi_binedges[..., -1] = self.psi_upper_edges[..., -1]

        # Create bin edges array for ang_err.
        s = np.array(self.ang_err_lower_edges.shape)
        s[-1] += 1
        self.ang_err_binedges = np.empty(s, dtype=np.double)
        self.ang_err_binedges[..., :-1] = self.ang_err_lower_edges
        self.ang_err_binedges[..., -1] = self.ang_err_upper_edges[..., -1]

    @property
    def n_log10_true_e_bins(self):
        """(read-only) The number of log10 true energy bins.
        """
        return len(self._true_e_bin_edges) - 1

    @property
    def true_e_bin_edges(self):
        """(read-only) The (n_true_e+1,)-shaped 1D numpy ndarray holding the
        bin edges of the true energy.

        Depricated! Use log10_true_enu_binedges instead!
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
    def log10_true_enu_binedges(self):
        """(read-only) The (n_log10_true_enu+1,)-shaped 1D numpy ndarray holding
        the bin edges of the log10 true neutrino energy.
        """
        return self._true_e_bin_edges

    @property
    def n_true_dec_bins(self):
        """(read-only) The number of true declination bins.
        """
        return len(self._true_dec_bin_edges) - 1

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

    @property
    def log10_reco_e_binedges_lower(self):
        """(read-only) The upper bin edges of the log10 reco energy axes.
        """
        return self.reco_e_lower_edges

    @property
    def log10_reco_e_binedges_upper(self):
        """(read-only) The upper bin edges of the log10 reco energy axes.
        """
        return self.reco_e_upper_edges

    @property
    def min_log10_reco_e(self):
        """(read-only) The minimum value of the reconstructed energy axis.
        """
        # Select only valid reco energy bins with bin widths greater than zero.
        m = (self.reco_e_upper_edges - self.reco_e_lower_edges) > 0
        return np.min(self.reco_e_lower_edges[m])

    @property
    def max_log10_reco_e(self):
        """(read-only) The maximum value of the reconstructed energy axis.
        """
        # Select only valid reco energy bins with bin widths greater than zero.
        m = (self.reco_e_upper_edges - self.reco_e_lower_edges) > 0
        return np.max(self.reco_e_upper_edges[m])

    @property
    def min_log10_psi(self):
        """(read-only) The minimum log10 value of the psi axis.
        """
        # Select only valid psi bins with bin widths greater than zero.
        m = (self.psi_upper_edges - self.psi_lower_edges) > 0
        return np.min(np.log10(self.psi_lower_edges[m]))

    @property
    def max_log10_psi(self):
        """(read-only) The maximum log10 value of the psi axis.
        """
        # Select only valid psi bins with bin widths greater than zero.
        m = (self.psi_upper_edges - self.psi_lower_edges) > 0
        return np.max(np.log10(self.psi_upper_edges[m]))

    @property
    def pdf(self):
        """(read-only) The probability-density-function
        P(E_reco,psi,ang_err|E_nu,dec_nu), which, by definition, is the
        histogram property divided by the 3D bin volumes for E_reco, psi, and
        ang_err.
        """
        log10_reco_e_bw = self.reco_e_upper_edges - self.reco_e_lower_edges
        psi_bw = self.psi_upper_edges - self.psi_lower_edges
        ang_err_bw = self.ang_err_upper_edges - self.ang_err_lower_edges

        bin_volumes = (
            log10_reco_e_bw[
                :, :, :, np.newaxis, np.newaxis
            ] *
            psi_bw[
                :, :, :, :, np.newaxis
            ] *
            ang_err_bw[
                :, :, :, :, :
            ]
        )

        # Divide the histogram bin probability values by their bin volume.
        # We do this only where the histogram actually has non-zero entries.
        pdf = np.copy(self.histogram)
        m = self.histogram != 0
        pdf[m] /= bin_volumes[m]

        return pdf

    def get_true_dec_idx(
            self,
            true_dec):
        """Returns the true declination index for the given true declination
        value.

        Parameters
        ----------
        true_dec : float
            The true declination value in radians.

        Returns
        -------
        true_dec_idx : int
            The index of the declination bin for the given declination value.
        """
        if (true_dec < self.true_dec_bin_edges[0]) or\
           (true_dec > self.true_dec_bin_edges[-1]):
            raise ValueError(
                f'The declination {true_dec} degrees is not supported by the '
                'smearing matrix!')

        true_dec_idx = np.digitize(true_dec, self.true_dec_bin_edges) - 1

        return true_dec_idx

    def get_log10_true_e_idx(
            self,
            log10_true_e):
        """Returns the bin index for the given true log10 energy value.

        Parameters
        ----------
        log10_true_e : float
            The log10 value of the true energy.

        Returns
        -------
        log10_true_e_idx : int
            The index of the true log10 energy bin for the given log10 true
            energy value.
        """
        if (log10_true_e < self.true_e_bin_edges[0]) or\
           (log10_true_e > self.true_e_bin_edges[-1]):
            raise ValueError(
                f'The log10 true energy value {log10_true_e} is not supported '
                'by the smearing matrix!')

        log10_true_e_idx = np.digitize(
            log10_true_e, self._true_e_bin_edges) - 1

        return log10_true_e_idx

    def get_reco_e_idx(
            self,
            true_e_idx,
            true_dec_idx,
            reco_e):
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
        lower_edges = self.reco_e_lower_edges[true_e_idx, true_dec_idx]
        upper_edges = self.reco_e_upper_edges[true_e_idx, true_dec_idx]

        m = (lower_edges <= reco_e) & (upper_edges > reco_e)
        idxs = np.nonzero(m)[0]
        if (len(idxs) == 0):
            return None

        reco_e_idx = idxs[0]

        return reco_e_idx

    def get_psi_idx(
            self,
            true_e_idx,
            true_dec_idx,
            reco_e_idx,
            psi):
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
        lower_edges = self.psi_lower_edges[true_e_idx, true_dec_idx, reco_e_idx]
        upper_edges = self.psi_upper_edges[true_e_idx, true_dec_idx, reco_e_idx]

        m = (lower_edges <= psi) & (upper_edges > psi)
        idxs = np.nonzero(m)[0]
        if len(idxs) == 0:
            return None

        psi_idx = idxs[0]

        return psi_idx

    def get_ang_err_idx(
            self,
            true_e_idx,
            true_dec_idx,
            reco_e_idx,
            psi_idx,
            ang_err):
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
            true_e_idx, true_dec_idx, reco_e_idx, psi_idx]
        upper_edges = self.ang_err_upper_edges[
            true_e_idx, true_dec_idx, reco_e_idx, psi_idx]

        m = (lower_edges <= ang_err) & (upper_edges > ang_err)
        idxs = np.nonzero(m)[0]
        if len(idxs) == 0:
            return None

        ang_err_idx = idxs[0]

        return ang_err_idx

    def get_true_log_e_range_with_valid_log_e_pdfs(
            self,
            dec_idx):
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
            (self.reco_e_upper_edges[:, dec_idx] -
             self.reco_e_lower_edges[:, dec_idx] > 0),
            axis=1) != 0
        min_log_true_e = np.min(self.true_e_bin_edges[:-1][m])
        max_log_true_e = np.max(self.true_e_bin_edges[1:][m])

        return (min_log_true_e, max_log_true_e)

    def get_log_e_pdf(
            self,
            log_true_e_idx,
            dec_idx):
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
            self,
            log_true_e_idx,
            dec_idx,
            log_e_idx):
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
            self,
            log_true_e_idx,
            dec_idx,
            log_e_idx,
            psi_idx):
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
            self,
            rss,
            dec_idx,
            log_true_e_idxs):
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
        log_e_idx = np.empty((n_evt,), dtype=np.int64)
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
            self,
            rss,
            dec_idx,
            log_true_e_idxs,
            log_e_idxs):
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
        if len(log_true_e_idxs) != len(log_e_idxs):
            raise ValueError(
                'The lengths of log_true_e_idxs and log_e_idxs must be equal!')

        n_evt = len(log_true_e_idxs)
        psi_idx = np.empty((n_evt,), dtype=np.int64)
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
            self,
            rss,
            dec_idx,
            log_true_e_idxs,
            log_e_idxs,
            psi_idxs):
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
        ang_err_idx = np.empty((n_evt,), dtype=np.int64)
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
