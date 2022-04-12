# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.storage import create_FileLoader


def load_smearing_histogram(pathfilenames):
    """Loads the 5D smearing histogram from the given data file.

    Parameters
    ----------
    pathfilenames : list of str
        The file name of the data file.

    Returns
    -------
    histogram : 5d ndarray
        The 5d histogram array holding the probability values of the smearing
        matrix.
        The axes are (true_e, true_dec, reco_e, psf, ang_err).
    true_e_bin_edges : 1d ndarray
        The ndarray holding the bin edges of the true energy axis.
    true_dec_bin_edges : 1d ndarray
        The ndarray holding the bin edges of the true declination axis.
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
    psf_lower_edges : 4d ndarray
        The 4d ndarray holding the lower bin edges of the PSF axis.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psf).
    psf_upper_edges : 4d ndarray
        The 4d ndarray holding the upper bin edges of the PSF axis.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psf).
    ang_err_lower_edges : 5d ndarray
        The 5d ndarray holding the lower bin edges of the angular error axis.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psf, n_ang_err).
    ang_err_upper_edges : 5d ndarray
        The 5d ndarray holding the upper bin edges of the angular error axis.
        The shape is (n_true_e, n_true_dec, n_reco_e, n_psf, n_ang_err).
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
        'PSF_min[deg]':        'psf_min',
        'PSF_max[deg]':        'psf_max',
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

    true_e_bin_edges = np.union1d(data['true_e_min'], data['true_e_max'])
    true_dec_bin_edges = np.union1d(data['true_dec_min'], data['true_dec_max'])

    n_true_e = len(true_e_bin_edges) - 1
    n_true_dec = len(true_dec_bin_edges) - 1

    n_reco_e = _get_nbins_from_edges(
        data['e_min'], data['e_max'])
    n_psf = _get_nbins_from_edges(
        data['psf_min'], data['psf_max'])
    n_ang_err = _get_nbins_from_edges(
        data['ang_err_min'], data['ang_err_max'])

    # Get reco energy bin_edges as a 3d array.
    idxs = np.array(
        range(len(data))
    ) % (n_psf * n_ang_err) == 0

    reco_e_lower_edges = np.reshape(
        data['e_min'][idxs],
        (n_true_e, n_true_dec, n_reco_e)
    )
    reco_e_upper_edges = np.reshape(
        data['e_max'][idxs],
        (n_true_e, n_true_dec, n_reco_e)
    )

    # Get psf bin_edges as a 4d array.
    idxs = np.array(
        range(len(data))
    ) % n_ang_err == 0

    psf_lower_edges = np.reshape(
        data['psf_min'][idxs],
        (n_true_e, n_true_dec, n_reco_e, n_psf)
    )
    psf_upper_edges = np.reshape(
        data['psf_max'][idxs],
        (n_true_e, n_true_dec, n_reco_e, n_psf)
    )

    # Get angular error bin_edges as a 5d array.
    ang_err_lower_edges = np.reshape(
        data['ang_err_min'],
        (n_true_e, n_true_dec, n_reco_e, n_psf, n_ang_err)
    )
    ang_err_upper_edges = np.reshape(
        data['ang_err_max'],
        (n_true_e, n_true_dec, n_reco_e, n_psf, n_ang_err)
    )


    # Create 5D histogram for the probabilities.
    histogram = np.reshape(
        data['norm_counts'],
        (
            n_true_e,
            n_true_dec,
            n_reco_e,
            n_psf,
            n_ang_err
        )
    )

    return (
        histogram,
        true_e_bin_edges,
        true_dec_bin_edges,
        reco_e_lower_edges,
        reco_e_upper_edges,
        psf_lower_edges,
        psf_upper_edges,
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
            self.true_e_bin_edges,
            self.true_dec_bin_edges,
            self.reco_e_lower_edges,
            self.reco_e_upper_edges,
            self.psf_lower_edges,
            self.psf_upper_edges,
            self.ang_err_lower_edges,
            self.ang_err_upper_edges
        ) = load_smearing_histogram(pathfilenames)

    def get_dec_idx(self, dec):
        """Returns the declination index for the given declination value.

        Parameters
        ----------
        dec : float
            The declination value in radians.

        Returns
        -------
        dec_idx : int
            The index of the declination bin for the given declination value.
        """
        dec = np.degrees(dec)

        if (dec < self.true_dec_bin_edges[0]) or\
           (dec > self.true_dec_bin_edges[-1]):
            raise ValueError('The declination {} degrees is not supported by '
                'the smearing matrix!'.format(dec))

        dec_idx = np.digitize(dec, self.true_dec_bin_edges) - 1

        return dec_idx

    def get_true_log_e_range_with_valid_log_e_pfds(self, dec_idx):
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
        pdf /= (np.sum(pdf * bin_widths))

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
        lower_bin_edges = self.psf_lower_edges[
            log_true_e_idx, dec_idx, log_e_idx
        ]
        upper_bin_edges = self.psf_upper_edges[
            log_true_e_idx, dec_idx, log_e_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf /= (np.sum(pdf * bin_widths))

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

        # Normalize the PDF.
        pdf = pdf / np.sum(pdf * bin_widths)

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

        return (psi_idx, np.radians(psi))

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

        return (ang_err_idx, np.radians(ang_err))
