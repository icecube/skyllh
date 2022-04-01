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
