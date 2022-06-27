# -*- coding: utf-8 -*-

import numpy as np
import pickle


def get_pd_atmo_Enu_sin_dec_nu(flux_pathfilename):
    """Constructs the atmospheric PDF p_atmo(E_nu|sin(dec_nu)) in unit 1/GeV.

    Parameters
    ----------
    flux_pathfilename : str
        The pathfilename of the file containing the MCEq flux.

    Returns
    -------
    pd_atmo : (n_sin_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the the atmospheric energy PDF in unit 1/GeV.
    sin_dec_binedges : numpy ndarray
        The (n_sin_dec+1,)-shaped 1D numpy ndarray holding the sin(dec) bin
        edges.
    log10_e_grid_edges : numpy ndarray
        The (n_e_grid+1,)-shaped 1D numpy ndarray holding the energy bin edges
        in log10.
    """
    with open(flux_pathfilename, 'rb') as f:
        ((e_grid, zenith_angle_binedges), flux_def) = pickle.load(f)

    # Select energy bins below 10**9 GeV.
    m_e_grid = e_grid <= 10**9
    e_grid = e_grid[m_e_grid]

    zenith_angles = 0.5*(zenith_angle_binedges[:-1]+ zenith_angle_binedges[1:])

    # Calculate the e_grid bin edges in log10.
    log10_e_grid_edges = np.empty((len(e_grid)+1),)
    d_log10_e_grid = np.diff(np.log10(e_grid))[0]
    log10_e_grid_edges[:-1] = np.log10(e_grid) - d_log10_e_grid/2
    log10_e_grid_edges[-1] = log10_e_grid_edges[-2] + d_log10_e_grid

    # Calculate the energy bin widths of the energy grid.
    dE = np.diff(10**log10_e_grid_edges)

    # Convert zenith angles into sin(declination) angles.
    sin_dec_binedges = np.sin(np.deg2rad(zenith_angle_binedges) - np.pi/2)
    sin_dec_angles = np.sin(np.deg2rad(zenith_angles) - np.pi/2)

    n_e_grid = len(e_grid)
    n_sin_dec = len(sin_dec_angles)

    # Calculate p_atmo(E_nu|sin(dec_nu)).
    pd_atmo = np.zeros((n_sin_dec, n_e_grid))
    for (sin_dec_idx, sin_dec) in enumerate(sin_dec_angles):
        if sin_dec < 0:
            fl = flux_def['numu_total'][:,sin_dec_idx][m_e_grid]
        else:
            # For up-going we use the flux calculation from the streight
            # downgoing.
            fl = flux_def['numu_total'][:,0][m_e_grid]
        pd_atmo[sin_dec_idx] = fl/np.sum(fl*dE)

    # Cross check the normalization of the PDF.
    if not np.all(np.isclose(np.sum(pd_atmo*dE[np.newaxis,:], axis=1), 1)):
        raise ValueError(
            'The atmospheric true energy PDF is not normalized!')

    return (pd_atmo, sin_dec_binedges, log10_e_grid_edges)
