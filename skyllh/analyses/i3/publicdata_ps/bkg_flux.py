# -*- coding: utf-8 -*-

import numpy as np
import pickle

from skyllh.core.binning import (
    get_bincenters_from_binedges,
)
from skyllh.core.flux_model import (
    PowerLawEnergyFluxProfile,
    SteadyPointlikeFFM,
)


def get_dOmega(dec_min, dec_max):
    """Calculates the solid angle given two declination angles.

    Parameters
    ----------
    dec_min : float | array of float
        The smaller declination angle.
    dec_max : float | array of float
        The larger declination angle.

    Returns
    -------
    solidangle : float | array of float
        The solid angle corresponding to the two given declination angles.
    """
    return 2*np.pi*(np.sin(dec_max) - np.sin(dec_min))


def southpole_zen2dec(zen):
    """Converts zenith angles at the South Pole to declination angles.

    Parameters
    ----------
    zen : (n,)-shaped 1d numpy ndarray
        The numpy ndarray holding the zenith angle values in radians.

    Returns
    -------
    dec : (n,)-shaped 1d numpy ndarray
        The numpy ndarray holding the declination angle values in radians.
    """
    dec = zen - np.pi/2
    return dec


def get_flux_atmo_decnu_log10enu(flux_pathfilename, log10_enu_max=9):
    """Constructs the atmospheric flux map function
    f_atmo(log10(E_nu/GeV),dec_nu) in unit 1/(GeV cm^2 sr s).

    Parameters
    ----------
    flux_pathfilename : str
        The pathfilename of the file containing the MCEq fluxes.
    log10_enu_max : float
        The log10(E/GeV) value of the maximum neutrino energy to be considered.

    Returns
    -------
    flux_atmo : (n_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the the atmospheric neutrino flux function in
        unit 1/(GeV cm^2 sr s).
    decnu_binedges : (n_decnu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the dec_nu bin edges.
    log10_enu_binedges : (n_enu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the neutrino energy bin edges in log10.
    """
    with open(flux_pathfilename, 'rb') as f:
        ((e_grid, zenith_angle_binedges), flux_def) = pickle.load(f)
    zenith_angle_binedges = np.deg2rad(zenith_angle_binedges)

    # Select energy bins below 10**log10_true_e_max GeV.
    m_e_grid = e_grid <= 10**log10_enu_max
    e_grid = e_grid[m_e_grid]

    decnu_binedges = southpole_zen2dec(zenith_angle_binedges)
    decnu_angles = get_bincenters_from_binedges(decnu_binedges)

    # Calculate the neutrino energy bin edges in log10.
    log10_enu_binedges = np.empty((len(e_grid)+1),)
    d_log10_enu = np.diff(np.log10(e_grid))[0]
    log10_enu_binedges[:-1] = np.log10(e_grid) - d_log10_enu/2
    log10_enu_binedges[-1] = log10_enu_binedges[-2] + d_log10_enu

    n_decnu = len(decnu_angles)
    n_enu = len(e_grid)

    # Calculate f_atmo(E_nu,dec_nu).
    f_atmo = np.zeros((n_decnu, n_enu))
    zero_zen_idx = np.digitize(0, zenith_angle_binedges) - 1
    for (decnu_idx, decnu) in enumerate(decnu_angles):
        if decnu < 0:
            fl = flux_def['numu_total'][:, decnu_idx][m_e_grid]
        else:
            # For up-going we use the flux calculation from the streight
            # downgoing.
            fl = flux_def['numu_total'][:, zero_zen_idx][m_e_grid]
        f_atmo[decnu_idx] = fl

    return (f_atmo, decnu_binedges, log10_enu_binedges)


def get_flux_astro_decnu_log10enu(decnu_binedges, log10_enu_binedges):
    """Constructs the astrophysical neutrino flux function
    f_astro(log10(E_nu/GeV),dec_nu) in unit 1/(GeV cm^2 sr s).

    It uses the best fit from the IceCube publication [1].

    Parameters
    ----------
    decnu_binedges : (n_decnu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the dec_nu bin edges.
    log10_enu_binedges : (n_enu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the log10 values of the neutrino energy bin
        edges in GeV.

    Returns
    -------
    f_astro : (n_decnu, n_log10enu)-shaped 2D numpy ndarray
        The numpy ndarray holding the astrophysical flux values in unit
        1/(GeV cm^2 sr s).

    References
    ----------
    [1] https://arxiv.org/pdf/2111.10299.pdf
    """
    fluxmodel = SteadyPointlikeFFM(
        Phi0=1.44e-18,
        energy_profile=PowerLawEnergyFluxProfile(
            E0=100e3,
            gamma=2.37))

    n_decnu = len(decnu_binedges) - 1

    enu_binedges = np.power(10, log10_enu_binedges)
    enu_bincenters = get_bincenters_from_binedges(enu_binedges)

    fl = fluxmodel(E=enu_bincenters).squeeze()
    f_astro = np.tile(fl, (n_decnu, 1))

    return f_astro


def convert_flux_bkg_to_pdf_bkg(f_bkg, decnu_binedges, log10_enu_binedges):
    """Converts the given background flux function f_bkg into a background flux
    PDF in unit 1/(log10(E/GeV) rad).

    Parameters
    ----------
    f_bkg : (n_decnu, n_enu)-shaped 2D numpy ndarray
        The numpy ndarray holding the background flux values in unit
        1/(GeV cm^2 s sr).
    decnu_binedges : (n_decnu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the dec_nu bin edges in radians.
    log10_enu_binedges : (n_enu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the log10 values of the neutrino energy bin
        edges in GeV.

    Returns
    -------
    p_bkg : (n_decnu, n_enu)-shaped 2D numpy ndarray
        The numpy ndarray holding the background flux pdf values.
    """
    d_decnu = np.diff(decnu_binedges)
    d_log10_enu = np.diff(log10_enu_binedges)

    bin_area = d_decnu[:, np.newaxis] * d_log10_enu[np.newaxis, :]
    p_bkg = f_bkg / np.sum(f_bkg*bin_area)

    # Cross-check the normalization of the PDF.
    if not np.isclose(np.sum(p_bkg*bin_area), 1):
        raise ValueError(
            'The background PDF is not normalized! The integral is '
            f'{np.sum(p_bkg*bin_area)}!')

    return p_bkg


def get_pd_atmo_decnu_Enu(flux_pathfilename, log10_true_e_max=9):
    """Constructs the atmospheric neutrino PDF p_atmo(E_nu,dec_nu) in unit
    1/(GeV rad).

    Parameters
    ----------
    flux_pathfilename : str
        The pathfilename of the file containing the MCEq flux.
    log10_true_e_max : float
        The log10(E/GeV) value of the maximum true energy to be considered.

    Returns
    -------
    pd_atmo : (n_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the the atmospheric neutrino PDF in unit
        1/(GeV rad).
    decnu_binedges : (n_decnu+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the dec_nu bin edges.
    log10_e_grid_edges : (n_e_grid+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the energy bin edges in log10.
    """
    with open(flux_pathfilename, 'rb') as f:
        ((e_grid, zenith_angle_binedges), flux_def) = pickle.load(f)

    # Select energy bins below 10**log10_true_e_max GeV.
    m_e_grid = e_grid <= 10**log10_true_e_max
    e_grid = e_grid[m_e_grid]

    zenith_angles = 0.5*(zenith_angle_binedges[:-1] + zenith_angle_binedges[1:])
    decnu_angles = np.deg2rad(zenith_angles) - np.pi/2

    decnu_binedges = np.deg2rad(zenith_angle_binedges) - np.pi/2
    d_decnu = np.diff(decnu_binedges)

    # Calculate the e_grid bin edges in log10.
    log10_e_grid_edges = np.empty((len(e_grid)+1),)
    d_log10_e_grid = np.diff(np.log10(e_grid))[0]
    log10_e_grid_edges[:-1] = np.log10(e_grid) - d_log10_e_grid/2
    log10_e_grid_edges[-1] = log10_e_grid_edges[-2] + d_log10_e_grid

    n_decnu = len(decnu_angles)
    n_e_grid = len(e_grid)

    # Calculate p_atmo(E_nu,dec_nu).
    pd_atmo = np.zeros((n_decnu, n_e_grid))
    for (decnu_idx, decnu) in enumerate(decnu_angles):
        if decnu < 0:
            fl = flux_def['numu_total'][:, decnu_idx][m_e_grid]
        else:
            # For up-going we use the flux calculation from the streight
            # downgoing.
            fl = flux_def['numu_total'][:, 0][m_e_grid]
        pd_atmo[decnu_idx] = fl
    # Normalize the PDF.
    bin_area = d_decnu[:, np.newaxis] * np.diff(log10_e_grid_edges)[np.newaxis, :]
    pd_atmo /= np.sum(pd_atmo*bin_area)

    # Cross-check the normalization of the PDF.
    if not np.isclose(np.sum(pd_atmo*bin_area), 1):
        raise ValueError(
            'The atmospheric true energy PDF is not normalized! The integral '
            f'is {np.sum(pd_atmo*bin_area)}!')

    return (pd_atmo, decnu_binedges, log10_e_grid_edges)


def get_pd_atmo_E_nu_sin_dec_nu(flux_pathfilename):
    """Constructs the atmospheric energy PDF p_atmo(E_nu|sin(dec_nu)) in
    unit 1/GeV.

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

    zenith_angles = 0.5*(zenith_angle_binedges[:-1] + zenith_angle_binedges[1:])

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
            fl = flux_def['numu_total'][:, sin_dec_idx][m_e_grid]
        else:
            # For up-going we use the flux calculation from the streight
            # downgoing.
            fl = flux_def['numu_total'][:, 0][m_e_grid]
        pd_atmo[sin_dec_idx] = fl/np.sum(fl*dE)

    # Cross-check the normalization of the PDF.
    if not np.all(np.isclose(np.sum(pd_atmo*dE[np.newaxis, :], axis=1), 1)):
        raise ValueError(
            'The atmospheric true energy PDF is not normalized!')

    return (pd_atmo, sin_dec_binedges, log10_e_grid_edges)


def get_pd_astro_E_nu_sin_dec_nu(sin_dec_binedges, log10_e_grid_edges):
    """Constructs the astrophysical energy PDF p_astro(E_nu|sin(dec_nu)) in
    unit 1/GeV.
    It uses the best fit from the IceCube publication [1].

    Parameters
    ----------
    sin_dec_binedges : (n_sin_dec+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the sin(dec) bin edges.
    log10_e_grid_edges : (n_e_grid+1,)-shaped 1D numpy ndarray
        The numpy ndarray holding the log10 values of the energy bin edges in
        GeV of the energy grid.

    Returns
    -------
    pd_astro : (n_sin_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the energy probability density values
        p(E_nu|sin_dec_nu) in unit 1/GeV.

    References
    ----------
    [1] https://arxiv.org/pdf/2111.10299.pdf
    """
    fluxmodel = SteadyPointlikeFFM(
        Phi0=1.44e-18,
        energy_profile=PowerLawEnergyFluxProfile(
            E0=100e3,
            gamma=2.37))

    n_sin_dec = len(sin_dec_binedges) - 1

    e_grid_edges = 10**log10_e_grid_edges
    e_grid_bc = 0.5*(e_grid_edges[:-1] + e_grid_edges[1:])

    dE = np.diff(e_grid_edges)

    fl = fluxmodel(E=e_grid_bc).squeeze()
    pd = fl / np.sum(fl*dE)
    pd_astro = np.tile(pd, (n_sin_dec, 1))

    # Cross-check the normalization of the PDF.
    if not np.all(np.isclose(np.sum(pd_astro*dE[np.newaxis, :], axis=1), 1)):
        raise ValueError(
            'The astrophysical energy PDF is not normalized!')

    return pd_astro


def get_pd_bkg_E_nu_sin_dec_nu(pd_atmo, pd_astro, log10_e_grid_edges):
    """Constructs the total background flux probability density
    p_bkg(E_nu|sin(dec_nu)) in unit 1/GeV.

    Parameters
    ----------
    pd_atmo : (n_sin_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the probability density values
        p(E_nu|sin(dec_nu)) in 1/GeV of the atmospheric flux.
    pd_astro : (n_sin_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding the probability density values
        p(E_nu|sin(dec_nu)) in 1/GeV of the astrophysical flux.
    log10_e_grid_edges : (n_e_grid+1,)-shaped numpy ndarray
        The numpy ndarray holding the log10 values of the energy grid bin edges
        in GeV.

    Returns
    -------
    pd_bkg : (n_sin_dec, n_e_grid)-shaped 2D numpy ndarray
        The numpy ndarray holding total background probability density values
        p_bkg(E_nu|sin(dec_nu)) in unit 1/GeV.
    """
    pd_bkg = pd_atmo + pd_astro

    dE = np.diff(10**log10_e_grid_edges)

    s = np.sum(pd_bkg*dE[np.newaxis, :], axis=1, keepdims=True)
    pd_bkg /= s

    if not np.all(np.isclose(np.sum(pd_bkg*dE[np.newaxis, :], axis=1), 1)):
        raise ValueError(
            'The background energy PDF is not normalized!')

    return pd_bkg
