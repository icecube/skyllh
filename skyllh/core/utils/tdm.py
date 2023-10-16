# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.utils.coords import (
    angular_separation,
)


def get_tdm_field_func_psi(psi_floor=None):
    """Returns the TrialDataManager (TDM) field function for psi with an
    optional psi value floor.

    Parameters
    ----------
    psi_floor : float | None
        The optional floor value for psi. This should be ``None`` for a standard
        point-source analysis that uses an analytic function for the detector's
        point-spread-function (PSF).

    Returns
    -------
    tdm_field_func_psi : function
        TrialDataManager (TDM) field function for psi.
    """
    def tdm_field_func_psi(
            tdm,
            shg_mgr,
            pmm):
        """TDM data field function to calculate the opening angle between the
        source positions and the event's reconstructed position.
        """
        (src_idxs, evt_idxs) = tdm.src_evt_idxs

        ra = np.take(tdm.get_data('ra'), evt_idxs)
        dec = np.take(tdm.get_data('dec'), evt_idxs)

        src_array = tdm.get_data('src_array')
        src_ra = np.take(src_array['ra'], src_idxs)
        src_dec = np.take(src_array['dec'], src_idxs)

        psi = angular_separation(
            ra1=ra,
            dec1=dec,
            ra2=src_ra,
            dec2=src_dec,
            psi_floor=psi_floor)

        return psi

    return tdm_field_func_psi
