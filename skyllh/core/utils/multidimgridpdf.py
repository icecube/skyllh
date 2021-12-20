# -*- coding: utf-8 -*-

"""This module contains utility functions for creating and managing
MultiDimGridPDF instances.
"""

import numpy as np
import os

from skyllh.core.binning import BinningDefinition
from skyllh.core.pdf import MultiDimGridPDF
from skyllh.core.signalpdf import SignalMultiDimGridPDF
from skyllh.core.backgroundpdf import BackgroundMultiDimGridPDF


def kde_pdf_sig_spatial_norm_factor_func(pdf, tdm, fitparams, eventdata):
    """This is the standard normalization factor function for the spatial signal
    MultiDimGridPDF, which is created from KDE PDF values.
    It can be used for the ``norm_factor_func`` argument of the
    ``create_MultiDimGridPDF_from_kde_pdf`` function.
    """
    log10_psi_idx = pdf._axes.axis_name_list.index('log10_psi')
    # psi = tdm.get_data('psi')
    # Convert to psi.
    psi = 10**eventdata[:, log10_psi_idx]
    norm = 1. / (2 * np.pi * np.log(10) * psi * np.sin(psi))
    return norm


def kde_pdf_bkg_norm_factor_func(pdf, tdm, fitparams, eventdata):
    """This is the standard normalization factor function for the background
    MultiDimGridPDF, which is created from KDE PDF values.
    It can be used for the ``norm_factor_func`` argument of the
    ``create_MultiDimGridPDF_from_kde_pdf`` function.
    """
    return 1. / (2 * np.pi)


def create_MultiDimGridPDF_from_photosplinetable(
        ds, data, info_key, splinetable_key, norm_factor_func=None,
        kind=None, tl=None):
    """
    Creates a MultiDimGridPDF instance with pdf values taken from photospline pdf,
    a spline interpolation of KDE PDF values stored in a splinetable on disk.

    Parameters
    ----------
    ds : Dataset instance
        The Dataset instance the PDF applies to.
    data : DatasetData instance
        The DatasetData instance that holds the auxiliary data of the data set.
    info_key : str
        The auxiliary data name for the file containing PDF information.
    splinetable_key : str
        The auxiliary data name for the name of the splinetablefile.
    norm_factor_func : callable | None
        The normalization factor function. It must have the following call
        signature:
            __call__(pdf, tdm, fitparams)
    kind : str | None
        The kind of PDF to create. This is either ``'sig'`` for a
        SignalMultiDimGridPDF or ``'bkg'`` for a BackgroundMultiDimGridPDF
        instance. If set to None, a MultiDimGridPDF instance is created.
    tl : TimeLord instance | None
        The optional TimeLord instance to use for measuring timing information.

    Returns
    -------
    pdf : SignalMultiDimGridPDF instance | BackgroundMultiDimGridPDF instance |
          MultiDimGridPDF instance
        The created PDF instance. Depending on the ``kind`` argument, this is
        a SignalMultiDimGridPDF, a BackgroundMultiDimGridPDF, or a
        MultiDimGridPDF instance.
    """

    if(kind is None):
        pdf_type = MultiDimGridPDF
    elif(kind == 'sig'):
        pdf_type = SignalMultiDimGridPDF
    elif(kind == 'bkg'):
        pdf_type = BackgroundMultiDimGridPDF
    else:
        raise ValueError('The kind argument must be None, "sig", or "bkg"! '
                         'Currently it is '+str(kind)+'!')

    # Load the PDF data from the auxilary files.
    num_dict = ds.load_aux_data(info_key, tl=tl)

    kde_pdf_axis_name_map = ds.load_aux_data('KDE_PDF_axis_name_map', tl=tl)
    kde_pdf_axis_name_map_inv = dict(
        [(v, k) for (k, v) in kde_pdf_axis_name_map.items()])
    for var in num_dict['vars']:
        if(var not in kde_pdf_axis_name_map_inv):
            kde_pdf_axis_name_map_inv[var] = var

    if 'bin_centers' in num_dict:
        bin_centers_key = 'bin_centers'
    elif 'bins' in num_dict:
        bin_centers_key = 'bins'
    else:
        raise KeyError(
            "The PDF information file is missing 'bin_centers' or 'bins' key.")

    axis_binnings = [
        BinningDefinition(
            kde_pdf_axis_name_map_inv[var], num_dict[bin_centers_key][idx])
        for (idx, var) in enumerate(num_dict['vars'])
    ]

    # Getting the name of the splinetable file
    splinetable_file_list = ds.get_aux_data_definition(splinetable_key)
    # This is a list with only one element.
    splinetable_file = os.path.join(ds.root_dir, splinetable_file_list[0])

    pdf = pdf_type(
        axis_binnings,
        path_to_pdf_splinetable=splinetable_file,
        pdf_grid_data=None,
        norm_factor_func=norm_factor_func)

    return pdf


def create_MultiDimGridPDF_from_kde_pdf(
        ds, data, numerator_key, denumerator_key=None, norm_factor_func=None,
        kind=None, tl=None):
    """Creates a MultiDimGridPDF instance with pdf values taken from KDE PDF
    values stored in the dataset's auxiliary data.

    Parameters
    ----------
    ds : Dataset instance
        The Dataset instance the PDF applies to.
    data : DatasetData instance
        The DatasetData instance that holds the auxiliary data of the data set.
    numerator_key : str
        The auxiliary data name for the PDF numerator array.
    denumerator_key : str | None
        The auxiliary data name for the PDF denumerator array.
        This can be None, if no denumerator array is required.
    norm_factor_func : callable | None
        The normalization factor function. It must have the following call
        signature:
            __call__(pdf, tdm, fitparams)
    kind : str | None
        The kind of PDF to create. This is either ``'sig'`` for a
        SignalMultiDimGridPDF or ``'bkg'`` for a BackgroundMultiDimGridPDF
        instance. If set to None, a MultiDimGridPDF instance is created.
    tl : TimeLord instance | None
        The optional TimeLord instance to use for measuring timing information.

    Returns
    -------
    pdf : SignalMultiDimGridPDF instance | BackgroundMultiDimGridPDF instance |
          MultiDimGridPDF instance
        The created PDF instance. Depending on the ``kind`` argument, this is
        a SignalMultiDimGridPDF, a BackgroundMultiDimGridPDF, or a
        MultiDimGridPDF instance.
    """
    if(kind is None):
        pdf_type = MultiDimGridPDF
    elif(kind == 'sig'):
        pdf_type = SignalMultiDimGridPDF
    elif(kind == 'bkg'):
        pdf_type = BackgroundMultiDimGridPDF
    else:
        raise ValueError('The kind argument must be None, "sig", or "bkg"! '
                         'Currently it is '+str(kind)+'!')

    # Load the PDF data from the auxilary files.
    num_dict = ds.load_aux_data(numerator_key, tl=tl)

    denum_dict = None
    if(denumerator_key is not None):
        denum_dict = ds.load_aux_data(denumerator_key, tl=tl)

    kde_pdf_axis_name_map = ds.load_aux_data('KDE_PDF_axis_name_map', tl=tl)
    kde_pdf_axis_name_map_inv = dict(
        [(v, k) for (k, v) in kde_pdf_axis_name_map.items()])
    for var in num_dict['vars']:
        if(var not in kde_pdf_axis_name_map_inv):
            kde_pdf_axis_name_map_inv[var] = var

    if 'bin_centers' in num_dict:
        bin_centers_key = 'bin_centers'
    elif 'bins' in num_dict:
        bin_centers_key = 'bins'
    else:
        raise KeyError(
            "The PDF information file is missing 'bin_centers' or 'bins' key.")

    axis_binnings = [
        BinningDefinition(
            kde_pdf_axis_name_map_inv[var], num_dict[bin_centers_key][idx])
        for (idx, var) in enumerate(num_dict['vars'])
    ]

    vals = num_dict['pdf_vals']
    if(denum_dict is not None):
        # A denumerator is required, so we need to divide the numerator pdf
        # values by the denumerator pdf values, by preserving the correct axis
        # order.
        # Construct the slicing selector for the denumerator pdf values array to
        # match the axis order of the numerator pdf values array.
        selector = []
        for var in num_dict['vars']:
            if(var in denum_dict['vars']):
                # The variable is present in both pdf value arrays. So select
                # all values of that dimension.
                selector.append(slice(None, None))
            else:
                # The variable is not present in the normalization pdf value
                # array, so we need to add a dimension for that variable.
                selector.append(np.newaxis)

        denum = denum_dict['pdf_vals'][tuple(selector)]
        denum_nonzero_mask = denum != 0
        out = np.zeros_like(vals)
        np.divide(vals, denum, where=denum_nonzero_mask, out=out)
        vals = out

    # Set infinite values to NaN.
    vals[np.isinf(vals)] = np.nan
    # Set NaN values to 0.
    vals[np.isnan(vals)] = 0

    pdf = pdf_type(
        axis_binnings,
        path_to_pdf_splinetable=None,
        pdf_grid_data=vals,
        norm_factor_func=norm_factor_func)

    return pdf
