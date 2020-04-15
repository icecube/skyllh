# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

"""This module contains utility functions for creating and managing
NDPhotosplinePDF instances.
"""

from skyllh.core.binning import BinningDefinition
from skyllh.core.pdf import NDPhotosplinePDF
from skyllh.core.backgroundpdf import BackgroundNDPhotosplinePDF
from skyllh.core.signalpdf import SignalNDPhotosplinePDF


def create_NDPhotosplinePDF_from_photosplinefit(
        ds, kind, info_key, splinefit_key, param_set=None,
        norm_factor_func=None,
        tl=None
    ):
    """Creates a new NDPhotosplinePDF instance from a photospline fits file that
    is defined in the given data set.

    Parameters
    ----------
    ds : Dataset instance
        The Dataset instance the PDF applies to.
    kind : str | None
        The kind of PDF to create. This is either ``'sig'`` for a
        SignalNDPhotosplinePDF, or ``'bkg'`` for a BackgroundNDPhotosplinePDF
        instance. If set to None, a NDPhotosplinePDF instance is created.
    info_key : str
        The auxiliary data name for the file containing PDF meta data
        information.
    splinefit_key : str
        The auxiliary data name defining the path to the file containing the
        photospline fit.
    param_set : Parameter | ParameterSet | None
        The Parameter instance or ParameterSet instance defining the
        parameters of the new PDF. The ParameterSet holds the information
        which parameters are fixed and which are floating (i.e. fitted).
    norm_factor_func : callable | None
        The normalization factor function. It must have the following call
        signature:
            __call__(pdf, tdm, params)
    tl : TimeLord instance | None
        The optional TimeLord instance to use for measuring timing information.

    Returns
    -------
    pdf : SignalNDPhotosplinePDF instance |
          BackgroundNDPhotosplinePDF instance | NDPhotosplinePDF instance
        The created PDF instance. Depending on the ``kind`` argument, this is
        a SignalNDPhotosplinePDF, a BackgroundNDPhotosplinePDF, or a
        NDPhotosplinePDF instance.
    """

    if(kind is None):
        pdf_type = NDPhotosplinePDF
    elif(kind == 'sig'):
        pdf_type = SignalNDPhotosplinePDF
    elif(kind == 'bkg'):
        pdf_type = BackgroundNDPhotosplinePDF
    else:
        raise ValueError(
            'The kind argument must be None, "sig", or "bkg"! '
            'Currently it is '+str(kind)+'!')

    # Load the PDF data from the auxilary files.
    info_dict = ds.load_aux_data(info_key, tl=tl)

    kde_pdf_axis_name_map = ds.load_aux_data('KDE_PDF_axis_name_map', tl=tl)
    kde_pdf_axis_name_map_inv = dict(
        [(v, k) for (k, v) in kde_pdf_axis_name_map.items()])
    for var in info_dict['vars']:
        if(var not in kde_pdf_axis_name_map_inv):
            kde_pdf_axis_name_map_inv[var] = var

    # Select the bin center information from the meta data information file.
    # The "bins" key is for backward compatibility.
    if('bin_centers' in info_dict):
        bin_centers_key = 'bin_centers'
    elif('bins' in info_dict):
        bin_centers_key = 'bins'
    else:
        raise KeyError(
            'The PDF information file is missing "bin_centers" or "bins" key!')

    axis_binnings = [
        BinningDefinition(
            kde_pdf_axis_name_map_inv[var], info_dict[bin_centers_key][idx])
        for (idx, var) in enumerate(info_dict['vars'])
    ]

    # Getting the name of the splinetable file
    splinefit_file_list = ds.get_abs_pathfilename_list(
        ds.get_aux_data_definition(splinefit_key))
    if(len(splinefit_file_list) != 1):
        raise ValueError(
            'The spline fit file list must contain only a single file name! '
            'Currently it contains {} file names!'.format(
                len(splinefit_file_list)))

    pdf = pdf_type(
        axis_binnings=axis_binnings,
        param_set=param_set,
        path_to_pdf_splinefit=splinefit_file_list[0],
        norm_factor_func=norm_factor_func)

    return pdf
