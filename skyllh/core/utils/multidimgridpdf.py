# -*- coding: utf-8 -*-

"""This module contains utility functions for creating and managing
MultiDimGridPDF instances.
"""

import numpy as np

from skyllh.core.binning import (
    BinningDefinition,
)
from skyllh.core.pdf import (
    MultiDimGridPDF,
)
from skyllh.core.py import (
    classname,
)


def get_kde_pdf_sig_spatial_norm_factor_func(
        log10_psi_name='log10_psi'):
    """Returns the standard normalization factor function for the spatial
    signal MultiDimGridPDF, which is created from KDE PDF values.
    It can be used for the ``norm_factor_func`` argument of the
    ``create_MultiDimGridPDF_from_photosplinetable`` and
    ``create_MultiDimGridPDF_from_kde_pdf`` function.

    Parameters
    ----------
    log10_psi_name : str
        The name of the event data field for the log10(psi) values.
    """
    def kde_pdf_sig_spatial_norm_factor_func(
            pdf,
            tdm,
            params_recarray,
            eventdata,
            evt_mask=None):

        log10_psi_idx = pdf._axes.get_index_by_name(log10_psi_name)

        if evt_mask is None:
            psi = 10**eventdata[log10_psi_idx]
        else:
            psi = 10**eventdata[log10_psi_idx][evt_mask]

        norm = 1. / (2 * np.pi * np.log(10) * psi * np.sin(psi))

        return norm

    return kde_pdf_sig_spatial_norm_factor_func


def get_kde_pdf_bkg_norm_factor_func():
    """Returns the standard normalization factor function for the background
    MultiDimGridPDF, which is created from KDE PDF values.
    It can be used for the ``norm_factor_func`` argument of the
    ``create_MultiDimGridPDF_from_photosplinetable`` and
    ``create_MultiDimGridPDF_from_kde_pdf`` function.
    """
    def kde_pdf_bkg_norm_factor_func(
            pdf,
            tdm,
            params_recarray,
            eventdata,
            evt_mask=None):

        return 1. / (2 * np.pi)

    return kde_pdf_bkg_norm_factor_func


def create_MultiDimGridPDF_from_photosplinetable(
        multidimgridpdf_cls,
        pmm,
        ds,
        data,
        info_key,
        splinetable_key,
        kde_pdf_axis_name_map_key='KDE_PDF_axis_name_map',
        norm_factor_func=None,
        cache_pd_values=False,
        tl=None,
        **kwargs,
):
    """
    Creates a MultiDimGridPDF instance with pdf values taken from a photospline
    pdf, i.e. a spline interpolation of KDE PDF values stored in a splinetable
    on disk.

    Parameters
    ----------
    multidimgridpdf_cls : subclass of MultiDimGridPDF
        The MultiDimGridPDF class, which should be used.
    pmm : instance of ParameterModelMapper
        The instance of ParameterModelMapper, which defines the mapping of
        global parameters to local model parameters.
    ds : instance of Dataset
        The instance of Dataset the PDF applies to.
    data : instance of DatasetData
        The instance of DatasetData that holds the experimental and monte-carlo
        data of the dataset.
    info_key : str
        The auxiliary data name for the file containing PDF information.
    splinetable_key : str
        The auxiliary data name for the name of the file containing the
        photospline spline table.
    kde_pdf_axis_name_map_key : str
        The auxiliary data name for the KDE PDF axis name map.
    norm_factor_func : callable | None
        The function that calculates a possible required normalization
        factor for the PDF value based on the event properties.
        For more information about this argument see the documentation of the
        :meth:`skyllh.core.pdf.MultiDimGridPDF.__init__` method.
    cache_pd_values : bool
        Flag if the probability density values should get cached by the
        MultiDimGridPDF class.
    tl : instance of TimeLord | None
        The optional instance of TimeLord to use for measuring timing
        information.

    Returns
    -------
    pdf : instance of ``multidimgridpdf_cls``
        The created PDF instance of MultiDimGridPDF.
    """
    if not issubclass(multidimgridpdf_cls, MultiDimGridPDF):
        raise TypeError(
            'The multidimgridpdf_cls argument must be a subclass of '
            'MultiDimGridPDF! '
            f'Its current type is {classname(multidimgridpdf_cls)}.')

    # Load the PDF data from the auxilary files.
    num_dict = ds.load_aux_data(info_key, tl=tl)

    kde_pdf_axis_name_map = ds.load_aux_data(kde_pdf_axis_name_map_key, tl=tl)
    kde_pdf_axis_name_map_inv = dict(
        [(v, k) for (k, v) in kde_pdf_axis_name_map.items()])
    for var in num_dict['vars']:
        if var not in kde_pdf_axis_name_map_inv:
            kde_pdf_axis_name_map_inv[var] = var

    if 'bin_centers' in num_dict:
        bin_centers_key = 'bin_centers'
    elif 'bins' in num_dict:
        bin_centers_key = 'bins'
    else:
        raise KeyError(
            'The PDF information file is missing "bin_centers" or "bins" key!')

    axis_binnings = [
        BinningDefinition(
            name=kde_pdf_axis_name_map_inv[var],
            binedges=num_dict[bin_centers_key][idx])
        for (idx, var) in enumerate(num_dict['vars'])
    ]

    # Getting the name of the splinetable file
    splinetable_file = ds.get_abs_pathfilename_list(
        ds.get_aux_data_definition(splinetable_key))[0]

    pdf = multidimgridpdf_cls(
        pmm=pmm,
        axis_binnings=axis_binnings,
        path_to_pdf_splinetable=splinetable_file,
        norm_factor_func=norm_factor_func,
        cache_pd_values=cache_pd_values,
        **kwargs)

    return pdf


def create_MultiDimGridPDF_from_kde_pdf(  # noqa: C901
        multidimgridpdf_cls,
        pmm,
        ds,
        data,
        numerator_key,
        denumerator_key=None,
        kde_pdf_axis_name_map_key='KDE_PDF_axis_name_map',
        norm_factor_func=None,
        cache_pd_values=False,
        tl=None,
        **kwargs,
):
    """Creates a MultiDimGridPDF instance with pdf values taken from KDE PDF
    values stored in the dataset's auxiliary data.

    Parameters
    ----------
    multidimgridpdf_cls : subclass of MultiDimGridPDF
        The MultiDimGridPDF class, which should be used.
    pmm : instance of ParameterModelMapper
        The instance of ParameterModelMapper, which defines the mapping of
        global parameters to local model parameters.
    ds : instance of Dataset
        The instance of Dataset the PDF applies to.
    data : instance of DatasetData
        The instance of DatasetData that holds the auxiliary data of the
        dataset.
    numerator_key : str
        The auxiliary data name for the PDF numerator array.
    denumerator_key : str | None
        The auxiliary data name for the PDF denumerator array.
        This can be ``None``, if no denumerator array is required.
    kde_pdf_axis_name_map_key : str
        The auxiliary data name for the KDE PDF axis name map.
    norm_factor_func : callable | None
        The function that calculates a possible required normalization
        factor for the PDF value based on the event properties.
        For more information about this argument see the documentation of the
        :meth:`skyllh.core.pdf.MultiDimGridPDF.__init__` method.
    cache_pd_values : bool
        Flag if the probability density values should get cached by the
        MultiDimGridPDF class.
    tl : instance of TimeLord | None
        The optional instance of TimeLord to use for measuring timing
        information.

    Returns
    -------
    pdf : instance of ``multidimgridpdf_cls``
        The created PDF instance of MultiDimGridPDF.
    """
    if not issubclass(multidimgridpdf_cls, MultiDimGridPDF):
        raise TypeError(
            'The multidimgridpdf_cls argument must be a subclass of '
            'MultiDimGridPDF! '
            f'Its current type is {classname(multidimgridpdf_cls)}.')

    # Load the PDF data from the auxilary files.
    num_dict = ds.load_aux_data(numerator_key, tl=tl)

    denum_dict = None
    if denumerator_key is not None:
        denum_dict = ds.load_aux_data(denumerator_key, tl=tl)

    kde_pdf_axis_name_map = ds.load_aux_data(kde_pdf_axis_name_map_key, tl=tl)
    kde_pdf_axis_name_map_inv = dict(
        [(v, k) for (k, v) in kde_pdf_axis_name_map.items()])
    for var in num_dict['vars']:
        if var not in kde_pdf_axis_name_map_inv:
            kde_pdf_axis_name_map_inv[var] = var

    if 'bin_centers' in num_dict:
        bin_centers_key = 'bin_centers'
    elif 'bins' in num_dict:
        bin_centers_key = 'bins'
    else:
        raise KeyError(
            'The PDF information file is missing "bin_centers" or "bins" key!')

    axis_binnings = [
        BinningDefinition(
            kde_pdf_axis_name_map_inv[var], num_dict[bin_centers_key][idx])
        for (idx, var) in enumerate(num_dict['vars'])
    ]

    vals = num_dict['pdf_vals']
    if denum_dict is not None:
        # A denumerator is required, so we need to divide the numerator pdf
        # values by the denumerator pdf values, by preserving the correct axis
        # order.
        # Construct the slicing selector for the denumerator pdf values array to
        # match the axis order of the numerator pdf values array.
        selector = []
        for var in num_dict['vars']:
            if var in denum_dict['vars']:
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

    pdf = multidimgridpdf_cls(
        pmm=pmm,
        axis_binnings=axis_binnings,
        pdf_grid_data=vals,
        norm_factor_func=norm_factor_func,
        cache_pd_values=cache_pd_values,
        **kwargs)

    return pdf
