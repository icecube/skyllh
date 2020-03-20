# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains possible background PDF models for the
likelihood function.
"""

from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF
)


class BackgroundMultiDimGridPDF(MultiDimGridPDF, IsBackgroundPDF):
    """This class provides a multi-dimensional background PDF. The PDF is
    created from pre-calculated PDF data on a grid. The grid data is
    interpolated using a :class:`scipy.interpolate.RegularGridInterpolator`
    instance.
    """

    def __init__(self, axis_binnings, pdf_path_to_splinetable=None, pdf_grid_data=None, norm_factor_func=None):
        """Creates a new background PDF instance for a multi-dimensional PDF
        given as PDF values on a grid. The grid data is interpolated with a
        :class:`scipy.interpolate.RegularGridInterpolator` instance. As grid
        points the bin edges of the axis binning definitions are used.

        Parameters
        ----------
        axis_binnings : sequence of BinningDefinition
            The sequence of BinningDefinition instances defining the binning of
            the PDF axes. The name of each BinningDefinition instance defines
            the event field name that should be used for querying the PDF.
        pdf_path_to_splinetable : str
            The path to the file that contains the spline table
            (a pre-computed fit to pdf_grid_data)
        pdf_grid_data : n-dimensional numpy ndarray
            The n-dimensional numpy ndarray holding the PDF values at given grid
            points. The grid points must match the bin edges of the given
            BinningDefinition instances of the `axis_binnings` argument.
        norm_factor_func : callable | None
            The function that calculates a possible required normalization
            factor for the PDF value based on the event properties.
            The call signature of this function
            must be `__call__(pdf, events, fitparams)`, where `pdf` is this PDF
            instance, `events` is a numpy record ndarray holding the events for
            which to calculate the PDF values, and `fitparams` is a dictionary
            with the current fit parameter names and values.
        """
        super(BackgroundMultiDimGridPDF, self).__init__(
            axis_binnings, pdf_path_to_splinetable, pdf_grid_data, norm_factor_func)
