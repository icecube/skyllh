# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains background PDF classes for the
likelihood function.
"""

from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF,
    NDPhotosplinePDF,
)


class BackgroundMultiDimGridPDF(
        MultiDimGridPDF,
        IsBackgroundPDF):
    """This class provides a multi-dimensional background PDF defined on a grid.
    The PDF is created from pre-calculated PDF data on a grid. The grid data is
    interpolated using a :class:`scipy.interpolate.RegularGridInterpolator`
    instance.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new :class:`~skyllh.core.pdf.MultiDimGridPDF` instance that
        is also derived from :class:`~skyllh.core.pdf.IsBackgroundPDF`.

        For the documentation of arguments see the documentation of the
        :meth:`~skyllh.core.pdf.MultiDimGridPDF.__init__` method.
        """
        super().__init__(*args, **kwargs)


class BackgroundNDPhotosplinePDF(
        NDPhotosplinePDF,
        IsBackgroundPDF):
    """This class provides a multi-dimensional background PDF created from a
    n-dimensional photospline fit. The photospline package is used to evaluate
    the PDF fit.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new :class:`~skyllh.core.pdf.NDPhotosplinePDF` instance
        that is also derived from :class:`~skyllh.core.pdf.IsBackgroundPDF`.

        For the documentation of arguments see the documentation of the
        :meth:`~skyllh.core.pdf.NDPhotosplinePDF.__init__` method.
        """
        super().__init__(*args, **kwargs)
