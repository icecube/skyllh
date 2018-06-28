# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains possible background PDF models for the
likelihood function.
The base class of all background pdf models is ``BackgroundPDF``.
"""

from skylab.core.pdf import PDF

class BackgroundPDF(PDF):
    """This is the base class for all background PDF models.
    """
    def __init__(self):
        super(BackgroundPDF, self).__init__()

class SpatialBackgroundPDF(BackgroundPDF):
    """This is the base class for all spatial background PDF models.
    """
    def __init__(self):
        super(SpatialBackgroundPDF, self).__init__()
