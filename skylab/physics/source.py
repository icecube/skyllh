# -*- coding: utf-8 -*-

"""The ``source`` module contains the base class ``SourceModel`` for modelling a
source in the sky. What kind of properties this source has is modeled by a
derived class. The most common one is the PointSource source model for a point
source at a given position in the sky with a given flux model.
"""

class SourceModel(object):
    """The base class for all sources in Skylab.
    """
    def __init__(self):
        pass

class PointSource(SourceModel):
    """The PointSource class is a source model for a point source like object
    in the sky at a given location with a given flux model.
    """
    def __init__(self, ra, dec, flux):
