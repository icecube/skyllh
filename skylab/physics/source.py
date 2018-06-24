# -*- coding: utf-8 -*-

"""The ``source`` module contains the base class ``SourceModel`` for modelling a
source in the sky. What kind of properties this source has is modeled by a
derived class. The most common one is the PointSource source model for a point
source at a given position in the sky with a given flux model.
"""
from skylab.core.py import ObjectCollection
from skylab.physics.flux import FluxModel

class SourceModel(object):
    """The base class for all source models in Skylab.
    """
    def __init__(self, fluxmodel):
        self.fluxmodel = fluxmodel

    @property
    def fluxmodel(self):
        """The flux model of the source. It's an instance of class derived from
        class FluxModel.
        """
        return self._fluxmodel
    @fluxmodel.setter
    def fluxmodel(self, obj):
        if(not isinstance(obj, FluxModel)):
            raise TypeError('The fluxmodel property must be an instance of FluxModel!')
        self._fluxmodel = obj

class SourceCollection(ObjectCollection):
    """This class describes a collection of sources. It can be used to group
    sources into a single object, for instance for a stacking analysis.
    """
    def __init__(self, source_t=None):
        """Creates a new source collection.

        Parameters
        ----------
        source_t : type | None
            The type of the source. If set to None (default), SourceModel will
            be used.
        """
        if(source_t is None):
            source_t = SourceModel
        super(SourceCollection, self).__init__(obj_t=source_t)

    @property
    def sources(self):
        """(read-only) The list of sources of type SourceModel.
        """
        return self.objects

class Catalog(SourceCollection):
    """This class describes a catalog of sources. It is derived from
    SourceCollection. A catalog has a name.
    """
    def __init__(self, name, source_t=None):
        """Creates a new source catalog.

        Parameters
        ----------
        name : str
            The name of the catalog.
        source_t : type | None
            The type of the source. If set to None (default), the default type
            defined by SourceCollection will be used.
        """
        super(Catalog, self).__init__(source_t=source_t)
        self.name = name

    @property
    def name(self):
        """The name of the catalog.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

class PointSource(SourceModel):
    """The PointSource class is a source model for a point source like object
    in the sky at a given location with a given flux model.
    """
    def __init__(self, ra, dec, fluxmodel):
        super(PointSource, self).__init__(fluxmodel)
        self.ra = ra
        self.dec = dec

    @property
    def ra(self):
        """The right-ascention angle in radian of the source position.
        """
        return self._ra
    @ra.setter
    def ra(self, v):
        if(not isinstance(v, float)):
            raise TypeError('The ra property must be of type float!')
        self._ra = v

    @property
    def dec(self):
        """The declination angle in radian of the source position.
        """
        return self._dec
    @dec.setter
    def dec(self, v):
        if(not isinstance(v, float)):
            raise TypeError('The dec property must be of type float!')
        self._dec = v

class PointSourceCatalog(Catalog):
    """Describes a catalog of point sources.
    """
    def __init__(self, name):
        """Creates a new point source catalog of the given name.

        Parameters
        ----------
        name : str
            The name of the point source catalog.
        """
        super(PointSourceCatalog, self).__init__(name, source_t=PointSource)
