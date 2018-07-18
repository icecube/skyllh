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
    def __init__(self, source_type=None, sources=None):
        """Creates a new source collection.

        Parameters
        ----------
        source_type : type | None
            The type of the source. If set to None (default), SourceModel will
            be used.
        sources : sequence of source_t instances | None
            The sequence of sources this collection should be initalized with.
        """
        if(source_t is None):
            source_t = SourceModel
        super(SourceCollection, self).__init__(obj_type=source_t, obj_list=sources)

    @property
    def source_type(self):
        """(read-only) The type of the source model.
        """
        return self.obj_type

    @property
    def sources(self):
        """(read-only) The list of sources of type ``source_t``.
        """
        return self.objects

class Catalog(SourceCollection):
    """This class describes a catalog of sources. It is derived from
    SourceCollection. A catalog has a name.
    """
    def __init__(self, name, source_type=None, sources=None):
        """Creates a new source catalog.

        Parameters
        ----------
        name : str
            The name of the catalog.
        source_type : type | None
            The type of the source. If set to None (default), the default type
            defined by SourceCollection will be used.
        sources : sequence of source_t | None
            The sequence of sources this catalog should be initalized with.
        """
        super(Catalog, self).__init__(source_type=source_type, sources=sources)
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

    def as_source_collection(self):
        """Creates a SourceCollection object for this catalog and returns it.
        """
        source_collection = SourceCollection(source_t=self.source_t, sources=self.sources)

class PointLikeSource(SourceModel):
    """The PointLikeSource class is a source model for a point-like source
    object in the sky at a given location (right-ascention and declination) with
    a given flux model.
    """
    def __init__(self, ra, dec, fluxmodel):
        super(PointLikeSource, self).__init__(fluxmodel)
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


class PointLikeSourceCollection(SourceCollection):
    """Describes a collection of point-like sources.
    """
    def __init__(self, sources=None):
        """Creates a new collection of PointLikeSource objects.

        Parameters
        ----------
        sources : sequence of PointLikeSource instances | None
            The sequence of PointLikeSource objects this collection should be
            initalized with.
        """
        super(PointLikeSourceCollection, self).__init__(
            source_type=PointLikeSource, sources=sources)


class PointLikeSourceCatalog(Catalog):
    """Describes a catalog of point-like sources. The difference to a
    PointLikeSourceCollection is the additional properties of a catalog, e.g.
    the name.
    """
    def __init__(self, name, sources=None):
        """Creates a new point source catalog of the given name.

        Parameters
        ----------
        name : str
            The name of the point-like source catalog.
        sources : sequence of PointLikeSource instances | None
            The sequence of PointLikeSource instances this catalog should be
            initalized with.
        """
        super(PointLikeSourceCatalog, self).__init__(
            name=name, source_type=PointLikeSource, sources=sources)
