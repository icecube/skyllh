# -*- coding: utf-8 -*-

"""The ``source`` module contains the base class ``SourceModel`` for modelling a
source in the sky. What kind of properties this source has is modeled by a
derived class. The most common one is the PointLikeSource source model for a
point-like source at a given location in the sky with a given flux model.
"""

import numpy as np

from skyllh.core.py import (
    ObjectCollection,
    classname,
    float_cast,
    issequence
)


class SourceLocation(object):
    """Stores the location of a source, i.e. right-ascention and declination.
    """
    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec

    @property
    def ra(self):
        """The right-ascention angle in radian of the source position.
        """
        return self._ra
    @ra.setter
    def ra(self, v):
        v = float_cast(v, 'The ra property must be castable to type float!')
        self._ra = v

    @property
    def dec(self):
        """The declination angle in radian of the source position.
        """
        return self._dec
    @dec.setter
    def dec(self, v):
        v = float_cast(v, 'The dec property must be castable to type float!')
        self._dec = v

class SourceWeights(object):
    """Stores the relative weights of a source, i.e. weights and gradients.
       There are two weights that should be included. one is the detector weight,
       which is declination dependent, and the other is a hypothesis weight, and that
       is provided by the user.
    """
    def __init__(self, src_w=None, src_w_grad=None, src_w_W=None):
        self.src_w      = src_w
        self.src_w_grad = src_w_grad
        self.src_w_W    = src_w_W
    @property
    def src_w(self):
        """The relative weight of the source(s).
        """
        return self._src_w
    @src_w.setter
    def src_w(self, v):
        v = float_cast(v, 'The src_w property must be castable to type float!')
        self._src_w = v

    @property
    def src_w_grad(self):
        """The relative weight gradients of the source(s).
        """
        return self._src_w_grad
    @src_w_grad.setter
    def src_w_grad(self, v):
        v = float_cast(v, 'The src_w_grad property must be castable to type float!')
        self._src_w_grad = v

    @property
    def src_w_W(self):
        """The hypothesis weight of the source(s).
        """
        return self._src_w_W
    @src_w_W.setter
    def src_w_W(self, v):
        v = float_cast(v, 'The src_w_W property must be castable to type float!')
        self._src_w_W = v


class SourceModel(object):
    """The base class for all source models in Skyllh. Each source has a central
    location given by a right-ascention and declination location.
    """
    def __init__(self, ra, dec, src_w=None, src_w_grad=None, src_w_W=None):
        self.loc = SourceLocation(ra, dec)
        src_w = np.ones_like(self.loc.ra, dtype=np.float64)
        src_w_grad = np.zeros_like(self.loc.ra, dtype=np.float64)

        if (src_w_W is None):
            src_w_W = np.ones_like(self.loc.ra, dtype=np.float64)

        self.weight = SourceWeights(src_w, src_w_grad, src_w_W)

    @property
    def loc(self):
        """The location of the source.
        """
        return self._loc
    @loc.setter
    def loc(self, srcloc):
        if(not isinstance(srcloc, SourceLocation)):
            raise TypeError('The loc property must be an instance of SourceLocation!')
        self._loc = srcloc

    @property
    def weight(self):
        """The weight of the source.
        """
        return self._weight
    @weight.setter
    def weight(self, w_src):
        if(not isinstance(w_src, SourceWeights)):
            raise TypeError('The weight property must be an instance of SourceWeights!')
        self._weight = w_src

    @property
    def id(self):
        """(read-only) The ID of the source. It's an integer generated with the
        id() function. Hence, it's related to the memory address of the object.
        """
        return id(self)


class SourceCollection(ObjectCollection):
    """This class describes a collection of sources. It can be used to group
    sources into a single object, for instance for a stacking analysis.
    """
    @staticmethod
    def cast(obj, errmsg):
        """Casts the given object to a SourceCollection object. If the cast
        fails, a TypeError with the given error message is raised.

        Parameters
        ----------
        obj : SourceModel | sequence of SourceModel | SourceCollection
            The object that should be casted to SourceCollection.
        errmsg : str
            The error message if the cast fails.

        Raises
        ------
        TypeError
            If the cast fails.
        """
        if(isinstance(obj, SourceModel)):
            obj = SourceCollection(SourceModel, [obj])
        if(not isinstance(obj, SourceCollection)):
            if(issequence(obj)):
                obj = SourceCollection(SourceModel, obj)
            else:
                raise TypeError(errmsg)
        return obj

    def __init__(self, source_type=None, sources=None):
        """Creates a new source collection.

        Parameters
        ----------
        source_type : type | None
            The type of the source. If set to None (default), SourceModel will
            be used.
        sources : sequence of source_type instances | None
            The sequence of sources this collection should be initalized with.
        """
        if(source_type is None):
            source_type = SourceModel
        super(SourceCollection, self).__init__(sources, obj_type=source_type)

    @property
    def source_type(self):
        """(read-only) The type of the source model.
        """
        return self.obj_type

    @property
    def sources(self):
        """(read-only) The list of sources of type ``source_type``.
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
        sources : sequence of source_type | None
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
        source_collection = SourceCollection(source_type=self.source_type, sources=self.sources)
        return source_collection


class PointLikeSource(SourceModel):
    """The PointLikeSource class is a source model for a point-like source
    object in the sky at a given location (right-ascention and declination).
    """
    def __init__(self, ra, dec, src_w=None, src_w_grad=None, src_w_W=None):
        super(PointLikeSource, self).__init__(ra, dec, src_w, src_w_grad, src_w_W)

    @property
    def ra(self):
        """(read-only) The right-ascention angle in radian of the source
        position.
        This is a short-cut for `self.loc.ra`.
        """
        return self._loc._ra

    @property
    def dec(self):
        """(read-only) The declination angle in radian of the source position.
        This is a short-cut for `self.loc.dec`.
        """
        return self._loc._dec

    def __str__(self):
        """Pretty string representation of this class instance.
        """
        s = classname(self) + ': { ra=%.3f deg, dec=%.3f deg }'%(
            np.rad2deg(self.ra), np.rad2deg(self.dec))
        return s


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

    @property
    def ra(self):
        """(read-only) The ndarray with the right-ascention of all the sources.
        """
        return np.array([ src.ra for src in self ])

    @property
    def dec(self):
        """(read-only) The ndarray with the declination of all the sources.
        """
        return np.array([ src.dec for src in self ])


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
