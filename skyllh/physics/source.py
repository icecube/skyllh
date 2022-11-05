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
    issequence,
    str_cast,
)
from skyllh.core.model import Model


# NOTE: This class should live somewhere close to the source stacking code.
class SourceWeights(object):
    """This is a helper class for the source stacking algorithm.
    It stores the relative weights of a source, i.e. weights and gradients.
    There are two weights that should be included. One is the detector weight,
    which is declination dependent, and the other is a hypothesis weight, and
    that is provided by the user.
    """
    def __init__(self, src_w=None, src_w_grad=None, src_w_W=None):
        self.src_w = src_w
        self.src_w_grad = src_w_grad
        self.src_w_W = src_w_W

    @property
    def src_w(self):
        """The relative weight of the source(s).
        """
        return self._src_w
    @src_w.setter
    def src_w(self, v):
        v = float_cast(
            v, 'The src_w property must be castable to type float!')
        self._src_w = v

    @property
    def src_w_grad(self):
        """The relative weight gradients of the source(s).
        """
        return self._src_w_grad
    @src_w_grad.setter
    def src_w_grad(self, v):
        v = float_cast(
            v, 'The src_w_grad property must be castable to type float!')
        self._src_w_grad = v

    @property
    def src_w_W(self):
        """The hypothesis weight of the source(s).
        """
        return self._src_w_W
    @src_w_W.setter
    def src_w_W(self, v):
        v = float_cast(
            v, 'The src_w_W property must be castable to type float!')
        self._src_w_W = v


class SourceModel(Model):
    """The base class for all source models in SkyLLH. A source can have a
    relative weight w.r.t. other sources.
    """
    def __init__(self, classification=None, weight=None, *args, **kwargs):
        """Creates a new source model instance.

        Parameters
        ----------
        classification : str | None
            The astronomical classification of the source.
        weight : float | None
            The relative weight of the source w.r.t. other sources.
            If set to None, unity will be used.
        """
        super().__init__(*args, **kwargs)

        self.classification = classification
        self.weight = weight

    @property
    def classification(self):
        """The astronomical classification of the source.
        """
        return self._classification
    @classification.setter
    def classification(self, c):
        self._classification = str_cast(
            c,
            'The classification property must be castable to type str!',
            allow_None=True)

    @property
    def weight(self):
        """The weight of the source. This can be None,
        """
        return self._weight
    @weight.setter
    def weight(self, w):
        if w is None:
            w = 1.
        w = float_cast(
            w, 'The weight property must be castable to type float!')
        self._weight = w


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

    def __init__(self, sources=None, source_type=None):
        """Creates a new source collection.

        Parameters
        ----------
        sources : sequence of source_type instances | None
            The sequence of sources this collection should be initalized with.
        source_type : type | None
            The type of the source. If set to None (default), SourceModel will
            be used.
        """
        if(source_type is None):
            source_type = SourceModel
        super(SourceCollection, self).__init__(sources, obj_type=source_type)

    @property
    def source_type(self):
        """(read-only) The type of the source model.
        This property is an alias for the `obj_type` property.
        """
        return self.obj_type

    @property
    def sources(self):
        """(read-only) The list of sources of type ``source_type``.
        """
        return self.objects


class SourceCatalog(SourceCollection):
    """This class describes a catalog of sources. It is derived from
    SourceCollection. A catalog has a name.
    """
    def __init__(self, name, sources=None, source_type=None):
        """Creates a new source catalog.

        Parameters
        ----------
        name : str
            The name of the catalog.
        sources : sequence of source_type | None
            The sequence of sources this catalog should be initalized with.
        source_type : type | None
            The type of the source class. If set to None (default), the
            default type defined by SourceCollection will be used.
        """
        super().__init__(source_type=source_type, sources=sources)

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

    def __str__(self):
        s = '"' + self.name + '" ' + super().__str__()
        return s

    def as_SourceCollection(self):
        """Creates a SourceCollection object for this catalog and returns it.
        """
        source_collection = SourceCollection(
            source_type=self.source_type,
            sources=self.sources)
        return source_collection


class IsPointlike(object):
    """This is a classifier class that can be used by other classes to indicate
    that the specific class describes a point-like object.
    """
    def __init__(
            self,
            ra_func_instance=None,
            get_ra_func=None,
            set_ra_func=None,
            dec_func_instance=None,
            get_dec_func=None,
            set_dec_func=None,
            **kwargs):
        """Constructor method. Gets called when the an instance of a class is
        created which derives from this IsPointlike class.

        Parameters
        ----------
        ra_func_instance : object
            The instance object the right-ascention property's getter and setter
            functions are defined in.
        get_ra_func : callable
            The callable object of the getter function of the right-ascention
            property. It must have the call signature
            `__call__(ra_func_instance)`.
        set_ra_func : callable
            The callable object of the setter function of the right-ascention
            property. It must have the call signature
            `__call__(ra_func_instance, value)`.
        dec_func_instance : object
            The instance object the declination property's getter and setter
            functions are defined in.
        get_dec_func : object
            The callable object of the getter function of the declination
            property. It must have the call signature
            `__call__(dec_func_instance)`.
        set_dec_func : object
            The callable object of the setter function of the declination
            property. It must have the call signature
            `__call__(dec_func_instance, value)`.
        """
        super().__init__(**kwargs)

        self._ra_func_instance = ra_func_instance
        self._get_ra_func = get_ra_func
        self._set_ra_func = set_ra_func

        self._dec_func_instance = dec_func_instance
        self._get_dec_func = get_dec_func
        self._set_dec_func = set_dec_func

    @property
    def ra(self):
        """The right-ascention coordinate of the point-like source.
        """
        return self._get_ra_func(self._ra_func_instance)
    @ra.setter
    def ra(self, v):
        v = float_cast(
            v, 'The ra property must be castable to type float!')
        self._set_ra_func(self._ra_func_instance, v)

    @property
    def dec(self):
        """The declination coordinate of the point-like source.
        """
        return self._get_dec_func(self._dec_func_instance)
    @dec.setter
    def dec(self, v):
        v = float_cast(
            v, 'The dec property must be castable to type float!')
        self._set_dec_func(self._dec_func_instance, v)


class PointLikeSource(IsPointlike, SourceModel):
    """The PointLikeSource class is a source model for a point-like source
    object in the sky at a given location (right-ascention and declination).
    """
    def __init__(self, ra, dec, name=None, weight=None, *args, **kwargs):
        """Creates a new PointLikeSource instance for defining a point-like
        source.

        Parameters
        ----------
        ra : float
            The right-ascention coordinate of the source in radians.
        dec : float
            The declination coordinate of the source in radians.
        name : str | None
            The name of the source.
        weight : float | None
            The relative weight of the source w.r.t. other sources.
            If set to None, unity will be used.
        """
        super().__init__(
            name=name,
            weight=weight,
            ra_func_instance=self,
            get_ra_func=type(self)._get_ra,
            set_ra_func=type(self)._set_ra,
            dec_func_instance=self,
            get_dec_func=type(self)._get_dec,
            set_dec_func=type(self)._set_dec,
            *args,
            **kwargs,
        )

        self.ra = ra
        self.dec = dec

    def _get_ra(self):
        return self._ra

    def _set_ra(self, ra):
        self._ra = ra

    def _get_dec(self):
        return self._dec

    def _set_dec(self, dec):
        self._dec = dec

    def __str__(self):
        """Pretty string representation.
        """
        c = ''
        if self.classification is not None:
            c = f', classification={self.classification}'
        s = classname(self) + ': "%s": { ra=%.3f deg, dec=%.3f deg%s }'%(
            self.name, np.rad2deg(self.ra), np.rad2deg(self.dec), c)
        return s
