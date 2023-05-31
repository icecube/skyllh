# -*- coding: utf-8 -*-

"""The :mod:`~skyllh.core.model` module contains the base class ``SourceModel``
for modelling a source in the sky. What kind of properties this source has is
modeled by a derived class. The most common one is the PointLikeSource source
model for a point-like source at a given location in the sky.
"""

import numpy as np

from skyllh.core.model import (
    Model,
    ModelCollection,
)
from skyllh.core.py import (
    classname,
    float_cast,
    issequenceof,
    str_cast,
    typename,
)


class SourceModel(
        Model,
):
    """The base class for all source models in SkyLLH. A source can have a
    relative weight w.r.t. other sources.
    """
    def __init__(
            self,
            name=None,
            classification=None,
            weight=None,
            **kwargs):
        """Creates a new source model instance.

        Parameters
        ----------
        name : str | None
            The name of the source model.
        classification : str | None
            The astronomical classification of the source.
        weight : float | None
            The relative weight of the source w.r.t. other sources.
            If set to None, unity will be used.
        """
        super().__init__(
            name=name,
            **kwargs)

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
        """The weight of the source.
        """
        return self._weight

    @weight.setter
    def weight(self, w):
        if w is None:
            w = 1.
        w = float_cast(
            w,
            'The weight property must be castable to type float!')
        self._weight = w


class SourceModelCollection(
        ModelCollection,
):
    """This class describes a collection of source models. It can be used to
    group sources into a single object, for instance for a stacking analysis.
    """
    @staticmethod
    def cast(
            obj,
            errmsg=None,
            **kwargs):
        """Casts the given object to a SourceModelCollection object. If the cast
        fails, a TypeError with the given error message is raised.

        Parameters
        ----------
        obj : SourceModel | sequence of SourceModel | SourceModelCollection |
                None
            The object that should be casted to SourceModelCollection.
            If set to None, an empty SourceModelCollection is created.
        errmsg : str | None
            The error message if the cast fails.
            If set to None, a generic error message will be used.

        Additional keyword arguments
        ----------------------------
        Additional keyword arguments are passed to the constructor of the
        SourceModelCollection class.

        Raises
        ------
        TypeError
            If the cast failed.
        """
        if obj is None:
            return SourceModelCollection(
                sources=None, source_type=SourceModel, **kwargs)

        if isinstance(obj, SourceModel):
            return SourceModelCollection(
                sources=[obj], source_type=SourceModel, **kwargs)

        if isinstance(obj, SourceModelCollection):
            return obj

        if issequenceof(obj, SourceModel):
            return SourceModelCollection(
                sources=obj, source_type=SourceModel, **kwargs)

        if errmsg is None:
            errmsg = (f'Cast of object "{str(obj)}" of type '
                      f'"{typename(obj)}" to SourceModelCollection failed!')
        raise TypeError(errmsg)

    def __init__(
            self,
            sources=None,
            source_type=None,
            **kwargs):
        """Creates a new source collection.

        Parameters
        ----------
        sources : sequence of source_type instances | None
            The sequence of sources this collection should be initalized with.
            If set to None, an empty SourceModelCollection instance is created.
        source_type : type | None
            The type of the source.
            If set to None (default), SourceModel will be used.
        """
        if source_type is None:
            source_type = SourceModel

        super().__init__(
            models=sources,
            model_type=source_type,
            **kwargs)

    @property
    def source_type(self):
        """(read-only) The type of the source model.
        This property is an alias for the `obj_type` property.
        """
        return self.model_type

    @property
    def sources(self):
        """(read-only) The list of sources of type ``source_type``.
        """
        return self.models


class IsPointlike(
        object):
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
            v,
            'The ra property must be castable to type float!')
        self._set_ra_func(self._ra_func_instance, v)

    @property
    def dec(self):
        """The declination coordinate of the point-like source.
        """
        return self._get_dec_func(self._dec_func_instance)

    @dec.setter
    def dec(self, v):
        v = float_cast(
            v,
            'The dec property must be castable to type float!')
        self._set_dec_func(self._dec_func_instance, v)


class PointLikeSource(
        SourceModel,
        IsPointlike):
    """The PointLikeSource class is a source model for a point-like source
    object in the sky at a given location (right-ascention and declination).
    """
    def __init__(
            self,
            ra,
            dec,
            name=None,
            weight=None,
            **kwargs):
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

        s = (
            f'{classname(self)}: "{self.name}": '
            '{ '
            f'ra={np.rad2deg(self.ra):.3f} deg, '
            f'dec={np.rad2deg(self.dec):.3f} deg'
            f'{c}'
            ' }'
        )

        return s
