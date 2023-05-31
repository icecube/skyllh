# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

"""This module defines the base class for any model class used in SkyLLH.
"""

from skyllh.core.py import (
    NamedObjectCollection,
    issequenceof,
    str_cast,
    typename,
)


class Model(
        object):
    """This class provides a base class for all model classes used in SkyLLH.
    Models could be for instance source models or background models.
    """
    def __init__(
            self,
            name=None,
            **kwargs):
        """Creates a new Model instance.

        Parameters
        ----------
        name : str | None
            The name of the model. If set to `None`, the id of the object is
            taken as name.
        """
        super().__init__(
            **kwargs)

        if name is None:
            name = self.id

        self.name = name

    @property
    def name(self):
        """The name of the model.
        """
        return self._name

    @name.setter
    def name(self, name):
        name = str_cast(
            name,
            'The name property must be castable to type str!')
        self._name = name

    @property
    def id(self):
        """(read-only) The ID of the model. It's an integer generated with
        Python's `id` function. Hence, it's related to the memory address
        of the object.
        """
        return id(self)


class ModelCollection(
        NamedObjectCollection):
    """This class describes a collection of Model instances. It can be
    used to group several models into a single object.
    """
    @staticmethod
    def cast(
            obj,
            errmsg=None,
            **kwargs):
        """Casts the given object to a ModelCollection object.
        If the cast fails, a TypeError with the given error message is raised.

        Parameters
        ----------
        obj : Model instance | sequence of Model instances |
                ModelCollection | None
            The object that should be casted to ModelCollection.
            If set to None, an empty ModelCollection is created.
        errmsg : str | None
            The error message if the cast fails.
            If set to None, a generic error message will be used.

        Additional keyword arguments
        ----------------------------
        Additional keyword arguments are passed to the constructor of the
        ModelCollection class.

        Raises
        ------
        TypeError
            If the cast failed.

        Returns
        -------
        model_collection : instance of ModelCollection
            The created ModelCollection instance. If `obj` is already a
            ModelCollection instance, it will be returned.
        """
        if obj is None:
            return ModelCollection(
                models=None, model_type=Model, **kwargs)

        if isinstance(obj, Model):
            return ModelCollection(
                models=[obj], model_type=Model, **kwargs)

        if isinstance(obj, ModelCollection):
            return obj

        if issequenceof(obj, Model):
            return ModelCollection(
                models=obj, model_type=Model, **kwargs)

        if errmsg is None:
            errmsg = (f'Cast of object "{str(obj)}" of type '
                      f'"{typename(type(obj))}" to ModelCollection failed!')
        raise TypeError(errmsg)

    def __init__(
            self,
            models=None,
            model_type=None,
            **kwargs):
        """Creates a new Model collection. The type of the model instances this
        collection holds can be restricted, by setting the model_type argument.

        Parameters
        ----------
        models : sequence of model_type instances | None
            The sequence of models this collection should be initalized with.
        model_type : type | None
            The type of the model. It must be a subclass of class ``Model``.
            If set to None (default), Model will be used.
        """
        if model_type is None:
            model_type = Model
        if not issubclass(model_type, Model):
            raise TypeError(
                'The model_type argument must be a subclass of Model!')

        super().__init__(
            objs=models,
            obj_type=model_type,
            **kwargs)

    @property
    def model_type(self):
        """(read-only) The type of the model.
        """
        return self.obj_type

    @property
    def models(self):
        """(read-only) The list of models of type `model_type`.
        """
        return self.objects


class DetectorModel(Model):
    """This class provides a base class for a detector model. It can be used
    in combination with the ParameterModelMapper class.
    """
    def __init__(self, name, **kwargs):
        """Creates a new DetectorModel instance.

        Parameters
        ----------
        name : str
            The name of the detector model.
        """
        super().__init__(
            name=name,
            **kwargs)
