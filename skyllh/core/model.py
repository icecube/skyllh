# -*- coding: utf-8 -*-
# Author: Martin Wolf <mail@martin-wolf.org>

"""This module defines the base class for any model class used in SkyLLH.
"""

from skyllh.core.py import (
    NamedObjectCollection,
    issequence,
    str_cast
)

class Model(object):
    """This class provides a base class for all model classes used in SkyLLH.
    Models could be for instance source models or background models.
    """
    def __init__(self, name=None):
        """Creates a new Model instance.

        Parameters
        ----------
        name : str | None
            The name of the model. If set to `None`, the id of the object is
            taken as name.
        """
        super(Model, self).__init__()

        if(name is None):
            name = self.id

        self.name = name

    @property
    def name(self):
        """The name of the model.
        """
        return self._name
    @name.setter
    def name(self, name):
        name = str_cast(name, 'The name property must be castable to type str!')
        self._name = name

    @property
    def id(self):
        """(read-only) The ID of the model. It's an integer generated with
        Python's `id` function. Hence, it's related to the memory address
        of the object.
        """
        return id(self)


class ModelCollection(NamedObjectCollection):
    """This class describes a collection of Model instances. It can be
    used to group several models into a single object.
    """
    @staticmethod
    def cast(obj, errmsg=None):
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

        Raises
        ------
        TypeError
            If the cast fails.

        Returns
        -------
        modelcollection : instance of ModelCollection
            The created ModelCollection instance. If `obj` is already a
            ModelCollection instance, it will be returned.
        """
        if(obj is None):
            obj = ModelCollection(models=None, model_type=Model)
            return obj

        if(isinstance(obj, Model)):
            obj = ModelCollection(models=[obj], model_type=Model)
            return obj

        if(isinstance(obj, ModelCollection)):
            return obj

        if(issequence(obj)):
            obj = ModelCollection(models=obj, model_type=Model)
            return obj

        if(errmsg is None):
            errmsg = 'Cast of object "%s" to ModelCollection failed!'%(str(obj))
        raise TypeError(errmsg)

    def __init__(self, models=None, model_type=None):
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
        if(model_type is None):
            model_type = Model

        if(not issubclass(model_type, Model)):
            raise TypeError('The model_type argument must be a subclass of '
                'class Model!')

        super(ModelCollection, self).__init__(
            objs=models,
            obj_type=model_type)

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
    in combination with the ModelParameterMapper class.
    """
    def __init__(self, name):
        """Creates a new DetectorModel instance.

        Parameters
        ----------
        name : str
            The name of the detector model.
        """
        super().__init__(name=name)
