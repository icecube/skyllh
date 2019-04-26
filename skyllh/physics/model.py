# -*- coding: utf-8 -*-

"""This module defines the base classes for any physics models used by SkyLLH.
"""

from skyllh.core.py import (
    ObjectCollection,
    issequence,
    str_cast
)


class PhysicsModel(object):
    """This class provides a base class for all physics models like source
    models or background models.
    """
    def __init__(self, name=None):
        """Creates a new PhysicsModel instance.

        Parameters
        ----------
        name : str | None
            The name of the physics model. If set to `None`, the id of the
            object is taken as name.
        """
        super(PhysicsModel, self).__init__()

        if(name is None):
            name = self.id

        self.name = name

    @property
    def name(self):
        """The name of the physics model.
        """
        return self._name
    @name.setter
    def name(self, n):
        n = str_cast(n, 'The name property must be castable to type str!')
        self._name = n

    @property
    def id(self):
        """(read-only) The ID of the physics model. It's an integer generated
        with Python's `id` function. Hence, it's related to the memory address
        of the object.
        """
        return id(self)


class PhysicsModelCollection(ObjectCollection):
    """This class describes a collection of PhysicsModel instances. It can be
    used to group several physics models into a single object.
    """
    @staticmethod
    def cast(obj, errmsg):
        """Casts the given object to a PhysicsModelCollection object.
        If the cast fails, a TypeError with the given error message is raised.

        Parameters
        ----------
        obj : PhysicsModel instance | sequence of PhysicsModel instances |
                PhysicsModelCollection | None
            The object that should be casted to PhysicsModelCollection.
        errmsg : str
            The error message if the cast fails.

        Raises
        ------
        TypeError
            If the cast fails.

        Returns
        -------
        physmodelcollection : instance of PhysicsModelCollection
            The created PhysicsModelCollection instance. If `obj` is already a
            PhysicsModelCollection instance, it will be returned.
        """
        if(obj is None):
            obj = PhysicsModelCollection(PhysicsModel, None)
            return obj

        if(isinstance(obj, PhysicsModel)):
            obj = PhysicsModelCollection(PhysicsModel, [obj])
            return obj

        if(isinstance(obj, PhysicsModelCollection)):
            return obj

        if(issequence(obj)):
            obj = PhysicsModelCollection(PhysicsModel, obj)
            return obj

        raise TypeError(errmsg)

    def __init__(self, model_type=None, models=None):
        """Creates a new PhysicsModel collection. The type of the physics model
        instances the collection holds can be restricted, by setting the
        model_type parameter.

        Parameters
        ----------
        model_type : type | None
            The type of the physics model. If set to None (default),
            PhysicsModel will be used.
        models : sequence of model_type instances | None
            The sequence of physics models this collection should be initalized
            with.
        """
        if(model_type is None):
            model_type = PhysicsModel
        super(PhysicsModelCollection, self).__init__(
            obj_type=model_type,
            obj_list=models)

    @property
    def model_type(self):
        """(read-only) The type of the physics model.
        """
        return self.obj_type

    @property
    def models(self):
        """(read-only) The list of models of type `model_type`.
        """
        return self.objects
