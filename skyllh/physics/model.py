# -*- coding: utf-8 -*-

"""This module defines the base classes for any physics models used by SkyLLH.
"""

from skyllh.core.py import ObjectCollection


class PhysicsModel(object):
    """This class provides a base class for all physics models like source
    models or background models.
    """
    def __init__(self):
        super(PhysicsModel, self).__init__()


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
                PhysicsModelCollection
            The object that should be casted to PhysicsModelCollection.
        errmsg : str
            The error message if the cast fails.

        Raises
        ------
        TypeError
            If the cast fails.
        """
        if(isinstance(obj, PhysicsModel)):
            obj = PhysicsModelCollection(PhysicsModel, [obj])
        if(not isinstance(obj, PhysicsModelCollection)):
            if(issequence(obj)):
                obj = PhysicsModelCollection(PhysicsModel, obj)
            else:
                raise TypeError(errmsg)
        return obj

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
