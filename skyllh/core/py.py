# -*- coding: utf-8 -*-

from __future__ import division

import abc
import copy
import inspect
import numpy as np
import sys


class PyQualifier(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for any Python qualifier class.
    An object can get qualified by calling a PyQualifier instance with that
    object. The PyQualifier class will be added to the ``__pyqualifiers__``
    attribute of the object.
    """

    def __init__(self):
        super(PyQualifier, self).__init__()

    def __call__(self, obj):
        """Declares the given Python object to be qualified with the
        PyQualifier class of `self`. It adds the class of `self` to the
        `__pyqualifiers__` tuple of the given object.

        Parameters
        ----------
        obj : object
            The Python object that should get qualified.

        Returns
        -------
        obj : object
            The given object, but modified to be declared for this Python
            qualifier.
        """
        if(not hasattr(obj, '__pyqualifiers__')):
            setattr(obj, '__pyqualifiers__', ())

        obj.__pyqualifiers__ += (self.__class__,)

        return obj

    def check(self, obj):
        """Checks if the given Python object is declared with the PyQualifier
        class of the `self` object.

        Parameters
        ----------
        obj : object
            The Python object to check.

        Returns
        -------
        check : bool
            The check result. `True` if the object is declared for this Python
            qualifier, and `False` otherwise.
        """
        if(not hasattr(obj, '__pyqualifiers__')):
            return False

        if(self.__class__ in obj.__pyqualifiers__):
            return True

        return False

class ConstPyQualifier(PyQualifier):
    """This class defines a PyQualifier for constant Python objects.
    """
    def __init__(self):
        super(ConstPyQualifier, self).__init__()

const = ConstPyQualifier()


def typename(t):
    """Returns the name of the given type ``t``.
    """
    return t.__name__

def classname(obj):
    """Returns the name of the class of the class instance ``obj``.
    """
    return typename(type(obj))

def get_byte_size_prefix(size):
    """Determines the biggest size prefix for the given size in bytes such that
    the new size is still greater one.

    Parameters
    ----------
    size : int
        The size in bytes.

    Returns
    -------
    newsize : float
        The new byte size accounting for the byte prefix.
    prefix : str
        The biggest byte size prefix.
    """
    prefix_factor_list = [
        ('', 1), ('K', 1024), ('M', 1024**2), ('G', 1024**3), ('T', 1024**4)]

    prefix_idx = 0
    for (prefix, factor) in prefix_factor_list[1:]:
        if(size / factor < 1):
            break
        prefix_idx += 1

    (prefix, factor) = prefix_factor_list[prefix_idx]
    newsize = size / factor

    return (newsize, prefix)

def getsizeof(objects):
    """Determines the size in bytes the given objects have in memory.
    If an object is a sequence, the size of the elements of the sequence will
    be estimated as well and added to the result. This does not account for the
    multiple occurence of the same object.

    Parameters
    ----------
    objects : sequence of instances of object | instance of object.

    Returns
    -------
    memsize : int
        The memory size in bytes of the given objects.
    """
    if(not issequence(objects)):
        objects = [objects]

    memsize = 0
    for obj in objects:
        if(issequence(obj)):
            memsize += getsizeof(obj)
        else:
            memsize += sys.getsizeof(obj)

    return memsize

def issequence(obj):
    """Checks if the given object ``obj`` is a sequence or not. The definition of
    a sequence in this case is, that the function ``len`` is defined for the
    object.

    .. note::

        A str object is NOT considered as a sequence!

    :return True: If the given object is a sequence.
    :return False: If the given object is a str object or not a sequence.

    """
    if(isinstance(obj, str)):
        return False

    try:
        len(obj)
    except TypeError:
        return False

    return True

def issequenceof(obj, T, pyqualifiers=None):
    """Checks if the given object ``obj`` is a sequence with items being
    instances of type ``T``, possibly qualified with the given Python
    qualifiers.

    Parameters
    ----------
    obj : object
        The Python object to check.
    T : type | tuple of types
        The type each item of the sequence should be. If a tuple of types is
        given, each item can be one of the given types.
    pyqualifiers : instance of PyQualifier |
            sequence of instances of PyQualifier | None
        One or more instances of PyQualifier. Each instance acts as a filter
        for each item.
        If any of the filters return `False` for any of the items, this check
        returns `False`. If set to `Ǹone`, no filters are applied.

    Returns
    -------
    check : bool
        The result of the check.
    """
    if(pyqualifiers is None):
        pyqualifiers = tuple()
    elif(not issequence(pyqualifiers)):
        pyqualifiers = (pyqualifiers,)

    if(not issequence(obj)):
        return False
    for item in obj:
        if(not isinstance(item, T)):
            return False
        for pyqualifier in pyqualifiers:
            if(not pyqualifier.check(item)):
                return False

    return True

def issequenceofsubclass(obj, T):
    """Checks if the given object ``obj`` is a sequence with items being
    sub-classes of class T.
    """
    if(not issequence(obj)):
        return False
    for item in obj:
        if(not issubclass(item, T)):
            return False
    return True

def isproperty(obj, name):
    """Checks if the given attribute is of type property. The attribute must
    exist in ``obj``.

    Parameters
    ----------
    obj : object
        The Python object whose attribute to check for being a property.
    name : str
        The name of the attribute.

    Returns
    -------
    check : bool
        True if the given attribute is of type property, False otherwise.

    Raises
    ------
    AttributeError
        If the given attribute is not an attribute of the class of ``obj``.
    """
    attr = type(obj).__class__.__getattribute__(type(obj), name)
    return isinstance(attr, property)

def func_has_n_args(func, n):
    """Checks if the given function `func` has `n` arguments.

    Parameters
    ----------
    func : callable
        The function to check.
    n : int
        The number of arguments the function must have.

    Returns
    -------
    check : bool
        True if the given function has `n` arguments. False otherwise.
    """
    check = (len(inspect.signature(func).parameters) == n)
    return check

def bool_cast(v, errmsg):
    """Casts the given value to a boolean value. If the cast is impossible, a
    TypeError is raised with the given error message.
    """
    try:
        v = bool(v)
    except:
        raise TypeError(errmsg)
    return v

def int_cast(v, errmsg, allow_None=False):
    """Casts the given value to an integer value. If the cast is impossible, a
    TypeError is raised with the given error message. If `allow_None` is set to
    `True` the value `v` can also be `None`.
    """
    if(allow_None and v is None):
        return v

    try:
        v = int(v)
    except:
        raise TypeError(errmsg)

    return v

def float_cast(v, errmsg, allow_None=False):
    """Casts the given value to a float. If the cast is impossible, a TypeError
    is raised with the given error message. If `allow_None` is set to `True`
    the value `v` can also be `None`.

    Parameters
    ----------
    v : to_float_castable object | sequence of to_float_castable objects
        The object that should get casted to a float. This can also be a
        sequence of objects that should get casted to floats.
    errmsg : str
        The error message in case the cast failed.
    allow_None : bool
        Flag if ``None`` is allowed as value for v. If yes, the casted result is
        ``None``.

    Returns
    -------
    float | None | list of float and or None
        The float / ``None`` value casted from ``v``. If a sequence of objects
        was provided, a list of casted values is returned.
    """
    # Define cast function for a single object.
    def _obj_float_cast(v, errmsg, allow_None):
        if(allow_None and v is None):
            return v

        try:
            v = float(v)
        except:
            raise TypeError(errmsg)

        return v

    if(issequence(v)):
        float_list = []
        for el_v in v:
            float_list.append(_obj_float_cast(el_v, errmsg, allow_None))
        return float_list

    return _obj_float_cast(v, errmsg, allow_None)

def str_cast(v, errmsg):
    """Casts the given value to a str object.
    If the cast is impossible, a TypeError is raised with the given error
    message.
    """
    try:
        v = str(v)
    except:
        raise TypeError(errmsg)
    return v

def list_of_cast(t, v, errmsg):
    """Casts the given value `v` to a list of items of type `t`.
    If the cast is impossible, a TypeError is raised with the given error
    message.
    """
    if(isinstance(v, t)):
        v = [v]
    if(not issequenceof(v, t)):
        raise TypeError(errmsg)
    v = list(v)
    return v

def get_smallest_numpy_int_type(values):
    """Returns the smallest numpy integer type that can represent the given
    integer values.

    Parameters
    ----------
    values : int | sequence of int
        The integer value(s) that need to be representable by the returned
        integer type.

    Returns
    -------
    inttype : numpy integer type
        The smallest numpy integer type that can represent the given values.
    """
    values = np.atleast_1d(values)

    vmin = np.min(values)
    vmax = np.max(values)

    if(vmin < 0):
        types = [np.int8, np.int16, np.int32, np.int64]
    else:
        types = [np.uint8, np.uint16, np.uint32, np.uint64]

    for inttype in types:
        ii = np.iinfo(inttype)
        if(vmin >= ii.min and vmax <= ii.max):
            return inttype

    raise ValueError("No integer type spans [%d, %d]!"%(vmin, vmax))

def get_number_of_float_decimals(value):
    """Determines the number of significant decimals the given float number has.
    The maximum number of supported decimals is 16.

    Parameters
    ----------
    value : float
        The float value whose number of significant decimals should get
        determined.

    Returns
    -------
    decimals : int
        The number of decimals of value which are non-zero.
    """
    val_str = '{:.16f}'.format(value)
    (val_num_str, val_decs_str) = val_str.split('.', 1)
    for idx in range(len(val_decs_str)-1, -1, -1):
        if(int(val_decs_str[idx]) != 0):
            return idx+1
    return 0


class ObjectCollection(object):
    """This class provides a collection of objects of a specific type. Objects
    can be added to the collection via the ``add`` method or can be removed
    from the collection via the ``pop`` method. The objects of another object
    collection can be added to this object collection via the ``add`` method as
    well.
    """
    def __init__(self, objs=None, obj_type=None):
        """Constructor of the ObjectCollection class. Must be called by the
        derived class.

        Parameters
        ----------
        objs : instance of obj_type | sequence of obj_type instances | None
            The sequence of objects of type ``obj_type`` with which this
            collection should get initialized with.
        obj_type : type
            The type of the objects, which can be added to the collection.
        """
        if(obj_type is None):
            obj_type = object
        if(not issubclass(obj_type, object)):
            raise TypeError(
                'The obj_type argument must be a subclass of object!')

        self._obj_type = obj_type
        self._objects = []

        # Add given list of objects.
        if(objs is not None):
            if(not issequence(objs)):
                objs = [ objs ]
            for obj in objs:
                self.add(obj)

    @property
    def obj_type(self):
        """(read-only) The object type.
        """
        return self._obj_type

    @property
    def objects(self):
        """(read-only) The list of objects of this object collection.
        All objects are of the same type as specified through the ``obj_type``
        property.
        """
        return self._objects

    def __len__(self):
        """Returns the number of objects being in this object collection.
        """
        return len(self._objects)

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return iter(self._objects)

    def __add__(self, other):
        """Implementation to support the operation ``oc = self + other``, where
        ``self`` is this ObjectCollection object and ``other`` something useful
        else. This creates a copy ``oc`` of ``self`` and adds ``other``
        to ``oc``.

        Parameters
        ----------
        other : obj_type | ObjectCollection of obj_type

        Returns
        -------
        oc : ObjectCollection
            The new ObjectCollection object with object from self and other.
        """
        oc = self.copy()
        oc.add(other)
        return oc

    def __str__(self):
        """Pretty string representation of this object collection.
        """
        return classname(self)+ ': ' + str(self._objects)

    def copy(self):
        """Creates a copy of this ObjectCollection. The objects of the
        collection are not copied!
        """
        oc = ObjectCollection(self._obj_type)
        oc._objects = copy.copy(self._objects)
        return oc

    def add(self, obj):
        """Adds the given object to this object collection.

        Parameters
        ----------
        obj : obj_type instance | ObjectCollection of obj_type
            An instance of ``obj_type`` that should be added to the collection.
            If given an ObjectCollection for objects of type obj_type, it will
            add all objects of the given collection to this collection.

        Returns
        -------
        self : ObjectCollection
            The instance of this ObjectCollection, in order to be able to chain
            several ``add`` calls.
        """
        if(isinstance(obj, ObjectCollection)):
            if(typename(obj.obj_type) != typename(self._obj_type)):
                raise TypeError('Cannot add objects from ObjectCollection for '
                    'objects of type "%s" to this ObjectCollection for objects '
                    'of type "%s"!'%(
                        typename(obj.obj_type), typename(self._obj_type)))
            self._objects.extend(obj.objects)
            return self

        if(not isinstance(obj, self._obj_type)):
            raise TypeError('The object of type "%s" cannot be added to the '
                'object collection for objects of type "%s"!'%(
                    classname(obj), typename(self._obj_type)))

        self._objects.append(obj)
        return self
    __iadd__ = add

    def index(self, obj):
        """Gets the index of the given object instance within this object
        collection.

        Parameters
        ----------
        obj : obj_type instance
            The instance of obj_type whose index should get retrieved.

        Returns
        -------
        idx : int
            The index of the object within this object collection.
        """
        return self._objects.index(obj)

    def pop(self, index=None):
        """Removes and returns the object at the given index (default last).
        Raises IndexError if the collection is empty or index is out of range.

        Parameters
        ----------
        index : int | None
            The index of the object to remove. If set to None, the index of the
            last object is used.

        Returns
        -------
        obj : obj_type
            The removed object.
        """
        if(index is None):
            index = len(self._objects)-1
        obj = self._objects.pop(index)
        return obj


class NamedObjectCollection(ObjectCollection):
    """This class provides a collection of objects, which have a name. Access
    via the object name is efficient because the index of each object is
    tracked w.r.t. its name.
    """
    def __init__(self, objs=None, obj_type=None):
        """Creates a new NamedObjectCollection instance. Must be called by the
        derived class.

        Parameters
        ----------
        objs : instance of obj_type | sequence of instances of obj_type | None
            The sequence of objects of type ``obj_type`` with which this collection
            should get initialized with.
        obj_type : type
            The type of the objects, which can be added to the collection.
            This type must have an attribute named ``name``.
        """
        self._obj_name_to_idx = dict()

        super(NamedObjectCollection, self).__init__(
            objs=objs,
            obj_type=obj_type)

    def __getitem__(self, key):
        """Returns an object based on its name or index.

        Parameters
        ----------
        key : str | int
            The object identification. Either its name or its index position
            within the object collection.

        Returns
        -------
        obj : instance of obj_type
            The requested object.

        Raises
        ------
        KeyError
            If the given object is not found within this object collection.
        """
        if(isinstance(key, str)):
            key = self.index_by_name(key)
        return super(NamedObjectCollection, self).__getitem__(key)

    def add(self, obj):
        """Adds the given object to this named object collection.

        Parameters
        ----------
        obj : obj_type instance | NamedObjectCollection of obj_type
            An instance of ``obj_type`` that should be added to this named
            object collection.
            If a NamedObjectCollection instance for objects of type ``obj_type``
            is given, all objects of that given collection will be added to this
            named object collection.

        Returns
        -------
        self : NamedObjectCollection
            The instance of this NamedObjectCollection, in order to be able to
            chain several ``add`` calls.
        """
        super(NamedObjectCollection, self).add(obj)

        if(isinstance(obj, NamedObjectCollection)):
            # Several objects have been added, so we recreate the name to index
            # dictionary.
            self._obj_name_to_idx = dict([
                (o.name,idx) for (idx,o) in enumerate(self._objects) ])
        else:
            # Only a single object was added at the end.
            self._obj_name_to_idx[obj.name] = len(self) - 1

        return self

    def index_by_name(self, name):
        """Gets the index of the object with the given name within this named
        object collection.

        Parameters
        ----------
        name : str
            The name of the object whose index should get retrieved.

        Returns
        -------
        idx : int
            The index of the object within this named object collection.
        """
        return self._obj_name_to_idx[name]

    def pop(self, index=None):
        """Removes and returns the object at the given index (default last).
        Raises IndexError if the collection is empty or index is out of range.

        Parameters
        ----------
        index : int | str | None
            The index of the object to remove. If set to None, the index of the
            last object is used.
            If a str instance is given, it specifies the name of the object.

        Returns
        -------
        obj : obj_type instance
            The removed object.
        """
        if(isinstance(index, str)):
            # Get the index of the object given its name.
            index = self.index_by_name(index)

        obj = super(NamedObjectCollection, self).pop(index)

        # Recreate the object name to index dictionary.
        self._obj_name_to_idx = dict([
            (o.name,idx) for (idx,o) in enumerate(self._objects) ])

        return obj
