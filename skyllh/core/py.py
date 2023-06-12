# -*- coding: utf-8 -*-

import abc
import copy
import functools
import inspect
import numpy as np
import sys

from collections import OrderedDict

from skyllh.core.display import INDENTATION_WIDTH


class PyQualifier(
        object,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for any Python qualifier class.
    An object can get qualified by calling a PyQualifier instance with that
    object. The PyQualifier class will be added to the ``__pyqualifiers__``
    attribute of the object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        if not hasattr(obj, '__pyqualifiers__'):
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
        if not hasattr(obj, '__pyqualifiers__'):
            return False

        if self.__class__ in obj.__pyqualifiers__:
            return True

        return False


class ConstPyQualifier(
        PyQualifier):
    """This class defines a PyQualifier for constant Python objects.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


const = ConstPyQualifier()


def get_class_of_func(f):
    """Determines the class object that defined the given method or function
    ``f``.

    Parameters
    ----------
    f : function | method
        The function or method whose parent class should be determined.

    Returns
    -------
    cls : class | None
        The class object which defines the given function or method. ``None``
        is returned when no class could be determined.
    """
    if isinstance(f, functools.partial):
        return get_class_of_func(f.func)

    if inspect.ismethod(f) or\
        ((inspect.isbuiltin(f)) and
         (getattr(f, '__self__', None) is not None) and
         (getattr(f.__self__, '__class__', None))):
        for cls in inspect.getmro(f.__self__.__class__):
            if hasattr(cls, '__dict__') and (f.__name__ in cls.__dict__):
                return cls
        # Fall back to normal function evaluation.
        f = getattr(f, '__func__', f)

    if inspect.isfunction(f):
        cls = getattr(
            inspect.getmodule(f),
            f.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
            None)
        if isinstance(cls, type):
            return cls

    # Handle special descriptor objects.
    cls = getattr(f, '__objclass__', None)
    return cls


def typename(t):
    """Returns the name of the given type ``t``.
    """
    return t.__name__


def classname(obj):
    """Returns the name of the class of the class instance ``obj``.
    """
    return typename(type(obj))


def module_classname(obj):
    """Returns the module and class name of the given instance ``obj``.
    """
    return f'{obj.__module__}.{classname(obj)}'


def module_class_method_name(obj, meth_name):
    """Returns the module, class, and method name of the given instance ``obj``.

    Parameters
    ----------
    obj : instance of object
        The object instance.
    meth_name : str
        The name of the method.
    """
    return f'{module_classname(obj)}.{meth_name}'


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
        if size / factor < 1:
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
    if not issequence(objects):
        objects = [objects]

    memsize = 0
    for obj in objects:
        if issequence(obj):
            memsize += getsizeof(obj)
        else:
            memsize += sys.getsizeof(obj)

    return memsize


def issequence(obj):
    """Checks if the given object ``obj`` is a sequence or not. The definition of
    a sequence in this case is, that the function ``len`` is defined for the
    object.

    .. note::

        A str object is NOT considered a sequence!

    Returns
    -------
    check : bool
        ``True`` if the given object is a sequence. ``False`` if the given
        object is an instance of str or not a sequence.

    """
    if isinstance(obj, str):
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
    obj : instance of object
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
    if pyqualifiers is None:
        pyqualifiers = tuple()
    elif not issequence(pyqualifiers):
        pyqualifiers = (pyqualifiers,)

    if not issequence(obj):
        return False

    for item in obj:
        if not isinstance(item, T):
            return False
        for pyqualifier in pyqualifiers:
            if not pyqualifier.check(item):
                return False

    return True


def issequenceofsubclass(obj, T):
    """Checks if the given object ``obj`` is a sequence with items being
    sub-classes of class T.

    Parameters
    ----------
    obj : instance of object
        The object to check.
    T : class
        The base class of the items of the given object.

    Returns
    -------
    check : bool
        ``True`` if the given object is a sequence of instances which are
        sub-classes of class ``T``. ``False`` if ``obj`` is not a sequence or
        any item is not a sub-class of class ``T``.
    """
    if not issequence(obj):
        return False

    for item in obj:
        if not issubclass(item, T):
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
    except Exception:
        raise TypeError(errmsg)

    return v


def int_cast(v, errmsg, allow_None=False):
    """Casts the given value to an integer value. If the cast is impossible, a
    TypeError is raised with the given error message. If `allow_None` is set to
    `True` the value `v` can also be `None`.
    """
    if allow_None and v is None:
        return v

    try:
        v = int(v)
    except Exception:
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
        if allow_None and v is None:
            return v

        try:
            v = float(v)
        except Exception:
            raise TypeError(errmsg)

        return v

    if issequence(v):
        float_list = []
        for el_v in v:
            float_list.append(_obj_float_cast(el_v, errmsg, allow_None))
        return float_list

    return _obj_float_cast(v, errmsg, allow_None)


def str_cast(v, errmsg, allow_None=False):
    """Casts the given value to a str object.
    If the cast is impossible, a TypeError is raised with the given error
    message.
    """
    if allow_None and v is None:
        return v

    try:
        v = str(v)
    except Exception:
        raise TypeError(errmsg)

    return v


def list_of_cast(t, v, errmsg):
    """Casts the given value `v` to a list of items of type `t`.
    If the cast is impossible, a TypeError is raised with the given error
    message.
    """
    if isinstance(v, t):
        v = [v]

    if not issequenceof(v, t):
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

    if vmin < 0:
        types = [np.int8, np.int16, np.int32, np.int64]
    else:
        types = [np.uint8, np.uint16, np.uint32, np.uint64]

    for inttype in types:
        ii = np.iinfo(inttype)
        if vmin >= ii.min and vmax <= ii.max:
            return inttype

    raise ValueError(
        f'No integer type spans [{vmin}, {vmax}]!')


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

    Raises
    ------
    ValueError
        If a nan value was provided.
    """
    if np.isnan(value):
        raise ValueError(
            'The provided value is nan!')

    val_str = '{:.16f}'.format(value)
    (val_num_str, val_decs_str) = val_str.split('.', 1)
    for idx in range(len(val_decs_str)-1, -1, -1):
        if int(val_decs_str[idx]) != 0:
            return idx+1

    return 0


def make_dict_hash(d):
    """Creates a hash value for the given dictionary.

    Parameters
    ----------
    d : dict | None
        The dictionary holding (name: value) pairs.
        If set to None, an empty dictionary is used.

    Returns
    -------
    hash : int
        The hash of the dictionary.
    """
    if d is None:
        d = {}

    if not isinstance(d, dict):
        raise TypeError(
            'The d argument must be of type dict!')

    # A note on the ordering of Python dictionary items: The items are ordered
    # internally according to the hash value of their keys. Hence, if we don't
    # insert more dictionary items, the order of the items won't change. Thus,
    # we can just take the items list and make a tuple to create a hash of it.
    # The hash will be the same for two dictionaries having the same items.
    return hash(tuple(d.items()))


class ObjectCollection(
        object):
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
        obj_type : type | None
            The type of the objects, which can be added to the collection.
            If set to None, the type will be determined from the given objects.
            If no objects are given, the object type will be `object`.
        """
        if obj_type is None:
            obj_type = object
            if objs is not None:
                if issequence(objs) and len(objs) > 0:
                    obj_type = type(objs[0])
                else:
                    obj_type = type(objs)

        if not issubclass(obj_type, object):
            raise TypeError(
                'The obj_type argument must be a subclass of object!')

        self._obj_type = obj_type
        self._objects = []

        # Add given list of objects.
        if objs is not None:
            if not issequence(objs):
                objs = [objs]
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
        obj_str = ",\n".join([
            ' '*INDENTATION_WIDTH + str(obj) for obj in self._objects
        ])
        return classname(self) + ": {\n" + obj_str + "\n}"

    def copy(self):
        """Creates a copy of this ObjectCollection. The objects of the
        collection are not copied!
        """
        oc = ObjectCollection(self._obj_type)
        oc._objects = copy.copy(self._objects)
        return oc

    def add(self, obj):
        """Adds the given object, sequence of objects, or object collection to
        this object collection.

        Parameters
        ----------
        obj : obj_type instance | sequence of obj_type |
              ObjectCollection of obj_type
            An instance of ``obj_type`` that should be added to the collection.
            If given an ObjectCollection for objects of type obj_type, it will
            add all objects of the given collection to this collection.

        Returns
        -------
        self : ObjectCollection
            The instance of this ObjectCollection, in order to be able to chain
            several ``add`` calls.
        """
        if issequence(obj) and not isinstance(obj, ObjectCollection):
            obj = ObjectCollection(obj)

        if isinstance(obj, ObjectCollection):
            if not issubclass(obj.obj_type, self.obj_type):
                raise TypeError(
                    'Cannot add objects from ObjectCollection for objects of '
                    f'type "{typename(obj.obj_type)}" to this ObjectCollection '
                    f'for objects of type "{typename(self._obj_type)}"!')
            self._objects.extend(obj.objects)
            return self

        if not isinstance(obj, self._obj_type):
            raise TypeError(
                f'The object of type "{classname(obj)}" cannot be added to the '
                'object collection for objects of type '
                f'"{typename(self._obj_type)}"!')

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
        if index is None:
            index = len(self._objects) - 1
        obj = self._objects.pop(index)
        return obj


class NamedObjectCollection(ObjectCollection):
    """This class provides a collection of objects, which have a name. Access
    via the object name is efficient because the index of each object is
    tracked w.r.t. its name.
    """
    def __init__(self, objs=None, obj_type=None, **kwargs):
        """Creates a new NamedObjectCollection instance. Must be called by the
        derived class.

        Parameters
        ----------
        objs : instance of obj_type | sequence of instances of obj_type | None
            The sequence of objects of type ``obj_type`` with which this
            collection should get initialized with.
        obj_type : type
            The type of the objects, which can be added to the collection.
            This type must have an attribute named ``name``.
        """
        self._obj_name_to_idx = OrderedDict()

        # The ObjectCollection class will call the add method to add individual
        # objects. This will update the _obj_name_to_idx attribute.
        super().__init__(
            objs=objs,
            obj_type=obj_type,
            **kwargs)

        if not hasattr(self.obj_type, 'name'):
            raise TypeError(
                f'The object type "{typename(self.obj_type)}" has no '
                'attribute named "name"!')

    @property
    def name_list(self):
        """(read-only) The list of the names of all the objects of this
        NamedObjectCollection instance.
        The order of this list of names is preserved to the order objects were
        added to this collection.
        """
        return list(self._obj_name_to_idx.keys())

    def _create_obj_name_to_idx_dict(self, start=None, end=None):
        """Creates the dictionary {obj.name: index} for object in the interval
        [`start`, `end`).

        Parameters
        ----------
        start : int | None
            The object start index position, which is inclusive.
        end : int | None
            The object end index position, which is exclusive.

        Returns
        -------
        obj_name_to_idx : dict
            The dictionary {obj.name: index}.
        """
        if start is None:
            start = 0

        return OrderedDict([
            (o.name, start+idx)
            for (idx, o) in enumerate(self._objects[start:end])
        ])

    def __contains__(self, name):
        """Returns ``True`` if an object of the given name exists in this
        NamedObjectCollection instance, ``False`` otherwise.

        Parameters
        ----------
        name : str
            The name of the object.

        Returns
        -------
        check : bool
            ``True`` if an object of name ``name`` exists in this
            NamedObjectCollection instance, ``False`` otherwise.
        """
        return name in self._obj_name_to_idx

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
        if isinstance(key, str):
            key = self.get_index_by_name(key)
        return super().__getitem__(key)

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
        n_objs = len(self)

        super().add(obj)

        self._obj_name_to_idx.update(
            self._create_obj_name_to_idx_dict(n_objs))

        return self
    __iadd__ = add

    def get_index_by_name(self, name):
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
        if isinstance(index, str):
            # Get the index of the object given its name.
            index = self.get_index_by_name(index)

        obj = super().pop(index)

        # Recreate the object name to index dictionary.
        self._obj_name_to_idx = self._create_obj_name_to_idx_dict()

        return obj
