import copy

def typename(t):
    """Returns the name of the given type ``t``.
    """
    return t.__name__

def classname(obj):
    """Returns the name of the class of the class instance ``obj``.
    """
    return typename(type(obj))

def issequence(obj):
    """Checks if the given object ``obj`` is a squence or not. The definition of
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

def issequenceof(obj, T):
    """Checks if the given object ``obj`` is a sequence with items of type ``T``.
    """
    if(not issequence(obj)):
        return False
    for item in obj:
        if(not isinstance(item, T)):
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
    """
    return isinstance(type(obj).__dict__[name], property)

def float_cast(v, errmsg):
    """Casts the given value to a float. If the cast is impossible, a TypeError
    is raised with the given error message.
    """
    if(isinstance(v, int)):
        v = float(v)
    elif(isinstance(v, str)):
        try:
            v = float(v)
        except:
            raise TypeError(errmsg)
    if(not isinstance(v, float)):
        raise TypeError(errmsg)
    return v

class ObjectCollection(object):
    """This class provides a collection of objects of a specific type. Objects
    can be added to the collection via the ``add`` method or can be removed
    from the collection via the ``pop`` method. The objects of another object
    collection can be added to this object collection via the ``add`` method as
    well.
    """
    def __init__(self, obj_type, obj_list=None):
        """Constructor of the ObjectCollection class. Must be called by the
        derived class.

        Parameters
        ----------
        obj_type : type
            The type of the objects, which can be added to the collection.
        obj_list : list of obj_t | None
            The list of objects of type ``obj_t`` with which this collection
            should get initialized with.
        """
        if(not issubclass(obj_type, object)):
            raise TypeError('The obj_t argument must be a subclass of object!')
        self._obj_type = obj_type
        self._objects = []

        # Add given list of objects.
        if(obj_list is not None):
            if(not issequence(obj_list)):
                raise TypeError('The obj_list argument must be a sequence!')
            for obj in obj_list:
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
        """Adds the given object to the collection.

        Parameters
        ----------
        obj : obj_type | ObjectCollection of obj_type
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
                raise TypeError('Cannot add objects from ObjectCollection for objects of type "%s" to this ObjectCollection for objects of type "%s"!'%(typename(obj.obj_type), typename(self._obj_type)))
            self._objects.extend(obj.objects)
            return self

        if(not isinstance(obj, self._obj_type)):
            raise TypeError('The object of type "%s" cannot be added to the object collection for objects of type "%s"!'%(classname(obj), typename(self._obj_type)))

        self._objects.append(obj)
        return self
    __iadd__ = add

    def index(self, obj):
        return self._objects.index(obj)

    def pop(self, index=None):
        """Removes and returns object at given index (default last).
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
