
def classname(obj):
    """Returns the name of the class of the class instance ``obj``.
    """
    return type(obj).__name__

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
