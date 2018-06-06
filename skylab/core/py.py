
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
    """Checks if the given object ``obj`` is a sequence of type ``T``.
    """
    if(not issequence(obj)):
        return False
    for item in obj:
        if(not isinstance(item, T)):
            return False
    return True
