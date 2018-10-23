"""The lh module provides functionality for constructing the final likelihood
function in an elegant way by using mathematical operators.

The likelihood function get constructed through its components. Each component
must be a callable object, i.e. a function or an object instance with an
implemented __call__ method. The call signature must be of the form
__call__(events, params), where events is an one-dimensional numpy.recarray
holding the event data, and params is a dictionary holding the current set of
parameters of the LH function.


Examples
--------
def signal_fraction(events, params):
    return params['nsignal'] / len(events)

def background_fraction(events, params):
    return 1 - signal_fraction(events, params)

lh = LHComponent(signal_fraction) + LHComponent(background_fraction)
"""

import numbers
import numpy as np

from py import is_sequence

def log(lhcomponent):
    """Convinient function to add the numpy.log operation to the likelihood
    function.

    Parameters
    ----------
    lhcomponent : LHComponent
        The LHComponent object the logarithm operation should be added to.

    Returns
    -------
    lhcomponent : LHComponent
        The same LHComponent object as given as input but the the numpy.log
        operator added.
    """
    return lhcomponent.add_operation(np.log)

def sum(lhcomponent):
    """Convinient function to add the numpy.sum operation to the likelihood
    function.

    Parameters
    ----------
    lhcomponent : LHComponent
        The LHComponent object the sum operation should be added to.

    Returns
    -------
    lhcomponent : LHComponent
        The same LHComponent object as given as input but the the numpy.sum
        operator added.
    """
    return lhcomponent.add_operation(np.sum)

class LHComponent(object):
    """A likelihood component is a part of the final likelihood function.
    A LHComponent has a source, e.g. a PDF. LHComponent objects can be combined
    by mathematical operators resulting into a CombinedLHComponent object.
    """
    def __init__(self,
        source=None,
        lhcomponent=None
    ):
        """Constructs a LHComponent object for a given source object, so that
        the source object can be used as component in the final likelihood
        function.

        Parameters
        ----------
        source : callable
            The callable source of the LH component. The __call__ function must
            have the following call signature:
                __call__(self, params, events),
            where events is a one-dimensional numpy.recarray where each entry
            represents an event, and params is a dictionary with the current
            set of parameters of the likelihood function.

        lhcomponent : LHComponent | None
            If specified a copy of that given object will be made.
        """
        if(isinstance(lhcomponent, LHComponent)):
            # This is a copy constructor.
            self.source = lhcomponent.source
            self.operation_list = list()
            for operation in lhcomponent.operation_list:
                self.operation_list.append(operation)

            return
        #-----------------------------------------------------------------------
        self.source = source
        self.operation_list = list()

    ###
    # Operator methods.
    #___________________________________________________________________________
    def __add__(self, x):
        """Implementation to support the operation ``b = a + x``, where
        ``a`` is this LHComponent object and ``x`` something useful else.

        ``x`` can be a :py:class:`LHComponent` or a
        :py:class:`CombinedLHComponent` object. In this case a
        :py:class:`CombinedLHComponent` object will be created, which combines
        the data of this LHComponent with the given data into one combined lh
        component, using the ``numpy.add`` function for the combination.

        Otherwise, it creates a copy of ``a``, calls its :py:meth:`add` method,
        and returns it as object ``b``.

        """
        if(isinstance(x, LHComponent)):
            clhc = CombinedLHComponent(
                lhcomponent_a = self,
                lhcomponent_b = x,
                operator      = np.add,
            )
            return clhc

        lhc = self.copy()
        lhc.add(x)
        return lhc

    #___________________________________________________________________________
    def __mul__(self, x):
        """Implementation to support the operation ``b = a * x``, where
        ``a`` is this LHComponent object and ``x`` something useful else.

        ``x`` can be a :py:class:`LHComponent` or a
        :py:class:`CombinedLHComponent` object. In this case a
        :py:class:`CombinedLHComponent` object will be created, which combines
        the data of this LHComponent with the given LHComponent into one
        combined data, using the ``numpy.multiply`` function for the combination.

        Otherwise, it creates a copy of ``a``, calls its :py:meth:`multiply`
        method, and returns it as object ``b``.

        """
        if(isinstance(x, LHComponent)):
            clhc = CombinedLHComponent(
                lhcomponent_a = self,
                lhcomponent_b = x,
                operator      = np.multiply
            )
            return clhc

        lhc = self.copy()
        lhc.multiply(x)
        return lhc

    ###
    # Public class methods.
    #___________________________________________________________________________
    def add(self, x):
        """Implements the operation ``a += x``, where ``a`` is this LHComponent
        object and ``x`` something useful else.

        ``x`` can be one of the following types:

        - ``numbers.Real``

        - :py:class:`LHComponent`:

            In this case a :py:class:`CombinedLHComponent` object with operator
            ``numpy.add`` of ``a`` and ``x`` is created replacing ``a``.

        Returns
        -------
        self : LHComponent
            The instance of this LHComponent, in order to be
            able to chain several operator functions.

        """
        if(isinstance(x, numbers.Real)):
            op = LHComponentOperator(np.add, x)
            self.operation_list.append(op)
        elif(isinstance(x, LHComponent)):
            return CombinedLHComponent(
                datastreambase_a = self,
                datastreambase_b = x,
                operator         = np.add
            )
        else:
            raise TypeError(
                'The type "%s" is not supported for a data stream addition!'%
                (type(x)))

        return self
    #---------------------------------------------------------------------------
    __iadd__ = add

    #___________________________________________________________________________
    def multiply(self, x):
        """Implements the in-place operation ``a *= x``, where ``a`` is this
        LHComponent object and ``x`` something useful else.

        ``x`` can be one of the following types:

        - ``numbers.Real``

        - :py:class:`LHComponent`:

            In this case a :py:class:`CombinedLHComponent` object with operator
            ``numpy.multiply`` of ``a`` and ``x`` is created replacing ``a``.

        Returns
        -------
        self : LHComponent
            The instance of this LHComponent class, in order to be
            able to chain several operator functions.

        """
        if(isinstance(x, numbers.Real)):
            op = LHComponentOperator(np.multiply, x)
            self.operation_list.append(op)
        elif(isinstance(x, LHComponent)):
            clhc = CombinedLHComponent(
                lhcomponent_a = self,
                lhcomponent_b = x,
                operator      = np.multiply
            )
            return clhc
        else:
            raise TypeError(
                'The type "%s" is not supported for a data stream '\
                'multiplication!'%
                (type(x)))

        return self
    #---------------------------------------------------------------------------
    __imul__ = multiply

    #___________________________________________________________________________
    def add_operation(self, operator, *operator_args, **operator_kwargs):
        """Adds the operation with the given operator with its given arguments
        and keyword arguments to this LHComponent object.

        See the documentation of the :py:meth:`LHComponentOperator.__init__`
        constructor method of the LHComponentOperator class for a full
        description.

        """
        self.operation_list.append(LHComponentOperator(
            operator        = operator,
            operator_args   = operator_args,
            operator_kwargs = operator_kwargs
        ))
        return self

    #___________________________________________________________________________
    def copy(self):
        """Creates a copy of this LHComponent object by using its
        copy-constructor.

        """
        return LHComponent(lhcomponent=self)

    #___________________________________________________________________________
    def evaluate(self, params, events):
        """Evaluates the LHComponent for the given events and set of parameters.

        Parameters
        ----------
        params : dict
            The dictionary with the set of current parameters of the likelihood
            function.
        events : numpy.recarray
            Each entry in this array represents an event for which the LH
            component should be evaluated.

        Returns
        -------
        data : numpy.ndarray
            The data values of the LH component for all the given events.
            Its shape equals the one of events.
        """
        # Get the data from the source.
        data = self.source(params, events)

        # Apply the operations on the data.
        for operator in self.operation_list:
            data = operator(data)

        return data

    #___________________________________________________________________________
    def as_function(self):
        """Creates a LHFunction object for this likelihood component.

        Returns:
        lhf : LHFunction
            The LHFunction object for this likelihood component.
        """
        return LHFunction(self)

class CombinedLHComponent(LHComponent):
    """A CombinedLHComponent object combines two LHComponent objects with an
    operator function.
    """
    def __init__(self,
        lhcomponent_a       = None,
        lhcomponent_b       = None,
        operator            = None,
        operator_args       = None,
        operator_kwargs     = None,
        combinedlhcomponent = None,
        **kwargs
    ):
        """Constructs a new CombinedLHComponent object for the LHComponent
        objects *lhcomponent_a*, and *lhcomponent_b*, which will be
        combined via the given operation to implement the following assignment
        operation::

            combinedlhcomponent = lhcomponent_a operator lhcomponent_b

        The additional keyword arguments will get passed to the
        :py:meth:`LHComponent.__init__` constructor method of the
        LHComponent class.

        Parameters:
        -----------
        lhcomponent_a : LHComponent
            The first LHComponent object in the above assignment operation.

        lhcomponent_b : LHComponent
            The second LHComponent object in the above assignment operation.

        operator : callable
            The operator callable object that does the combination
            of both LHComponent objects.

            It must be a callable object with the following call signature::

                data_new = operator(data_a, data_b, *operator_args, **operator_kwargs)

            where ``data_a``, and ``data_b`` are the numpy.ndarray objects
            holding a data chunk from the ``lhcomponent_a`` and
            ``lhcomponent_b`` objects, respectively. They must be
            of the same shape, or can at least be broadcasted to the same shape.
            The ``operator_args`` arguments, and the ``operator_kwargs`` keyword
            arguments are the ones, which are specified through the
            *operator_args* and *operator_kwargs* parameters of this
            constructor method.

            The operator object must return a numpy.ndarray (``data_new``) of
            the same shape as the (broadcasted) input numpy.ndarray objects
            (``data_a`` and ``data_b``).

        operator_args : list | None
            The list of arguments for the operator function.

            If set to ``None``, an empty list will be used.

        operator_kwargs: dict | None
            The dictionary with the keyword arguments for the operator function.

            If set to ``None``, an empty dictionary will be used.

        combinedlhcomponent: CombinedLHComponent | None
            If set to a CombinedLHComponent object, this
            new CombinedLHComponent object will be a copy of this given
            CombinedLHComponent object. The constructor becomes a
            copy-constructor.

        """
        if(isinstance(combinedlhcomponent, CombinedLHComponent)):
            # This is the copy constructor.
            super(CombinedLHComponent, self).__init__(
                lhcomponent = combinedlhcomponent
            )

            self.lhcomponent_a   = combinedlhcomponent.lhcomponent_a
            self.lhcomponent_b   = combinedlhcomponent.lhcomponent_b
            self.operator        = combinedlhcomponent.operator
            self.operator_args   = combinedlhcomponent.operator_args
            self.operator_kwargs = combinedlhcomponent.operator_kwargs

            return
        #-----------------------------------------------------------------------
        super(CombinedLHComponent, self).__init__(**kwargs)

        self.lhcomponent_a   = lhcomponent_a
        self.lhcomponent_b   = lhcomponent_b
        self.operator        = operator
        self.operator_args   = operator_args
        self.operator_kwargs = operator_kwargs

    ###
    # Class properties.
    #___________________________________________________________________________
    @property
    def lhcomponent_a(self):
        """The first :py:class:`LHComponent` object in the assignment
        operation
        ``combinedlhcomponent = lhcomponent_a operator lhcomponent_b``.

        """
        return self._lhcomponent_a
    #---------------------------------------------------------------------------
    @lhcomponent_a.setter
    def lhcomponent_a(self, lhcomponent):
        if(not isinstance(lhcomponent, LHComponent)):
            raise TypeError(
                'The component set to the "lhcomponent_a" property must be an '\
                'instance of LHComponent!')
        self._lhcomponent_a = lhcomponent

    #___________________________________________________________________________
    @property
    def lhcomponent_b(self):
        """The second :py:class:`LHComponent` object in the assignment
        operation
        ``combinedlhcomponent = lhcomponent_a operator lhcomponent_b``.

        """
        return self._lhcomponent_b
    #---------------------------------------------------------------------------
    @lhcomponent_b.setter
    def lhcomponent_b(self, lhcomponent):
        if(not isinstance(lhcomponent, LHComponent)):
            raise TypeError(
                'The component set to the "lhcomponent_b" property must be an '\
                'instance of LHComponent!')
        self._lhcomponent_b = lhcomponent

    #___________________________________________________________________________
    @property
    def operator(self):
        """The LHOperator object that combines the two LHComponent
        objects done in the operation
        ``combinedlhcomponent = lhcomponent_a operator lhcomponent_b``.

        It must have the call signature
        ``data_new = operator(data_a, data_b, *operator_args, **operator_kwargs)``.

        """
        return self._operator
    #---------------------------------------------------------------------------
    @operator.setter
    def operator(self, op):
        if(not callable(op)):
            raise TypeError(
                'The data set to the "operator" keyword argument must be '\
                'callable!')
        self._operator = op

    #___________________________________________________________________________
    @property
    def operator_args(self):
        """The list of the arguments for the operator function specified via
        the :py:attr:`operator` property.

        If set to ``None``, it will be set to an empty list.

        """
        return self._operator_args
    #---------------------------------------------------------------------------
    @operator_args.setter
    def operator_args(self, args):
        if(args is None):
            args = list()
        if(not is_sequence(args)):
            args = [args]
        self._operator_args = list(args)

    #___________________________________________________________________________
    @property
    def operator_kwargs(self):
        """The dictionary with the keyword arguments for the operator function
        specified via the :py:attr:`operator` property.

        If set to ``None``, it will be set to an empty dictionary.

        """
        return self._operator_kwargs
    #---------------------------------------------------------------------------
    @operator_kwargs.setter
    def operator_kwargs(self, kwargs):
        if(kwargs is None):
            kwargs = dict()
        if(not isinstance(kwargs, dict)):
            raise TypeError(
                'The data set to the "operator_kwargs" property must be None, '\
                'or an instance of dict!')
        self._operator_kwargs = dict(kwargs)

    ###
    # Public class methods.
    #___________________________________________________________________________
    def copy(self):
        """Creates a copy of this CombinedLHComponent object by using its
        copy-constructor.

        """
        return CombinedLHComponent(combinedlhcomponent=self)

    #___________________________________________________________________________
    def evaluate(self, params, events):
        """Evaluates this combined lh component for the given events and set of
        parameters. First it evaluates the two LH components. Afterwards it
        combines the two data streams via the defined operator. Finally, it
        applies any additional operators on the combined data.
        """
        data_a = self._lhcomponent_a.evaluate(params, events)
        data_b = self._lhcomponent_b.evaluate(params, events)

        data = self._operator(data_a, data_b,
            *self._operator_args,
            **self._operator_kwargs)

        # Apply all (for the combined data stream) defined operations on the
        # combined data.
        for operation in self.operation_list:
            data = operation(data)

        return data

class LHComponentOperator(object):
    """The LHComponentOperator class defines an operator function that operates
    on the data of a LHComponent.
    """
    def __init__(self, operator, operator_args, operator_kwargs):
        self.operator = operator
        self.operator_args = operator_args
        self.operator_kwargs = operator_kwargs

    #___________________________________________________________________________
    @property
    def operator(self):
        """The operator function object that operates on the data of a
        LHComponent.
        It must have the call signature
        ``data_new = operator(data_a, data_b, *operator_args, **operator_kwargs)``.

        """
        return self._operator
    #---------------------------------------------------------------------------
    @operator.setter
    def operator(self, op):
        if(not callable(op)):
            raise TypeError(
                'The data set to the "operator" keyword argument must be '\
                'callable!')
        self._operator = op

    #___________________________________________________________________________
    @property
    def operator_args(self):
        """The list of the arguments for the operator function specified via
        the :py:attr:`operator` property.

        If set to ``None``, it will be set to an empty list.

        """
        return self._operator_args
    #---------------------------------------------------------------------------
    @operator_args.setter
    def operator_args(self, args):
        if(args is None):
            args = list()
        if(not is_sequence(args)):
            args = [args]
        self._operator_args = list(args)

    #___________________________________________________________________________
    @property
    def operator_kwargs(self):
        """The dictionary with the keyword arguments for the operator function
        specified via the :py:attr:`operator` property.

        If set to ``None``, it will be set to an empty dictionary.

        """
        return self._operator_kwargs
    #---------------------------------------------------------------------------
    @operator_kwargs.setter
    def operator_kwargs(self, kwargs):
        if(kwargs is None):
            kwargs = dict()
        if(not isinstance(kwargs, dict)):
            raise TypeError(
                'The data set to the "operator_kwargs" property must be None, '\
                'or an instance of dict!')
        self._operator_kwargs = dict(kwargs)

    #___________________________________________________________________________
    def __call__(self, data):
        """Applies the defined operation on the given data.

        """
        return self.operator(data, *self.operator_args, **self.operator_kwargs)

class LHFunctionParams(object):
    """The LHFunctionParams class provides a container for the parameters used
    in the likelihood function and its components.
    It provides the method `def_param` to define a parameter with its name,
    initial value, and boundaries.
    """
    def __init__(self):
        self.clear()

    def __eq__(self, other):
        """Implements the `self == other` comparison.
        It compares if all the parameters are the same for this object and the
        specified other one.
        """
        # Check if the number of parameters are equal.
        if(self.N != other.N):
            return False

        # Check if all the names are equal.
        self_names = self.names
        other_names = other.names
        for i in xrange(self.N):
            if(self_names[i] != other_names[i]):
                return False

        # Check if all the initials are equal.
        if(not np.all(np.array(self.initials) == np.array(other.initials))):
            return False

        # Check if all the isconst flags are equal.
        if(not np.all(np.array(self.isconst_list) == np.array(other.isconst_list))):
            return False

        # Check if all the bounds are equal.
        if(not np.all(np.array(self.bounds) == np.array(other.bounds))):
            return False

        return True

    def __ne__(self, other):
        """Implements the `self != other` comparison.
        It's defined as `not (self == other)`.
        """
        return not (self == other)

    def __str__(self):
        """Implements the pretty string output of this LHFunctionParams object.
        """
        s = ''
        print self.bounds
        for i in xrange(self.N):
            if(self.isconst_list[i] is True):
                s += '%s: initial %e (constant)\n'%(self.names[i], self.initials[i])
            else:
                s += '%s: initial %e, bounds (%e,%e)\n'%(self.names[i], self.initials[i], self.bounds[i][0], self.bounds[i][1])
        return s

    def def_param(self, name, initial=0, isconst=False, valmin=None, valmax=None):
        """Defines a parameter of the likelihood function. The order of the
        definition is important and is conserved.

        Parameters
        ----------
        name : str | LHFunctionParams
            The parameter's name.
            If name is an instance of LHFunctionParams, the parameters will be
            a copy of the parameters defined in that instance.

        initial : float
            The (initial) value (guess) of the parameter, which will be used as
            start point for the minimizer.

        isconst : bool
            Flag if the new parameter has a constant (True) or a
            variable (False).

        valmin : float
            The minimal bound value of the parameter.
            A finite number must be specified for variable parameters.

        valmax : float
            The maximal bound value of the parameter.
            A finite number must be specified for variable parameters.

        Returns
        -------
        self : LHFunction
            The object itself to allow for chained parameter definitions.
        """
        if(isinstance(name, LHFunctionParams)):
            # Copy the parameter definitions from the given LHFunctionParams
            # object.
            self.clear()
            for i in xrange(name.N):
                self.def_param(name.names[i], name.initials[i], name.isconst_list[i], name.bounds[i][0], name.bounds[i][1])
            return self

        if(not isinstance(name, str)):
            raise TypeError('The `name` argument must be of type str!')
        if(not isinstance(initial, float)):
            raise TypeError('The `initial` argument must be of type float!')
        if(not isinstance(isconst, bool)):
            raise TypeError('The `isconst` argument must be of type bool!')
        if(valmin is None):
            if(not isconst):
                raise ValueError('A lower bound must be specified for parameter "%s"!'%(name))
            else:
                valmin = -np.inf
        if(not isinstance(valmin, float)):
            raise TypeError('The `valmin` argument must be of type float!')
        if(valmax is None):
            if(not isconst):
                raise ValueError('An upper bound must be specified for parameter "%s"!'%(name))
            else:
                valmax = +np.inf
        if(not isinstance(valmax, float)):
            raise TypeError('The `valmax` argument must be of type float!')

        if(not isconst):
            if(np.abs(valmin) == np.inf or np.abs(valmax) == np.inf):
                raise ValueError('The variable parameter "%s" must have finite value bounds!'%(name))
            if(valmin >= valmax):
                raise ValueError('The lower bound must be smaller than the upper bound for parameter "%s"!'%(name))

        self._param_name_list.append(name)
        self._param_value_list.append(initial)
        self._param_isconst_list.append(isconst)
        self._param_bounds_list.append((valmin, valmax))

        return self

    def clear(self):
        """Deletes all the parameter definitions.
        """
        self._param_name_list = list()
        self._param_value_list = list()
        self._param_isconst_list = list()
        self._param_bounds_list = list()

    def generate_random_initials(self):
        """Generates a set of random initials for all variable parameters.
        A new random initial is defined as

            lower_bound + RAND * (upper_bound - lower_bound),

        where RAND is a uniform random variable between 0 and 1.
        """
        vb = self.variable_bounds
        # Do random_initial = lower_bound + RAND * (upper_bound - lower_bound)
        ri = vb[:,0] + np.random.uniform(size=vb.shape[0]) * (vb[:,1] - vb[:,0])
        print 'random initials:', ri
        return ri

    def get_param_idx(self, name):
        """Returns the parameter index for the given parameter name.
        It raises a KeyError if the given parameter does not exist.
        """
        for i in xrange(len(self._param_name_list)):
            if(self._param_name_list[i] == name):
                return i
        raise KeyError('The parameter "%s" is not defined!'%(name))

    @property
    def N(self):
        """The number of parameters defined.
        """
        return len(self._param_name_list)

    @property
    def N_var(self):
        """The number of variable parameters defined.
        """
        return np.count_nonzero(~np.array(self.isconst_list))

    @property
    def names(self):
        """The list of names of all the defined parameters.
        """
        return self._param_name_list

    @property
    def initials(self):
        """The 1d numpy.ndarray holding the initial values (guess) of all the
        parameters.
        """
        return np.array(self._param_value_list)

    @property
    def isconst_list(self):
        """The list of bools specifying which parameter is a constant.
        """
        return self._param_isconst_list

    @property
    def variable_initials(self):
        """The ndarray with the initial values of the variable parameters.
        """
        initials = np.array(self.initials, dtype=np.float64)
        isconst = np.array(self.isconst_list, dtype=np.bool)
        return initials[~isconst]

    @property
    def bounds(self):
        """The list of tuple holding the boundaries for all the parameters.
        """
        return self._param_bounds_list

    @property
    def variable_bounds(self):
        """The 2-dimensional numpy.ndarray holding the boundaries for all the
        variable parameters.
        """
        bounds = np.array(self._param_bounds_list, dtype=np.float64)
        isconst = np.array(self.isconst_list, dtype=np.bool)
        return bounds[~isconst,:]

class LHFunction(object):
    """This is the high level likelihood function class. It represents the final
    likelihood function, which gets constructed by the likelihood components.
    It also holds the (variable) parameters of the likelihood function.
    """
    def __init__(self, lhcomponent):
        """Constructs the likelihood function based on the given LHComponent
        object, which can also be a CombinedLHComponent object.

        Parameters
        ----------
        lhcomponent : LHComponent | callable
            The likelihood component (or object thereof) that describes the
            mathematical expression of the LH function.
            It can also be just a callable object that has the call signature
            __call__(params, events). In that case a LHComponent object will
            be created for that callable object automatically.
        """
        if(not isinstance(lhcomponent, LHComponent)):
            lhcomponent = LHComponent(lhcomponent)
        self._lh = lhcomponent
        self._params = LHFunctionParams()

    def evaluate(self, params, events):
        """Evaluates the LH function for the given set of parameters given the
        array of events.

        Parameters
        ----------
        params : dict
            The `name => value` dictionary with the current set of parameters.
        events: numpy.recarray
            The data, i.e. the events, for the likelihood function.
        """
        return self._lh.evaluate(params, events)

    def __call__(self, params, events):
        """Implements the call operator which is just a shortcut for the
        evaluate method.
        """
        return self.evaluate(params, events)

    @property
    def params(self):
        """The LHFunctionParams object holding the parameters for this
        likelihood function.
        """
        return self._params
    @params.setter
    def params(self, params):
        if(not isinstance(params, LHFunctionParams)):
            raise TypeError('The parameter object for the likelihood function must be of type LHFunctionParams!')
        self._params = params

    @property
    def ndim(self):
        """The dimensionality of the likelihood function. It's the number of
        parameters defined for this function.
        """
        return self._params.N

    @property
    def n_var_dim(self):
        """The number of variable parameters of this likelihood function.
        """
        return self._params.N_var

class LHModel(object):
    """The LHModel class is the base class for all likelihood models.
    """
    def __init__(self):
        pass
