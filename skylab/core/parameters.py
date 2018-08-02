# -*- coding: utf-8 -*-

import abc
import itertools
import numpy as np
from copy import deepcopy

from skylab.physics.source import SourceModel, SourceCollection
from skylab.core.py import ObjectCollection, issequence, float_cast

def make_linear_parameter_grid_1d(name, low, high, delta):
    """Utility function to create a ParameterGrid object for a 1-dimensional
    linear parameter grid.

    Parameters
    ----------
    name : str
        The name of the parameter.
    low : float
        The lowest value of the parameter.
    high : float
        The highest value of the parameter.
    delta : float
        The constant distance between the grid values. By definition this
        defines also the precision of the parameter values.

    Returns
    -------
    obj : ParameterGrid
        The ParameterGrid object holding the discrete parameter grid values.
    """
    grid = np.linspace(low, high, np.round((high-low)/delta)+1)
    return ParameterGrid(name, grid, delta)

def make_params_hash(params):
    """Utility function to create a hash value for a given parameter dictionary.

    Parameters
    ----------
    params : dict
        The dictionary holding the parameter (name: value) pairs.

    Returns
    -------
    hash : int
        The hash of the parameter dictionary.
    """
    if(not isinstance(params, dict)):
        raise TypeError('The params argument must be of type dict!')

    # A note on the ordering of Python dictionary items: The items are ordered
    # internally according to the hash value of their keys. Hence, if we don't
    # insert more dictionary items, the order of the items won't change. Thus,
    # we can just take the items list and make a tuple to create a hash of it.
    # The hash will be the same for two dictionaries having the same items.
    return hash(tuple(params.items()))

class ParameterGrid(object):
    """This class provides a data holder for a parameter that has a set of
    discrete values, i.e. has a value grid.
    """
    def __init__(self, name, grid, precision):
        """Creates a new parameter grid.

        Parameters
        ----------
        name : str
            The name of the parameter.
        grid : numpy.ndarray
            The numpy ndarray holding the discrete grid values of the parameter.
        precision : float
            The precision of the parameter values.
        """
        self.name = name
        self.precision = precision
        # Setting the grid, will automatically round the grid values to the
        # precision of the grid. Hence, we need to set the grid property after
        # setting the precision property.
        self.grid = grid

    @property
    def name(self):
        """The name of the parameter.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

    @property
    def grid(self):
        """The numpy.ndarray with the grid values of the parameter.
        """
        return self._grid
    @grid.setter
    def grid(self, arr):
        if(not isinstance(arr, np.ndarray)):
            raise TypeError('The grid property must be of type numpy.ndarray!')
        if(arr.ndim != 1):
            raise ValueError('The grid property must be a 1D numpy.ndarray!')
        self._grid = self.round_to_precision(arr)

    @property
    def precision(self):
        """The precision (float) of the parameter values.
        """
        return self._precision
    @precision.setter
    def precision(self, value):
        if(isinstance(value, int)):
            value = float(value)
        if(not isinstance(value, float)):
            raise TypeError('The precision property must be of type float!')
        self._precision = value

    @property
    def ndim(self):
        """The dimensionality of the parameter grid.
        """
        return self._grid.ndim

    def add_extra_lower_and_upper_bin(self):
        """Adds an extra lower and upper bin to this parameter grid. This is
        usefull when interpolation or gradient methods require an extra bin on
        each side of the grid.
        """
        newgrid = np.empty((self._grid.size+2,))
        newgrid[1:-1] = self._grid
        newgrid[0] = newgrid[1] - self._precision
        newgrid[-1] = newgrid[-2] + self._precision
        self.grid = newgrid

    def copy(self):
        """Copies this ParameterGrid object and returns the copy.
        """
        copy = deepcopy(self)
        return copy

    def round_to_precision(self, value):
        """Rounds the given value to the precision of the parameter.

        Parameters
        ----------
        value : float | ndarray
            The value(s) to round.
        """
        return np.around(value / self._precision) * self._precision

class ParameterGridSet(ObjectCollection):
    """Describes a set of parameter grids.
    """
    def __init__(self, param_grid_list=None):
        """Constructs a new ParameterGridSet object.

        Parameters
        ----------
        param_grid_list : list of ParameterGrid | ParameterGrid | None
            The list of ParameterGrid objects with which this set should get
            initialized with.
        """
        super(ParameterGridSet, self).__init__(obj_type=ParameterGrid, obj_list=param_grid_list)

    @property
    def ndim(self):
        """The dimensionality of this parameter grid set. By definition it's the
        number of parameters of the set.
        """
        return len(self)

    @property
    def parameter_names(self):
        """(read-only) The list of the parameter names.
        """
        return [ paramgrid.name for paramgrid in self.objects ]

    @property
    def parameter_permutation_dict_list(self):
        """(read-only) The list of parameter dictionaries constructed from all
        permutations of all the parameter values.
        """
        # Get the list of parameter names.
        param_names = [ paramgrid.name for paramgrid in self.objects ]
        # Get the list of parameter grids, in same order than the parameter
        # names.
        param_grids = [ paramgrid.grid for paramgrid in self.objects ]

        dict_list = [ dict([ (p_i, t_i)
                            for (p_i, t_i) in zip(param_names, tup) ])
                     for tup in itertools.product(*param_grids) ]
        return dict_list

    def add_extra_lower_and_upper_bin(self):
        """Adds an extra lower and upper bin to all the parameter grids. This is
        usefull when interpolation or gradient methods require an extra bin on
        each side of the grid.
        """
        for paramgrid in self.objects:
            paramgrid.add_extra_lower_and_upper_bin()

    def copy(self):
        """Copies this ParameterGridSet object and returns the copy.
        """
        copy = deepcopy(self)
        return copy

class FitParameterManifoldGridInterpolationMethod(object):
    """This is an abstract base class for implementing a method to interpolate
    a set of fit parameters on a fit parameter manifold grid. In general the
    number of fit parameters can be arbitrary and hence the manifold's
    dimensionality can be arbitrary too. However, in practice the interpolation
    on a multi-dimensional manifold can be rather difficult.
    Nevertheless, we provide this interface to allow for different kinds of
    manifold grids with different dimensionality.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, f, fitparams_grid_set):
        """Constructor for a FitParameterManifoldGridInterpolationMethod object.
        It must be called by the derived class.

        Parameters
        ----------
        f : callable R^d -> R
            The function that takes d fit parameters as input and returns the
            value of the manifold at this d-dimensional point for each given
            event.
            The call signature of f must be:
                __call__(gridfitparams, eventdata)
            where gridfitparams is the dictionary with the fit parameter values
            on the grid and ``eventdata`` is a 2-dimensional (N,V)-shaped numpy
            ndarray holding the event data, where N is the number of events, and
            V the dimensionality of the event data.
        fitparams_grid_set : ParameterGridSet
            The set of d fit parameter grids. This defines the grid of the
            manifold.
        """
        self.f = f
        self.fitparams_grid_set = fitparams_grid_set

    @property
    def f(self):
        """The R^d -> R manifold function.
        """
        return self._f
    @f.setter
    def f(self, func):
        if(not callable(func)):
            raise TypeError('The f property must be a callable object!')
        self._f = func

    @property
    def fitparams_grid_set(self):
        """The ParameterGridSet object defining the set of d fit parameter
        grids. This defines the grid of the manifold.
        """
        return self._fitparams_grid_set
    @fitparams_grid_set.setter
    def fitparams_grid_set(self, obj):
        if(not isinstance(obj, ParameterGridSet)):
            raise TypeError('The fitparams_grid_set property must be an instance of ParameterGridSet!')
        self._fitparams_grid_set = obj

    @property
    def ndim(self):
        """(read-only) The dimensionality of the manifold.
        """
        return len(self.fitparams_grid_set)

    @abc.abstractmethod
    def get_value_and_gradients(self, events, fitparams):
        """Retrieves the interpolated value of the manifold at the d-dimensional
        point ``fitparams`` for all given events, along with the d gradients,
        i.e. partial derivatives.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
            manifold value should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values, defining the point
            on the manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of fit parameters. The order of the D parameters is defined
            by the ParameterGridSet that has been provided at construction time
            of this interpolation method object.
        """
        pass


class ParabolaFitParameterInterpolationMethod(FitParameterManifoldGridInterpolationMethod):
    """This fit parameter manifold grid interpolation method interpolates the
    1-dimensional fit parameter manifold using a parabola.
    """
    def __init__(self, f, fitparams_grid_set):
        super(ParabolaFitParameterInterpolationMethod, self).__init__(f, fitparams_grid_set)

        self.p_grid = self.fitparams_grid_set[0]

        # Create a cache for the parabola parameterization for the last
        # manifold grid point for the different events.
        self._create_cache(None, np.array([]), np.array([]), np.array([]))

    def _create_cache(self, x1, M1, a, b):
        """Create a cache for the parabola parameterization for the last
        manifold grid point p1 for the nevents different events.

        Parameters
        ----------
        x1 : float | None
        M1 : 1d ndarray
        a : 1d ndarray
        b : 1d ndarray
        """
        self._cache = {
            'x1': x1,
            'M1': M1,
            'a': a,
            'b': b
        }

    def get_value_and_gradients(self, events, fitparams):
        """Calculates the interpolted manifold value and its gradient for each
        given event at the point ``fitparams``.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record ndarray holding the data events for which the
            manifold value should get calculated.
        fitparams : dict
            The dictionary with the fit parameter values, defining the point
            on the manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of fit parameters.
        """
        (xname, x) = fitparams.items()[0]

        # Determine the nearest grid point x1 and the grid precision.
        x1 = self.p_grid.round_to_precision(x)
        dx = self.p_grid.precision

        # Check if the parabola parametrization for x1 is already cached.
        if((self._cache['x1'] == x1) and
           (len(events) == len(self._cache['M1']))
          ):
            M1 = self._cache_parabola['M1']
            a = self._cache_parabola['a']
            b = self._cache_parabola['b']
        else:
            # Calculate the neighboring gridponts to x1: x0 and x2.
            x0 = self.p_grid.round_to_precision(x1 - dx)
            x2 = self.p_grid.round_to_precision(x1 + dx)

            # Parameterize the parabola with parameters a, b, and M1.
            M0 = self.f({xname:x0}, eventdata)
            M1 = self.f({xname:x1}, eventdata)
            M2 = self.f({xname:x2}, eventdata)

            a = 0.5*(M0 - 2.*M1 + M2) / dx**2
            b = 0.5*(M2 - M0) / dx

            # Cache the parabola parametrization.
            self._create_cache(x1, M1, a, b)

        # Calculate the interpolated manifold value.
        value = a * (x - x1)**2 + b * (x - x1) + M1
        # Calculate the gradient of the manifold.
        gradients = 2. * a * (x - x1) + b

        return (value, np.atleast_2d(gradients))


class FitParameter(object):
    """This class describes a single fit parameter. A fit parameter has a name,
    a value range, an initial value, and a current value. The current value will
    be updated in the fitting process.
    """
    def __init__(self, name, valmin, valmax, initial):
        """Creates a new fit parameter object.

        Parameters
        ----------
        name : str
            The name of the fit parameter.
        valmin : float
            The minimal bound value of the fit parameter.
        valmax : float
            The maximal bound value of the fit parameter.
        initial : float
            The (initial) value (guess) of the parameter, which will be used as
            start point for the fitting procedure.
        """
        self.name = name
        self.valmin = valmin
        self.valmax = valmax
        self.initial = initial

        self.value = self.initial

    @property
    def name(self):
        """The name of the fit parameter.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be of type str!')
        self._name = name

    @property
    def valmin(self):
        """The minimal bound value of the fit parameter.
        """
        return self._valmin
    @valmin.setter
    def valmin(self, v):
        v = float_cast(v, 'The valmin property must castable to type float!')
        self._valmin = v

    @property
    def valmax(self):
        """The maximal bound value of the fit parameter.
        """
        return self._valmax
    @valmax.setter
    def valmax(self, v):
        v = float_cast(v, 'The valmax property must be castable to type float!')
        self._valmax = v

    @property
    def initial(self):
        """The initial value of the fit parameter.
        """
        return self._initial
    @initial.setter
    def initial(self, v):
        v = float_cast(v, 'The initial property must be castable to type float!')
        self._initial = v


class SourceFitParameterManager(object):
    """This class provides the functionality to manage the fit parameters of the
    sources. It provides a translator functionality to translate a fit
    parameter to a source model parameter of a source.
    Sometimes it's necessary to define a fit parameter, which relates to a
    source model parameter for a set of sources, while another fit parameter
    relates to the same source model parameter, but for another set of sources.

    At construction time this manager takes the collection of sources. Each
    source gets an index, which is defined as the position of the source within
    the collection.
    """
    def __init__(self, sources):
        """Constructs a new source fit parameter manager.

        Parameters
        ----------
        sources : SourceCollection | SourceModel
            The instance of a SourceCollection defining the list of sources.
            If a single source model is specified a SourceCollection object with
            this source will be constructed automatically.

        """
        self.sources = sources

        # Define the list of fit parameters and the list of source parameter
        # names, which map to the fit parameters.

        # Define the (N_fitparams,)-shaped numpy array of FitParameter objects.
        self._fit_params = np.empty((0,), dtype=np.object)
        # Define the (N_fitparams,)-shaped numpy array of str objects.
        self._src_param_names = np.empty((0,), dtype=np.object)
        # (N_fitparams, N_sources) shaped boolean ndarray defining what fit
        # parameter applies to which source.
        self._fit_param_2_src_mask = np.zeros((0, len(self.sources)), dtype=np.bool)

    @property
    def sources(self):
        """The SourceCollection defining the sources.
        """
        return self._sources
    @sources.setter
    def sources(self, obj):
        obj = SourceCollection.cast(obj, 'The sources property must be castable to an instance of SourceCollection!')
        self._sources = obj

    @property
    def initials(self):
        """(read-only) The 1D ndarray holding the initial values of all the fit
        parameters.
        """
        return np.array([ fit_param.initial
                         for fit_param in self._fit_params ], dtype=np.float)

    def def_fit_parameter(self, fit_param, src_param_name=None, sources=None):
        """Defines a new fit parameter that maps to a given source parameter
        for a list of sources. If no list of sources is given, it maps to all
        sources.

        Parameters
        ----------
        fit_param : FitParameter
            The FitParameter instance defining the fit parameter.
        src_param_name : str | None
            The name of the source parameter. It must match the name of a source
            model property. If set to None (default) the name of the fit
            parameter will be used.
        sources : SourceCollection | None
            The instance of SourceCollection with the sources for which the fit
            parameter applies. If None (the default) is specified, the fit
            parameter will apply to all sources.
        """
        if(not isinstance(fit_param, FitParameter)):
            raise TypeError('The fit_param argument must be an instance of FitParameter!')

        if(not isinstance(src_param_name, str)):
            raise TypeError('The src_param_name argument must be of type str!')

        if(sources is None):
            sources = self.sources
        sources = SourceCollection.cast(sources,
            'The sources argument must be castable to an instance of SourceCollection!')

        # Append the fit parameter and source parameter name to the internal
        # arrays.
        self._fit_params = np.concatenate((self._fit_params, [fit_param]))
        self._src_param_names = np.concatenate((self._src_param_names, [src_param_name]))

        # Get the list of source indices for which the fit parameter applies.
        mask = np.zeros((len(self.sources),), dtype=np.bool)
        for ((idx,src), applied_src) in itertools.product(enumerate(self.sources), sources):
            if(applied_src.id == src.id):
                mask[idx] = True
        self._fit_param_2_src_mask = np.vstack((self._fit_param_2_src_mask, mask))

    def get_source_fit_params(self, src_idx, fit_param_values):
        """Constructs a dictionary with the source parameters that are beeing
        fitted. As values the given global fit parameter values will be used.
        Hence, this method translates the global fit parameter values into the
        source parameters.

        Parameters
        ----------
        src_idx : int
            The index of the source for which the parameters should get
            retieved.
        fit_param_values : 1D ndarray
            The array holding the current global fit parameter values.

        Returns
        -------
        src_fitparams : dict
            The dictionary holding the translated source parameters that are
            beeing fitted.
        """
        # Get the mask of global fit parameters that apply to the requested
        # source.
        fp_mask = self._fit_param_2_src_mask[:,src_idx]

        # Get the source parameter names.
        src_param_names = self._src_param_names[fp_mask]
        src_param_values = fit_param_values[fp_mask]

        src_fitparams = dict(zip(src_param_names, src_param_values))

        return src_fitparams
