# -*- coding: utf-8 -*-

"""This module provides functionality for interpolation.
"""

import abc
import numpy as np

from skyllh.core.parameters import (
    ParameterGrid,
    ParameterGridSet
)
from skyllh.core.py import classname


class GridManifoldInterpolationMethod(object, metaclass=abc.ABCMeta):
    """This is an abstract base class for implementing a method to interpolate
    a manifold that is defined on a grid of parameter values. In general the
    number of parameters can be arbitrary and hence the manifold's
    dimensionality can be arbitrary, too. However, in practice the interpolation
    on a multi-dimensional manifold can be rather difficult.
    Nevertheless, we provide this interface to allow for manifold grids with
    different dimensionality.
    """

    def __init__(self, func, param_grid_set, **kwargs):
        """Constructor for a GridManifoldInterpolationMethod object.
        It must be called by the derived class.

        Parameters
        ----------
        func : callable R^D -> R
            The function that takes D parameter grid values as input and returns
            the value of the D-dimensional manifold at this point for each given
            trial event and source.
            The call signature of func must be:

                ``__call__(tdm, eventdata, gridparams_recarray, src_idxs)``

            where ``tdm`` is the TrialDataManager instance holding the trial
            data, ``gridparams_recarray`` is the numpy record ndarray of length
            ``len(src_idxs)`` with the D parameter names and values on the grid
            for all sources, and ``eventdata`` is a 2-dimensional
            (N_events,V)-shaped numpy ndarray holding the event data, where
            N_events is the number of events, and V the dimensionality of the
            event data.
            The return value of ``func`` should be the (N,)-shaped 1D ndarray
            holding the values for each set of parameter values of the sources
            given via the ``gridparams_recarray``. The length of the array, i.e.
            N, depends on the ``src_evt_idx`` property of the TrialDataManager
            and the requested set of sources via the ``src_idxs`` argument.
            In the worst case N is N_sources * N_events.
        param_grid_set : instance of ParameterGrid |
                         instance of ParameterGridSet
            The set of D parameter grids. This defines the grid of the
            manifold.
        """
        super().__init__(**kwargs)

        self.func = func
        self.param_grid_set = param_grid_set

    @property
    def func(self):
        """The R^d -> R manifold function.
        """
        return self._func

    @func.setter
    def func(self, f):
        if not callable(f):
            raise TypeError(
                'The func property must be a callable object!')
        self._func = f

    @property
    def param_grid_set(self):
        """The ParameterGridSet instance defining the set of d parameter grids.
        This defines the grid of the manifold.
        """
        return self._param_grid_set

    @param_grid_set.setter
    def param_grid_set(self, obj):
        if isinstance(obj, ParameterGrid):
            obj = ParameterGridSet([obj])
        elif not isinstance(obj, ParameterGridSet):
            raise TypeError(
                'The param_grid_set property must be an instance '
                'of ParameterGrid or ParameterGridSet!')
        self._param_grid_set = obj

    @property
    def ndim(self):
        """(read-only) The dimensionality of the manifold.
        """
        return len(self._param_grid_set)

    def _broadcast_params_recarray_to_values_array(
            self,
            tdm,
            params_recarray):
        """Broadcasts the given source parameter record array to all values of
        a PDF evaluation, i.e. to all sources and trial events. The result is
        a numpy record array of length N_values.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial events and the
            mapping of trial events to the sources.
        params_recarray : instance of numpy record array
            The numpy record array of length N_sources holding the parameter
            names and values for all sources.

        Returns
        -------
        values_params_recarray : instance of numpy record ndarray
            The numpy record array of length N_values holding the source
            parameter names and values for all sources and trial events.
        """
        if len(params_recarray) != tdm.n_sources:
            raise ValueError(
                f'The length of params_recarray array ({len(params_recarray)}) '
                f'must be equal to the number of sources ({tdm.n_sources})!')

        tdm_src_evt_idxs = tdm.src_evt_idxs
        if tdm_src_evt_idxs is None:
            # All trial events contibute to all sources.
            values_params_recarray = np.repeat(
                params_recarray, tdm.n_selected_events)

            return values_params_recarray

        # A mapping of trial events to sources is defined.
        (src_idxs, evt_idxs) = tdm_src_evt_idxs
        values_params_recarray = np.empty(
            (len(src_idxs),), dtype=params_recarray.dtype)

        v_start = 0
        for (src_idx, src_params_recarray) in enumerate(params_recarray):
            n = len(evt_idxs[src_idxs == src_idx])
            values_params_recarray[v_start:v_start+n] = np.tile(
                src_params_recarray, n)
            v_start += n

        return values_params_recarray

    @abc.abstractmethod
    def __call__(self, tdm, eventdata, params_recarray):
        """Retrieves the interpolated value of the manifold at the D-dimensional
        point ``params_recarray`` for all given events and sources, along with
        the D gradients, i.e. partial derivatives.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data.
        eventdata : numpy ndarray
            The 2D (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of trial events, and V the
            dimensionality of the event data.
        params_recarray : instance of numpy record ndarray
            The numpy record ndarray holding the N_sources set of parameter
            names and values, that define the point (for each source) on the
            manifold for which the value should get calculated for each event.

        Returns
        -------
        values : ndarray of float
            The (N,)-shaped numpy ndarray holding the interpolated manifold
            values for the given events and sources.
        grads : ndarray of float
            The (D,N)-shaped numpy ndarray holding the D manifold gradients for
            the N given values, where D is the number of parameters.
            The order of the D parameters is defined by the ParameterGridSet
            that has been provided at construction time of this interpolation
            method object.
        """
        pass


class NullGridManifoldInterpolationMethod(GridManifoldInterpolationMethod):
    """This grid manifold interpolation method performes no interpolation. When
    the
    :meth:`~skyllh.core.interpolate.NullGridManifoldInterpolationMethod.get_value_and_gradients`
    method is called, it rounds the parameter values to their nearest grid
    point values. All gradients are set to zero.
    """
    def __init__(
            self,
            func,
            param_grid_set,
            **kwargs):
        """Creates a new NullGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func : callable R^d -> R
            The function that takes d parameter grid values as input and returns
            the value of the d-dimensional manifold at this point for each given
            trial event and source.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set : instance of ParameterGrid | instance of ParameterGridSet
            The set of d parameter grids. This defines the grid of the
            manifold.
        """
        super().__init__(
            func=func,
            param_grid_set=param_grid_set,
            **kwargs)

    def __call__(self, tdm, eventdata, params_recarray):
        """Calculates the non-interpolated manifold value and its gradient
        (zero) for each given event and source at the points given by
        ``params_recarray``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data.
        eventdata : numpy ndarray
            The (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params_recarray : instance of numpy record ndarray
            The numpy record ndarray holding the D parameter values,
            defining the point on the manifold for which the values should get
            calculated.

        Returns
        -------
        values : ndarray of float
            The (N,)-shaped numpy ndarray holding the interpolated manifold
            values for the given events and sources.
        grads :  ndarray of float
            The (D,N)-shaped ndarray of float holding the D manifold gradients
            for the N values, where D is the number of parameters.
            By definition, all gradients are zero.
        """
        # Round the given parameter values to their nearest grid values.
        gridparams_recarray = np.empty_like(params_recarray)
        for pname in params_recarray.dtype.fields.keys():
            p_grid = self._param_grid_set[pname]
            pvalues = params_recarray[pname]
            gridparams_recarray[pname] = p_grid.round_to_nearest_grid_point(
                pvalues)

        values = self._func(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=params_recarray)

        grads = np.zeros(
            (len(params_recarray.dtype.fields), len(values)),
            dtype=np.float64)

        return (values, grads)


class Linear1DGridManifoldInterpolationMethod(GridManifoldInterpolationMethod):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a line.
    """
    def __init__(
            self,
            func,
            param_grid_set,
            **kwargs):
        """Creates a new Linear1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func : callable R -> R
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given source and trial event.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set : instance of ParameterGrid | instance of ParameterGridSet
            The one parameter grid. This defines the grid of the manifold.
        """
        super().__init__(
            func=func,
            param_grid_set=param_grid_set,
            **kwargs)

        if len(self._param_grid_set) != 1:
            raise ValueError(
                f'The {classname(self)} class supports only 1D grid manifolds. '
                'The param_grid_set argument must contain 1 ParameterGrid '
                f'instance! Currently it has {len(self._param_grid_set)}!')
        self.p_grid = self._param_grid_set[0]

        # Create a cache for the line parameterization for the last
        # manifold grid point for the different events.
        self._cache = self._create_cache(
            trial_data_state_id=None,
            x0=None,
            m=None,
            b=None)

    def _create_cache(self, trial_data_state_id, x0, m, b):
        """Creates a cache for the line parameterization for the last manifold
        grid point for the nevents different events.

        Parameters
        ----------
        trial_data_state_id : int | None
            The trial data state id of the TrialDataManager.
        x0 : instance of ndarray | None
            The (N_sources,)-shaped numpy ndarray holding the parameter grid
            value of the lower point of the grid manifold for each source used
            to estimate the line.
        m : instance of ndarray | None
            The (N_values,)-shaped numpy ndarray holding the slope of the line
            for each trial event and source.
        b : instance of ndarray | None
            The (N_values,)-shaped numpy ndarray holding the offset coefficient
            of the line for each trial event and source.
        """
        cache = {
            'trial_data_state_id': trial_data_state_id,
            'x0': x0,
            'm': m,
            'b': b
        }

        return cache

    def __call__(self, tdm, eventdata, params_recarray):
        """Calculates the interpolated manifold value and its gradient for each
        given event and source at the point ``params_recarray``.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance holding the trial data.
        eventdata : numpy ndarray
            The (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params_recarray : numpy record ndarray
            The numpy record ndarray of length N_sources holding the parameter
            names and values for each source, defining the point on the manifold
            for which the value should get calculated.

        Returns
        -------
        values : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of parameters.
        """
        # Broadcast params_recarray to all sources and trial events.
        values_params_recarray = \
            self._broadcast_params_recarray_to_values_array(
                tdm=tdm,
                params_recarray=params_recarray)

        xname = next(iter(params_recarray.dtype.fields.keys()))
        x = params_recarray[xname]

        # TODO: Implement optimization when all x values are the same, i.e.
        # all sources share the same parameter.

        self__p_grid = self.p_grid

        # Determine the nearest grid point that is lower than x and use that as
        # x0.
        x0 = self__p_grid.round_to_lower_grid_point(x)

        # Check if the line parametrization for x0 is already cached.
        self__cache = self._cache
        if (np.all(np.isclose(self__cache['x0'], x0))) and\
           (self__cache['trial_data_state_id'] is not None) and\
           (self__cache['trial_data_state_id'] == tdm.trial_data_state_id):
            m = self__cache['m']
            b = self__cache['b']

            x = values_params_recarray[xname]
            values = m*x + b

            return (values, np.atleast_2d(m))

        # The line parametrization is not cached.
        # Calculate the line parametrization for all the given events.
        self__f = self._func

        x1 = self__p_grid.round_to_upper_grid_point(x)

        n_values = tdm.get_n_values()
        N_sources = len(params_recarray)

        all_src_idxs = np.arange(N_sources)

        values = np.empty((n_values,), dtype=np.float64)
        m = np.empty((n_values,), dtype=np.float64)

        if tdm.src_evt_idxs is None:
            tdm_src_idxs = np.repeat(all_src_idxs, N_sources)
        else:
            tdm_src_idxs = tdm.src_evt_idxs[0]

        def get_make_values_mask(src_idxs):
            return np.vectorize(lambda sidx: sidx in src_idxs)

        # Calculate the values where the parameter values fall onto a grid
        # point.
        # In that case x0 and x1 are equal.
        # The value will be of that grid point x0, but the gradient is
        # calculated based on the two neighboring grid points of x0.
        src_mask = x1 == x0

        src_idxs = all_src_idxs[src_mask]

        values_mask = get_make_values_mask(src_idxs)(tdm_src_idxs)
        N_values = np.count_nonzero(values_mask)

        gridparams_recarray_x0 = np.array(
            x0[src_mask],
            dtype=[(xname, np.float64)])
        values[values_mask] = self__f(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=gridparams_recarray_x0,
            src_idxs=src_idxs,
            N_values=N_values,
            ret_gridparams_recarray=False)

        gridparams_recarray_x0 = np.array(
            self__p_grid.round_to_nearest_grid_point(
                x0[src_mask] - self__p_grid.delta),
            dtype=[(xname, np.float64)])
        (M0, x0_recarr) = self__f(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=gridparams_recarray_x0,
            N_values=N_values,
            ret_gridparams_recarray=True)

        gridparams_recarray_x1 = np.array(
            self__p_grid.round_to_nearest_grid_point(
                x1[src_mask] + self__p_grid.delta),
            dtype=[(xname, np.float64)])
        (M1, x1_recarr) = self__f(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=gridparams_recarray_x1,
            N_values=N_values,
            ret_gridparams_recarray=True)

        m[values_mask] = (M1 - M0) / (x1_recarr[xname] - x0_recarr[xname])

        # Calculate the values where the parameter values do not fall onto a
        # grid point.
        # In that case x0 and x1 are not equal.
        src_mask = np.invert(src_mask)

        src_idxs = all_src_idxs[src_mask]

        values_mask = get_make_values_mask(src_idxs)(tdm_src_idxs)
        N_values = np.count_nonzero(values_mask)

        gridparams_recarray_x0 = np.array(
            x0[src_mask],
            dtype=[(xname, np.float64)])
        (M0, x0_recarr) = self__f(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=gridparams_recarray_x0,
            N_values=N_values,
            ret_gridparams_recarray=True)

        gridparams_recarray_x1 = np.array(
            x1[src_mask],
            dtype=[(xname, np.float64)])
        (M1, x1_recarr) = self__f(
            tdm=tdm,
            eventdata=eventdata,
            gridparams_recarray=gridparams_recarray_x1,
            N_values=N_values,
            ret_gridparams_recarray=True)

        m[values_mask] = (M1 - M0) / (x1_recarr[xname] - x0_recarr[xname])
        b = M0 - m[values_mask]*x0_recarr[xname]

        # Cache the line parametrization.
        self._cache = self._create_cache(
            trial_data_state_id=tdm.trial_data_state_id,
            x0=x0,
            m=m,
            b=b)

        # Calculate the interpolated manifold values. The gradient is m.
        x = values_params_recarray[xname]
        values = m*x + b

        return (values, np.atleast_2d(m))


class Parabola1DGridManifoldInterpolationMethod(GridManifoldInterpolationMethod):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a parabola.
    """
    def __init__(
            self,
            func,
            param_grid_set,
            **kwargs):
        """Creates a new Parabola1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func : callable R -> R
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given source and trial event.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set : instance of ParameterGrid | instance of ParameterGridSet
            The one parameter grid. This defines the grid of the manifold.
        """
        super().__init__(
            func=func,
            param_grid_set=param_grid_set,
            **kwargs)

        if len(self._param_grid_set) != 1:
            raise ValueError(
                f'The {classname(self)} class supports only 1D grid manifolds. '
                'The param_grid_set argument must contain 1 ParameterGrid '
                f'instance! Currently it has {len(self._param_grid_set)}!')
        self._p_grid = self._param_grid_set[0]

        # Create a cache for the parabola parameterization for the last
        # manifold grid point for the different events.
        self._cache = self._create_cache(
            trial_data_state_id=None,
            x1=None,
            M1=None,
            a=None,
            b=None)

    def _create_cache(self, trial_data_state_id, x1, M1, a, b):
        """Creates a cache for the parabola parameterization for the last
        manifold grid point for the nevents different events.

        Parameters
        ----------
        trial_data_state_id : int | None
            The trial data state ID of the TrialDataManager.
        x1 : float | None
            The parameter grid value for middle point of the grid manifold used
            to estimate the parabola.
        M1 : 1d ndarray
            The grid manifold value for each event of the middle point (x1,).
        a : 1d ndarray
            The parabola coefficient ``a`` for each event.
        b : 1d ndarray
            The parabola coefficient ``b`` for each event.
        """
        self._cache = {
            'trial_data_state_id': trial_data_state_id,
            'x1': x1,
            'M1': M1,
            'a': a,
            'b': b
        }

    # FIXME adapt to params_recarray!
    def get_value_and_gradients(self, tdm, eventdata, params):
        """Calculates the interpolated manifold value and its gradient for each
        given event at the point ``params``.

        Parameters
        ----------
        tdm : TrialDataManager
            The TrialDataManager instance holding the trial data.
        eventdata : numpy (N_events,V)-shaped 2D ndarray
            The 2D (N_events,V)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params : dict
            The dictionary with the parameter values, defining the point on the
            manifold for which the value should get calculated.

        Returns
        -------
        value : (N,) ndarray of float
            The interpolated manifold value for the N given events.
        gradients : (D,N) ndarray of float
            The D manifold gradients for the N given events, where D is the
            number of parameters.
        """
        (xname, x) = tuple(params.items())[0]

        # Create local variable name alias to avoid Python dot lookups.
        self__p_grid = self._p_grid
        self__p_grid__round_to_nearest_grid_point = \
            self__p_grid.round_to_nearest_grid_point
        self__cache = self._cache

        tdm_trial_data_state_id = tdm.trial_data_state_id

        # Determine the nearest grid point x1.
        x1 = self__p_grid__round_to_nearest_grid_point(x)

        # Check if the parabola parametrization for x1 is already cached.
        if (self__cache['x1'] == x1) and\
           (self__cache['trial_data_state_id'] is not None) and\
           (self__cache['trial_data_state_id'] == tdm_trial_data_state_id):
            M1 = self__cache['M1']
            a = self__cache['a']
            b = self__cache['b']
        else:
            dx = self__p_grid.delta

            # Calculate the neighboring grid points to x1: x0 and x2.
            x0 = self__p_grid__round_to_nearest_grid_point(x1 - dx)
            x2 = self__p_grid__round_to_nearest_grid_point(x1 + dx)

            # Parameterize the parabola with parameters a, b, and M1.
            self__f = self.f
            M0 = self__f(tdm, {xname: x0}, eventdata)
            M1 = self__f(tdm, {xname: x1}, eventdata)
            M2 = self__f(tdm, {xname: x2}, eventdata)

            a = 0.5*(M0 - 2.*M1 + M2) / dx**2
            b = 0.5*(M2 - M0) / dx

            # Cache the parabola parametrization.
            self._create_cache(tdm_trial_data_state_id, x1, M1, a, b)

        # Calculate the interpolated manifold value.
        value = a * (x - x1)**2 + b * (x - x1) + M1
        # Calculate the gradient of the manifold.
        gradients = 2. * a * (x - x1) + b

        return (value, np.atleast_2d(gradients))
