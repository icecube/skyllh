"""This module provides functionality for interpolation."""

import abc
from collections.abc import Callable
from typing import cast

import numpy as np

from skyllh.core.parameters import ParameterGrid, ParameterGridSet
from skyllh.core.py import (
    classname,
)
from skyllh.core.trialdata import TrialDataManager


class GridManifoldInterpolationMethod(
    metaclass=abc.ABCMeta,
):
    """This is an abstract base class for implementing a method to interpolate
    a manifold that is defined on a grid of parameter values. In general the
    number of parameters can be arbitrary and hence the manifold's
    dimensionality can be arbitrary, too. However, in practice the interpolation
    on a multi-dimensional manifold can be rather difficult.
    Nevertheless, we provide this interface to allow for manifold grids with
    different dimensionality.
    """

    def __init__(
        self,
        func: Callable,
        param_grid_set: ParameterGrid | ParameterGridSet,
        **kwargs,
    ):
        """Constructor for a GridManifoldInterpolationMethod object.
        It must be called by the derived class.

        Parameters
        ----------
        func
            The function that takes D parameter grid values as input and returns
            the value of the D-dimensional manifold at this point for each given
            trial event and source.
            The call signature of func must be:

                ``__call__(tdm, eventdata, gridparams_recarray, n_values, **kwargs)``

            The arguments are as follows:

                tdm
                    The TrialDataManager instance holding the trial event data.
                eventdata
                    A two-dimensional (V,N_events)-shaped numpy ndarray holding
                    the event data, where N_events is the number of trial
                    events, and V the dimensionality of the event data.
                gridparams_recarray
                    The structured numpy ndarray of length ``len(src_idxs)``
                    with the D parameter names and values on the grid for all
                    sources.
                n_values
                    The length of the output numpy ndarray of shape (n_values,).
                **kwargs
                    Additional keyword arguments required by ``func``.

            The return value of ``func`` should be the (n_values,)-shaped
            one-dimensional ndarray holding the values for each set of parameter
            values of the sources given via the ``gridparams_recarray``.
            The length of the array, i.e. n_values, depends on the
            ``src_evt_idx`` property of the TrialDataManager. In the worst case
            n_values is N_sources * N_events.
        param_grid_set
            The set of D parameter grids. This defines the grid of the
            manifold.
        """
        super().__init__(**kwargs)

        self.func = func
        self.param_grid_set = param_grid_set

    @property
    def func(self):
        """The R^d -> R manifold function."""
        return self._func

    @func.setter
    def func(self, f):
        if not callable(f):
            raise TypeError('The func property must be a callable object!')
        self._func = f

    @property
    def param_grid_set(self) -> 'ParameterGridSet':
        """The ParameterGridSet instance defining the set of D parameter grids.
        This defines the grid of the manifold.
        """
        return self._param_grid_set

    @param_grid_set.setter
    def param_grid_set(self, obj):
        if isinstance(obj, ParameterGrid):
            obj = ParameterGridSet([obj])
        elif not isinstance(obj, ParameterGridSet):
            raise TypeError('The param_grid_set property must be an instance of ParameterGrid or ParameterGridSet!')
        self._param_grid_set = obj

    @property
    def ndim(self):
        """(read-only) The dimensionality of the manifold."""
        return len(self._param_grid_set)

    @abc.abstractmethod
    def __call__(
        self,
        tdm: TrialDataManager,
        eventdata: np.ndarray,
        params_recarray: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieves the interpolated value of the manifold at the D-dimensional
        point ``params_recarray`` for all given events and sources, along with
        the D gradients, i.e. partial derivatives.

        Parameters
        ----------
        tdm
            The TrialDataManager instance holding the trial data.
        eventdata
            The 2D (V,N_events)-shaped numpy ndarray holding the event data,
            where N_events is the number of trial events, and V the
            dimensionality of the event data.
        params_recarray
            The structured numpy ndarray holding the N_sources set of parameter
            names and values, that define the point (for each source) on the
            manifold for which the value should get calculated for each event.
        **kwargs
            Additional keyword arguments required by the function ``func``.

        Returns
        -------
        values
            The (N,)-shaped numpy ndarray holding the interpolated manifold
            values for the given events and sources.
        grads
            The (D,N)-shaped numpy ndarray holding the D manifold gradients for
            the N given values, where D is the number of parameters.
            The order of the D parameters is defined by the ParameterGridSet
            that has been provided at construction time of this interpolation
            method object.
        """
        pass


class NullGridManifoldInterpolationMethod(
    GridManifoldInterpolationMethod,
):
    """This grid manifold interpolation method performs no interpolation. When
    the
    :meth:`~skyllh.core.interpolate.NullGridManifoldInterpolationMethod.__call__`
    method is called, it rounds the parameter values to their nearest grid
    point values. All gradients are set to zero.
    """

    def __init__(
        self,
        func: Callable,
        param_grid_set: ParameterGrid | ParameterGridSet,
        **kwargs,
    ):
        """Creates a new NullGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func
            The function that takes d parameter grid values as input and returns
            the value of the d-dimensional manifold at this point for each given
            trial event and source.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set
            The set of d parameter grids. This defines the grid of the
            manifold.
        """
        super().__init__(func=func, param_grid_set=param_grid_set, **kwargs)

    def __call__(
        self,
        tdm: TrialDataManager,
        eventdata: np.ndarray,
        params_recarray: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the non-interpolated manifold value and its gradient
        (zero) for each given event and source at the points given by
        ``params_recarray``.

        Parameters
        ----------
        tdm
            The TrialDataManager instance holding the trial data.
        eventdata
            The (V,N_events)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params_recarray
            The structured numpy ndarray of length N_sources holding the
            parameter names and values of the sources, defining the point on the
            manifold for which the values should get calculated.
        **kwargs
            Additional keyword arguments required by the function ``func``.

        Returns
        -------
        values
            The (N,)-shaped numpy ndarray holding the interpolated manifold
            values for the given events and sources.
        grads
            The (D,N)-shaped ndarray of float holding the D manifold gradients
            for the N values, where D is the number of parameters of the
            manifold.
            By definition, all gradients are zero.
        """
        # Round the given parameter values to their nearest grid values.
        gridparams_recarray_dtype = [(p_grid.name, np.float64) for p_grid in self._param_grid_set]

        gridparams_recarray = np.empty(params_recarray.shape, dtype=gridparams_recarray_dtype)

        for p_grid in self._param_grid_set:
            pname = p_grid.name
            pvalues = params_recarray[pname]
            gridparams_recarray[pname] = p_grid.round_to_nearest_grid_point(pvalues)

        values = cast(
            np.ndarray,
            self._func(
                tdm=tdm,
                eventdata=eventdata,
                gridparams_recarray=gridparams_recarray,
                n_values=tdm.get_n_values(),
                **kwargs,
            ),
        )

        grads = np.zeros((len(self.param_grid_set), len(values)), dtype=np.float64)

        return (values, grads)


class Linear1DGridManifoldInterpolationMethod(
    GridManifoldInterpolationMethod,
):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a line.
    """

    def __init__(
        self,
        func: Callable,
        param_grid_set: ParameterGrid | ParameterGridSet,
        **kwargs,
    ):
        """Creates a new Linear1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given source and trial event.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set
            The one parameter grid. This defines the grid of the manifold.
        """
        super().__init__(func=func, param_grid_set=param_grid_set, **kwargs)

        if len(self._param_grid_set) != 1:
            raise ValueError(
                f'The {classname(self)} class supports only 1D grid manifolds. '
                'The param_grid_set argument must contain 1 ParameterGrid '
                f'instance! Currently it has {len(self._param_grid_set)}!'
            )
        self._p_grid = self._param_grid_set[0]

        # Create a cache for the line parameterization for the last
        # manifold grid point for the different events.
        self._cache = self._create_cache(trial_data_state_id=None, x0=None, m=None, b=None)

    def _create_cache(
        self,
        trial_data_state_id: int | None,
        x0: np.ndarray | None,
        m: np.ndarray | None,
        b: np.ndarray | None,
    ):
        """Creates a cache for the line parameterization for the last manifold
        grid point for the N_events different events.

        Parameters
        ----------
        trial_data_state_id
            The trial data state id of the TrialDataManager.
        x0
            The (N_sources,)-shaped numpy ndarray holding the parameter grid
            value of the lower point of the grid manifold for each source used
            to estimate the line.
        m
            The (N_values,)-shaped numpy ndarray holding the slope of the line
            for each trial event and source.
        b
            The (N_values,)-shaped numpy ndarray holding the offset coefficient
            of the line for each trial event and source.
        """
        cache = {'trial_data_state_id': trial_data_state_id, 'x0': x0, 'm': m, 'b': b}

        return cache

    def _is_cached(
        self,
        trial_data_state_id,
        x0,
    ) -> bool:
        """Checks if the given line parametrization are already cached for the
        given x0 values.

        Returns
        -------
        check
            ``True`` if the line parametrization for x0 is already cached,
            ``False`` otherwise.
        """
        self__cache = self._cache
        return bool(
            (self__cache['trial_data_state_id'] is not None)
            and (self__cache['trial_data_state_id'] == trial_data_state_id)
            and (np.all(np.isclose(self__cache['x0'], x0)))
        )

    def __call__(
        self,
        tdm: TrialDataManager,
        eventdata: np.ndarray,
        params_recarray: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the interpolated manifold value and its gradient for each
        given source and trial event at the point ``params_recarray``.

        Parameters
        ----------
        tdm
            The TrialDataManager instance holding the trial data.
        eventdata
            The (V,N_events)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params_recarray
            The numpy record ndarray of length N_sources holding the parameter
            names and values for each source, defining the point on the manifold
            for which the value should get calculated.
            This record ndarray can be of length 1. In that case the single set
            of parameters is used for all sources.
        **kwargs
            Additional keyword arguments required by the function ``func``.

        Returns
        -------
        values
            The (N_values,)-shaped numpy ndarray of float holding the
            interpolated manifold values for all sources and trial events.
        grads
            The (D,N_values)-shaped numpy ndarray of float holding the D
            manifold gradients for the N_values values for all sources and trial
            events, where D is the number of interpolation parameters.
        """
        xname = self._p_grid.name

        x = params_recarray[xname]

        # Determine the nearest grid point that is lower than x and use that as
        # x0.
        x0 = self._p_grid.round_to_lower_grid_point(x)

        # Check if the line parametrization for x0 is already cached.
        if self._is_cached(tdm.trial_data_state_id, x0):
            m = self._cache['m']
            b = self._cache['b']

            (x,) = tdm.broadcast_sources_arrays_to_values_arrays((x,))

            values = m * x + b

            return (values, np.atleast_2d(m))

        # The line parametrization is not cached.
        # Calculate the line parametrization for all the given events.
        self__func = self._func

        x1 = self._p_grid.round_to_upper_grid_point(x)

        n_values = tdm.get_n_values()

        values = np.empty((n_values,), dtype=np.float64)
        m = np.empty((n_values,), dtype=np.float64)

        gridparams_recarray = np.array(x0, dtype=[(xname, np.float64)])
        M0: np.ndarray = cast(
            np.ndarray,
            self__func(
                tdm=tdm, eventdata=eventdata, gridparams_recarray=gridparams_recarray, n_values=n_values, **kwargs
            ),
        )

        gridparams_recarray = np.array(x1, dtype=[(xname, np.float64)])
        M1: np.ndarray = cast(
            np.ndarray,
            self__func(
                tdm=tdm, eventdata=eventdata, gridparams_recarray=gridparams_recarray, n_values=n_values, **kwargs
            ),
        )

        # Broadcast x0 and x1 to the values array.
        (x, v_x0, v_x1) = tdm.broadcast_sources_arrays_to_values_arrays((x, x0, x1))

        m = (M1 - M0) / (v_x1 - v_x0)
        b = M0 - m * v_x0

        # Cache the line parametrization.
        self._cache = self._create_cache(trial_data_state_id=tdm.trial_data_state_id, x0=x0, m=m, b=b)

        # Calculate the interpolated manifold values. The gradient is m.
        values = m * x + b

        return (values, np.atleast_2d(m))


class Parabola1DGridManifoldInterpolationMethod(
    GridManifoldInterpolationMethod,
):
    """This grid manifold interpolation method interpolates the 1-dimensional
    grid manifold using a parabola.
    """

    def __init__(
        self,
        func: Callable,
        param_grid_set: ParameterGrid | ParameterGridSet,
        **kwargs,
    ):
        """Creates a new Parabola1DGridManifoldInterpolationMethod instance.

        Parameters
        ----------
        func
            The function that takes the parameter grid value as input and
            returns the value of the 1-dimensional manifold at this point for
            each given source and trial event.
            See the documentation of the
            :class:`~skyllh.core.interpolate.GridManifoldInterpolationMethod`
            class for more details.
        param_grid_set
            The one parameter grid. This defines the grid of the manifold.
        """
        super().__init__(func=func, param_grid_set=param_grid_set, **kwargs)

        if len(self._param_grid_set) != 1:
            raise ValueError(
                f'The {classname(self)} class supports only 1D grid manifolds. '
                'The param_grid_set argument must contain 1 ParameterGrid '
                f'instance! Currently it has {len(self._param_grid_set)}!'
            )
        self._p_grid = self._param_grid_set[0]

        # Create a cache for the parabola parameterization for the last
        # manifold grid point for the different events.
        self._cache = self._create_cache(trial_data_state_id=None, x1=None, M1=None, a=None, b=None)

    def _create_cache(
        self,
        trial_data_state_id: int | None,
        x1: np.ndarray | None,
        M1: np.ndarray | None,
        a: np.ndarray | None,
        b: np.ndarray | None,
    ) -> dict:
        """Creates a cache for the parabola parameterization for the last
        manifold grid point for the N_events different events.

        Parameters
        ----------
        trial_data_state_id
            The trial data state ID of the TrialDataManager.
        x1
            The (N_sources,)-shaped numpy ndarray of float holding the parameter
            grid value for the middle point of the grid manifold for all sources
            used to estimate the parabola.
        M1
            The (N_values,)-shaped numpy ndarray of float holding the grid
            manifold value for each source and trial event of the middle point
            (x1,).
        a
            The (N_values,)-shaped numpy ndarray of float holding the parabola
            coefficient ``a`` for each source and trial event.
        b
            The (N_values,)-shaped numpy ndarray of float holding the parabola
            coefficient ``b`` for each source and trial event.

        Returns
        -------
        cache
            The dictionary holding the cache data.
        """
        cache = {
            'trial_data_state_id': trial_data_state_id,
            'x1': x1,
            'M1': M1,
            'a': a,
            'b': b,
        }

        return cache

    def _is_cached(
        self,
        trial_data_state_id,
        x1,
    ):
        """Checks if the parabola parametrization is already cached for the
        given x1 values.
        """
        self__cache = self._cache
        if (self__cache['trial_data_state_id'] is not None) and (
            self__cache['trial_data_state_id'] == trial_data_state_id
        ):
            return not np.any(np.not_equal(self__cache['x1'], x1))

        return False

    def __call__(
        self,
        tdm: TrialDataManager,
        eventdata: np.ndarray,
        params_recarray: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the interpolated manifold value and its gradient for each
        given source and trial event at the point ``params_recarray``.

        Parameters
        ----------
        tdm
            The TrialDataManager instance holding the trial data.
        eventdata
            The (V,N_events)-shaped numpy ndarray holding the event data,
            where N_events is the number of events, and V the dimensionality of
            the event data.
        params_recarray
            The numpy record ndarray of length N_sources holding the parameter
            names and values for each source, defining the point on the manifold
            for which the value should get calculated.
            This record ndarray can be of length 1. In that case the single set
            of parameters is used for all sources.
        **kwargs
            Additional keyword arguments required by the function ``func``.

        Returns
        -------
        values
            The interpolated manifold value for the N given events.
        grads
            The D manifold gradients for the N given events, where D is the
            number of parameters.
        """
        xname = self._p_grid.name

        x = params_recarray[xname]

        # Determine the nearest grid point x1.
        x1 = self._p_grid.round_to_nearest_grid_point(x)

        # Check if the parabola parametrization for x1 is already cached.
        if self._is_cached(tdm.trial_data_state_id, x1):
            M1 = self._cache['M1']
            a = self._cache['a']
            b = self._cache['b']
        else:
            dx = self._p_grid.delta

            # Calculate the neighboring grid points to x1: x0 and x2.
            x0 = self._p_grid.round_to_nearest_grid_point(x1 - dx)
            x2 = self._p_grid.round_to_nearest_grid_point(x1 + dx)

            # Parameterize the parabola with parameters a, b, and M1.
            self__func = self._func

            n_values = tdm.get_n_values()

            gridparams_recarray = np.array(x0, dtype=[(xname, np.float64)])
            M0: np.ndarray = cast(
                np.ndarray,
                self__func(
                    tdm=tdm, eventdata=eventdata, gridparams_recarray=gridparams_recarray, n_values=n_values, **kwargs
                ),
            )

            gridparams_recarray = np.array(x1, dtype=[(xname, np.float64)])
            M1: np.ndarray = cast(
                np.ndarray,
                self__func(
                    tdm=tdm, eventdata=eventdata, gridparams_recarray=gridparams_recarray, n_values=n_values, **kwargs
                ),
            )

            gridparams_recarray = np.array(x2, dtype=[(xname, np.float64)])
            M2: np.ndarray = cast(
                np.ndarray,
                self__func(
                    tdm=tdm, eventdata=eventdata, gridparams_recarray=gridparams_recarray, n_values=n_values, **kwargs
                ),
            )

            a = 0.5 * (M0 - 2.0 * M1 + M2) / dx**2
            b = 0.5 * (M2 - M0) / dx

            # Cache the parabola parametrization.
            self._cache = self._create_cache(trial_data_state_id=tdm.trial_data_state_id, x1=x1, M1=M1, a=a, b=b)

        # Broadcast x, x1, and (x-x1) to the values array.
        (x, x1, x_minus_x1) = tdm.broadcast_sources_arrays_to_values_arrays((x, x1, x - x1))

        # Calculate the interpolated manifold values.
        values = a * x_minus_x1**2 + b * x_minus_x1 + M1
        # Calculate the gradient of the manifold for all values.
        grads = 2.0 * a * x_minus_x1 + b

        return (values, np.atleast_2d(grads))
