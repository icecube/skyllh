# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.py import (
    classname,
    int_cast,
)


class RandomStateService(
        object):
    """The RandomStateService class provides a container for a
    numpy.random.RandomState object, initialized with a given seed. This service
    can then be passed to any function or method within skyllh that requires a
    random number generator.
    """
    def __init__(
            self,
            seed=None,
            **kwargs,
    ):
        """Creates a new random state service. The ``random`` property can then
        be used to draw random numbers.

        Parameters
        ----------
        seed : int | None
            The seed to use. If None, the random number generator will be seeded
            randomly. See the numpy documentation for numpy.random.RandomState
            what that means.
        """
        super().__init__(**kwargs)

        self._seed = int_cast(
            seed,
            'The seed argument must be None, or cast-able to type int!',
            allow_None=True)
        self.random = np.random.RandomState(self._seed)

    @property
    def seed(self):
        """(read-only) The seed (int) of the random number generator.
        None, if not set. To change the seed, use the `reseed` method.
        """
        return self._seed

    @property
    def random(self):
        """The numpy.random.RandomState object.
        """
        return self._random

    @random.setter
    def random(self, random):
        if not isinstance(random, np.random.RandomState):
            raise TypeError(
                'The random property must be of type numpy.random.RandomState!')
        self._random = random

    def reseed(self, seed):
        """Reseeds the random number generator with the given seed.

        Parameters
        ----------
        seed : int | None
            The seed to use. If None, the random number generator will be seeded
            randomly. See the numpy documentation for numpy.random.RandomState
            what that means.
        """
        self._seed = int_cast(
            seed,
            'The seed argument must be None or cast-able to type int!',
            allow_None=True)
        self.random.seed(self._seed)


class RandomChoice(
        object,
):
    """This class provides an efficient numpy.random.choice functionality
    specialized for SkyLLH. The advantage is that it stores the cumulative
    distribution function (CDF), which is assumed to be constant.
    """

    def __init__(
            self,
            items,
            probabilities,
            **kwargs,
    ):
        """Creates a new instance of RandomChoice holding the probabilities
        and their cumulative distribution function (CDF).

        Parameters
        ----------
        items : instance of numpy.ndarray
            The (N,)-shaped numpy.ndarray holding the items from which to
            choose.
        probabilities : instance of numpy.ndarray
            The (N,)-shaped numpy.ndarray holding the probability for each item.
        """
        super().__init__(**kwargs)

        self._assert_items(items)
        self._items = items

        self._assert_probabilities(probabilities, self._items.size)
        self._probabilities = probabilities

        # Create the cumulative distribution function (CDF). We use float64 to
        # avoid a possible overflow when doing the summation.
        self._cdf = np.cumsum(self._probabilities, dtype=np.float64)
        self._cdf /= self._cdf[-1]

    @property
    def items(self):
        """(read-only) The (N,)-shaped numpy.ndarray holding the items from
        which to choose.
        """
        return self._items

    @property
    def probabilities(self):
        """(read-only) The (N,)-shaped numpy.ndarray holding the probability
        for each item.
        """
        return self._probabilities

    def _assert_items(
            self,
            items,
    ):
        """Checks for the correct type and shape of the items.

        Parameters
        ----------
        items : The (N,)-shaped numpy.ndarray holding the items from which to
            choose.

        Raises
        ------
        TypeError
            If the type of items is not numpy.ndarray.
        ValueError
            If the items do not have the correct type and shape.
        """
        if not isinstance(items, np.ndarray):
            raise TypeError(
                'The items must be an instance of numpy.ndarray! '
                f'Its current type is {classname(items)}!')

        if items.ndim != 1:
            raise ValueError(
                'The items must be a 1-dimensional numpy.ndarray!')

    def _assert_probabilities(
            self,
            p,
            n_items,
    ):
        """Checks for correct values of the probabilities.

        Parameters
        ----------
        p : instance of numpy.ndarray
            The (N,)-shaped numpy.ndarray holding the probability for each item.
        n_items : int
            The number of items.

        Raises
        ------
        ValueError
            If the probabilities do not have the correct type, shape and values.
        """
        atol = np.sqrt(np.finfo(np.float64).eps)
        atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

        if p.ndim != 1:
            raise ValueError(
                'The probabilities must be provided as a 1-dimensional '
                'numpy.ndarray!')

        if p.size != n_items:
            raise ValueError(
                f'The size ({p.size}) of the probabilities array must match '
                f'the number of items ({n_items})!')

        if np.any(p < 0):
            raise ValueError(
                'The probabilities must be greater or equal zero!')

        p_sum = np.sum(p)
        if abs(p_sum - 1.) > atol:
            raise ValueError(
                f'The sum of the probabilities ({p_sum}) must be 1!')

    def __call__(
            self,
            rss,
            size,
    ):
        """Chooses ``size`` random items from ``self.items`` according to
        ``self.probabilities``.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService from which random numbers are
            drawn from.
        size : int
            The number of items to draw.

        Returns
        -------
        random_items : instance of numpy.ndarray
            The (size,)-shaped numpy.ndarray holding the randomly selected items
            from ``self.items``.
        """
        uniform_values = rss.random.random(size)

        # The np.searchsorted function is much faster when the values are
        # sorted. But we want to keep the randomness of the returned items.
        idxs_of_sort = np.argsort(uniform_values)
        sorted_idxs = np.searchsorted(
            self._cdf,
            uniform_values[idxs_of_sort],
            side='right')
        idxs = np.empty_like(sorted_idxs)
        idxs[idxs_of_sort] = sorted_idxs
        random_items = self._items[idxs]

        return random_items
