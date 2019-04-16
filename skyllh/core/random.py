# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.py import int_cast

class RandomStateService(object):
    """The RandomStateService class provides a container for a
    numpy.random.RandomState object, initialized with a given seed. This service
    can then be passed to any function or method within skyllh that requires a
    random number generator.
    """
    def __init__(self, seed=None):
        """Creates a new random state service. The ``random`` property can then
        be used to draw random numbers.

        Parameters
        ----------
        seed : int | None
            The seed to use. If None, the random number generator will be seeded
            randomly. See the numpy documentation for numpy.random.RandomState
            what that means.
        """
        self._seed = int_cast(seed, 'The seed argument must be None, or '
            'castable to type int!', allow_None=True)
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
        if(not isinstance(random, np.random.RandomState)):
            raise TypeError('The random property must be of type numpy.random.RandomState!')
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
        self._seed = int_cast(seed, 'The seed argument must be None or '
            'castable to type int!', allow_None=True)
        self.random.seed(self._seed)
