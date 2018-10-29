# -*- coding: utf-8 -*-

import numpy as np

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
        self.seed = seed
        self.random = np.random.RandomState(self.seed)

    @property
    def seed(self):
        """The seed (int) of the random number generator. None, if not set.
        """
        return self._seed
    @seed.setter
    def seed(self, seed):
        if(seed is not None):
            if(not isinstance(seed, int)):
                raise TypeError('The seed for the random number generator must be of type int!')
        self._seed = seed

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
        self.seed = seed
        self.random.seed(self.seed)
