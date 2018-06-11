# -*- coding: utf-8 -*-

import abc

import numpy as np

class DataScramblingMethod(object):
    """Base class (type) for implementing a data scrambling method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def scramble(self, random, data):
        """The scramble method implements the actual scrambling of the given
        data, which is method dependent. The scrambling must be performed
        in-place, i.e. it alters the data inside the given data array.

        Parameters
        ----------
        random : numpy.random.RandomState
            The numpy.random.RandomState instance, which must be used to get
            random numbers from.
        data : numpy.ndarray
            The ndarray containing the to be scrambled data.

        Returns
        -------
        scrambled_data : numpy.ndarray
            The ndarray with scrambled data.
        """
        pass

class RAScrambling(DataScramblingMethod):
    """The RAScrambling method performs right-ascention scrambling within a
    given RA range. By default it's (0, 2\pi).
    """
    def __init__(self, ra_range=None):
        """Initializes a new RAScrambling instance.

        Parameters
        ----------
        ra_range : tuple | None
            The two-element tuple holding the range in radians within the RA
            values should get drawn from. If set to None, the default (0, 2\pi)
            will be used.
        """
        super(RAScrambling, self).__init__()
        self.ra_range = ra_range

    @property
    def ra_range(self):
        """The two-element tuple holding the range within the RA values
        should get drawn from.
        """
        return self._ra_range
    @ra_range.setter
    def ra_range(self, ra_range):
        if(ra_range is None):
            ra_range = (0, 2*np.pi)
        if(not isinstance(ra_range, tuple)):
            raise TypeError('The ra_range property must be a tuple!')
        if(len(ra_range) != 2):
            raise ValueError('The ra_range tuple must contain 2 elements!')
        self._ra_range = ra_range

    def scramble(self, random, data):
        data["ra"] = random.uniform(*self.ra_range, size=data.size)


class DataScrambler(object):
    def __init__(self, method, seed=None):
        """Creates a data scrambler instance with a given defined scrambling
        method.

        Parameters
        ----------
        method : DataScramblingMethod
            The instance of DataScramblingMethod that defines the method of
            the data scrambling.
        seed : int | None
            The seed for the random number generator (RNG).
        """
        self.method = method
        self.seed = seed

    @property
    def method(self):
        """The underlaying scrambling method that should be used to scramble
        the data. This must be an instance of the DataScramblingMethod class.
        """
        return self._method
    @method.setter
    def method(self, method):
        if(not isinstance(method, DataScramblingMethod)):
            raise TypeError('The data scrambling method must be an instance of DataScramblingMethod!')
        self._method = method

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

    def scramble(self, data):
        """Sets the seed of the RNG and scrambles the given data by calling the
        scramble method of the scrambling method class, that was configured for
        the data scrambler.
        """
        random = np.random.RandomState(self.seed)
        self._method.scramble(random, data)
