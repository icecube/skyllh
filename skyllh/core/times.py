# -*- coding: utf-8 -*-

import abc

from skyllh.core.random import RandomStateService
from skyllh.core.livetime import Livetime

class TimeGenerationMethod(object):
    """Base class (type) for implementing a method to generate times.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_times(self, size):
        """The ``generate_times`` method implements the actual generation of
        times, which is method dependent.

        Parameters
        ----------
        size : int
            The number of times that should get generated.

        Returns
        -------
        times : ndarray
            The 1d numpy ndarray holding the generated times.
        """
        pass

class LivetimeTimeGenerationMethod(TimeGenerationMethod):
    """The LivetimeTimeGenerationMethod provides the method to generate times
    from a Livetime object. It will uniformely generate times that will coincide
    with the on-time intervals of the detector, by calling the ``draw_ontimes``
    method of the Livetime class.
    """
    def __init__(self, rss, livetime):
        """Creates a new LivetimeTimeGeneration instance.

        Parameters
        ----------
        rss : RandomStateService | None
            The random state service providing the random number generator (RNG).
        livetime : Livetime
            The Livetime instance that should be used to generate times from.
        """
        self.rss = rss
        self.livetime = livetime

    @property
    def rss(self):
        """The RandomStateService object providing the random number generator.
        """
        return self._rss
    @rss.setter
    def rss(self, rss):
        if(not isinstance(rss, RandomStateService)):
            raise TypeError('The random state service (rss) must be an instance of RandomStateService!')
        self._rss = rss

    @property
    def livetime(self):
        """The Livetime instance used to draw times from.
        """
        return self._livetime
    @livetime.setter
    def livetime(self, livetime):
        if(not isinstance(livetime, Livetime)):
            raise TypeError('The livetime property must be an instance of Livetime!')
        self._livetime = livetime

    def generate_times(self, size):
        return self.livetime.draw_ontimes(self.rss, size)

class TimeGenerator(object):
    def __init__(self, method):
        """Creates a time generator instance with a given defined time
        generation method.

        Parameters
        ----------
        method : TimeGenerationMethod
            The instance of TimeGenerationMethod that defines the method of
            generating times.
        """
        self.method = method

    @property
    def method(self):
        """The TimeGenerationMethod object that should be used to generate
        the times.
        """
        return self._method
    @method.setter
    def method(self, method):
        if(not isinstance(method, TimeGenerationMethod)):
            raise TypeError('The time generation method must be an instance of TimeGenerationMethod!')
        self._method = method

    def generate_times(self, size):
        """Generates ``size`` amount of times by calling the ``generate_times``
        method of the TimeGenerationMethod class.

        Parameters
        ----------
        size : int
            The number of time that should get generated.

        Returns
        -------
        times : ndarray
            The 1d ndarray holding the generated times.
        """
        return self._method.generate_times(size)
