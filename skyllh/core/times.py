# -*- coding: utf-8 -*-

import abc

from skyllh.core.livetime import Livetime


class TimeGenerationMethod(
        object,
        metaclass=abc.ABCMeta,
):
    """Base class (type) for implementing a method to generate times.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate_times(
            self,
            rss,
            size,
    ):
        """The ``generate_times`` method implements the actual generation of
        times, which is method dependent.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number
            generator (RNG).
        size : int
            The number of times that should get generated.

        Returns
        -------
        times : ndarray
            The 1d numpy ndarray holding the generated times.
        """
        pass


class LivetimeTimeGenerationMethod(
        TimeGenerationMethod,
):
    """The LivetimeTimeGenerationMethod provides the method to generate times
    from a Livetime object. It will uniformely generate times that will coincide
    with the on-time intervals of the detector, by calling the `draw_ontimes`
    method of the Livetime class.
    """
    def __init__(self, livetime, **kwargs):
        """Creates a new LivetimeTimeGeneration instance.

        Parameters
        ----------
        livetime : Livetime
            The Livetime instance that should be used to generate times from.
        """
        super().__init__(**kwargs)

        self.livetime = livetime

    @property
    def livetime(self):
        """The Livetime instance used to draw times from.
        """
        return self._livetime

    @livetime.setter
    def livetime(self, livetime):
        if not isinstance(livetime, Livetime):
            raise TypeError(
                'The livetime property must be an instance of Livetime!')
        self._livetime = livetime

    def generate_times(
            self,
            rss,
            size,
            **kwargs,
    ):
        """Generates `size` MJD times according to the detector on-times
        provided by the Livetime instance.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number
            generator (RNG).
        size : int
            The number of times that should get generated.

        Returns
        -------
        times : ndarray
            The 1d (`size`,)-shaped numpy ndarray holding the generated times.
        """
        times = self._livetime.draw_ontimes(
            rss=rss,
            size=size,
            **kwargs)

        return times


class TimeGenerator(
        object):
    def __init__(self, method):
        """Creates a time generator instance with a given defined time
        generation method.

        Parameters
        ----------
        method : instance of TimeGenerationMethod
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
        if not isinstance(method, TimeGenerationMethod):
            raise TypeError(
                'The time generation method must be an instance of '
                'TimeGenerationMethod!')
        self._method = method

    def generate_times(
            self,
            rss,
            size,
            **kwargs,
    ):
        """Generates ``size`` amount of times by calling the ``generate_times``
        method of the TimeGenerationMethod class.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number generator
            (RNG).
        size : int
            The number of time that should get generated.
        **kwargs
            Additional keyword arguments are passed to the ``generate_times``
            method of the TimeGenerationMethod class.

        Returns
        -------
        times : ndarray
            The 1d (``size``,)-shaped ndarray holding the generated times.
        """
        times = self._method.generate_times(
            rss=rss,
            size=size,
            **kwargs)

        return times
