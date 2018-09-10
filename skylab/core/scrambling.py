# -*- coding: utf-8 -*-

import abc

import numpy as np

from skylab.core.random import RandomStateService
from skylab.core.times import TimeGenerator

class DataScramblingMethod(object):
    """Base class (type) for implementing a data scrambling method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def scramble(self, data):
        """The scramble method implements the actual scrambling of the given
        data, which is method dependent. The scrambling must be performed
        in-place, i.e. it alters the data inside the given data array.

        Parameters
        ----------
        data : numpy.ndarray
            The ndarray containing the to be scrambled data.

        """
        pass

class UniformRAScramblingMethod(DataScramblingMethod):
    """The UniformRAScramblingMethod method performs right-ascention scrambling
    uniformly within a given RA range. By default it's (0, 2\pi).

    Note: This alters only the ``ra`` values of the data!
    """
    def __init__(self, rss, ra_range=None):
        """Initializes a new RAScramblingMethod instance.

        Parameters
        ----------
        rss : RandomStateService | None
            The random state service providing the random number generator (RNG).
        ra_range : tuple | None
            The two-element tuple holding the range in radians within the RA
            values should get drawn from. If set to None, the default (0, 2\pi)
            will be used.
        """
        super(UniformRAScramblingMethod, self).__init__()
        self.rss = rss
        self.ra_range = ra_range

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

    def scramble(self, data):
        data["ra"] = self.rss.random.uniform(*self.ra_range, size=data.size)

class TimeScramblingMethod(DataScramblingMethod):
    """The TimeScramblingMethod class provides a data scrambling method to
    perform data coordinate scrambling based on a generated time. It draws a
    random time from a time generator and transforms the horizontal (local)
    coordinates into equatorial coordinates using a specified transformation
    function.
    """
    def __init__(self, timegen, hor_to_equ_transform):
        """Initializes a new time scramling method instance.

        Parameters
        ----------
        timegen : TimeGenerator
            The time generator that should be used to generate random MJD times.
        hor_to_equ_transform : callable
            The transformation function to transform coordinates from the
            horizontal system into the equatorial system.

            The call signature must be:

                __call__(azi, zen, mjd)

            The return signature must be: (ra, dec)

        """
        super(TimeScramblingMethod, self).__init__()

        self.timegen = timegen
        self.hor_to_equ_transform = hor_to_equ_transform

    @property
    def timegen(self):
        """The TimeGenerator instance that should be used to generate random MJD
        times.
        """
        return self._timegen
    @timegen.setter
    def timegen(self, timegen):
        if(not isinstance(timegen, TimeGenerator)):
            raise TypeError('The timegen property must be an instance of TimeGenerator!')
        self._timegen = timegen

    @property
    def hor_to_equ_transform(self):
        """The transformation function to transform coordinates from the
        horizontal system into the equatorial system.
        """
        return self._hor_to_equ_transform
    @hor_to_equ_transform.setter
    def hor_to_equ_transform(self, transform):
        if(not callable(transform)):
            raise TypeError('The hor_to_equ_transform property must be a callable object!')
        self._hor_to_equ_transform = transform

    def scramble(self, data):
        mjds = self.timegen.generate_times(data.size)
        data['time'] = mjds
        (data['ra'], data['dec']) = self.hor_to_equ_transform(data['azi'], data['zen'], mjds)

class DataScrambler(object):
    def __init__(self, method, inplace_scrambling=True):
        """Creates a data scrambler instance with a given defined scrambling
        method.

        Parameters
        ----------
        method : DataScramblingMethod
            The instance of DataScramblingMethod that defines the method of
            the data scrambling.
        inplace_scrambling : bool
            Flag if the scrambler should perform an in-place scrambling of the
            data. If set to False, a copy of the given data is created before
            the scrambling is performed on the data, otherwise (the default)
            the scrambling is performed on the given data directly.
        """
        self.method = method
        self.inplace_scrambling = inplace_scrambling

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
    def inplace_scrambling(self):
        """Flag if the scrambler should perform an in-place scrambling of the
        data. If set to False, a copy of the given data is created before the
        scrambling is performed on the data, otherwise (the default) the
        scrambling is performed on the given data directly.
        """
        return self._inplace_scrambling
    @inplace_scrambling.setter
    def inplace_scrambling(self, flag):
        if(not isinstance(flag, bool)):
            raise TypeError('The inplace_scrambling property must be of type bool!')
        self._inplace_scrambling = flag

    def scramble_data(self, data):
        """Scrambles the given data by calling the scramble method of the
        scrambling method class, that was configured for the data scrambler.
        If the ``inplace_scrambling`` property is set to False, a copy of the
        data is created before the scrambling is performed.

        Parameters
        ----------
        data : numpy record array
            The numpy record array holding the data, which should get scrambled.

        Returns
        -------
        data : numpy record array
            The numpy record array with the scrambled data. If the
            ``inplace_scrambling`` property is set to True, this output array is
            the same array as the input array, otherwise it's a new array.
        """
        if(not self._inplace_scrambling):
            data = np.copy(data)

        self._method.scramble(data)

        return data
