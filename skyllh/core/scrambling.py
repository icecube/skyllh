# -*- coding: utf-8 -*-

import abc

import numpy as np

from skyllh.core.times import (
    TimeGenerator,
)


class DataScramblingMethod(
        object,
        metaclass=abc.ABCMeta,
):
    """Base class for implementing a data scrambling method.
    """

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs)

    @abc.abstractmethod
    def scramble(
            self,
            rss,
            dataset,
            data,
    ):
        """The scramble method implements the actual scrambling of the given
        data, which is method dependent. The scrambling must be performed
        in-place, i.e. it alters the data inside the given data array.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number
            generator (RNG).
        dataset : instance of Dataset
            The instance of Dataset for which the data should get scrambled.
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray containing the to be scrambled data.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The given DataFieldRecordArray holding the scrambled data.
        """
        pass


class UniformRAScramblingMethod(
        DataScramblingMethod,
):
    r"""The UniformRAScramblingMethod method performs right-ascention scrambling
    uniformly within a given RA range. By default it's (0, 2\pi).

    :note::

        This alters only the ``ra`` values of the data!

    """
    def __init__(
            self,
            ra_range=None,
            **kwargs,
    ):
        r"""Initializes a new RAScramblingMethod instance.

        Parameters
        ----------
        ra_range : tuple | None
            The two-element tuple holding the range in radians within the RA
            values should get drawn from. If set to None, the default (0, 2\pi)
            will be used.
        """
        super().__init__(
            **kwargs)

        self.ra_range = ra_range

    @property
    def ra_range(self):
        """The two-element tuple holding the range within the RA values
        should get drawn from.
        """
        return self._ra_range

    @ra_range.setter
    def ra_range(self, ra_range):
        if ra_range is None:
            ra_range = (0, 2*np.pi)
        if not isinstance(ra_range, tuple):
            raise TypeError(
                'The ra_range property must be a tuple!')
        if len(ra_range) != 2:
            raise ValueError(
                'The ra_range tuple must contain 2 elements!')
        self._ra_range = ra_range

    def scramble(
            self,
            rss,
            dataset,
            data,
    ):
        """Scrambles the given data uniformly in right-ascention.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number
            generator (RNG).
        dataset : instance of Dataset
            The instance of Dataset for which the data should get scrambled.
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray instance containing the to be scrambled
            data.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The given DataFieldRecordArray holding the scrambled data.
        """
        dt = data['ra'].dtype

        data['ra'] = rss.random.uniform(
            *self.ra_range, size=len(data)).astype(dt, copy=False)

        return data


class TimeScramblingMethod(
        DataScramblingMethod):
    """The TimeScramblingMethod class provides a data scrambling method to
    perform data coordinate scrambling based on a generated time. It draws a
    random time from a time generator and transforms the horizontal (local)
    coordinates into equatorial coordinates using a specified transformation
    function.
    """
    def __init__(
            self,
            timegen,
            hor_to_equ_transform,
            **kwargs,
    ):
        """Initializes a new time scramling method instance.

        Parameters
        ----------
        timegen : instance of TimeGenerator
            The time generator that should be used to generate random MJD times.
        hor_to_equ_transform : callable
            The transformation function to transform coordinates from the
            horizontal system into the equatorial system.

            The call signature must be:

                __call__(azi, zen, mjd)

            The return signature must be: (ra, dec)

        """
        super().__init__(
            **kwargs)

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
        if not isinstance(timegen, TimeGenerator):
            raise TypeError(
                'The timegen property must be an instance of TimeGenerator!')
        self._timegen = timegen

    @property
    def hor_to_equ_transform(self):
        """The transformation function to transform coordinates from the
        horizontal system into the equatorial system.
        """
        return self._hor_to_equ_transform

    @hor_to_equ_transform.setter
    def hor_to_equ_transform(self, transform):
        if not callable(transform):
            raise TypeError(
                'The hor_to_equ_transform property must be a callable object!')
        self._hor_to_equ_transform = transform

    def scramble(
            self,
            rss,
            dataset,
            data,
    ):
        """Scrambles the given data based on random MJD times, which are
        generated from a TimeGenerator instance. The event's right-ascention and
        declination coordinates are calculated via a horizontal-to-equatorial
        coordinate transformation and the generated MJD time of the event.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number
            generator (RNG).
        dataset : instance of Dataset
            The instance of Dataset for which the data should get scrambled.
        data : instance of DataFieldRecordArray
            The DataFieldRecordArray instance containing the to be scrambled
            data.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The given DataFieldRecordArray holding the scrambled data.
        """
        mjds = self.timegen.generate_times(rss, len(data))

        data['time'] = mjds

        (data['ra'], data['dec']) = self.hor_to_equ_transform(
            data['azi'], data['zen'], mjds)

        return data


class DataScrambler(
        object,
):
    def __init__(
            self,
            method,
            **kwargs,
    ):
        """Creates a data scrambler instance with a given defined scrambling
        method.

        Parameters
        ----------
        method : instance of DataScramblingMethod
            The instance of DataScramblingMethod that defines the method of
            the data scrambling.
        """
        super().__init__(
            **kwargs)

        self.method = method

    @property
    def method(self):
        """The underlaying scrambling method that should be used to scramble
        the data. This must be an instance of the DataScramblingMethod class.
        """
        return self._method

    @method.setter
    def method(self, method):
        if not isinstance(method, DataScramblingMethod):
            raise TypeError(
                'The data scrambling method must be an instance of '
                'DataScramblingMethod!')
        self._method = method

    def scramble_data(
            self,
            rss,
            dataset,
            data,
            copy=False,
    ):
        """Scrambles the given data by calling the scramble method of the
        scrambling method class, that was configured for the data scrambler.
        If the ``inplace_scrambling`` property is set to False, a copy of the
        data is created before the scrambling is performed.

        Parameters
        ----------
        rss : instance of RandomStateService
            The random state service providing the random number generator
            (RNG).
        dataset : instance of Dataset
            The instance of Dataset for which the data should get scrambled.
        data : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the data, which should
            get scrambled.
        copy : bool
            Flag if a copy of the given data should be made before scrambling
            the data. The default is False.

        Returns
        -------
        data : instance of DataFieldRecordArray
            The given DataFieldRecordArray instance with the scrambled data.
            If the ``inplace_scrambling`` property is set to True, this output
            array is the same array as the input array, otherwise it's a new
            array.
        """
        if copy:
            data = data.copy()

        data = self._method.scramble(
            rss=rss,
            dataset=dataset,
            data=data)

        return data
