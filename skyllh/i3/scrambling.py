# -*- coding: utf-8 -*-

from skylab.core.scrambling import TimeScramblingMethod
from skylab.i3.coords import hor_to_equ_transform, azi_to_ra_transform

class I3TimeScramblingMethod(TimeScramblingMethod):
    """The I3TimeScramblingMethod class provides a data scrambling method to
    perform time scrambling of the data,
    by drawing a MJD time from a given time generator.
    """
    def __init__(self, timegen):
        """Initializes a new I3 time scrambling instance.

        Parameters
        ----------
        timegen : TimeGenerator
            The time generator that should be used to generate random MJD times.
        """
        super(I3TimeScramblingMethod, self).__init__(timegen, hor_to_equ_transform)

    # We override the scramble method because for IceCube we only need to change
    # the ``ra`` field.
    def scramble(self, data):
        """Draws a time from the time generator and calculates the right
        ascention coordinate from the azimuth angle according to the time.
        Sets the values of the ``time`` and ``ra`` keys of data.
        """
        mjds = self.timegen.generate_times(data.size)
        data['time'] = mjds
        data['ra'] = azi_to_ra_transform(data['azi'], mjds)

