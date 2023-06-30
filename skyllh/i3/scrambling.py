# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.scrambling import (
    DataScramblingMethod,
    TimeScramblingMethod,
)

from skyllh.i3.utils.coords import (
    azi_to_ra_transform,
    hor_to_equ_transform,
)


class I3TimeScramblingMethod(
        TimeScramblingMethod,
):
    """The I3TimeScramblingMethod class provides a data scrambling method to
    perform time scrambling of the data,
    by drawing a MJD time from a given time generator.
    """
    def __init__(
            self,
            timegen,
            **kwargs,
    ):
        """Initializes a new I3 time scrambling instance.

        Parameters
        ----------
        timegen : TimeGenerator
            The time generator that should be used to generate random MJD times.
        """
        super().__init__(
            timegen=timegen,
            hor_to_equ_transform=hor_to_equ_transform,
            **kwargs)

    # We override the scramble method because for IceCube we only need to change
    # the ``ra`` field.
    def scramble(
            self,
            rss,
            dataset,
            data,
    ):
        """Draws a time from the time generator and calculates the right
        ascention coordinate from the azimuth angle according to the time.
        Sets the values of the ``time`` and ``ra`` keys of data.

        Parameters
        ----------
        rss : RandomStateService
            The random state service providing the random number
            generator (RNG).
        dataset : instance of Dataset
            The instance of Dataset for which the data should get scrambled.
        data : DataFieldRecordArray instance
            The DataFieldRecordArray instance containing the to be scrambled
            data.

        Returns
        -------
        data : numpy record ndarray
            The given numpy record ndarray holding the scrambled data.
        """
        mjds = self._timegen.generate_times(rss, len(data))

        data['time'] = mjds
        data['ra'] = azi_to_ra_transform(data['azi'], mjds)

        return data


class I3SeasonalVariationTimeScramblingMethod(
        DataScramblingMethod,
):
    """The I3SeasonalVariationTimeScramblingMethod class provides a data
    scrambling method to perform data coordinate scrambling based on a generated
    time, which follows seasonal variations within the experimental data.
    """
    def __init__(
            self,
            data,
            **kwargs,
    ):
        """Initializes a new seasonal time scrambling instance.

        Parameters
        ----------
        data : instance of I3DatasetData
            The instance of I3DatasetData holding the experimental data and
            good-run-list information.
        """
        super().__init__(**kwargs)

        # The run weights are the number of events in each run relative to all
        # the events to account for possible seasonal variations.
        self.run_weights = np.zeros((len(data.grl),), dtype=np.float64)
        n_events = len(data.exp['time'])
        for (i, (start, stop)) in enumerate(
                zip(data.grl['start'], data.grl['stop'])):
            mask = (data.exp['time'] >= start) & (data.exp['time'] < stop)
            self.run_weights[i] = len(data.exp[mask]) / n_events
        self.run_weights /= np.sum(self.run_weights)

        self.grl = data.grl

    def scramble(
            self,
            rss,
            dataset,
            data,
    ):
        """Scrambles the given data based on random MJD times, which are
        generated uniformely within the data runs, where the data runs are
        weighted based on their amount of events compared to the total events.

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
        # Get run indices based on their seasonal weights.
        run_idxs = rss.random.choice(
            self.grl['start'].size,
            size=len(data['time']),
            p=self.run_weights)

        # Draw random times uniformely within the runs.
        times = rss.random.uniform(
            self.grl['start'][run_idxs],
            self.grl['stop'][run_idxs])

        # Get the correct right ascension.
        data['time'] = times
        data['ra'] = azi_to_ra_transform(
            azi=data['azi'],
            mjd=times)

        return data
