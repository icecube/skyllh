# -*- coding: utf-8 -*-

from skyllh.core.background_generation import BackgroundGenerationMethod
from skyllh.core.scrambling import DataScrambler


class FixedScrambledExpDataI3BkgGenMethod(BackgroundGenerationMethod):
    """This class implements the background event generation method for the
    IceCube detector using scrambled experimental data as background hypothesis
    with a fixed number of background events equal to the number of events in
    the dataset. This background generation method is the one used in SkyLab.
    """
    def __init__(self, data_scrambler):
        """Creates a new background generation method instance to generate
        background events from scrambled experimental data with a fixed number
        of events equal to the number of events in the dataset.

        Parameters
        ----------
        data_scrambler : instance of DataScrambler
            The DataScrambler instance to use to generate scrambled experimental
            data.
        """
        super(FixedScrambledExpDataI3BkgGenMethod, self).__init__()

        self.data_scrambler = data_scrambler

    @property
    def data_scrambler(self):
        """The DataScrambler instance that implements the data scrambling.
        """
        return self._data_scrambler
    @data_scrambler.setter
    def data_scrambler(self, scrambler):
        if(not isinstance(scrambler, DataScrambler)):
            raise TypeError('The data_scrambler property must be an instance '
                'of DataScrambler!')
        self._data_scrambler = scrambler

    def generate_events(self, rss, dataset, data, mean):
        """Generates background events from the given data, by scrambling the
        data. The number of events is equal to the size of the given dataset.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from. It is used to scramble the experimental data.
        dataset : instance of Dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data : instance of DatasetData
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.
        mean : float
            Ignored argument. The number of generated events corresponds to the
            number of events in the dataset.

        Returns
        -------
        bkg_events : numpy record array
            The numpy record arrays holding the generated background events.
        """
        bkg_events = self._data_scrambler.scramble_data(rss, data.exp)

        return bkg_events
