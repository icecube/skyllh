# -*- coding: utf-8 -*-

import abc


class BackgroundGenerationMethod(object):
    """This is the abstract base class for a detector specific background
    generation method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Constructs a new background generation method instance.
        """
        super(BackgroundGenerationMethod, self).__init__()

    @abc.abstractmethod
    def generate_events(self, rss, dataset, data, mean, **kwargs):
        """This method is supposed to generate a `mean` number of background
        events for the given dataset and its data.

        Parameters
        ----------
        rss : instance of RandomStateService
            The instance of RandomStateService that should be used to generate
            random numbers from.
        dataset : instance of Dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data : instance of DatasetData
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.
        mean : float
            The mean number of background events to generate.
        **kwargs
            Additional keyword arguments, which might be required for a
            particular background generation method.

        Returns
        -------
        bkg_events : numpy record array
            The numpy record arrays holding the generated background events.
        """
        pass
