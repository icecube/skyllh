# -*- coding: utf-8 -*-

"""This module contains utility functions related analysis trials.
"""

import pickle

from skyllh.core.timing import TaskTimer


def create_pseudo_data_file(
        ana,
        rss,
        filename,
        mean_n_bkg_list=None,
        mean_n_sig=0,
        bkg_kwargs=None,
        sig_kwargs=None,
        tl=None
    ):
    """Creates a pickle file that contains the pseudo data for a single trial.

    Parameters
    ----------
    ana : Analysis
        The Analysis instance that should be used to generate the pseudo data.
    rss : RandomStateService
        The RandomStateService instance to use for generating random numbers.
    filename : str
        The data file name into which the generated pseudo data should get
        written to.
    mean_n_bkg_list : list of float | None
        The mean number of background events that should be generated for
        each dataset. If set to None (the default), the background
        generation method needs to obtain this number itself.
    mean_n_sig : float
        The mean number of signal events that should be generated for the
        trial. The actual number of generated events will be drawn from a
        Poisson distribution with this given signal mean as mean.
    bkg_kwargs : dict | None
        Additional keyword arguments for the `generate_events` method of the
        background generation method class. An usual keyword argument is
        `poisson`.
    sig_kwargs : dict | None
        Additional keyword arguments for the `generate_signal_events` method
        of the `SignalGenerator` class. An usual keyword argument is
        `poisson`.
    tl : TimeLord | None
        The instance of TimeLord that should be used to time individual tasks.

    """
    with TaskTimer(tl, 'Generating pseudo data.'):
        (n_sig, n_events_list, events_list) = ana.generate_pseudo_data(
            rss = rss,
            mean_n_bkg_list = mean_n_bkg_list,
            mean_n_sig = mean_n_sig,
            bkg_kwargs = bkg_kwargs,
            sig_kwargs = sig_kwargs,
            tl = tl
        )

    trial_data = dict(
        mean_n_bkg_list = mean_n_bkg_list,
        mean_n_sig = mean_n_sig,
        bkg_kwargs = bkg_kwargs,
        sig_kwargs = sig_kwargs,
        n_sig = n_sig,
        n_events_list = n_events_list,
        events_list = events_list
    )

    with TaskTimer(tl, 'Writing pseudo data to file.'):
        with open(filename, 'wb') as fp:
            pickle.dump(trial_data, fp)


def load_pseudo_data(filename, tl=None):
    """Loads the pseudo data for a single trial from the given file name.

    Parameters
    ----------
    filename : str
        The name of the file that contains the pseudo data.
    tl : TimeLord | None
        The instance of TimeLord that should be used to time individual tasks.

    Returns
    -------
    mean_n_sig : float
        The mean number of signal events that was used to generate the pseudo
        data.
    n_sig : int
        The actual total number of signal events in the pseudo data.
    n_events_list : list of int
        The total number of events for each data set of the pseudo data.
    events_list : list of DataFieldRecordArray instances
        The list of DataFieldRecordArray instances containing the pseudo
        data events for each data sample. The number of events for each
        data sample can be less than the number of events given by
        `n_events_list` if an event selection method was already utilized
        when generating background events.
    """
    with TaskTimer(tl, 'Loading pseudo data from file.'):
        with open(filename, 'rb') as fp:
            trial_data = pickle.load(fp)

    return (
        trial_data['mean_n_sig'],
        trial_data['n_sig'],
        trial_data['n_events_list'],
        trial_data['events_list']
    )
