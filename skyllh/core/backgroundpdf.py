# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains background PDF classes for the
likelihood function.
"""

import numpy as np

from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF,
    TimePDF,
)
from skyllh.core.py import (
    classname,
)


class BackgroundMultiDimGridPDF(
        MultiDimGridPDF,
        IsBackgroundPDF):
    """This class provides a multi-dimensional background PDF defined on a grid.
    The PDF is created from pre-calculated PDF data on a grid. The grid data is
    interpolated using a :class:`scipy.interpolate.RegularGridInterpolator`
    instance.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new :class:`~skyllh.core.pdf.MultiDimGridPDF` instance that
        is also derived from :class:`~skyllh.core.pdf.IsBackgroundPDF`.

        For the documentation of arguments see the documentation of the
        :meth:`~skyllh.core.pdf.MultiDimGridPDF.__init__` method.
        """
        super().__init__(*args, **kwargs)


class BackgroundTimePDF(
        TimePDF,
        IsBackgroundPDF):
    """This class provides a background time PDF class.
    """

    def __init__(
            self,
            livetime,
            time_flux_profile,
            **kwargs):
        """Creates a new signal time PDF instance for a given time flux profile
        and detector live time.

        Parameters
        ----------
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_flux_profile : instance of TimeFluxProfile
            The signal's time flux profile.
        """
        super().__init__(
            pmm=None,
            livetime=livetime,
            time_flux_profile=time_flux_profile,
            **kwargs)

        self._pd = None

    def initialize_for_new_trial(
            self,
            tdm,
            tl=None,
            **kwargs):
        """Initializes the background time PDF with new trial data. Because this
        PDF does not depend on any parameters, the probability density values
        can be pre-computed here.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            ``'time'`` : float
                The MJD time of the event.

        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.
        """
        times = tdm.get_data('time')

        self._pd = np.zeros((len(times),), dtype=np.float64)

        # Get a mask of the event times which fall inside a detector on-time
        # interval.
        on = self._livetime.is_on(times)

        self._pd[on] = self._time_flux_profile(t=times[on]) / self._S

    def get_pd(
            self,
            tdm,
            params_recarray=None,
            tl=None):
        """
        Parameters
        ----------
        tdm : instance of TrialDataManager
            The instance of TrialDataManager holding the trial event data for
            which to calculate the PDF value. The following data fields must
            exist:

            ``'time'`` : float
                The MJD time of the event.

        params_recarray : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used to measure
            timing information.

        Returns
        -------
        pd : instance of numpy ndarray
            The (N_events,)-shaped numpy ndarray holding the background
            probability density value for each event.
        grads : dict
            The dictionary holding the gradients of the probability density
            w.r.t. each global fit parameter.
            The background PDF does not depend on any global fit parameter,
            hence, this is an empty dictionary.
        """
        if self._pd is None:
            raise RuntimeError(
                f'The {classname(self)} was not initialized with trial data!')

        grads = dict()

        return (self._pd, grads)
