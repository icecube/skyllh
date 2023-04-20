# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains background PDF classes for the
likelihood function.
"""

from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF,
    NDPhotosplinePDF,
    TimePDF,
)

import numpy as np


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
            pmm,
            livetime,
            time_flux_profile,
            **kwargs):
        """Creates a new signal time PDF instance for a given time flux profile
        and detector live time.

        Parameters
        ----------
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper defining the mapping of the
            global parameters to the local source parameters.
        livetime : instance of Livetime
            An instance of Livetime, which provides the detector live-time
            information.
        time_flux_profile : instance of TimeFluxProfile
            The signal's time flux profile.
        """
        super().__init__(
            pmm=pmm,
            livetime=livetime,
            time_flux_profile=time_flux_profile,
            **kwargs)

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
        times = tdm.get_data('time')

        pd = np.zeros((len(times),), dtype=np.float64)

        # Get a mask of the event times which fall inside a detector on-time
        # interval.
        on = self._livetime.is_on(times)

        norm = 1 / (self._I * self._S)

        pd[on] = self._time_flux_profile(t=times[on]) * norm

        return (pd, dict())


class BackgroundNDPhotosplinePDF(
        NDPhotosplinePDF,
        IsBackgroundPDF):
    """DEPRECATED This class provides a multi-dimensional background PDF created
    from a n-dimensional photospline fit. The photospline package is used to
    evaluate the PDF fit.
    """

    def __init__(
            self,
            *args,
            **kwargs):
        """Creates a new :class:`~skyllh.core.pdf.NDPhotosplinePDF` instance
        that is also derived from :class:`~skyllh.core.pdf.IsBackgroundPDF`.

        For the documentation of arguments see the documentation of the
        :meth:`~skyllh.core.pdf.NDPhotosplinePDF.__init__` method.
        """
        super().__init__(*args, **kwargs)


class BackgroundUniformTimePDF(TimePDF, IsBackgroundPDF):

    def __init__(self, grl):
        """Creates a new background time PDF instance as uniform background

        Parameters
        ----------
        grl : ndarray
            Array of the detector good run list

        """
        super(BackgroundUniformTimePDF, self).__init__()
        self.start = grl["start"][0]
        self.end = grl["stop"][-1]
        self.grl = grl

    def cdf(self, t):
        """Compute the cumulative density function for the box pdf. This is
        needed for normalization.

        Parameters
        ----------
        t : float, ndarray
            MJD times

        Returns
        -------
        cdf : float, ndarray
            Values of cumulative density function evaluated at t
        """
        t_start = self.grl["start"][0]
        t_end = self.grl["stop"][-1]
        t = np.atleast_1d(t)

        cdf = np.zeros(t.size, float)

        # values between start and stop times
        mask = (t_start <= t) & (t <= t_end)
        cdf[mask] = (t[mask] - t_start) / [t_end - t_start]

        # take care of values beyond stop time in sample

        return cdf

    def norm_uptime(self):
        """Compute the normalization with the dataset uptime. Distributions like
        scipy.stats.norm are normalized (-inf, inf).
        These must be re-normalized such that the function sums to 1 over the
        finite good run list domain.

        Returns
        -------
        norm : float
            Normalization such that cdf sums to 1 over good run list domain
        """

        integral = (self.cdf(self.grl["stop"]) - self.cdf(self.grl["start"])).sum()

        if np.isclose(integral, 0):
            return 0

        return 1. / integral

    def get_prob(self, tdm, fitparams=None, tl=None):
        """Calculates the background time probability density of each event.

        tdm : TrialDataManager
            Unused interface argument.
        fitparams : None
            Unused interface argument.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        pd : array of float
            The (N,)-shaped ndarray holding the probability density for each event.
        grads : empty array of float
            Does not depend on fit parameter, so no gradient.
        """
        livetime = self.grl["stop"][-1] - self.grl["start"][0]
        pd = 1./livetime
        grads = np.array([], dtype=np.double)

        return (pd, grads)
