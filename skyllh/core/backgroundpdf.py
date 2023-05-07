# -*- coding: utf-8 -*-

"""The ``backgroundpdf`` module contains background PDF classes for the
likelihood function.
"""

import numpy as np

from skyllh.core.pdf import (
    IsBackgroundPDF,
    MultiDimGridPDF,
    NDPhotosplinePDF,
    TimePDF,
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

        norm = self._S / self._I**2

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
