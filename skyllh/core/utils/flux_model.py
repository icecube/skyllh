# -*- coding: utf-8 -*-

from scipy.stats import (
    rv_continuous,
)

from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.py import (
    classname,
)


def create_scipy_stats_rv_continuous_from_TimeFluxProfile(
        profile,
):
    """This function builds a scipy.stats.rv_continuous instance for a given
    :class:`~skyllh.core.flux_model.TimeFluxProfile` instance.

    It can be used to generate random numbers according to the given time flux
    profile function.

    Parameters
    ----------
    profile : instance of TimeFluxProfile
        The instance of TimeFluxProfile providing the function of the time flux
        profile.

    Returns
    -------
    rv : instance of rv_continuous_frozen
        The instance of rv_continuous_frozen representing the time flux profile
        as a continuous random variate instance.
    """
    if not isinstance(profile, TimeFluxProfile):
        raise TypeError(
            'The profile argument must be an instance of TimeFluxProfile! '
            f'Its current type is {classname(profile)}!')

    norm = 0
    tot_integral = profile.get_total_integral()
    if tot_integral != 0:
        norm = 1 / tot_integral

    class rv_continuous_from_TimeFluxProfile(
            rv_continuous):

        def __init__(self, *args, **kwargs):
            """Creates a new instance of the subclass of rv_continuous using
            the time flux profile.
            """
            self._profile = profile
            self._norm = norm

            super().__init__(*args, **kwargs)

        def _pdf(self, t):
            """Calculates the probability density of the time flux profile
            function for given time values.
            """
            pd = self._profile(t=t) * self._norm

            return pd

        def _cdf(self, t):
            """Calculates the cumulative distribution function values for rhe
            given time values. If the time flux profile instance provides a
            ``cdf`` method, it will be used. Otherwise the generic ``_cdf``
            method of the ``rv_continuous`` class will be used.
            """
            if hasattr(self._profile, 'cdf') and callable(self._profile.cdf):
                return self._profile.cdf(t=t)

            return super()._cdf(t)

    rv = rv_continuous_from_TimeFluxProfile(
        a=profile.t_start,
        b=profile.t_stop,
    ).freeze(loc=0, scale=1)

    return rv
