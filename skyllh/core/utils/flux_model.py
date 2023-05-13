# -*- coding: utf-8 -*-

from scipy.stats import (
    rv_continuous,
)

from skyllh.core.py import (
    classname,
)
from skyllh.core.random import (
    RandomStateService,
)

from skyllh.physics.flux_model import (
    TimeFluxProfile,
)


def create_scipy_stats_rv_continuous_from_TimeFluxProfile(
        rss,
        profile
):
    """This function builds a scipy.stats.rv_continuous instance for a given
    :class:`~skyllh.physics.flux_model.TimeFluxProfile` instance.

    It can be used to generate random numbers according to the given time flux
    profile function.

    Parameters
    ----------
    rss : instance of RandomStateService
        The instance of RandomStateService which should be used to draw random
        numbers from.
    profile : instance of TimeFluxProfile
        The instance of TimeFluxProfile providing the function of the time flux
        profile.

    Returns
    -------
    rv : instance of rv_continuous_frozen
        The instance of rv_continuous_frozen representing the time flux profile
        as a continuous random variate instance.
    """
    if not isinstance(rss, RandomStateService):
        raise TypeError(
            'The rss argument must be an instance of RandomStateService! '
            f'Its current type is {classname(rss)}!')

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

    rv = rv_continuous_from_TimeFluxProfile(
        a=profile.t_start,
        b=profile.t_stop,
        seed=rss.random,
    ).freeze(loc=0, scale=1)

    return rv
