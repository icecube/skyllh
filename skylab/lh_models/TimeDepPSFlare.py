from skylab.core.lh import LHModel

class TimeDepPSFlareLHModel(LHModel):
    """The time dependent point-source flare likelihood model searches for a
    point source with a steady flare at a given location in the sky.

    The test statistic for the time dependent
    hypothesis test defined as:

    TS = 2 log [ T_w/T * L(Phi_0,gamma,T_0,T_w)/L(0) ]

    where T_w is the best fit time window
    """
    def __init__(self):
        super(TimeDepPSFlareLHModel, self).__init__(self)
