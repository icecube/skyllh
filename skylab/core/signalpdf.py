# -*- coding: utf-8 -*-

"""The ``signalpdf`` module contains possible signal PDF models for the
likelihood function.
The base class of all signal pdf models is ``SignalPDF``.
"""

import numpy as np

from skylab.core.pdf import SpatialPDF

class SpatialSignalPDF(SpatialPDF):
    """This is the base class for all spatial signal PDF models.
    """
    def __init__(self):
        super(SpatialSignalPDF, self).__init__()

class GaussianSpatialSignalPDF(SpatialSignalPDF):
    """The gaussian spatial signal PDF models the spatial signal contribution of
    an event as a gaussian distribution of the form

        1/(2*\pi*\sigma^2) * exp(-1/2*(r / \sigma)**2),

    where \sigma is the spatial uncertainty of the event and r the distance on
    the sphere between the source and the data event.

    This spatial signal PDF model is only meaning full for a point-like source
    model, i.e. a point-source.
    """
    def __init__(self):
        super(GaussianSpatialSignalPDF, self).__init__()

    def assert_is_valid_for_exp_data(self, data_exp):
        """Checks if this PDF is valid for all the given experimental data.
        Since this PDF is a function which is defined everywhere, it just
        returns True.
        """
        return True

    def get_prob(self, events, params):
        """Calculates the spatial signal probability of each event for all given
        sources.

        Parameters
        ----------
        events : numpy record ndarray
            The numpy record array holding the event data. The following data
            fields need to be present:

            'ra' : float
                The right-ascention in radian of the data event.
            'dec' : float
                The declination in radian of the data event.
            'sigma': float
                The reconstruction uncertainty in radian of the data event.
        params : dict
            The dictionary holding the spatial source parameters, i.e. the
            source position(s). It must contain the following keys:

            'src_ra' : 1d ndarray
                The right-ascention values in radian of the source positions.
            'src_dec': 1d ndarray
                The declination values in radian of the source positions.

        Returns
        -------
        prob : (N_events,N_sources) shaped 2d ndarray
            The ndarray holding the spatial signal probability on the sphere for
            each event and source.
        """
        src_ra = params['src_ra']
        src_dec = params['src_dec']
        ra = events['ra']
        dec = events['dec']
        sigma = events['sigma']

        # Make the source position angles two-dimensional so the PDF value can
        # be calculated via numpy broadcasting automatically for several
        # sources. This is useful for stacking analyses.
        src_ra = src_ra[:,np.newaxis]
        src_dec = src_dec[:,np.newaxis]

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(src_dec) * np.sin(dec)

        # Handle possible floating precision errors.
        cos_r[cos_r < -1.] = -1.
        cos_r[cos_r > 1.] = 1.
        r = np.arccos(cos_r)

        prob = 0.5/(np.pi*sigma**2) * np.exp(-0.5*(dist / sigma)**2)

        return prob
