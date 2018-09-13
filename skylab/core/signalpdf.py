# -*- coding: utf-8 -*-

"""The ``signalpdf`` module contains possible signal PDF models for the
likelihood function.
"""

import numpy as np

from skylab.core.py import issequenceof
from skylab.core.pdf import SpatialPDF, IsSignalPDF
from skylab.core.source_hypothesis import SourceHypoGroupManager
from skylab.physics.source import PointLikeSource

class GaussianPSFPointLikeSourceSignalSpatialPDF(SpatialPDF, IsSignalPDF):
    """This spatial signal PDF model describes the spatial PDF for a point
    source smeared with a 2D gaussian point-spread-function (PSF).
    Mathematically, it's the convolution of a point in the sky, i.e. the source
    location, with the PSF. The result of this convolution has the gaussian form

        1/(2*\pi*\sigma^2) * exp(-1/2*(r / \sigma)**2),

    where \sigma is the spatial uncertainty of the event and r the distance on
    the sphere between the source and the data event.
    """
    def __init__(self, src_hypo_group_manager):
        """Creates a new spatial signal PDF for point-like sources with a
        gaussian point-spread-function (PSF).

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance, that defines the sources, for
            which the spatial PDF values should get calculated for.
        """
        super(GaussianPSFPointLikeSourceSignalSpatialPDF, self).__init__(
            ra_range=(0, 2*np.pi),
            dec_range=(-np.pi/2, np.pi/2))

        if(not isinstance(src_hypo_group_manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager argument must be an instance of SourceHypoGroupManager!')

        # For the calculation ndarrays for the right-ascention and declination
        # of the different point-like sources is more efficient.
        self._src_arr = self.source_to_array(src_hypo_group_manager.source_list)

    def source_to_array(self, sources):
        """Converts the given sequence of PointLikeSource instances into a numpy
        record array holding the necessary source information for calculating
        the spatial signal PDF values.

        Parameters
        ----------
        sources : sequence of PointLikeSource instances
            The sequence of PointLikeSource instances for which the signal PDF
            values should get calculated for.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the necessary source
            information needed for this spatial signal PDF.
        """
        if(not issequenceof(sources, PointLikeSource)):
            raise TypeError('The sources argument must be a sequence of PointLikeSource instances!')

        arr = np.empty(
            (len(sources),),
            dtype=[('ra', np.float), ('dec', np.float)],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.ra
            arr['dec'][i] = src.dec

        return arr

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the SourceHypoGroupManager instance that defines the sources,
        for which the spatial signal PDF values should get calculated for.
        This recreates the internal source array.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance that should be used for this
            spatial signal PDF.
        """
        self._src_arr = self.source_to_array(src_hypo_group_manager.source_list)

    def get_prob(self, events, params=None):
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
        params : None
            Unused interface argument.

        Returns
        -------
        prob : (N_sources,N_events) shaped 2D ndarray
            The ndarray holding the spatial signal probability on the sphere for
            each source and event.
        """
        ra = events['ra']
        dec = events['dec']
        sigma = events['sigma']

        # Make the source position angles two-dimensional so the PDF value can
        # be calculated via numpy broadcasting automatically for several
        # sources. This is useful for stacking analyses.
        src_ra = self._src_arr['ra'][:,np.newaxis]
        src_dec = self._src_arr['dec'][:,np.newaxis]

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(src_dec) * np.sin(dec)

        # Handle possible floating precision errors.
        cos_r[cos_r < -1.] = -1.
        cos_r[cos_r > 1.] = 1.
        r = np.arccos(cos_r)

        prob = 0.5/(np.pi*sigma**2) * np.exp(-0.5*(r / sigma)**2)

        return prob
