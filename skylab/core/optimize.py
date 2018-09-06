# -*- coding: utf-8 -*-

import abc

import numpy as np

class EventSelectionMethod(object):
    """This is the abstract base class for all event selection method classes.
    The idea is to pre-select only events that contribute to the likelihood
    function, i.e. are more signal than background like. The different methods
    are implemented through derived classes of this base class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(EventSelectionMethod, self).__init__()

    @abc.abstractmethod
    def select_events(self, events):
        """This method selects the events, which will contribute to the
        log-likelihood ratio function.

        Parameters
        ----------
        events : numpy record array
            The set of all events.

        Returns
        -------
        selected_events : numpy record array
            The set of selected events, i.e. a subset of the events parameter.
        """
        pass


class AllEventSelectionMethod(EventSelectionMethod):
    """This event selection method selects all events.
    """
    def __init__(self):
        super(AllEventSelectionMethod, self).__init__()

    def select_events(self, events):
        return events


class SpatialBoxEventSelectionMethod(EventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of sources positions.
    """
    def __init__(self, src_ra, src_dec, delta_angle):
        """Creates and configures a spatial box event selection method object.

        Parameters
        ----------
        src_ra : 1D ndarray
            The ndarray holding the right-ascention coordinate of all the
            sources.
        src_dec : 1D ndarray
            The ndarray holding the declination coordinate of all the sources.
        delta_angle : float
            The half-opening angle around the source for which events should
            get selected.
        """
        super(SpatialBoxEventSelectionMethod, self).__init__()

        self.src_ra = src_ra
        self.src_dec = src_dec
        self.delta_angle = delta_angle

    @property
    def src_ra(self):
        """The ndarray holding the right-ascention coordinate of all the
        sources.
        """
        return self._src_ra
    @src_ra.setter
    def src_ra(self, arr):
        arr = np.atleast_1d(arr)
        self._src_ra = arr

    @property
    def src_dec(self):
        """The ndarray holding the declination coordinate of all the sources.
        """
        return self._src_dec
    @src_dec.setter
    def src_dec(self, arr):
        arr = np.atleast_1d(arr)
        self._src_dec = arr

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle
    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(angle, 'The delta_angle property must be castable to type float!')
        self._delta_angle = angle

    def select_events(self, events):
        """Selects the events within the spatial box in right-ascention and
        declination.

        The solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec) is a
        function of declination, i.e. for a constant dOmega, the right-ascension
        value has to change with declination.

        Parameters
        ----------
        events : numpy record array
            The numpy record array with the event data.
            The following data fields must exist:
                ra : The right-ascention of the event.
                dec : The declination of the event.

        Returns
        -------
        selected_events : numpy record array
            The numpy record array holding only the selected events.
        """

        # Calculate the minus and plus declination around the source and bound
        # it to -90deg and +90deg, respectively.
        src_dec_minus = np.maximum(-np.pi/2, self.src_dec - self.delta_angle)
        src_dec_plus = np.minimum(self.src_dec + self.delta_ang, np.pi/2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA = np.amin([np.repeat(2*np.pi, len(self.src_ra)), 2*self.delta_ang / cosfact], axis=0)

        # Calculate the right-ascension distance of the events w.r.t. the
        # source. We make sure to use the smaller distance on the circle, thus
        # the maximal distance is 180deg, i.e. pi.
        # ra_dist is a 2D ndarray with the source on axis 0 and the events on
        # axis 1.
        ra_dist = np.fabs(np.mod(events['ra'] - self.src_ra[:,np.newaxis] + np.pi, 2*np.pi) - np.pi)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        mask_ra = ra_dist < dRA[:,np.newaxis]/2.

        # Determine the mask for the events which fall inside the declination
        # window.
        # mask_dec is a (N_sources,N_events)-shaped ndarray.
        mask_dec = ((events['dec'] > src_dec_minus[:,np.newaxis]) &
                    (events['dec'] < src_dec_plus[:,np.newaxis]))

        # Determine the mask for the events which fall inside the
        # right-ascension and declination window.
        # mask_sky is a (N_sources,N_events)-shaped ndarray.
        mask_sky = mask_ra & mask_dec

        # Determine the mask for the events that fall inside at least one
        # source sky window.
        # mask is a (N_events,)-shaped ndarray.
        mask = np.any(mask_sky, axis=0)

        # Reduce the events according to the mask.
        selected_events = events[mask]

        return selected_events
