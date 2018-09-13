# -*- coding: utf-8 -*-

import abc
import numpy as np

from skylab.core.py import float_cast, issequenceof
from skylab.core.source_hypothesis import SourceHypoGroupManager
from skylab.physics.source import SourceModel


class EventSelectionMethod(object):
    """This is the abstract base class for all event selection method classes.
    The idea is to pre-select only events that contribute to the likelihood
    function, i.e. are more signal than background like. The different methods
    are implemented through derived classes of this base class.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, src_hypo_group_manager):
        """Creates a new event selection method instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        """
        super(EventSelectionMethod, self).__init__()

        self.src_hypo_group_manager = src_hypo_group_manager

        # The _src_arr variable holds a numpy record array with the necessary
        # source information needed for the event selection method.
        self._src_arr = None

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance, which defines the list of
        sources.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Changes the SourceHypoGroupManager instance of the event selection
        method. This will also recreate the internal source numpy record array.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The new SourceHypoGroupManager instance, that should be used for
            this event selection method.
        """
        self.src_hypo_group_manager = src_hypo_group_manager
        self._src_arr = self.source_to_array(self._src_hypo_group_manager.source_list)

    @abc.abstractmethod
    def source_to_array(self, sources):
        """This method is supposed to convert a sequence of SourceModel
        instances into a structured numpy ndarray with the source information
        in a format that is best understood my the actual event selection
        method.

        Parameters
        ----------
        sources : sequence of SourceModel
            The sequence of source models containing the necessary information
            of the source.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the necessary information
            for each source.
        """
        pass

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
    def __init__(self, src_hypo_group_manager):
        """Creates a new event selection method instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances. For this particular
            event selection method it has no meaning, but it is an interface
            parameter.
        """
        super(AllEventSelectionMethod, self).__init__(src_hypo_group_manager)

    def source_to_array(self, sources):
        return None

    def select_events(self, events):
        return events


class SpatialBoxEventSelectionMethod(EventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of point-like sources
    positions.
    """
    def __init__(self, src_hypo_group_manager, delta_angle):
        """Creates and configures a spatial box event selection method object.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source for which events should
            get selected.
        """
        super(SpatialBoxEventSelectionMethod, self).__init__(src_hypo_group_manager)

        self.delta_angle = delta_angle

        self._src_arr = self.source_to_array(self._src_hypo_group_manager.source_list)

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

    def source_to_array(self, sources):
        """Converts the given sequence of SourceModel instances into a
        structured numpy ndarray holding the necessary source information needed
        for this event selection method.

        Parameters
        ----------
        sources : sequence of SourceModel
            The sequence of source models containing the necessary information
            of the source.

        Returns
        -------
        arr : numpy record ndarray
            The generated numpy record ndarray holding the necessary information
            for each source. It contains the following data fields: 'ra', 'dec'.
        """
        if(not issequenceof(sources, SourceModel)):
            raise TypeError('The sources argument must be a sequence of SourceModel instances!')

        arr = np.empty(
            (len(sources),),
            dtype=[('ra', np.float), ('dec', np.float)],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.loc.ra
            arr['dec'][i] = src.loc.dec

        return arr

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
        src_dec_minus = np.maximum(-np.pi/2, self._src_arr['dec'] - self.delta_angle)
        src_dec_plus = np.minimum(self._src_arr['dec'] + self.delta_angle, np.pi/2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA = np.amin([np.repeat(2*np.pi, len(self._src_arr['ra'])), 2*self.delta_angle / cosfact], axis=0)

        # Calculate the right-ascension distance of the events w.r.t. the
        # source. We make sure to use the smaller distance on the circle, thus
        # the maximal distance is 180deg, i.e. pi.
        # ra_dist is a 2D ndarray with the source on axis 0 and the events on
        # axis 1.
        ra_dist = np.fabs(np.mod(events['ra'] - self._src_arr['ra'][:,np.newaxis] + np.pi, 2*np.pi) - np.pi)

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
