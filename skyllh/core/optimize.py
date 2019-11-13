# -*- coding: utf-8 -*-

import abc
import numpy as np

from skyllh.core.py import float_cast, issequenceof
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.timing import TaskTimer
from skyllh.physics.source import SourceModel


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
    def select_events(self, events, retmask=False):
        """This method selects the events, which will contribute to the
        log-likelihood ratio function.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events.
        retmask : bool
            Flag if also the (N_events,)-shaped boolean ndarray mask of the
            selected events should get returned.
            Default is False.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        mask : (N_events,)-shaped ndarray of bools
            The mask of the selected events, in case `retmask` is set to True.
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

    def select_events(self, events, retmask=False):
        """Selects all of the given events. Hence, the returned event array is
        the same as the given array.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events, for which
            the selection method should get applied.
        retmask : bool
            Flag if also the mask of the selected events should get returned.
            Default is False.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        mask : (N_events,)-shaped ndarray of bools
            The mask of the selected events, in case `retmask` is set to True.
        """
        if(retmask):
            return (events, np.ones((len(events),), dtype=np.bool))
        return events


class SpatialEventSelectionMethod(EventSelectionMethod):
    """This class defines the base class for all spatial event selection
    methods.
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
        super(SpatialEventSelectionMethod, self).__init__(
            src_hypo_group_manager)


class DecBandEventSectionMethod(SpatialEventSelectionMethod):
    """This event selection method selects events within a declination band
    around a list of point-like source positions.
    """
    def __init__(self, src_hypo_group_manager, delta_angle):
        """Creates and configures a spatial declination band event selection
        method object.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source in declination for which
            events should get selected.
        """
        super(DecBandEventSectionMethod, self).__init__(
            src_hypo_group_manager)

        self.delta_angle = delta_angle

        self._src_arr = self.source_to_array(
            self._src_hypo_group_manager.source_list)

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle
    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(angle, 'The delta_angle property must be castable '
            'to type float!')
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
            raise TypeError('The sources argument must be a sequence of '
                'SourceModel instances!')

        arr = np.empty(
            (len(sources),),
            dtype=[('ra', np.float), ('dec', np.float)],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.loc.ra
            arr['dec'][i] = src.loc.dec

        return arr

    def select_events(self, events, retmask=False, tl=None):
        """Selects the events within the declination band.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            - 'dec' : float
                The declination of the event.
        retmask : bool
            Flag if also the mask of the selected events should get returned.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        mask : (N_events,)-shaped ndarray of bools
            The mask of the selected events, in case `retmask` is set to True.
        """
        delta_angle = self._delta_angle
        src_arr = self._src_arr

        # Calculates the minus and plus declination around each source and
        # bound it to -90deg and +90deg, respectively.
        src_dec_minus = np.maximum(-np.pi/2, src_arr['dec'] - delta_angle)
        src_dec_plus = np.minimum(src_arr['dec'] + delta_angle, np.pi/2)

        # Determine the mask for the events which fall inside the declination
        # window.
        # mask_dec is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM-DecBand: Calculate mask_dec'):
            mask_dec = ((events['dec'] > src_dec_minus[:,np.newaxis]) &
                        (events['dec'] < src_dec_plus[:,np.newaxis]))

        # Determine the mask for the events that fall inside at least one
        # source declination band.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM-DecBand: Calculate mask.'):
            mask = np.any(mask_dec, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM-DecBand: Create selected_events.'):
            idx = events.indices[mask]
            selected_events = events[idx]

        if(retmask):
            return (selected_events, mask)
        return selected_events


class RABandEventSectionMethod(SpatialEventSelectionMethod):
    """This event selection method selects events within a right-ascension band
    around a list of point-like source positions.
    """
    def __init__(self, src_hypo_group_manager, delta_angle):
        """Creates and configures a right-ascension band event selection
        method object.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source in right-ascension for
            which events should get selected.
        """
        super(RABandEventSectionMethod, self).__init__(
            src_hypo_group_manager)

        self.delta_angle = delta_angle

        self._src_arr = self.source_to_array(
            self._src_hypo_group_manager.source_list)

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle
    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(angle,
            'The delta_angle property must be castable to type float!')
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
            raise TypeError('The sources argument must be a sequence of '
                'SourceModel instances!')

        arr = np.empty(
            (len(sources),),
            dtype=[('ra', np.float), ('dec', np.float)],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.loc.ra
            arr['dec'][i] = src.loc.dec

        return arr

    def select_events(self, events, retmask=False, tl=None):
        """Selects the events within the right-ascention band.

        The solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec) is a
        function of declination, i.e. for a constant dOmega, the right-ascension
        value has to change with declination.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            - 'ra' : float
                The right-ascention of the event.
            - 'dec' : float
                The declination of the event.
        retmask : bool
            Flag if also the mask of the selected events should get returned.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        mask : (N_events,)-shaped ndarray of bools
            The mask of the selected events, in case `retmask` is set to True.
        """
        delta_angle = self._delta_angle
        src_arr = self._src_arr

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi/2, src_arr['dec'] - delta_angle)
        src_dec_plus = np.minimum(src_arr['dec'] + delta_angle, np.pi/2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2*np.pi, len(src_arr['ra'])),
             np.fabs(delta_angle / cosfact)], axis=0)

        # Calculate the right-ascension distance of the events w.r.t. the
        # source. We make sure to use the smaller distance on the circle, thus
        # the maximal distance is 180deg, i.e. pi.
        # ra_dist is a (N_sources,N_events)-shaped 2D ndarray.
        with TaskTimer(tl, 'ESM-RaBand: Calculate ra_dist.'):
            ra_dist = np.fabs(
                np.mod(events['ra'] - src_arr['ra'][:,np.newaxis] + np.pi, 2*np.pi) - np.pi)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM-RaBand: Calculate mask_ra.'):
            mask_ra = ra_dist < dRA_half[:,np.newaxis]

        # Determine the mask for the events that fall inside at least one
        # source sky window.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM-RaBand: Calculate mask.'):
            mask = np.any(mask_ra, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM-RaBand: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            idx = events.indices[mask]
            selected_events = events[idx]

        if(retmask):
            return (selected_events, mask)
        return selected_events


class SpatialBoxEventSelectionMethod(SpatialEventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of point-like source
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
        super(SpatialBoxEventSelectionMethod, self).__init__(
            src_hypo_group_manager)

        self.delta_angle = delta_angle

        self._src_arr = self.source_to_array(
            self._src_hypo_group_manager.source_list)

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle
    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(angle,
            'The delta_angle property must be castable to type float!')
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
            raise TypeError('The sources argument must be a sequence of '
                'SourceModel instances!')

        arr = np.empty(
            (len(sources),),
            dtype=[('ra', np.float), ('dec', np.float)],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.loc.ra
            arr['dec'][i] = src.loc.dec

        return arr

    def select_events(self, events, retmask=False, tl=None):
        """Selects the events within the spatial box in right-ascention and
        declination.

        The solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec) is a
        function of declination, i.e. for a constant dOmega, the right-ascension
        value has to change with declination.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            - 'ra' : float
                The right-ascention of the event.
            - 'dec' : float
                The declination of the event.
        retmask : bool
            Flag if also the mask of the selected events should get returned.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        mask : (N_events,)-shaped ndarray of bools
            The mask of the selected events, in case `retmask` is set to True.
        """
        delta_angle = self._delta_angle
        src_arr = self._src_arr

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi/2, src_arr['dec'] - delta_angle)
        src_dec_plus = np.minimum(src_arr['dec'] + delta_angle, np.pi/2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2*np.pi, len(src_arr['ra'])),
             np.fabs(delta_angle / cosfact)], axis=0)

        # Calculate the right-ascension distance of the events w.r.t. the
        # source. We make sure to use the smaller distance on the circle, thus
        # the maximal distance is 180deg, i.e. pi.
        # ra_dist is a (N_sources,N_events)-shaped 2D ndarray.
        with TaskTimer(tl, 'ESM: Calculate ra_dist.'):
            ra_dist = np.fabs(
                np.mod(events['ra'] - src_arr['ra'][:,np.newaxis] + np.pi, 2*np.pi) - np.pi)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_ra.'):
            mask_ra = ra_dist < dRA_half[:,np.newaxis]

        # Determine the mask for the events which fall inside the declination
        # window.
        # mask_dec is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_dec'):
            mask_dec = ((events['dec'] > src_dec_minus[:,np.newaxis]) &
                        (events['dec'] < src_dec_plus[:,np.newaxis]))

        # Determine the mask for the events which fall inside the
        # right-ascension and declination window.
        # mask_sky is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_sky.'):
            mask_sky = mask_ra & mask_dec

        # Determine the mask for the events that fall inside at least one
        # source sky window.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask.'):
            mask = np.any(mask_sky, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            idx = events.indices[mask]
            selected_events = events[idx]

        if(retmask):
            return (selected_events, mask)
        return selected_events
