# -*- coding: utf-8 -*-

import abc
import inspect
import numpy as np

from skyllh.core.py import (
    classname,
    float_cast,
    func_has_n_args,
    issequenceof
)
from skyllh.core.source_hypothesis import SourceHypoGroupManager
from skyllh.core.timing import TaskTimer
from skyllh.physics.source import SourceModel


class EventSelectionMethod(object, metaclass=abc.ABCMeta):
    """This is the abstract base class for all event selection method classes.
    The idea is to pre-select only events that contribute to the likelihood
    function, i.e. are more signal than background like. The different methods
    are implemented through derived classes of this base class.
    """

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
        self._src_arr = self.source_to_array(
            self._src_hypo_group_manager.source_list)

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance, which defines the list of
        sources.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError(
                'The src_hypo_group_manager property must be an instance of '
                'SourceHypoGroupManager!')
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
    def select_events(self, events, retidxs=False, tl=None):
        """This method selects the events, which will contribute to the
        log-likelihood ratio function.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events.
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
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
        super(AllEventSelectionMethod, self).__init__(
            src_hypo_group_manager)

    def source_to_array(self, sources):
        return None

    def select_events(self, events, retidxs=False, tl=None):
        """Selects all of the given events. Hence, the returned event array is
        the same as the given array.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events, for which
            the selection method should get applied.
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
        """
        if(retidxs):
            return (events, events.indices)
        return events


class SpatialEventSelectionMethod(EventSelectionMethod, metaclass=abc.ABCMeta):
    """This class defines the base class for all spatial event selection
    methods.
    """

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

    def select_events(self, events, retidxs=False, tl=None):
        """Selects the events within the declination band.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            - 'dec' : float
                The declination of the event.
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
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
            idxs = events.indices[mask]
            selected_events = events[idxs]

        if(retidxs):
            return (selected_events, idxs)
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

    def select_events(self, events, retidxs=False, tl=None):
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
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
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
            idxs = events.indices[mask]
            selected_events = events[idxs]

        if(retidxs):
            return (selected_events, idxs)
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

    def select_events(self, events, retidxs=False, tl=None):
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
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
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
            #remove all events that are not close to any source
            mask_sky = mask_sky.T[mask].T
        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.

            idxs = np.argwhere(mask_sky)
            src_idxs = idxs[:,0]
            ev_idxs = idxs[:,1]

            selected_events = events[events.indices[mask]]

        if(retidxs):
            return (selected_events, (src_idxs, ev_idxs))
        return selected_events, None


class PsiFuncEventSelectionMethod(EventSelectionMethod):
    """This event selection method selects events whose psi value, i.e. the
    great circle distance of the event to the source, is smaller than the value
    of the provided function.
    """
    def __init__(self, src_hypo_group_manager, psi_name, func, axis_name_list):
        """Creates a new PsiFuncEventSelectionMethod instance.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        psi_name : str
            The name of the data field that provides the psi value of the event.
        func : callable
            The function that should get evaluated for each event. The call
            signature must be ``func(*axis_data)``, where ``*axis_data`` is the
            event data of each required axis. The number of axes must match the
            provided axis names through the ``axis_name_list``.
        axis_name_list : list of str
            The list of data field names for each axis of the function ``func``.
            All field names must be valid field names of the trial data's
            DataFieldRecordArray instance.
        """
        super(PsiFuncEventSelectionMethod, self).__init__(
            src_hypo_group_manager)

        self.psi_name = psi_name
        self.func = func
        self.axis_name_list = axis_name_list

        if(not (len(inspect.signature(self._func).parameters) >=
                len(self._axis_name_list))):
            raise TypeError(
                'The func argument must be a callable instance with at least '
                '%d arguments!'%(
                    len(self._axis_name_list)))

    @property
    def psi_name(self):
        """The name of the data field that provides the psi value of the event.
        """
        return self._psi_name
    @psi_name.setter
    def psi_name(self, name):
        if(not isinstance(name, str)):
            raise TypeError(
                'The psi_name property must be an instance of type str!')
        self._psi_name = name

    @property
    def func(self):
        """The function that should get evaluated for each event. The call
        signature must be ``func(*axis_data)``, where ``*axis_data`` is the
        event data of each required axis. The number of axes must match the
        provided axis names through the ``axis_name_list`` property.
        """
        return self._func
    @func.setter
    def func(self, f):
        if(not callable(f)):
            raise TypeError(
                'The func property must be a callable instance!')
        self._func = f

    @property
    def axis_name_list(self):
        """The list of data field names for each axis of the function defined
        through the ``func`` property.
        """
        return self._axis_name_list
    @axis_name_list.setter
    def axis_name_list(self, names):
        if(not issequenceof(names, str)):
            raise TypeError(
                'The axis_name_list property must be a sequence of str '
                'instances!')
        self._axis_name_list = list(names)

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
        arr : None
            Because this event selection method does not depend directly on the
            source (only indirectly through the psi values), no source array
            is required.
        """
        return None

    def select_events(self, events, retidxs=False, tl=None):
        """Selects the events whose psi value is smaller than the value of the
        predefined function.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            - <psi_name> : float
                The great circle distance of the event with the source.
            - <*axis_name_list> : float
                The name of the axis required for the function ``func`` to be
                evaluated.

        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
        """
        cls_name = classname(self)

        with TaskTimer(tl, '%s: Get psi values.'%(cls_name)):
            psi = events[self._psi_name]

        with TaskTimer(tl, '%s: Get axis data values.'%(cls_name)):
            func_args = [ events[axis] for axis in self._axis_name_list ]

        with TaskTimer(tl, '%s: Creating mask.'%(cls_name)):
            mask = psi < self._func(*func_args)

        with TaskTimer(tl, '%s: Create selected_events.'%(cls_name)):
            idxs = events.indices[mask]
            selected_events = events[idxs]

        if(retidxs):
            return (selected_events, idxs)
        return selected_events, None





class SpatialBoxAndPsiFuncEventSelectionMethod(SpatialBoxEventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of point-like source
    positions.
    """
    def __init__(self, src_hypo_group_manager, delta_angle, psi_name, func, axis_name_list):
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
        super(SpatialBoxAndPsiFuncEventSelectionMethod, self).__init__(
            src_hypo_group_manager, delta_angle )

        self.psi_name = psi_name
        self.func = func
        self.axis_name_list = axis_name_list

        if(not (len(inspect.signature(self._func).parameters) >=
                len(self._axis_name_list))):
            raise TypeError(
                'The func argument must be a callable instance with at least '
                '%d arguments!'%(
                    len(self._axis_name_list)))


    @property
    def psi_name(self):
        """The name of the data field that provides the psi value of the event.
        """
        return self._psi_name
    @psi_name.setter
    def psi_name(self, name):
        if(not isinstance(name, str)):
            raise TypeError(
                'The psi_name property must be an instance of type str!')
        self._psi_name = name

    @property
    def func(self):
        """The function that should get evaluated for each event. The call
        signature must be ``func(*axis_data)``, where ``*axis_data`` is the
        event data of each required axis. The number of axes must match the
        provided axis names through the ``axis_name_list`` property.
        """
        return self._func
    @func.setter
    def func(self, f):
        if(not callable(f)):
            raise TypeError(
                'The func property must be a callable instance!')
        self._func = f

    @property
    def axis_name_list(self):
        """The list of data field names for each axis of the function defined
        through the ``func`` property.
        """
        return self._axis_name_list
    @axis_name_list.setter
    def axis_name_list(self, names):
        if(not issequenceof(names, str)):
            raise TypeError(
                'The axis_name_list property must be a sequence of str '
                'instances!')
        self._axis_name_list = list(names)


    def _get_psi(self, events, idxs):
        """Function to calculate the log10 of the opening angle between the
        source position and the event's reconstructed position.
        """

        ra = events['ra']
        dec = events['dec']

        src_idxs, ev_idxs = idxs
        src_ra = self._src_arr['ra'][src_idxs]
        src_dec = self._src_arr['dec'][src_idxs]

        cos_ev = np.sqrt(1. - np.sin(dec)**2)
        cos_psi = (np.cos(src_ra - np.take(ra,ev_idxs)) * \
                np.cos(src_dec) * np.take(cos_ev, ev_idxs) + \
                np.sin(src_dec) * np.take(np.sin(dec), ev_idxs))

        # Handle possible floating precision errors.
        cos_psi[cos_psi < -1.] = -1.
        cos_psi[cos_psi > 1.] = 1.

        psi = np.arccos(cos_psi)
        psi = np.where(psi == 0, 1e-10, psi)

        return psi

    def select_events(self, events, retidxs=False, tl=None):
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
        retidxs : bool
            Flag if also the indices of of the selected events should get
            returned as 1d ndarray.
            Default is False.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        idxs : ndarray of ints
            The indices of the selected events, in case `retidxs` is set to
            True.
        """
        #print('Do psi event selection!')

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
        #with TaskTimer(tl, 'ESM: Calculate ra_dist.'):
        #    ra_dist = np.fabs(
        #        np.mod(events['ra'] - src_arr['ra'][:,np.newaxis] + np.pi, 2*np.pi) - np.pi)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        #print('length background events', len(events['ra']))

        with TaskTimer(tl, 'ESM: Calculate mask_ra.'):
            nsrc = len(src_arr['ra'])
           
            # fill in batch sizes of 200 maximum to save memory
            batch_size=200
            if nsrc > batch_size:
                mask_ra = np.zeros((nsrc, len(events['ra'])), dtype=bool)
                n_batches = np.ceil(nsrc / float(batch_size))
                for bi in range(n_batches):
                    if not (bi == n_batches-1):
                        mask_ra[bi*batch_size : (bi+1)*batch_size,...] = (np.fabs(
                            np.mod(events['ra'] - src_arr['ra'][bi*batch_size : (bi+1)*batch_size][:,np.newaxis] + np.pi, 2*np.pi) - 
                            np.pi) < dRA_half[:,np.newaxis])
                    else:
                        mask_ra[bi*batch_size : ,...] = (np.fabs(
                            np.mod(events['ra'] - src_arr['ra'][bi*batch_size:][:,np.newaxis] + np.pi, 2*np.pi) - 
                            np.pi) < dRA_half[:,np.newaxis])

            else:
                mask_ra = np.fabs(
                    np.mod(events['ra'] - src_arr['ra'][:,np.newaxis] + np.pi, 2*np.pi) - np.pi) < dRA_half[:,np.newaxis]

            #ra_dist < dRA_half[:,np.newaxis]
            #del ra_dist

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
            #remove all events that are not close to any source
            mask_sky = mask_sky.T[mask].T
        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.

            idxs = np.argwhere(mask_sky)
            src_idxs = idxs[:,0]
            ev_idxs = idxs[:,1]

            selected_events = events[mask]

        # Now in addition add the psi selection
        cls_name = classname(self)

        with TaskTimer(tl, '%s: Get psi values.'%(cls_name)):
            psi = self._get_psi(selected_events, (src_idxs, ev_idxs))

        #with TaskTimer(tl, '%s: Get axis data values.'%(cls_name)):
        #    func_args = [selected_events[axis][ev_idxs] for axis in self._axis_name_list ]

        with TaskTimer(tl, '%s: Creating mask.'%(cls_name)):
            #mask = psi < self._func(*func_args)
            mask = (self._func(psi) <= selected_events['ang_err'][ev_idxs]) | (psi < np.deg2rad(5.))
            #mask = psi>=0
        with TaskTimer(tl, '%s: Create selected_events.'%(cls_name)):
            m_tot = np.bincount(ev_idxs[mask]) > 0
            if not (m_tot.shape[0] == (np.max(ev_idxs)+1)):
                _m_tot = np.zeros((np.max(ev_idxs)+1,), dtype=bool)

                _m_tot[:m_tot.shape[0]] = m_tot
                m_tot = _m_tot

            src_idxs = src_idxs[mask]
            ev_idxs = ev_idxs[mask]

            n_old = len(idxs[:,0])

            idxs = [src_idxs, ev_idxs]

            #remove all events that are not important to any source
            final_sky_mask = np.zeros_like(mask_sky, dtype=bool)
            final_sky_mask[idxs] = mask_sky[idxs]
            final_sky_mask = final_sky_mask.T[m_tot].T

            idxs = np.argwhere(final_sky_mask)
            src_idxs = idxs[:,0]
            ev_idxs = idxs[:,1]

            final_selected_events = selected_events[m_tot]

            #print('number of events:', len(src_idxs))

        if(retidxs):
            return (final_selected_events, (src_idxs, ev_idxs))

        return final_selected_events, None



