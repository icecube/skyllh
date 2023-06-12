# -*- coding: utf-8 -*-

import abc
import inspect
import numpy as np
import scipy.sparse

from skyllh.core.py import (
    classname,
    float_cast,
    issequenceof,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    SourceModel,
)
from skyllh.core.timing import (
    TaskTimer,
)
from skyllh.core.utils.coords import (
    angular_separation,
)


class EventSelectionMethod(
        object,
        metaclass=abc.ABCMeta):
    """This is the abstract base class for all event selection method classes.
    The idea is to pre-select only events that contribute to the likelihood
    function, i.e. are more signal than background like. The different methods
    are implemented through derived classes of this base class.
    """

    def __init__(
            self,
            shg_mgr,
            **kwargs):
        """Creates a new event selection method instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager | None
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
            It can be ``None`` if the event selection method does not depend on
            the sources.
        """
        super().__init__(
            **kwargs)

        self._src_arr = None

        self._shg_mgr = shg_mgr
        if self._shg_mgr is not None:
            if not isinstance(self._shg_mgr, SourceHypoGroupManager):
                raise TypeError(
                    'The shg_mgr argument must be None or an instance of '
                    'SourceHypoGroupManager! '
                    f'Its current type is {classname(self._shg_mgr)}.')

            # The _src_arr variable holds a numpy record array with the
            # necessary source information needed for the event selection
            # method.
            self._src_arr = self.sources_to_array(
                sources=self._shg_mgr.source_list)

    @property
    def shg_mgr(self):
        """(read-only) The instance of SourceHypoGroupManager, which defines the
        list of sources.
        """
        return self._shg_mgr

    def __and__(self, other):
        """Implements the AND operator (&) for creating an event selection
        method, which is the intersection of this event selection method and
        another one using the expression ``intersection = self & other``.

        Parameters
        ----------
        other : instance of EventSelectionMethod
            The instance of EventSelectionMethod that is the other event
            selection method.

        Returns
        -------
        intersection : instance of IntersectionEventSelectionMethod
            The instance of IntersectionEventSelectionMethod that creates the
            intersection of this event selection method and the other.
        """
        return IntersectionEventSelectionMethod(self, other)

    def change_shg_mgr(self, shg_mgr):
        """Changes the SourceHypoGroupManager instance of the event selection
        method. This will also recreate the internal source numpy record array.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager | None
            The new SourceHypoGroupManager instance, that should be used for
            this event selection method.
            It can be ``None`` if the event selection method does not depend on
            the sources.
        """
        self._shg_mgr = shg_mgr
        self._src_arr = None

        if self._shg_mgr is not None:
            if not isinstance(self._shg_mgr, SourceHypoGroupManager):
                raise TypeError(
                    'The shg_mgr argument must be None or an instance of '
                    'SourceHypoGroupManager! '
                    f'Its current type is {classname(self._shg_mgr)}.')

            self._src_arr = self.sources_to_array(
                sources=self._shg_mgr.source_list)

    def sources_to_array(self, sources):
        """This method is supposed to convert a sequence of SourceModel
        instances into a structured numpy ndarray with the source information
        in a format that is best understood by the actual event selection
        method.

        Parameters
        ----------
        sources : sequence of SourceModel
            The sequence of source models containing the necessary information
            of the source.

        Returns
        -------
        arr : numpy record ndarray | None
            The generated numpy record ndarray holding the necessary information
            for each source.
            By default ``None`` is returned.
        """
        return None

    @abc.abstractmethod
    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """This method selects the events, which will contribute to the
        log-likelihood ratio function.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray of length N_events, holding the
            events.
        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray of length N_selected_events,
            holding the selected events, i.e. a subset of the ``events``
            argument.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The two 1d ndarrays of int of length N_values, holding the indices
            of the sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        pass


class IntersectionEventSelectionMethod(
        EventSelectionMethod):
    """This class provides an event selection method for the intersection of two
    event selection methods. It can be created using the ``&`` operator:
    ``evt_sel_method1 & evt_sel_method2``.
    """
    def __init__(
            self,
            evt_sel_method1,
            evt_sel_method2,
            **kwargs):
        """Creates a compounded event selection method of two given event
        selection methods.

        Parameters
        ----------
        evt_sel_method1 : instance of EventSelectionMethod
            The instance of EventSelectionMethod for the first event selection
            method.
        evt_sel_method2 : instance of EventSelectionMethod
            The instance of EventSelectionMethod for the second event selection
            method.
        """
        super().__init__(
            shg_mgr=None,
            **kwargs)

        self.evt_sel_method1 = evt_sel_method1
        self.evt_sel_method2 = evt_sel_method2

    @property
    def evt_sel_method1(self):
        """The instance of EventSelectionMethod for the first event selection
        method.
        """
        return self._evt_sel_method1

    @evt_sel_method1.setter
    def evt_sel_method1(self, method):
        if not isinstance(method, EventSelectionMethod):
            raise TypeError(
                'The evt_sel_method1 property must be an instance of '
                'EventSelectionMethod!'
                f'Its current type is {classname(method)}.')
        self._evt_sel_method1 = method

    @property
    def evt_sel_method2(self):
        """The instance of EventSelectionMethod for the second event selection
        method.
        """
        return self._evt_sel_method2

    @evt_sel_method2.setter
    def evt_sel_method2(self, method):
        if not isinstance(method, EventSelectionMethod):
            raise TypeError(
                'The evt_sel_method2 property must be an instance of '
                'EventSelectionMethod!'
                f'Its current type is {classname(method)}.')
        self._evt_sel_method2 = method

    def change_shg_mgr(self, shg_mgr):
        """Changes the SourceHypoGroupManager instance of the event selection
        method. This will call the ``change_shg_mgr`` of the individual event
        selection methods.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager | None
            The new SourceHypoGroupManager instance, that should be used for
            this event selection method.
            It can be ``None`` if the event selection method does not depend on
            the sources.
        """
        self._evt_sel_method1.change_shg_mgr(shg_mgr=shg_mgr)
        self._evt_sel_method2.change_shg_mgr(shg_mgr=shg_mgr)

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects events by calling the ``select_events`` methods of the
        individual event selection methods.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events.
        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of the sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        if ret_original_evt_idxs:
            (events, src_evt_idxs, org_evt_idxs1) =\
                self._evt_sel_method1.select_events(
                    events=events,
                    src_evt_idxs=src_evt_idxs,
                    ret_original_evt_idxs=True)

            (events, src_evt_idxs, org_evt_idxs2) =\
                self._evt_sel_method2.select_events(
                    events=events,
                    src_evt_idxs=src_evt_idxs,
                    ret_original_evt_idxs=True)

            org_evt_idxs = np.take(org_evt_idxs1, org_evt_idxs2)

            return (events, src_evt_idxs, org_evt_idxs)

        (events, src_evt_idxs) = self._evt_sel_method1.select_events(
            events=events,
            src_evt_idxs=src_evt_idxs)

        (events, src_evt_idxs) = self._evt_sel_method2.select_events(
            events=events,
            src_evt_idxs=src_evt_idxs)

        return (events, src_evt_idxs)


class AllEventSelectionMethod(
        EventSelectionMethod):
    """This event selection method selects all events.
    """
    def __init__(self, shg_mgr):
        """Creates a new event selection method instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances. For this particular
            event selection method it has no meaning, but it is an interface
            parameter.
        """
        super().__init__(
            shg_mgr=shg_mgr)

    def sources_to_array(self, sources):
        """Creates the source array from the given list of sources. This event
        selection method does not depend on the sources. Hence, ``None`` is
        returned.

        Returns
        -------
        arr : None
            The generated numpy record ndarray holding the necessary information
            for each source. Since this event selection method does not depend
            on any source, ``None`` is returned.
        """
        return None

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects all of the given events. Hence, the returned event array is
        the same as the given array.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding the events, for which
            the selection method should get applied.
        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : DataFieldRecordArray
            The instance of DataFieldRecordArray holding the selected events,
            i.e. a subset of the `events` argument.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        with TaskTimer(tl, 'ESM: Calculate indices of selected events.'):
            if src_evt_idxs is None:
                n_sources = self.shg_mgr.n_sources
                src_idxs = np.repeat(np.arange(n_sources), len(events))
                evt_idxs = np.tile(events.indices, n_sources)
            else:
                (src_idxs, evt_idxs) = src_evt_idxs

        if ret_original_evt_idxs:
            return (events, (src_idxs, evt_idxs), events.indices)

        return (events, (src_idxs, evt_idxs))


class SpatialEventSelectionMethod(
        EventSelectionMethod,
        metaclass=abc.ABCMeta):
    """This abstract base class defines the base class for all spatial event
    selection methods.
    """

    def __init__(
            self,
            shg_mgr,
            **kwargs):
        """Creates a new event selection method instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            **kwargs)

    def sources_to_array(self, sources):
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
        if not issequenceof(sources, SourceModel):
            raise TypeError(
                'The sources argument must be a sequence of SourceModel '
                'instances! '
                f'Its current type is {classname(sources)}.')

        arr = np.empty(
            (len(sources),),
            dtype=[
                ('ra', np.float64),
                ('dec', np.float64)
            ],
            order='F')

        for (i, src) in enumerate(sources):
            arr['ra'][i] = src.ra
            arr['dec'][i] = src.dec

        return arr


class DecBandEventSectionMethod(
        SpatialEventSelectionMethod):
    """This event selection method selects events within a declination band
    around a list of point-like source positions.
    """
    def __init__(
            self,
            shg_mgr,
            delta_angle):
        """Creates and configures a spatial declination band event selection
        method object.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source in declination for which
            events should get selected.
        """
        super().__init__(
            shg_mgr=shg_mgr)

        self.delta_angle = delta_angle

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle

    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(
            angle,
            'The delta_angle property must be castable to type float!')
        self._delta_angle = angle

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects the events within the declination band.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

                ``'dec'`` : float
                    The declination of the event.

        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
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
        with TaskTimer(tl, 'ESM-DecBand: Calculate mask_dec.'):
            mask_dec = (
                (events['dec'] > src_dec_minus[:, np.newaxis]) &
                (events['dec'] < src_dec_plus[:, np.newaxis])
            )

        # Determine the mask for the events that fall inside at least one
        # source declination band.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM-DecBand: Calculate mask.'):
            mask = np.any(mask_dec, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM-DecBand: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            selected_events_idxs = events.indices[mask]
            selected_events = events[selected_events_idxs]

        # Get selected events indices.
        idxs = np.argwhere(mask_dec[:, mask])
        src_idxs = idxs[:, 0]
        evt_idxs = idxs[:, 1]

        if ret_original_evt_idxs:
            return (selected_events, (src_idxs, evt_idxs), selected_events_idxs)

        return (selected_events, (src_idxs, evt_idxs))


class RABandEventSectionMethod(
        SpatialEventSelectionMethod):
    """This event selection method selects events within a right-ascension band
    around a list of point-like source positions.
    """
    def __init__(
            self,
            shg_mgr,
            delta_angle):
        """Creates and configures a right-ascension band event selection
        method object.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source in right-ascension for
            which events should get selected.
        """
        super().__init__(
            shg_mgr=shg_mgr)

        self.delta_angle = delta_angle

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle

    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(
            angle,
            'The delta_angle property must be castable to type float!')
        self._delta_angle = angle

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects the events within the right-ascention band.

        The solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec) is a
        function of declination, i.e. for a constant dOmega, the right-ascension
        value has to change with declination.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            ``'ra'`` : float
                The right-ascention of the event.
            ``'dec'`` : float
                The declination of the event.

        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of the sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
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
                np.mod(
                    events['ra'] - src_arr['ra'][:, np.newaxis] + np.pi,
                    2*np.pi) - np.pi)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM-RaBand: Calculate mask_ra.'):
            mask_ra = ra_dist < dRA_half[:, np.newaxis]

        # Determine the mask for the events that fall inside at least one
        # source sky window.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM-RaBand: Calculate mask.'):
            mask = np.any(mask_ra, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM-RaBand: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            selected_events_idxs = events.indices[mask]
            selected_events = events[selected_events_idxs]

        # Get selected events indices.
        idxs = np.argwhere(mask_ra[:, mask])
        src_idxs = idxs[:, 0]
        evt_idxs = idxs[:, 1]

        if ret_original_evt_idxs:
            return (selected_events, (src_idxs, evt_idxs), selected_events_idxs)

        return (selected_events, (src_idxs, evt_idxs))


class SpatialBoxEventSelectionMethod(
        SpatialEventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of point-like source
    positions.
    """
    def __init__(
            self,
            shg_mgr,
            delta_angle):
        """Creates and configures a spatial box event selection method object.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source for which events should
            get selected.
        """
        super().__init__(
            shg_mgr=shg_mgr)

        self.delta_angle = delta_angle

    @property
    def delta_angle(self):
        """The half-opening angle around the source in declination and
        right-ascention for which events should get selected.
        """
        return self._delta_angle

    @delta_angle.setter
    def delta_angle(self, angle):
        angle = float_cast(
            angle,
            'The delta_angle property must be castable to type float!')
        self._delta_angle = angle

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
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

            ``'ra'`` : float
                The right-ascention of the event.
            ``'dec'`` : float
                The declination of the event.

        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        (src_idxs, evt_idxs) : 1d ndarrays of ints | None
            The indices of sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        delta_angle = self._delta_angle
        src_arr = self._src_arr
        n_sources = len(src_arr)

        srcs_ra = src_arr['ra']
        srcs_dec = src_arr['dec']

        # Get the minus and plus declination around the sources.
        src_dec_minus = np.maximum(-np.pi/2, srcs_dec - delta_angle)
        src_dec_plus = np.minimum(srcs_dec + delta_angle, np.pi/2)

        # Calculate the cosine factor for the largest declination distance from
        # the source. We use np.amin here because smaller cosine values are
        # larger angles.
        # cosfact is a (N_sources,)-shaped ndarray.
        cosfact = np.amin(np.cos([src_dec_minus, src_dec_plus]), axis=0)

        # Calculate delta RA, which is a function of declination.
        # dRA is a (N_sources,)-shaped ndarray.
        dRA_half = np.amin(
            [np.repeat(2*np.pi, n_sources),
             np.fabs(delta_angle / cosfact)], axis=0)

        # Determine the mask for the events which fall inside the
        # right-ascention window.
        # mask_ra is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_ra.'):
            evts_ra = events['ra']
            # Fill in batch sizes of 128 maximum to save memory.
            batch_size = 128
            if n_sources > batch_size:
                mask_ra = np.zeros((n_sources, len(evts_ra)), dtype=bool)
                n_batches = int(np.ceil(n_sources / float(batch_size)))
                for bi in range(n_batches):
                    if bi == n_batches-1:
                        # We got the last batch of sources.
                        srcs_slice = slice(bi*batch_size, None)
                    else:
                        srcs_slice = slice(bi*batch_size, (bi+1)*batch_size)

                    ra_diff = np.fabs(
                        evts_ra - srcs_ra[srcs_slice][:, np.newaxis])
                    ra_mod = np.where(
                        ra_diff >= np.pi, 2*np.pi - ra_diff, ra_diff)
                    mask_ra[srcs_slice, :] = (
                        ra_mod < dRA_half[srcs_slice][:, np.newaxis]
                    )
            else:
                ra_diff = np.fabs(evts_ra - srcs_ra[:, np.newaxis])
                ra_mod = np.where(ra_diff >= np.pi, 2*np.pi-ra_diff, ra_diff)
                mask_ra = ra_mod < dRA_half[:, np.newaxis]

        # Determine the mask for the events which fall inside the declination
        # window.
        # mask_dec is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_dec.'):
            mask_dec = (
                (events['dec'] > src_dec_minus[:, np.newaxis]) &
                (events['dec'] < src_dec_plus[:, np.newaxis])
            )

        # Determine the mask for the events which fall inside the
        # right-ascension and declination window.
        # mask_sky is a (N_sources,N_events)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask_sky.'):
            mask_sky = mask_ra & mask_dec
            del mask_ra
            del mask_dec

        # Determine the mask for the events that fall inside at least one
        # source sky window.
        # mask is a (N_events,)-shaped ndarray.
        with TaskTimer(tl, 'ESM: Calculate mask.'):
            mask = np.any(mask_sky, axis=0)

        # Reduce the events according to the mask.
        with TaskTimer(tl, 'ESM: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            selected_events_idxs = events.indices[mask]
            selected_events = events[selected_events_idxs]

        # Get selected events indices.
        idxs = np.argwhere(mask_sky[:, mask])
        src_idxs = idxs[:, 0]
        evt_idxs = idxs[:, 1]

        if ret_original_evt_idxs:
            return (selected_events, (src_idxs, evt_idxs), selected_events_idxs)

        return (selected_events, (src_idxs, evt_idxs))


class PsiFuncEventSelectionMethod(
        EventSelectionMethod):
    """This event selection method selects events whose psi value, i.e. the
    great circle distance of the event to the source, is smaller than the value
    of the provided function.
    """
    def __init__(
            self,
            shg_mgr,
            psi_name,
            func,
            axis_name_list):
        """Creates a new PsiFuncEventSelectionMethod instance.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        psi_name : str
            The name of the data field that provides the psi value of the event.
        func : callable
            The function that should get evaluated for each event. The call
            signature must be

                ``func(*axis_data)``,

            where ``*axis_data`` is the event data of each required axis. The
            number of axes must match the provided axis names through the
            ``axis_name_list``.
        axis_name_list : list of str
            The list of data field names for each axis of the function ``func``.
            All field names must be valid field names of the trial data's
            DataFieldRecordArray instance.
        """
        super().__init__(
            shg_mgr=shg_mgr)

        self.psi_name = psi_name
        self.func = func
        self.axis_name_list = axis_name_list

        n_func_args = len(inspect.signature(self._func).parameters)
        if n_func_args < len(self._axis_name_list):
            raise TypeError(
                'The func argument must be a callable instance with at least '
                f'{len(self._axis_name_list)} arguments! Its current number '
                f'of arguments is {n_func_args}.')

        n_sources = self.shg_mgr.n_sources
        if n_sources != 1:
            raise ValueError(
                'The `PsiFuncEventSelectionMethod.select_events` currently '
                'supports only a single source. It was called with '
                f'{n_sources} sources.')

    @property
    def psi_name(self):
        """The name of the data field that provides the psi value of the event.
        """
        return self._psi_name

    @psi_name.setter
    def psi_name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                'The psi_name property must be an instance of type str! '
                f'Its current type is {classname(name)}.')
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
        if not callable(f):
            raise TypeError(
                'The func property must be a callable instance! '
                f'Its current type is {classname(f)}.')
        self._func = f

    @property
    def axis_name_list(self):
        """The list of data field names for each axis of the function defined
        through the ``func`` property.
        """
        return self._axis_name_list

    @axis_name_list.setter
    def axis_name_list(self, names):
        if not issequenceof(names, str):
            raise TypeError(
                'The axis_name_list property must be a sequence of str '
                'instances! '
                f'Its current type is {classname(names)}.')
        self._axis_name_list = list(names)

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects the events whose psi value is smaller than the value of the
        predefined function.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            <psi_name> : float
                The great circle distance of the event with the source.
            <axis_name(s)> : float
                The name of the axis required for the function ``func`` to be
                evaluated.

        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of the sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        cls_name = classname(self)

        with TaskTimer(tl, f'{cls_name}: Get psi values.'):
            psi = events[self._psi_name]

        with TaskTimer(tl, f'{cls_name}: Get axis data values.'):
            func_args = [events[axis] for axis in self._axis_name_list]

        with TaskTimer(tl, f'{cls_name}: Creating mask.'):
            mask = psi < self._func(*func_args)

        with TaskTimer(tl, f'{cls_name}: Create selected_events.'):
            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            selected_events_idxs = events.indices[mask]
            selected_events = events[selected_events_idxs]

        # Get selected events indices.
        idxs = np.argwhere(np.atleast_2d(mask))
        src_idxs = idxs[:, 0]
        evt_idxs = idxs[:, 1]

        if ret_original_evt_idxs:
            return (selected_events, (src_idxs, evt_idxs), selected_events_idxs)

        return (selected_events, (src_idxs, evt_idxs))


class AngErrOfPsiEventSelectionMethod(
        SpatialEventSelectionMethod):
    """This event selection method selects events within a spatial box in
    right-ascention and declination around a list of point-like source
    positions and performs an additional selection of events whose ang_err value
    is larger than the value of the provided function at a given psi value.
    """
    def __init__(
            self,
            shg_mgr,
            func,
            psi_floor=None,
            **kwargs):
        """Creates and configures a spatial box and psi func event selection
        method object.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the list of
            sources, i.e. the list of SourceModel instances.
        delta_angle : float
            The half-opening angle around the source for which events should
            get selected.
        psi_name : str | None
            The name of the data field that provides the psi value of the event.
            If set to ``None``, the psi value will be calculated automatically.
        func : callable
            The function that should get evaluated for each event. The call
            signature must be

                ``func(psi)``,

            where ``psi`` is the opening angle between the source and the event.
        psi_floor : float | None
            The psi func event selection is excluded for events having psi value
            below the ``psi_floor``. If None, set it to default 5 degrees.
        """
        super().__init__(
            shg_mgr=shg_mgr,
            **kwargs)

        self.func = func

        if psi_floor is None:
            psi_floor = np.deg2rad(5)
        self.psi_floor = psi_floor

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
        if not callable(f):
            raise TypeError(
                'The func property must be a callable instance! '
                f'Its current type is {classname(f)}.')
        self._func = f

    @property
    def psi_floor(self):
        """The psi func event selection is excluded for events having psi value
        below the `psi_floor`.
        """
        return self._psi_floor

    @psi_floor.setter
    def psi_floor(self, psi):
        psi = float_cast(
            psi,
            'The psi_floor property must be castable to type float!')
        self._psi_floor = psi

    def select_events(
            self,
            events,
            src_evt_idxs=None,
            ret_original_evt_idxs=False,
            tl=None):
        """Selects the events within the spatial box in right-ascention and
        declination and performs an additional selection of events whose ang_err
        value is larger than the value of the provided function at a given psi
        value.

        The solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec) is a
        function of declination, i.e. for a constant dOmega, the right-ascension
        value has to change with declination.

        Parameters
        ----------
        events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray that holds the event data.
            The following data fields must exist:

            ``'ra'`` : float
                The right-ascention of the event.
            ``'dec'`` : float
                The declination of the event.

        src_evt_idxs : 2-tuple of 1d ndarrays of ints | None
            The 2-element tuple holding the two 1d ndarrays of int of length
            N_values, specifying to which sources the given events belong to.
            If set to ``None`` all given events will be considered to for all
            sources.
        ret_original_evt_idxs : bool
            Flag if the original indices of the selected events should get
            returned as well.
        tl : instance of TimeLord | None
            The optional instance of TimeLord that should be used to collect
            timing information about this method.

        Returns
        -------
        selected_events : instance of DataFieldRecordArray
            The instance of DataFieldRecordArray holding only the selected
            events.
        (src_idxs, evt_idxs) : 1d ndarrays of ints
            The indices of the sources and the selected events.
        original_evt_idxs : 1d ndarray of ints
            The (N_selected_events,)-shaped numpy ndarray holding the original
            indices of the selected events, if ``ret_original_evt_idxs`` is set
            to ``True``.
        """
        if src_evt_idxs is None:
            n_sources = len(self._src_arr)
            n_events = len(events)
            src_idxs = np.repeat(np.arange(n_sources), n_events)
            evt_idxs = np.tile(np.arange(n_events), n_sources)
        else:
            (src_idxs, evt_idxs) = src_evt_idxs

        # Perform selection based on psi values.
        with TaskTimer(tl, 'ESM: Calculate psi values.'):
            psi = angular_separation(
                ra1=np.take(self._src_arr['ra'], src_idxs),
                dec1=np.take(self._src_arr['dec'], src_idxs),
                ra2=np.take(events['ra'], evt_idxs),
                dec2=np.take(events['dec'], evt_idxs),
            )

        with TaskTimer(tl, 'ESM: Create mask_psi.'):
            mask_psi = (
                (events['ang_err'][evt_idxs] >= self._func(psi)) |
                (psi < self.psi_floor)
            )

        with TaskTimer(tl, 'ESM: Create selected_events.'):
            # Have to define the shape argument in order to not truncate
            # the mask in case last events are not selected.
            mask_sky = scipy.sparse.csr_matrix(
                (mask_psi, (src_idxs, evt_idxs)),
                shape=(len(self._src_arr), len(events))
            ).toarray()
            mask = np.any(mask_sky, axis=0)

            # Using an integer indices array for data selection is several
            # factors faster than using a boolean array.
            selected_events_idxs = events.indices[mask]
            selected_events = events[selected_events_idxs]

        # Get final selected events indices.
        idxs = np.argwhere(mask_sky[:, mask])
        src_idxs = idxs[:, 0]
        evt_idxs = idxs[:, 1]

        if ret_original_evt_idxs:
            return (selected_events, (src_idxs, evt_idxs), selected_events_idxs)

        return (selected_events, (src_idxs, evt_idxs))
