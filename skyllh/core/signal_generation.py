# -*- coding: utf-8 -*-

import abc

from skyllh.core.py import (
    issequence,
    float_cast,
)


class SignalGenerationMethod(
        object,
        metaclass=abc.ABCMeta
):
    """This is a base class for a source and detector specific signal generation
    method, that calculates the source flux for a given monte-carlo event, which
    is needed to calculate the MC event weights for the signal generator.
    """

    def __init__(
            self,
            energy_range,
            **kwargs,
    ):
        """Constructs a new signal generation method instance.

        Parameters
        ----------
        energy_range : 2-element tuple of float | None
            The energy range from which to take MC events into account for
            signal event generation.
            If set to None, the entire energy range [0, +inf] is used.
        """
        super().__init__(**kwargs)

        self.energy_range = energy_range

    @property
    def energy_range(self):
        """The 2-element tuple of floats holding the energy range from which to
        take MC events into account for signal event generation.
        """
        return self._energy_range

    @energy_range.setter
    def energy_range(self, r):
        if r is not None:
            if not issequence(r):
                raise TypeError(
                    'The energy_range property must be a sequence!')
            if len(r) != 2:
                raise ValueError(
                    'The energy_range property must be a sequence of 2 '
                    'elements!')
            r = (
                float_cast(
                    r[0],
                    'The first element of the energy_range '
                    'sequence must be castable to type float!'),
                float_cast(
                    r[1],
                    'The second element of the energy_range '
                    'sequence must be castable to type float!')
            )
        self._energy_range = r

    @abc.abstractmethod
    def calc_source_signal_mc_event_flux(
            self,
            data_mc,
            shg,
    ):
        """This method is supposed to calculate the signal flux of each given
        MC event for each source hypothesis of the given source hypothesis
        group.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The numpy record array holding all the MC events.
        shg : instance of SourceHypoGroup
            The source hypothesis group instance, which defines the list of
            sources, and their flux model.

        Returns
        -------
        ev_idx_arr : ndarray
            The (N_selected_signal_events,)-shaped 1D ndarray holding the index
            of the MC event.
        shg_src_idx_arr : ndarray
            The (N_selected_signal_events,)-shaped 1D ndarray holding the index
            of the source within the given source hypothesis group for each
            signal candidate event.
        flux_arr : ndarray
            The (N_selected_signal_events,)-shaped 1D ndarray holding the flux
            value of each signal candidate event.
        """
        pass

    def signal_event_post_sampling_processing(
        self,
        shg,
        shg_sig_events_meta,
        shg_sig_events,
    ):
        """This method should be reimplemented by the derived class if there
        is some processing needed after the MC signal events have been sampled
        from the global MC data.

        Parameters
        ----------
        shg : SourceHypoGroup instance
            The source hypothesis group instance holding the sources and their
            locations.
        shg_sig_events_meta : numpy record ndarray
            The numpy record ndarray holding meta information about the
            generated signal events for the given source hypothesis group.
            The length of this array must be the same as shg_sig_events.
            It needs to contain the following data fields:

                shg_src_idx : int
                    The source index within the source hypothesis group.

        shg_sig_events : numpy record ndarray
            The numpy record ndarray holding the generated signal events for
            the given source hypothesis group and in the format of the original
            MC events.

        Returns
        -------
        shg_sig_events : numpy record array
            The processed signal events. In the default implementation of this
            method this is just the shg_sig_events input array.
        """
        return shg_sig_events
