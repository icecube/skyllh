# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.py import (
    get_smallest_numpy_int_type,
    float_cast
)
from skyllh.core.coords import rotate_spherical_vector
from skyllh.core.signal_generation import SignalGenerationMethod
from skyllh.physics.source import PointLikeSource
from skyllh.physics.flux import get_conversion_factor_to_internal_flux_unit


def source_sin_dec_shift_linear(x, w, L, U):
    """Calculates the shift of the sine of the source declination, in order to
    allow the construction of the source sine declination band with
    sin(dec_src) +/- w. This shift function, S(x), is implemented as a line
    with the following points:

        S(L) = w
        S((L+U)/2) = 0
        S(U) = -w

    Parameters
    ----------
    x : 1D numpy ndarray
        The sine of the source declination for each source.
    w : float
        The half size of the sin(dec)-window.
    L : float
        The lower value of the allowed sin(dec) range.
    U : float
        The upper value of the allowed sin(dec) range.

    Returns
    -------
    S : 1D numpy ndarray
        The sin(dec) shift of the sin(dec) values of the given sources, such
        that ``sin(dec_src) + S`` is the new sin(dec) of the source, and
        ``sin(dec_src) + S +/- w`` is always within the sin(dec) range [L, U].
    """
    x = np.atleast_1d(x)

    m = -2*w/(U-L)
    b = w*(L+U)/(U-L)
    S = m*x+b

    return S


def source_sin_dec_shift_cubic(x, w, L, U):
    """Calculates the shift of the sine of the source declination, in order to
    allow the construction of the source sine declination band with
    sin(dec_src) +/- w. This shift function, S(x), is implemented as a cubic
    function with the following points:

        S(L) = w
        S((L+U)/2) = 0
        S(U) = -w

    Parameters
    ----------
    x : 1D numpy ndarray
        The sine of the source declination for each source.
    w : float
        The half size of the sin(dec)-window.
    L : float
        The lower value of the allowed sin(dec) range.
    U : float
        The upper value of the allowed sin(dec) range.

    Returns
    -------
    S : 1D numpy ndarray
        The sin(dec) shift of the sin(dec) values of the given sources, such
        that ``sin(dec_src) + S`` is the new sin(dec) of the source, and
        ``sin(dec_src) + S +/- w`` is always within the sin(dec) range [L, U].
    """
    # TODO: Make sure that this function does what it's supposed to do
    x = np.atleast_1d(x)

    m = w / (x - 0.5*(L+U))**3
    S = m * np.power(x-0.5*(L+U),3)

    return S


class PointLikeSourceI3SignalGenerationMethod(SignalGenerationMethod):
    """This class provides a signal generation method for a point-like source
    seen in the IceCube detector.
    """
    def __init__(
        self,
        src_sin_dec_half_bandwidth=np.sin(np.radians(1)),
        src_sin_dec_shift_func=None,
        energy_range=None
    ):
        """Constructs a new signal generation method instance for a point-like
        source detected with IceCube.

        Parameters
        ----------
        src_sin_dec_half_bandwidth : float
            The half-width of the sin(dec) band to take MC events from around a
            source. The default is sin(1deg), i.e. a 1deg half-bandwidth.
        src_sin_dec_shift_func : callable | None
            The function that provides the source sin(dec) shift needed for
            constructing the source declination bands from where to draw
            monte-carlo events from. If set to None, the default function
            ``source_sin_dec_shift_linear`` will be used.
        energy_range : 2-element tuple of float | None
            The energy range from which to take MC events into account for
            signal event generation.
            If set to None, the entire energy range [0, +inf] is used.
        """
        super(PointLikeSourceI3SignalGenerationMethod, self).__init__(
            energy_range=energy_range)

        self.src_sin_dec_half_bandwidth = src_sin_dec_half_bandwidth

        if(src_sin_dec_shift_func is None):
            src_sin_dec_shift_func = source_sin_dec_shift_linear
        self.src_sin_dec_shift_func = src_sin_dec_shift_func

    @property
    def src_sin_dec_half_bandwidth(self):
        """The half-width of the sin(dec) band to take MC events from around a
        source.
        """
        return self._src_sin_dec_half_bandwidth
    @src_sin_dec_half_bandwidth.setter
    def src_sin_dec_half_bandwidth(self, v):
        v = float_cast(v, 'The src_sin_dec_half_bandwidth property must be '
            'castable to a float type!')
        self._src_sin_dec_half_bandwidth = v

    @property
    def src_sin_dec_shift_func(self):
        """The function that provides the source sin(dec) shift needed for
        constructing the source declination bands from where to draw
        monte-carlo events from.
        """
        return self._src_sin_dec_shift_func
    @src_sin_dec_shift_func.setter
    def src_sin_dec_shift_func(self, func):
        if(not callable(func)):
            raise TypeError('The src_sin_dec_shift_func property must be a '
                'callable object!')
        self._src_sin_dec_shift_func = func

    def _get_src_dec_bands(self, src_dec, max_sin_dec_range):
        """Calculates the minimum and maximum sin(dec) values for each source
        to use with a specified maximal sin(dec) range, which should get
        determined from the available MC data itself.

        Parameters
        ----------
        src_dec : 1D ndarray
            The declination values of the sources.
        max_sin_dec_range : 2-element tuple of floats
            The maximal sin(dec) range from where MC events are available.

        Returns
        -------
        src_sin_dec_band_min : (N_sources,)-shaped 1D ndarray
            The array holding the lower value of the sin(dec) band for each
            source.
        src_sin_dec_band_max : (N_sources,)-shaped 1D ndarray
            The array holding the upper value of the sin(dec) band for each
            source.
        src_dec_band_omega : (N_sources,)-shaped 1D ndarray
            The solid angle of the declination band for each source.
        """
        # Shift the source declination in order to be able to always create the
        # declination band via src_dec +/- half-bandwidth
        src_sin_dec = np.sin(src_dec)
        src_sin_dec += self._src_sin_dec_shift_func(
            src_sin_dec, self._src_sin_dec_half_bandwidth, *max_sin_dec_range)

        src_sin_dec_band_min = src_sin_dec - self._src_sin_dec_half_bandwidth
        src_sin_dec_band_max = src_sin_dec + self._src_sin_dec_half_bandwidth

        # Calculate the solid angle of the declination band.
        src_dec_band_omega = 2*np.pi*(src_sin_dec_band_max - src_sin_dec_band_min)

        return (src_sin_dec_band_min, src_sin_dec_band_max, src_dec_band_omega)

    def calc_source_signal_mc_event_flux(self, data_mc, shg):
        """Calculates the signal flux of each given MC event for each source
        hypothesis of the given source hypothesis group. The unit of the signal
        flux must be 1/(GeV cm^2 s sr).

        Parameters
        ----------
        data_mc : numpy record ndarray
            The numpy record array holding the MC events of a dataset.
        shg : SourceHypoGroup instance
            The source hypothesis group, which defines the list of sources, and
            their flux model.

        Returns
        -------
        indices_list : list of 1D ndarrays
            The list of event indices arrays specifying which MC events have
            been selected as signal candidate events for each source of the
            given source hypothesis group. Hence, the length of that list is the
            number of sources of the source hypothesis group. The length of the
            different 1D ndarrays is variable and depends on the source.
        flux_list : list of 1D ndarrays
            The list of 1D ndarrays holding the flux value of the selected
            signal candidate events. One array for each source of the given
            source hypothesis group. Hence, the length of that list is the
            number of sources of the source hypothesis group. The length of the
            different 1D ndarrays is variable and depends on the source.
        """
        indices = np.arange(
            0, len(data_mc),
            dtype=get_smallest_numpy_int_type((0, len(data_mc)))
        )
        n_sources = shg.n_sources

        # Get 1D array of source declination.
        src_dec = np.empty((n_sources,), dtype=np.float64)
        for (k, source) in enumerate(shg.source_list):
            if(not isinstance(source, PointLikeSource)):
                raise TypeError('The source instance must be an instance of '
                    'PointLikeSource!')
            src_dec[k] = source.dec

        data_mc_sin_true_dec = data_mc['sin_true_dec']
        data_mc_true_energy = data_mc['true_energy']

        # Calculate the source declination bands and their solid angle.
        max_sin_dec_range = (
            np.min(data_mc_sin_true_dec),
            np.max(data_mc_sin_true_dec)
        )
        (src_sin_dec_band_min, src_sin_dec_band_max, src_dec_band_omega) = self._get_src_dec_bands(src_dec, max_sin_dec_range)

        # Get the flux model of this source hypo group.
        fluxmodel = shg.fluxmodel

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)

        # Select the events that belong to a given source.
        indices_list = []
        flux_list = []

        for k in range(n_sources):
            # Create the sin(true_dec) range event mask for the source.
            src_event_mask = (
                (data_mc_sin_true_dec >= src_sin_dec_band_min[k]) &
                (data_mc_sin_true_dec <= src_sin_dec_band_max[k])
            )
            # Apply energy range cut if an energy range is defined.
            if(self.energy_range is not None):
                src_event_mask &= (
                    (data_mc_true_energy >= self.energy_range[0]) &
                    (data_mc_true_energy <= self.energy_range[1])
                )

            indices_list.append(indices[src_event_mask])
            flux = fluxmodel(data_mc_true_energy[src_event_mask])*toGeVcm2s / src_dec_band_omega[k]
            flux_list.append(flux)

        return (indices_list, flux_list)

    def signal_event_post_sampling_processing(
        self, shg, shg_sig_events_meta, shg_sig_events
    ):
        """Rotates the generated signal events to their source location for a
        given source hypothesis group.

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

            - 'shg_src_idx': int
                The source index within the source hypothesis group.

        shg_sig_events : numpy record ndarray
            The numpy record ndarray holding the generated signal events for
            the given source hypothesis group and in the format of the original
            MC events.

        Returns
        -------
        shg_sig_events : numpy record ndarray
            The numpy record ndarray with the processed MC signal events.
        """
        # Get the unique source indices of that source hypo group.
        shg_src_idxs = np.unique(shg_sig_events_meta['shg_src_idx'])
        # Go through each (sampled) source.
        for shg_src_idx in shg_src_idxs:
            source = shg.source_list[shg_src_idx]
            # Get the signal events of the source hypo group, that belong to the
            # source index.
            shg_src_mask = shg_sig_events_meta['shg_src_idx'] == shg_src_idx
            shg_src_sig_events = shg_sig_events[shg_src_mask]
            n_sig = len(shg_src_sig_events)

            # Rotate the signal events to the source location.
            (ra, dec) = rotate_spherical_vector(
                shg_src_sig_events['true_ra'], shg_src_sig_events['true_dec'],
                source.ra*np.ones(n_sig), source.dec*np.ones(n_sig),
                shg_src_sig_events['ra'], shg_src_sig_events['dec']
            )

            shg_src_sig_events['ra'] = ra
            shg_src_sig_events['dec'] = dec
            shg_src_sig_events['sin_dec'] = np.sin(dec)

            shg_sig_events[shg_src_mask] = shg_src_sig_events

        return shg_sig_events


class MultiPointLikeSourceI3SignalGenerationMethod(
        PointLikeSourceI3SignalGenerationMethod):
    """This class provides a signal generation method for a multiple point-like
    sources seen in the IceCube detector.
    """
    def __init__(
        self,
        src_sin_dec_half_bandwidth=np.sin(np.radians(1)),
        src_sin_dec_shift_func=None,
        energy_range=None,
        batch_size=200
    ):
        """Constructs a new signal generation method instance for a point-like
        source detected with IceCube.

        Parameters
        ----------
        src_sin_dec_half_bandwidth : float
            The half-width of the sin(dec) band to take MC events from around a
            source. The default is sin(1deg), i.e. a 1deg half-bandwidth.
        src_sin_dec_shift_func : callable | None
            The function that provides the source sin(dec) shift needed for
            constructing the source declination bands from where to draw
            monte-carlo events from. If set to None, the default function
            ``source_sin_dec_shift_linear`` will be used.
        energy_range : 2-element tuple of float | None
            The energy range from which to take MC events into account for
            signal event generation.
            If set to None, the entire energy range [0, +inf] is used.
        batch_size : int, optional
            Batch size for signal generation.
        """
        super(MultiPointLikeSourceI3SignalGenerationMethod, self).__init__(
                    src_sin_dec_half_bandwidth=src_sin_dec_half_bandwidth,
                    src_sin_dec_shift_func=src_sin_dec_shift_func,
                    energy_range=energy_range
                    )
        self.batch_size = batch_size

    def calc_source_signal_mc_event_flux(self, data_mc, shg):
        """Calculates the signal flux of each given MC event for each source
        hypothesis of the given source hypothesis group. The unit of the signal
        flux must be 1/(GeV cm^2 s sr).

        Parameters
        ----------
        data_mc : numpy record ndarray
            The numpy record array holding the MC events of a dataset.
        shg : SourceHypoGroup instance
            The source hypothesis group, which defines the list of sources, and
            their flux model.

        Returns
        -------
        ev_indices : 1D ndarray
            Event indices array specifying which MC events have been selected as
            signal candidate events for each source of the given source
            hypothesis group. The length of the 1D ndarray is variable and
            depends on the source.
        src_indices : 1D ndarray
            Source indices array specifying which source corresponds to the
            event in ev_indices array.
        flux_list : list of 1D ndarrays
            The list of 1D ndarrays holding the flux value of the selected
            signal candidate events. One array for each source of the given
            source hypothesis group. Hence, the length of that list is the
            number of sources of the source hypothesis group. The length of the
            different 1D ndarrays is variable and depends on the source.
        """
        indices = np.arange(
            0, len(data_mc),
            dtype=get_smallest_numpy_int_type((0, len(data_mc)))
        )
        n_sources = shg.n_sources

        # Get 1D array of source declination.
        src_dec = np.empty((n_sources,), dtype=np.float64)
        for (k, source) in enumerate(shg.source_list):
            if(not isinstance(source, PointLikeSource)):
                raise TypeError(
                    'The source instance must be an instance of '
                    'PointLikeSource!')
            src_dec[k] = source.dec

        data_mc_sin_true_dec = data_mc['sin_true_dec']
        data_mc_true_energy = data_mc['true_energy']

        # Calculate the source declination bands and their solid angle.
        max_sin_dec_range = (
            np.min(data_mc_sin_true_dec),
            np.max(data_mc_sin_true_dec)
        )
        (src_sin_dec_band_min, src_sin_dec_band_max, src_dec_band_omega) = self._get_src_dec_bands(src_dec, max_sin_dec_range)

        # Get the flux model and source weights of this source hypo group.
        fluxmodel = shg.fluxmodel
        src_weights = shg.source_weights

        # Calculate conversion factor from the flux model unit into the internal
        # flux unit GeV^-1 cm^-2 s^-1.
        toGeVcm2s = get_conversion_factor_to_internal_flux_unit(fluxmodel)

        # Select the events that belong to a given source.
        ev_indices = np.empty(
            (0,), dtype=get_smallest_numpy_int_type((0, len(data_mc))))
        src_indices = np.empty(
            (0,), dtype=get_smallest_numpy_int_type((0, n_sources)))
        flux = np.empty((0,), dtype='float32')

        n_batches = int(np.ceil(n_sources / float(self.batch_size)))

        for bi in range(n_batches):
            if(bi != n_batches-1):
                band_mask = np.logical_and(
                            (data_mc_sin_true_dec >=
                                src_sin_dec_band_min[bi*self.batch_size:(bi+1)*self.batch_size][:, np.newaxis]),
                            (data_mc_sin_true_dec <=
                                src_sin_dec_band_max[bi*self.batch_size:(bi+1)*self.batch_size][:, np.newaxis])
                            )
                if(self.energy_range is not None):
                    band_mask &= np.logical_and(
                        (data_mc_true_energy >= self.energy_range[0]),
                        (data_mc_true_energy <= self.energy_range[1]))

                ev_indi = np.tile(indices, self.batch_size)[band_mask.ravel()]
                src_indi = bi*self.batch_size + np.repeat(
                    np.arange(self.batch_size),
                    band_mask.sum(axis=1)
                )
                del band_mask
            else:
                n_final_batch = int(n_sources - bi*self.batch_size)
                band_mask = np.logical_and(
                            (data_mc_sin_true_dec >=
                                src_sin_dec_band_min[bi*self.batch_size:][:, np.newaxis]),
                            (data_mc_sin_true_dec <=
                                src_sin_dec_band_max[bi*self.batch_size:][:, np.newaxis])
                            )
                if(self.energy_range is not None):
                    band_mask &= np.logical_and(
                        (data_mc_true_energy >= self.energy_range[0]),
                        (data_mc_true_energy <= self.energy_range[1]))

                ev_indi = np.tile(indices, n_final_batch)[band_mask.ravel()]
                src_indi = bi*self.batch_size + np.repeat(
                    np.arange(n_final_batch),
                    band_mask.sum(axis=1)
                )
                del band_mask

            if(src_weights is None):
                fluxi = fluxmodel(data_mc_true_energy[ev_indi])*toGeVcm2s / src_dec_band_omega[src_indi]
            else:
                fluxi = src_weights[src_indi]*fluxmodel(data_mc_true_energy[ev_indi])*toGeVcm2s / src_dec_band_omega[src_indi]

            ev_indices = np.append(ev_indices, ev_indi)
            src_indices = np.append(src_indices, src_indi)
            flux = np.append(flux, fluxi)
        return (ev_indices, src_indices, flux)
