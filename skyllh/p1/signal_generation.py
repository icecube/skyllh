# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.py import (
    get_smallest_numpy_int_type,
    float_cast,
    int_cast,
    issequenceof,
)
from skyllh.core.utils.coords import (
    rotate_signal_events_on_sphere,
)
from skyllh.core.signal_generation import (
    SignalGenerationMethod,
)
from skyllh.core.source_model import (
    PointLikeSource,
)

from skyllh.core.sidereal_time import (SiderealTimeService)

def sources_to_recarray(sources):
    """Converts the sequence of PointLikeSource sources into a numpy
    structured array holding the information of the sources needed for the
    detector signal yield calculation.

    Parameters
    ----------
    sources : SourceModel | sequence of SourceModel
        The source model(s) containing the information of the source(s).

    Returns
    -------
    arr : instance of numpy.ndarray
        The generated (N_sources,)-shaped structured numpy ndarray holding
        the information for each source. This array contains the following
        fields:

            ``'dec'`` : float
                The declination of the source.
            ``'ra'`` : float
                The right-ascension of the source.

    """
    if isinstance(sources, PointLikeSource):
        sources = [sources]
    if not issequenceof(sources, PointLikeSource):
        raise TypeError(
            'The sources argument must be an instance or a sequence of '
            'instances of PointLikeSource!')

    arr_dtype = [
        ('dec', np.float64),
        ('ra', np.float64),
    ]

    arr = np.empty((len(sources),), dtype=arr_dtype)
    for (i, src) in enumerate(sources):
        arr['dec'][i] = src.dec
        arr['ra'][i] = src.ra

    return arr

def compute_alt_array(
        shg,
        detector_model,
        livetime,
        st_bin_width_deg
):
        st_service = SiderealTimeService(detector_model=detector_model,
                                         livetime=livetime, 
                                         st_bin_width_deg=st_bin_width_deg)
        
        src_recarray = sources_to_recarray(shg.source_list)
        (src_st_alt_arr, src_st_az_arr) = st_service.create_src_st_alt_array(
                src_array=src_recarray,
            )
        return (src_st_alt_arr, src_st_az_arr)

class PointLikeSourceSignalGenerationMethod(SignalGenerationMethod):
    """This class provides a signal generation method for point-like sources
    seen in the IceCube detector.
    """
    def __init__(
        self,
        band_deg_alt_range = 1.0,
        src_batch_size=128,
        energy_range=None,
        **kwargs
    ):
        """Constructs a new signal generation method instance for a point-like
        source detected with IceCube.

        Parameters
        ----------
        band_deg_alt_range : float
            The width of the alt band to take MC events from around a
            source altitude position in AltAz coordinates.
        energy_range : 2-element tuple of float | None
            The energy range from which to take MC events into account for
            signal event generation.
            If set to None, the entire energy range [0, +inf] is used.
        src_batch_size : int
            The source processing batch size used for the signal event flux
            calculation.
        """
        super().__init__(
            energy_range=energy_range,
            **kwargs)

        self.band_deg_alt_range = band_deg_alt_range
        self.src_batch_size = src_batch_size

        @property
        def band_deg_alt_range(self):
            """The width of the alt band to take MC events from around a
            source altitude position in AltAz coordinates.
            """
            return self._band_deg_alt_range

        @band_deg_alt_range.setter
        def band_deg_alt_range(self, v):
            v = float_cast(
                v,
                'The band_deg_alt_range property must be cast-able to type '
                'float!')
            self._band_deg_alt_range = v

        @property
        def src_batch_size(self):
            """The source processing batch size used for the signal event flux
            calculation.
            """
            return self._src_batch_size

        @src_batch_size.setter
        def src_batch_size(self, v):
            v = int_cast(
                v,
                'The src_batch_size property must be cast-able to type int!')
            self._src_batch_size = v

    def calc_source_signal_mc_event_flux(self, data_mc, shg):
        """Calculates the signal flux of each given MC event for each source
        hypothesis of the given source hypothesis group.

        Parameters
        ----------
        data_mc : numpy record ndarray
            The numpy record array holding the MC events of a dataset.
        shg : SourceHypoGroup instance
            The source hypothesis group, which defines the list of sources, and
            their flux model.

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
        indices = np.arange(
            0, len(data_mc),
            dtype=get_smallest_numpy_int_type((0, len(data_mc)))
        )
        n_sources = shg.n_sources

        # Get 1D array of source declination.
        src_dec = np.empty((n_sources,), dtype=np.float64)
        src_ra = np.empty((n_sources,), dtype=np.float64)
        for (k, source) in enumerate(shg.source_list):
            if not isinstance(source, PointLikeSource):
                raise TypeError(
                    'The source instance must be an instance of '
                    'PointLikeSource!')
            src_dec[k] = source.dec
            src_ra[k]  = source.ra
        
        data_mc_sin_true_alt = data_mc['sin_true_alt']
        data_mc_true_energy = data_mc['true_energy']

        alt = np.array([0.5969206])  # fake data

        # Calculate the source altitude bands and their solid angle.
        band_deg_alt_range = self.band_deg_alt_range  # self._band_deg_alt_range does not work!
        src_sin_alt_band_min = np.sin(np.deg2rad(np.rad2deg(alt) - band_deg_alt_range/2))
        src_sin_alt_band_max = np.sin(np.deg2rad(np.rad2deg(alt) + band_deg_alt_range/2))

        src_alt_band_omega = (
            2 * np.pi * (src_sin_alt_band_max - src_sin_alt_band_min)
        )

        # Get the flux model of this source hypo group (SHG).
        fluxmodel = shg.fluxmodel

        # Get the theoretical weights of all the sources of this SHG.
        src_weights = shg.get_source_weights()
        
        # Calculate conversion factor from the flux model unit into the internal
        # flux unit.
        to_internal_flux_unit =\
            fluxmodel.to_internal_flux_unit()
        
        # Select the events that belong to a given source.
        ev_idx_arr = np.empty(
            (0,),
            dtype=get_smallest_numpy_int_type((0, len(data_mc))))
        shg_src_idx_arr = np.empty(
            (0,),
            dtype=get_smallest_numpy_int_type((0, n_sources)))
        flux_arr = np.empty(
            (0,),
            dtype=np.float32)
        
        src_batch_size = self.src_batch_size  # self._src_batch_size does not work!
        n_batches = int(np.ceil(n_sources / src_batch_size))

        for bi in range(n_batches):
            src_start = bi*src_batch_size
            src_end = np.min([(bi+1)*src_batch_size, n_sources])
            bs = src_end - src_start

            src_slice = slice(src_start, src_end)

            # Create an event mask of shape (N_sources,N_events)
            ev_mask = np.logical_and(
                    (data_mc_sin_true_alt >=
                        src_sin_alt_band_min[src_slice][:, np.newaxis]),
                    (data_mc_sin_true_alt <=
                        src_sin_alt_band_max[src_slice][:, np.newaxis])
                )
            if self.energy_range is not None:
                ev_mask &= np.logical_and(
                    (data_mc_true_energy >= self.energy_range[0]),
                    (data_mc_true_energy <= self.energy_range[1])
                )

            print(src_sin_alt_band_min, src_sin_alt_band_max)
            print(np.count_nonzero(ev_mask))

            ev_idxs = np.tile(indices, bs)[ev_mask.ravel()]
            #print(ev_idxs)
            shg_src_idxs = bi*src_batch_size + np.repeat(
                    np.arange(bs),
                    ev_mask.sum(axis=1)
                )
            del ev_mask
            flux = (
                    fluxmodel(E=data_mc_true_energy[ev_idxs]).squeeze() *
                    to_internal_flux_unit /
                    src_alt_band_omega[shg_src_idxs]
                )
            if src_weights is not None:
                flux *= src_weights[shg_src_idxs]

            ev_idx_arr = np.append(ev_idx_arr, ev_idxs)
            shg_src_idx_arr = np.append(shg_src_idx_arr, shg_src_idxs)
            flux_arr = np.append(flux_arr, flux)

        return (ev_idx_arr, shg_src_idx_arr, flux_arr)
    

    def signal_event_post_sampling_processing(
            self,
            shg,
            shg_sig_events_meta,
            shg_sig_events,
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
        
        alt_value = np.array([0.5969206])  # fake data
        azi_value = np.array([2.41163])  # fake data

        # Get the unique source indices of that source hypo group.
        shg_src_idxs = np.unique(shg_sig_events_meta['shg_src_idx'])
        # Go through each (sampled) source.
        for shg_src_idx in shg_src_idxs:
            #source = shg.source_list[shg_src_idx]
            # Get the signal events of the source hypo group, that belong to the
            # source index.
            # Get the signal event of the alt
            shg_src_mask = shg_sig_events_meta['shg_src_idx'] == shg_src_idx
            shg_src_sig_events = shg_sig_events[shg_src_mask]
            n_sig = len(shg_src_sig_events)

            # Rotate the signal events to the source location.
            (ra, dec) = rotate_signal_events_on_sphere(
                src_ra=np.full(n_sig, azi_value),
                src_dec=np.full(n_sig, alt_value),
                evt_true_ra=shg_src_sig_events['true_ra'],
                evt_true_dec=shg_src_sig_events['true_dec'],
                evt_reco_ra=shg_src_sig_events['ra'],
                evt_reco_dec=shg_src_sig_events['dec']
            )

            shg_src_sig_events['ra'] = ra
            shg_src_sig_events['dec'] = dec
            shg_src_sig_events['sin_dec'] = np.sin(dec)

            shg_sig_events[shg_src_mask] = shg_src_sig_events

        return shg_sig_events