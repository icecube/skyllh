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

class PointLikeSourceSignalGenerationMethod(SignalGenerationMethod):
    """This class provides a signal generation method for point-like sources
    seen in the IceCube detector.
    """

    def __init__(
        self,
        detector_model,
        livetime,
        st_bin_width_deg,
        band_deg_alt_range,
        src_batch_size=128,
        energy_range=None,
        **kwargs
    ):
        """Constructs a new signal generation method instance for a point-like
        source detected with IceCube.

        Parameters
        ----------
        detector_model : instance of DetectorModel
            The instance of DetectorModel defining the detector for this
            detector signal yield.
        livetime : instance of Livetime
            The instance of Livetime defining the live-time of the dataset.
        st_bin_width_deg : float
            The sidereal time bin width in degree. This value must be in the
            range [0, 360].
        
        band_deg_alt_range : float
            The withd of the altitude bands in degree.
        source_list : list of instance of PointLikeSource
            The list of instance of PointLikeSource defining the sources
            for this detector signal yield.
        src_sin_dec_half_bandwidth : float
            The half-width of the sin(dec) band to take MC events from around a
            source. The default is sin(1deg), i.e. a 1deg half-bandwidth.
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

        self.src_batch_size = src_batch_size
        self.detector_model = detector_model
        self.livetime = livetime
        self.st_bin_width_deg = st_bin_width_deg
        self.band_deg_alt_range = band_deg_alt_range

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
        
        src_batch_size = self.src_batch_size
        n_batches = int(np.ceil(n_sources / src_batch_size))


        st_service = SiderealTimeService(detector_model=self.detector_model,
                                         livetime=self.livetime, 
                                         st_bin_width_deg=self.st_bin_width_deg)
        
        src_recarray = sources_to_recarray(shg.source_list)
        (src_st_alt_arr, _) = st_service.create_src_st_alt_array(
                src_array=src_recarray,
            )

        for bi in range(n_batches):
            src_start = bi*src_batch_size
            src_end = np.min([(bi+1)*src_batch_size, n_sources])
            bs = src_end - src_start

            src_slice = slice(src_start, src_end)
            
            for alt in src_st_alt_arr:
                # Create an event mask of shape (N_sources,N_events)
                src_sin_alt_band_min = np.sin(np.deg2rad(np.rad2deg(alt) - self.band_deg_alt_range/2))
                src_sin_alt_band_max = np.sin(np.deg2rad(np.rad2deg(alt) + self.band_deg_alt_range/2))

                # Calculate the solid angle of the declination band.
                src_alt_band_omega = (
                    2 * np.pi * (src_sin_alt_band_max - src_sin_alt_band_min)
                )
                
                ev_mask = np.logical_and(
                        (data_mc_sin_true_alt >=
                            src_sin_alt_band_min[src_slice][:, np.newaxis]),
                        (data_mc_sin_true_alt <=
                            src_sin_alt_band_max[src_slice][:, np.newaxis])
                    )
                #print(src_sin_alt_band_min, src_sin_alt_band_max)
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
                ev_idx_arr = np.append(ev_idx_arr, ev_idxs)
                shg_src_idx_arr = np.append(shg_src_idx_arr, shg_src_idxs)
                flux_arr = np.append(flux_arr, flux)

        return (ev_idx_arr, shg_src_idx_arr, flux_arr)
    
    def signal_event_post_sampling_processing(self):
        pass