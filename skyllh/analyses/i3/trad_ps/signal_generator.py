import numpy as np
from copy import deepcopy
import os.path

from skyllh.physics.flux import FluxModel
from skyllh.analyses.i3.trad_ps.utils import load_smearing_histogram


class signal_injector(object):
    r"""
    """

    def __init__(
        self,
        name: str,
        declination: float,
        right_ascension: float,
        flux_model: FluxModel,
        data_path="/home/mwolf/projects/publicdata_ps/icecube_10year_ps/irfs"
    ):
        r"""
        Parameters
        ----------
        - name : str
        Dataset identifier. Must be one among:
        ['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II'].

        - declination : float
        Source declination in degrees.

        - right_ascension : float
        Source right ascension in degrees.

        - flux_model : FluxModel
        Instance of the `FluxModel` class.

        - data_path : str
        Path to the smearing matrix data.
        """

        self.flux_model = flux_model
        self.dec = declination
        self.ra = right_ascension

        (
            self.histogram,
            self.true_e_bin_edges,
            self.true_dec_bin_edges,
            self.reco_e_lower_edges,
            self.reco_e_upper_edges,
            self.psf_lower_edges,
            self.psf_upper_edges,
            self.ang_err_lower_edges,
            self.ang_err_upper_edges
        ) = load_smearing_histogram(os.path.join(data_path, f"{name}_smearing.csv"))

        # Find the declination bin
        if(self.dec < self.true_dec_bin_edges[0] or self.dec > self.true_dec_bin_edges[-1]):
            raise ValueError("NotImplemented")
        self.dec_idx = np.digitize(self.dec, self.true_dec_bin_edges) - 1

    @staticmethod
    def _get_bin_centers(low_edges, high_edges):
        r"""Given an array of lower bin edges and an array of upper bin edges,
        returns the corresponding bin centers.
        """
        # bin_edges = np.union1d(low_edges, high_edges)
        # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_centers = 0.5 * (low_edges + high_edges)
        return bin_centers

    def get_weighted_marginalized_pdf(
        self, true_e_idx, reco_e_idx=None, psf_idx=None
    ):
        r"""Get the reconstructed muon energy pdf for a specific true neutrino
        energy weighted with the assumed flux model.
        The function returns both the bin center values and the pdf values,
        which might be useful for plotting.
        If no pdf is given for the assumed true neutrino energy, returns None.
        """

        if reco_e_idx is None:
            # Get the marginalized distribution of the reconstructed energy
            # for a given (true energy, true declination) bin.
            pdf = deepcopy(self.histogram[true_e_idx, self.dec_idx, :])
            pdf = np.sum(pdf, axis=(-2, -1))
            label = "reco_e"
        elif psf_idx is None:
            # Get the marginalized distribution of the neutrino-muon opening
            # angle for a given (true energy, true declination, reco energy)
            # bin.
            pdf = deepcopy(
                self.histogram[true_e_idx, self.dec_idx, reco_e_idx, :]
            )
            pdf = np.sum(pdf, axis=-1)
            label = "psf"
        else:
            # Get the marginalized distribution of the neutrino-muon opening
            # angle for a given
            # (true energy, true declination, reco energy, psi) bin.
            pdf = deepcopy(
                self.histogram[true_e_idx, self.dec_idx,
                               reco_e_idx, psf_idx, :]
            )
            label = "ang_err"

        # Check whether there is no pdf in the table for this neutrino energy.
        if np.sum(pdf) == 0:
            return None, None, None, None, None

        if label == "reco_e":
            # Get the reco energy bin centers.
            lower_bin_edges = (
                self.reco_e_lower_edges[true_e_idx, self.dec_idx, :]
            )
            upper_bin_edges = (
                self.reco_e_upper_edges[true_e_idx, self.dec_idx, :]
            )
            bin_centers = self._get_bin_centers(
                lower_bin_edges, upper_bin_edges
            )

        elif label == "psf":
            lower_bin_edges = (
                self.psf_lower_edges[true_e_idx, self.dec_idx, reco_e_idx, :]
            )
            upper_bin_edges = (
                self.psf_upper_edges[true_e_idx, self.dec_idx, reco_e_idx, :]
            )

        elif label == "ang_err":
            lower_bin_edges = (
                self.ang_err_lower_edges[
                    true_e_idx, self.dec_idx, reco_e_idx, psf_idx, :
                ]
            )
            upper_bin_edges = (
                self.ang_err_upper_edges[
                    true_e_idx, self.dec_idx, reco_e_idx, psf_idx, :
                ]
            )

        bin_centers = self._get_bin_centers(
            lower_bin_edges, upper_bin_edges
        )
        bin_width = upper_bin_edges - lower_bin_edges

        # Re-normalize in case some bins were cut.
        pdf /= (np.sum(pdf * bin_width))

        return lower_bin_edges, upper_bin_edges, bin_centers, bin_width, pdf

    def _get_reconstruction_from_histogram(
        self, rs, idxs, value=None, bin_centers=None
    ):
        if value is not None:
            if bin_centers is None:
                raise RuntimeError("NotImplemented.")
            value_idx = np.argmin(abs(value - bin_centers))
            idxs[idxs.index(None)] = value_idx

        (low_edges, up_edges, new_bin_centers, bin_width, hist) = (
            self.get_weighted_marginalized_pdf(idxs[0], idxs[1], idxs[2])
        )
        if low_edges is None:
            return None, None, None, None
        reco_bin = rs.choice(new_bin_centers, p=(hist * bin_width))
        reco_idx = np.argmin(abs(reco_bin - new_bin_centers))
        reco_value = np.random.uniform(low_edges[reco_idx], up_edges[reco_idx])

        return reco_value, reco_bin, new_bin_centers, idxs

    def circle_parametrization(self, rs, psf):
        psf = np.atleast_1d(psf)
        # Transform everything in radians and convert the source declination
        # to source zenith angle
        a = np.radians(psf)
        b = np.radians(90 - self.dec)
        c = np.radians(self.ra)

        # Random rotation angle for the 2D circle
        t = rs.uniform(0, 2*np.pi, size=len(psf))

        # Parametrize the circle
        x = (
            (np.sin(a)*np.cos(b)*np.cos(c)) * np.cos(t) +
            (np.sin(a)*np.sin(c)) * np.sin(t) -
            (np.cos(a)*np.sin(b)*np.cos(c))
        )
        y = (
            -(np.sin(a)*np.cos(b)*np.sin(c)) * np.cos(t) +
            (np.sin(a)*np.cos(c)) * np.sin(t) +
            (np.cos(a)*np.sin(b)*np.sin(c))
        )
        z = (
            (np.sin(a)*np.sin(b)) * np.cos(t) +
            (np.cos(a)*np.cos(b))
        )

        # Convert back to right ascension and declination
        # This is to distinguish between diametrically opposite directions.
        zen = np.arccos(z)
        azi = np.arctan2(y, x)

        return (np.degrees(np.pi - azi), np.degrees(np.pi/2 - zen))

    def get_log_e_pdf(self, log_true_e_idx):
        if log_true_e_idx == -1:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, self.dec_idx]
        pdf = np.sum(pdf, axis=(-2, -1))

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the reco energy bin edges and widths.
        lower_bin_edges = self.reco_e_lower_edges[
            log_true_e_idx, self.dec_idx
        ]
        upper_bin_edges = self.reco_e_upper_edges[
            log_true_e_idx, self.dec_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf /= (np.sum(pdf * bin_widths))

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def get_psi_pdf(self, log_true_e_idx, log_e_idx):
        if log_true_e_idx == -1 or log_e_idx == -1:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, self.dec_idx, log_e_idx]
        pdf = np.sum(pdf, axis=-1)

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the PSI bin edges and widths.
        lower_bin_edges = self.psf_lower_edges[
            log_true_e_idx, self.dec_idx, log_e_idx
        ]
        upper_bin_edges = self.psf_upper_edges[
            log_true_e_idx, self.dec_idx, log_e_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf /= (np.sum(pdf * bin_widths))

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def get_ang_err_pdf(self, log_true_e_idx, log_e_idx, psi_idx):
        if log_true_e_idx == -1 or log_e_idx == -1 or psi_idx == -1:
            return (None, None, None, None)

        pdf = self.histogram[log_true_e_idx, self.dec_idx, log_e_idx, psi_idx]

        if np.sum(pdf) == 0:
            return (None, None, None, None)

        # Get the ang_err bin edges and widths.
        lower_bin_edges = self.ang_err_lower_edges[
            log_true_e_idx, self.dec_idx, log_e_idx, psi_idx
        ]
        upper_bin_edges = self.ang_err_upper_edges[
            log_true_e_idx, self.dec_idx, log_e_idx, psi_idx
        ]
        bin_widths = upper_bin_edges - lower_bin_edges

        # Normalize the PDF.
        pdf = pdf / np.sum(pdf * bin_widths)

        return (pdf, lower_bin_edges, upper_bin_edges, bin_widths)

    def get_log_e_from_log_true_e_idxs(self, rs, log_true_e_idxs):
        n_evt = len(log_true_e_idxs)
        log_e_idx = np.empty((n_evt,), dtype=int)
        log_e = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            b_size = np.count_nonzero(m)
            (pdf, low_bin_edges, up_bin_edges,
             bin_widths) = self.get_log_e_pdf(b_log_true_e_idx)
            if pdf is None:
                log_e_idx[m] = -1
                log_e[m] = np.nan
                continue

            b_log_e_idx = rs.choice(
                np.arange(len(pdf)),
                p=(pdf * bin_widths),
                size=b_size)
            b_log_e = rs.uniform(
                low_bin_edges[b_log_e_idx],
                up_bin_edges[b_log_e_idx],
                size=b_size)

            log_e_idx[m] = b_log_e_idx
            log_e[m] = b_log_e

        return (log_e_idx, log_e)

    def get_psi_from_log_true_e_idxs_and_log_e_idxs(
            self, rs, log_true_e_idxs, log_e_idxs):
        if(len(log_true_e_idxs) != len(log_e_idxs)):
            raise ValueError('The lengths of log_true_e_idxs '
                             'and log_e_idxs must be equal!')

        n_evt = len(log_true_e_idxs)
        psi_idx = np.empty((n_evt,), dtype=int)
        psi = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            bb_unique_log_e_idxs = np.unique(log_e_idxs[m])
            for bb_log_e_idx in bb_unique_log_e_idxs:
                mm = m & (log_e_idxs == bb_log_e_idx)
                bb_size = np.count_nonzero(mm)
                (pdf, low_bin_edges, up_bin_edges, bin_widths) = (
                    self.get_psi_pdf(b_log_true_e_idx, bb_log_e_idx)
                )
                if pdf is None:
                    psi_idx[mm] = -1
                    psi[mm] = np.nan
                    continue

                bb_psi_idx = rs.choice(
                    np.arange(len(pdf)),
                    p=(pdf * bin_widths),
                    size=bb_size)
                bb_psi = rs.uniform(
                    low_bin_edges[bb_psi_idx],
                    up_bin_edges[bb_psi_idx],
                    size=bb_size)

                psi_idx[mm] = bb_psi_idx
                psi[mm] = bb_psi

        return (psi_idx, psi)

    def get_ang_err_from_log_true_e_idxs_and_log_e_idxs_and_psi_idxs(
            self, rs, log_true_e_idxs, log_e_idxs, psi_idxs):
        if (len(log_true_e_idxs) != len(log_e_idxs)) and\
           (len(log_e_idxs) != len(psi_idxs)):
            raise ValueError('The lengths of log_true_e_idxs, '
                             'log_e_idxs, and psi_idxs must be equal!')

        n_evt = len(log_true_e_idxs)
        ang_err = np.empty((n_evt,), dtype=np.double)

        unique_log_true_e_idxs = np.unique(log_true_e_idxs)
        for b_log_true_e_idx in unique_log_true_e_idxs:
            m = log_true_e_idxs == b_log_true_e_idx
            bb_unique_log_e_idxs = np.unique(log_e_idxs[m])
            for bb_log_e_idx in bb_unique_log_e_idxs:
                mm = m & (log_e_idxs == bb_log_e_idx)
                bbb_unique_psi_idxs = np.unique(psi_idxs[mm])
                for bbb_psi_idx in bbb_unique_psi_idxs:
                    mmm = mm & (psi_idxs == bbb_psi_idx)
                    bbb_size = np.count_nonzero(mmm)
                    (pdf, low_bin_edges, up_bin_edges, bin_widths) = (
                        self.get_ang_err_pdf(
                            b_log_true_e_idx, bb_log_e_idx, bbb_psi_idx)
                    )
                    if pdf is None:
                        ang_err[mmm] = np.nan
                        continue

                    bbb_ang_err_idx = rs.choice(
                        np.arange(len(pdf)),
                        p=(pdf * bin_widths),
                        size=bbb_size)
                    bbb_ang_err = rs.uniform(
                        low_bin_edges[bbb_ang_err_idx],
                        up_bin_edges[bbb_ang_err_idx],
                        size=bbb_size)

                    ang_err[mmm] = bbb_ang_err

        return ang_err

    def _generate_fast_n_events(self, rs, n_events):
        # Initialize the output:
        out_dtype = [
            ('log_true_e', np.double),
            ('log_e', np.double),
            ('psi', np.double),
            ('ra', np.double),
            ('dec', np.double),
            ('ang_err', np.double),
        ]
        events = np.empty((n_events,), dtype=out_dtype)

        # Determine the true energy range for which log_e PDFs are available.
        m = np.sum(
            (self.reco_e_upper_edges[:, self.dec_idx] -
             self.reco_e_lower_edges[:, self.dec_idx] > 0),
            axis=1) != 0
        min_log_true_e = np.min(self.true_e_bin_edges[:-1][m])
        max_log_true_e = np.max(self.true_e_bin_edges[1:][m])

        # First draw a true neutrino energy from the hypothesis spectrum.
        log_true_e = np.log10(self.flux_model.get_inv_normed_cdf(
            rs.uniform(size=n_events),
            E_min=10**min_log_true_e,
            E_max=10**max_log_true_e
        ))

        events['log_true_e'] = log_true_e

        log_true_e_idxs = (
            np.digitize(log_true_e, bins=self.true_e_bin_edges) - 1
        )
        # Get reconstructed energy given true neutrino energy.
        (log_e_idxs, log_e) = self.get_log_e_from_log_true_e_idxs(
            rs, log_true_e_idxs)
        events['log_e'] = log_e

        # Get reconstructed psi given true neutrino energy and reconstructed energy.
        (psi_idxs, psi) = self.get_psi_from_log_true_e_idxs_and_log_e_idxs(
            rs, log_true_e_idxs, log_e_idxs)
        events['psi'] = psi

        # Get reconstructed ang_err given true neutrino energy, reconstructed energy,
        # and psi.
        ang_err = self.get_ang_err_from_log_true_e_idxs_and_log_e_idxs_and_psi_idxs(
            rs, log_true_e_idxs, log_e_idxs, psi_idxs)
        events['ang_err'] = ang_err

        # Convert the psf into a set of (r.a. and dec.)
        (ra, dec) = self.circle_parametrization(rs, psi)
        events['ra'] = ra
        events['dec'] = dec

        return events

    def generate_fast(
            self, n_events, seed=1):
        rs = np.random.RandomState(seed)

        events = None
        n_evt_generated = 0
        while n_evt_generated != n_events:
            n_evt = n_events - n_evt_generated

            events_ = self._generate_fast_n_events(rs, n_evt)

            # Cut events that failed to be generated due to missing PDFs.
            m = np.invert(
                np.isnan(events_['log_e']) |
                np.isnan(events_['psi']) |
                np.isnan(events_['ang_err'])
            )
            events_ = events_[m]

            n_evt_generated += len(events_)
            if events is None:
                events = events_
            else:
                events = np.concatenate((events, events_))

        return events

    def _generate_n_events(self, rs, n_events):

        if not isinstance(n_events, int):
            raise TypeError("The number of events must be an integer.")

        if n_events < 0:
            raise ValueError("The number of events must be positive!")

        # Initialize the output:
        out_dtype = [
            ('log_true_e', np.double),
            ('log_e', np.double),
            ('psi', np.double),
            ('ra', np.double),
            ('dec', np.double),
            ('ang_err', np.double),
        ]

        if n_events == 0:
            print("Warning! Zero events are being generated")
            return np.array([], dtype=out_dtype)

        events = np.empty((n_events, ), dtype=out_dtype)

        # Determine the true energy range for which log_e PDFs are available.
        m = np.sum(
            (self.reco_e_upper_edges[:, self.dec_idx] -
             self.reco_e_lower_edges[:, self.dec_idx] > 0),
            axis=1) != 0
        min_log_true_e = np.min(self.true_e_bin_edges[:-1][m])
        max_log_true_e = np.max(self.true_e_bin_edges[1:][m])

        # First draw a true neutrino energy from the hypothesis spectrum.
        true_energies = np.log10(self.flux_model.get_inv_normed_cdf(
            rs.uniform(size=n_events),
            E_min=10**min_log_true_e,
            E_max=10**max_log_true_e
        ))

        true_e_idx = (
            np.digitize(true_energies, bins=self.true_e_bin_edges) - 1
        )

        for i in range(n_events):
            # Get a reconstructed energy according to P(E_reco | E_true)
            idxs = [true_e_idx[i], None, None]

            reco_energy, reco_e_bin, reco_e_bin_centers, idxs = (
                self._get_reconstruction_from_histogram(rs, idxs)
            )
            if reco_energy is not None:
                # Get an opening angle according to P(psf | E_true,E_reco).
                psf, psf_bin, psf_bin_centers, idxs = (
                    self._get_reconstruction_from_histogram(
                        rs, idxs, reco_e_bin, reco_e_bin_centers
                    )
                )

                if psf is not None:
                    # Get an angular error according to P(ang_err | E_true,E_reco,psf).
                    ang_err, ang_err_bin, ang_err_bin_centers, idxs = (
                        self._get_reconstruction_from_histogram(
                            rs, idxs, psf_bin, psf_bin_centers
                        )
                    )

                    # Convert the psf set of (r.a. and dec.)
                    ra, dec = self.circle_parametrization(rs, psf)

                    events[i] = (true_energies[i], reco_energy,
                                 psf, ra, dec, ang_err)
                else:
                    events[i] = (true_energies[i], reco_energy,
                                 np.nan, np.nan, np.nan, np.nan)
            else:
                events[i] = (true_energies[i], np.nan,
                             np.nan, np.nan, np.nan, np.nan)

        return events

    def generate(self, n_events, seed=1):
        rs = np.random.RandomState(seed)

        events = None
        n_evt_generated = 0
        while n_evt_generated != n_events:
            n_evt = n_events - n_evt_generated

            events_ = self._generate_n_events(rs, n_evt)

            # Cut events that failed to be generated due to missing PDFs.
            m = np.invert(
                np.isnan(events_['log_e']) |
                np.isnan(events_['psi']) |
                np.isnan(events_['ang_err'])
            )
            events_ = events_[m]
            n_evt_generated += len(events_)
            if events is None:
                events = events_
            else:
                events = np.concatenate((events, events_))

        return events
