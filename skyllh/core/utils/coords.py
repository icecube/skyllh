# -*- coding: utf-8 -*-

import numpy as np


def rotate_spherical_vector(ra1, dec1, ra2, dec2, ra3, dec3):
    """Calculates the rotation matrix R to rotate the spherical vector
    (ra1,dec1) onto the direction (ra2,dec2), and performs this rotation on the
    spherical vector (ra3,dec3).

    In practice (ra1,dec1) refers to the true location of a MC event,
    (ra2,dec2) the true location of the signal source, and (ra3,dec3) the
    reconstructed location of the MC event, which should get rotated according
    to the rotation of the two true directions.
    """
    # Make sure, the inputs are 1D arrays.
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert (len(ra1) == len(dec1) ==
            len(ra2) == len(dec2) ==
            len(ra3) == len(dec3)
           ), 'All input argument arrays must be of the same length!'

    N_event = len(ra1)

    # Calculate the space angle alpha between vector 1 and vector 2, and
    # correct for possible rounding erros.
    cos_alpha = (np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2)
                 + np.sin(dec1) * np.sin(dec2))
    cos_alpha[cos_alpha >  1] =  1
    cos_alpha[cos_alpha < -1] = -1
    alpha = np.arccos(cos_alpha)

    # Define the three 3D-vectors in spherical coordinate system. Each vector
    # is a (N_event,3)-shaped 2D array.
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T

    # Calculate the normalized rotation axis vector, nrot. nrot is a
    # (N_event,3)-shaped ndarray.
    nrot = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nrot**2, axis=1))
    nrot[norm > 0] /= norm[np.newaxis, norm > 0].T

    # Define the diagonal 3D unit matrix.
    one = np.diagflat(np.ones(3))

    # Calculate the rotation matrix R_i for each event i and perform the
    # rotation on vector 3 for each event.
    vec = np.empty((N_event,3), dtype=np.float64)

    sin_alpha = np.sin(alpha)
    twopi = 2*np.pi
    # Remap functions to avoid Python's (.)-resolution millions of times.
    (np_outer, np_dot, np_roll, np_diag, np_T) = (
     np.outer, np.dot, np.roll, np.diag, np.transpose)
    for i in range(N_event):
        cos_alpha_i = cos_alpha[i]
        nrot_i = nrot[i]
        nrotTnrot_i = np_outer(nrot_i, nrot_i)

        # Calculate cross product matrix, nrotx_i:
        # A[ij] = x_i * y_j - y_i * x_j
        skv = np_roll(np_roll(np_diag(nrot_i), shift=1, axis=1), shift=-1, axis=0)
        nrotx_i = skv - np_T(skv)

        # Calculate rotation matrix, R_i.
        R_i = ((1. - cos_alpha_i) * nrotTnrot_i
               + one*cos_alpha_i
               + sin_alpha[i] * nrotx_i)
        vec[i] = np_dot(R_i, np_T(vec3[i]))

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    ra += np.where(ra < 0., twopi, 0.)
    dec = np.arcsin(vec[:, 2])

    return (ra, dec)
