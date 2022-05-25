# -*- coding: utf-8 -*-

"""Generate specific data files for tests.
"""

import numpy as np
import numpy.lib.recfunctions as np_rfn


def generate_testdata():
    exp_testdata_dtype = np.dtype(
        [('time', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('ang_err', '<f8'),
         ('log_energy', '<f8'), ('sin_dec', '<f8')]
    )

    mc_testdata_dtype = np.dtype(
        [('time', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('ang_err', '<f8'),
         ('log_energy', '<f8'), ('sin_dec', '<f8'), ('true_ra', '<f8'),
         ('true_dec', '<f8'), ('true_energy', '<f8'), ('mcweight', '<f8')]
    )

    grl_testdata_dtype = np.dtype([('run', '<i8'), ('start', '<f8'),
        ('stop', '<f8'), ('livetime', '<f8'), ('events', '<i8')])

    exp_testdata = np.array(
        [(58443.0, 2.72271553, 0.49221455, 2.99055817, 2.06154594, 0.00655059),
         (58443.5, 0.0912451, 1.1717919, 5.62599504, 2.74410951, 0.03066161),
         (58444.0, 1.87636917, 0.2423321, 3.85740203, 1.81258875, 0.0147304),
         (58444.5, 4.30905534e+00, -0.1928237, 1.42919624, 1.37737515, 0.00438893)],
        dtype=exp_testdata_dtype)

    mc_testdata = np.array(
        [(58443.0, 0.29918625, 0.33339297, 0.29918625, 1.90418929, 0.05866254, 2.77096718, 0.38677653, 0.35868533, 0.38677653),
         (58443.5, 3.80573909, 1.02692302, 3.80573909, 2.59771935, 0.00686186, 2.94901297, 3.79586657, 1.03187752, 3.79586657),
         (58444.0, 1.00158215, -0.00708026, 1.00158215, 1.56371607, 0.00682492, 3.0267893 , 1.02335589, 0.00740956, 1.02335589),
         (58444.5, 6.19181196, 0.67255195, 6.19181196, 2.24334827, 0.00754302, 2.83277749, 6.2037248 , 0.65695732, 6.2037248)],
        dtype=mc_testdata_dtype)

    livetime_testdata = np.array([[58443.0, 58443.25],
                                  [58443.5, 58443.75],
                                  [58444.0, 58444.25], 
                                  [58444.5, 58444.75]])
    
    # Generate events.
    rng = np.random.default_rng(0)
    n_events = 1000
    events = rng.random((n_events, 3), )
    # Emulate ra and dec.
    events[:, 0] = 2 * np.pi * events[:, 0]
    events[:, 1] = np.pi * (events[:, 1] - 0.5)

    # Emulate angular error.
    events[:, 2] = 2 * np.pi * events[:, 2]
    events_dtype = np.dtype(
        [('ra', '<f8'), ('dec', '<f8'), ('ang_err', '<f8')]
    )
    events = np_rfn.unstructured_to_structured(events, dtype=events_dtype)

    testdata = {
        'exp_testdata': exp_testdata,
        'mc_testdata': mc_testdata,
        'livetime_testdata': livetime_testdata,
        'events': events
    }
    return testdata

if(__name__ == '__main__'):
    testdata = generate_testdata()
    np.save('exp_testdata.npy', testdata.get('exp_testdata'))
    np.save('mc_testdata.npy', testdata.get('mc_testdata'))
    np.save('livetime_testdata.npy', testdata.get('livetime_testdata'))
