.. _dataset_creation:

*******************
i3 Dataset creation
*******************

Introduction
============

i3SkyLLH uses IceCube datasets for data analysis. The main data unit can be accessed using `data_samples` variable. It is a dictionary with all the available data samples, where each data sample provides a collection of datasets.

The dataset collection contains all the seasons as individual :class:`~skyllh.i3.dataset.I3Dataset` objects. The object names have to be unique and follow ``ICxx`` or ``ICxx, year`` naming scheme, where ``xx`` is a number of working IceCube strings.


Creating dataset
================

#. Initialize :class:`~skyllh.core.dataset.DatasetCollection` object.
#. Use :meth:`~skyllh.core.dataset.DatasetCollection.add_datasets` method to add :class:`~skyllh.i3.dataset.I3Dataset` objects to the dataset collection.
#. Add data preparation functions with :meth:`~skyllh.core.dataset.DatasetCollection.add_data_preparation` method.
#. Define binning with :meth:`~skyllh.core.dataset.DatasetCollection.define_binning` method.


Notes
=====

Following datasets do **not** have GRL file:

#. GammaRays5yr_GalPlane
#. GammaRays5yr_PointSrc
#. NorthernTracks_v001p00
#. NorthernTracks_v001p01
#. PointSourceTracks_v001p00
#. PointSourceTracks_v002p00
#. TransientTracks_v001p00
   
File names are following neutrino sources `documentation <https://wiki.icecube.wisc.edu/images/b/b6/Nu-sources-data-format.pdf>`_.
   
GRL file should contain following fields::

    ('run', 'start', 'stop', 'livetime', 'events')

Experimental :class:`~skyllh.i3.dataset.I3Dataset` data must contain following fields (additional fields are allowed)::

    ('ra', 'dec', 'ang_err', 'time', 'log_energy', 'sin_dec')

Monte Carlo :class:`~skyllh.i3.dataset.I3Dataset` data must contain following fields (additional fields are allowed)::

    ('true_ra', 'true_dec', 'true_energy', 'mcweight', 'sin_true_dec')

In i3SkyLLH experimental data fields are renamed to be consistant with PEP8::

    'logE':     'log_energy',
    'angErr':   'ang_err'

Monte Carlo data fields are renamed to::

    'logE':     'log_energy',
    'angErr':   'ang_err',
    'trueE':    'true_energy',
    'trueRa':   'true_ra',
    'trueDec':  'true_dec',
    'ow':       'mcweight'


Full code sample
================

.. literalinclude:: code_samples/dataset_sample.py