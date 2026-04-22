.. _tutorials_index:

*********
Tutorials
*********

The tutorials below walk through common SkyLLH analysis tasks using IceCube public data.
They progress from a basic steady-state point-source fit to more specialised techniques.

The tutorials support both the `10-year <https://doi.org/10.7910/DVN/VKL316>`_ and 14-year (TODO: add link) IceCube public point-source datasets.
To load respective datasets, import the appropriate module and select the desired detector configurations:

.. code-block:: python

    from skyllh.datasets.i3.PublicData_14y_ps import create_dataset_collection

    dsc = create_dataset_collection(cfg=cfg)
    datasets = dsc['IC40', 'IC59', 'IC79', 'IC86_I-XI']

.. code-block:: python

    from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection

    dsc = create_dataset_collection(cfg=cfg)
    datasets = dsc['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II_VII']

:doc:`fitting_a_source`
   Fit a steady point source (NGC 1068) using the IceCube 14-year public track data.
   Covers loading datasets, maximising the log-likelihood ratio, computing the test statistic,
   and deriving flux normalisations.

:doc:`fixed_spectral_index`
   Repeat the point-source fit with a fixed (non-free) spectral index.
   Also demonstrates converting between mean signal event counts and flux normalisations.

:doc:`sky_scan`
   Produce a 2-D test-statistic map by scanning source positions on a sky grid around a
   candidate source. Visualises best-fit position and confidence contours using Wilks' theorem.

:doc:`likelihood_scan`
   Scan the log-likelihood ratio over a 2-D grid of flux normalisation and spectral index.
   Produces confidence contours in the gamma-flux parameter space.

:doc:`p_value_method_comparison`
   Compare two methods for computing a local p-value from a test statistic: generating
   background-only trials versus fitting a truncated gamma distribution to the TS distribution.

:doc:`injecting_signal_events`
   Generate synthetic signal events within a user-defined energy window.
   Illustrates the distinction between the signal-injection energy range and the
   likelihood hypothesis energy range.

:doc:`smearing_matrix`
   Access and visualise IceCube instrument response functions: the 5-D smearing matrix
   and the effective area as a function of energy and declination.

:doc:`time_dependent_point_source`
   Perform a time-dependent point-source analysis.
   Uses the Expectation-Maximisation (EM) algorithm to fit the timing and duration of a
   neutrino flare (TXS 0506+056).

.. toctree::
    :maxdepth: 3
    :hidden:

    fitting_a_source
    fixed_spectral_index
    sky_scan
    likelihood_scan
    p_value_method_comparison
    injecting_signal_events
    smearing_matrix
    time_dependent_point_source
