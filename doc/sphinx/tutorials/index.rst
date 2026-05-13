.. _tutorials_index:

*********
Tutorials
*********

The tutorials below walk through common SkyLLH analysis tasks using IceCube public data.
They progress from a basic steady-state point-source fit to more specialised techniques.

The tutorials support both the `10-year <https://doi.org/10.7910/DVN/VKL316>`_ (IceTracks-DR1) and 14-year (TODO: add link) (IceTracks-DR2) IceCube public point-source datasets. They are automatically downloaded from `dataverse.harvard.edu <https://dataverse.harvard.edu/dataverse/icecube>`_ to a local cache directory (``~/.skyllh/cache``). To use custom dataset locations, set the ``cfg['repository']['base_path']`` to the desired path.

To load respective datasets:

.. code-block:: python

    import skyllh
    from skyllh.core.config import Config

    cfg = Config()

    dsc = create_dataset_collection(cfg=cfg)
    datasets = skyllh.create_datasets('IceTracks-DR2', cfg=cfg)

.. code-block:: python

    datasets = skyllh.create_datasets('IceTracks-DR1', cfg=cfg)

Additional information about the IceCube public datasets can be found in the following references:
- IceTracks-DR1: `IceCube Collaboration, "IceCube Data for Neutrino Point-Source Searches Years 2008-2018", arXiv:2101.09836 (2021) <https://arxiv.org/abs/2101.09836>`_
- IceTracks-DR2: TBD

We provide an incomplete list of tutorials below. They are meant to illustrate how to perform a time-integrated point-source analysis using SkyLLH, but they are not exhaustive. We encourage users to explore the documentation and contribute additional tutorials covering other analysis types and techniques.

:doc:`fitting_a_source`
   Fit a steady point source (NGC 1068) using both the IceCube 10-year and 14-year public track data.
   Covers loading datasets, maximising the log-likelihood ratio, computing the test statistic,
   and deriving flux normalisations.

:doc:`dataset_collections`
   Introduces the concept of dataset collections, which are used to manage multiple datasets in a unified way.
   Covers loading datasets, inspecting available datasets, and accessing individual datasets.

:doc:`fixed_spectral_index`
   Repeat the point-source fit with a fixed (non-free) spectral index.
   Also demonstrates converting between mean signal event counts and flux normalisations.

:doc:`setting_an_energy_range`
   Set a user-defined energy range for the signal injection.
   Illustrates the distinction between the signal-injection energy range and the
   likelihood hypothesis energy range.

:doc:`sky_scan`
   Produce a 2-D test-statistic map by scanning source positions on a sky grid around a
   candidate source. Visualises best-fit position and confidence contours using Wilks' theorem.

:doc:`likelihood_scan`
   Scan the log-likelihood ratio over a 2-D grid of flux normalisation and spectral index.
   Produces confidence contours in the gamma-flux parameter space.

:doc:`p_value_method_comparison`
   Compare two methods for computing a local p-value from a test statistic: generating
   background-only trials versus fitting a truncated gamma distribution to the TS distribution.

:doc:`sensitivity_study`
   Estimate sensitivity and discovery potential for a point source.
   Demonstrates the analysis construction and how to convert signal counts to flux.

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
    dataset_collections
    fixed_spectral_index
    setting_an_energy_range
    sky_scan
    likelihood_scan
    p_value_method_comparison
    sensitivity_study
    smearing_matrix
    time_dependent_point_source
