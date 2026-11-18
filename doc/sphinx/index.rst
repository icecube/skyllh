.. SkyLLH

SkyLLH documentation
====================

The SkyLLH framework is an open-source Python-based package licensed under the
GPLv3 license. It provides a modular framework for implementing custom
likelihood functions and executing log-likelihood ratio hypothesis tests.
The idea is to provide a class structure tied to the mathematical objects of the
likelihood functions, rather than to entire abstract likelihood models.

The math formalism used in SkyLLH is described in the
math formalism `document <https://icecube.github.io/skyllh/user_manual.pdf>`_.

See :ref:`tutorials_index` section for hands-on examples using IceCube's public datasets.

v26.2.0 Release Notes
---------------------

This release contains **important bug fixes for analyses based on the IceCube public data**. Results obtained with v26.1.0 or earlier may be affected.

**Major fixes**

- **Signal energy PDFs now follow the source position in** ``skyllh.core.analysis.change_source``
  (#342). Before this fix, changing the source location after creating the
  analysis left the signal energy PDFs at the *original* source declination.
  Any workflow that reuses one analysis instance for many positions was
  affected, such as sky scans, source catalog loops and declination-dependent
  sensitivities.
- **The detector signal yield is now built in sin(dec)** (#341, fixes #337).
  The yield spline was built on a declination axis in radians, but it was
  evaluated at sin(dec). This made the expected signal yields wrong, and with
  them the ns-to-flux conversion, fitted fluxes, sensitivities and upper
  limits.

For other minor fixes, please read the `full release notes on GitHub <https://github.com/icecube/skyllh/releases>`_.

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Getting started

    installation
    tutorials/index
    examples/index
    concepts/index
    faq/index
    contributing

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Development

    dev_docs/logging
    dev_docs/unit_tests
    api_reference
