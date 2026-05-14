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
