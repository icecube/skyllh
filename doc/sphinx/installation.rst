.. _installation:

************
Installation
************

Python >= 3.11 is required.


Using pip
=========

The latest `skyllh` release can be installed from the
`PyPI <https://pypi.org/project/skyllh/>`_ repository:

.. code:: bash

    pip install skyllh

Optional dependency groups can be installed with extras:

.. code:: bash

    pip install skyllh[extras]   # iminuit, pyarrow
    pip install skyllh[dev]      # pre-commit, pytest
    pip install skyllh[docs]     # sphinx and doc-build tools

The current development version can be installed using pip:

.. code:: bash

    pip install git+https://github.com/icecube/skyllh.git

Optionally, a specific reference can be installed by:

.. code:: bash

    pip install git+https://github.com/icecube/skyllh.git@[ref]

where ``[ref]`` is a commit hash, branch name, or tag.


Using conda
===========

.. code:: bash

    conda install -c conda-forge skyllh


i3skyllh
========

The `i3skyllh <https://github.com/icecube/i3skyllh>`_ package provides
complementary pre-defined common analyses and datasets for the
`IceCube Neutrino Observatory <https://icecube.wisc.edu>`_ detector in a private
`repository <https://github.com/icecube/i3skyllh>`_.
