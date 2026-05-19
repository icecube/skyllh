.. _installation:

************
Installation
************

Python >= 3.11 is required.


Package install
===============

The `skyllh` package is available on `PyPI <https://pypi.org/project/skyllh/>`_ and `conda-forge <https://anaconda.org/channels/conda-forge/packages/skyllh/overview>`_ repositories, and can be installed with pip, uv, or conda:

.. tab-set::
    :class: outline

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install skyllh

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            uv pip install skyllh

    .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            conda install -c conda-forge skyllh

Optional dependency groups can be installed with extras:

.. code:: bash

    pip install "skyllh[extras]"   # iminuit, pyarrow
    pip install "skyllh[dev]"      # pre-commit, pytest
    pip install "skyllh[docs]"     # sphinx and doc-build tools

The current development version can be installed from GitHub:

.. code:: bash

    pip install git+https://github.com/icecube/skyllh.git

Optionally, a specific reference can be installed by:

.. code:: bash

    pip install git+https://github.com/icecube/skyllh.git@[ref]

where ``[ref]`` is a commit hash, branch name, or tag.


i3skyllh
========

The `i3skyllh <https://github.com/icecube/i3skyllh>`_ package provides
complementary pre-defined common analyses and datasets for the
`IceCube Neutrino Observatory <https://icecube.wisc.edu>`_ detector in a private
`repository <https://github.com/icecube/i3skyllh>`_.
