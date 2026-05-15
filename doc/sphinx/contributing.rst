.. _contributing:

************
Contributing
************

SkyLLH is an open-source project and contributions of all kinds are welcome.
Whether you found a bug, have a feature request, or want to share an analysis
tutorial, we encourage you to get involved.


Reporting issues
================

Bug reports and feature requests are tracked on the
`GitHub issue tracker <https://github.com/icecube/skyllh/issues>`_.

When reporting a bug, please include:

- A short, self-contained code snippet that reproduces the problem.
- The SkyLLH version (``python -c "import skyllh; from importlib.metadata import version; print(version('skyllh'))"``).
- The Python version and operating system.

For feature requests, describe the use case and why the feature would be
useful to the broader community.


Pull requests
=============

Code contributions are made through
`pull requests <https://github.com/icecube/skyllh/pulls>`_ on GitHub.

To submit a pull request:

1. Fork the `skyllh repository <https://github.com/icecube/skyllh>`_ and
   create a new branch from ``master``.
2. Install the development dependencies:

   .. code:: bash

       pip install -e ".[dev]"

3. Install the pre-commit hooks:

   .. code:: bash

       pre-commit install

4. Make your changes and ensure the test suite passes:

   .. code:: bash

       pytest

5. Push your branch and open a pull request against ``master``. Describe what
   the change does and reference any related issue.

Pull requests are reviewed by the maintainers. Feedback may be given before
merging, so please check back after submission.


Contributing to tutorials
=========================

Tutorials are Jupyter notebooks stored under ``doc/sphinx/tutorials/`` and
rendered into the documentation with `nbsphinx <https://nbsphinx.readthedocs.io>`_.

To add a new tutorial:

1. Place your notebook in the appropriate subdirectory under
   ``doc/sphinx/tutorials/``.
2. Add the notebook path (without the ``.ipynb`` extension) to
   ``doc/sphinx/tutorials/index.rst``.
3. Ensure the notebook can be read without execution (``nbsphinx_execute = 'never'``
   is the project default), so all output cells should be pre-computed and saved
   before committing.
4. Open a pull request following the steps above.

Tutorials using IceCube's public datasets are especially encouraged, as they
help new users get started quickly.


Code style
==========

SkyLLH uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting.
The pre-commit hooks run ``ruff`` automatically on every commit. To check and
fix style issues manually:

.. code:: bash

    ruff check --fix .
    ruff format .
