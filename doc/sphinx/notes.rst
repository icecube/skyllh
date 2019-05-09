.. _notes:

*****
Notes
*****


Quick setup
===========

To create autodoc files for the module run this in software directory::

    sphinx-apidoc -e -f -M -T -o docs/_source/autodoc skyllh/skyllh/

Referencing to auto generated docstrings can be looked up at http://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects

To create HTML documentation use

::

    docs/make html

command.


Docstrings
==========

To generate consistent reference documentantion NumPy `docstring convention <https://www.numpy.org/devdocs/docs/howto_document.html>`_ should be followed.

Some important notes:

* Variable, module, function, and class names should be written between single back-ticks (\`numpy\`).
* Use backquotes (\``print a\``) for code samples.
* List of available `sections <https://developer.lsst.io/python/numpydoc.html#py-docstring-sections>`_ in docstrings that appear in a common order.


Random scribbles
================

.. math:: \frac{a}{b} = a^3

Inline math test: :math:`\frac{a}{b} = a^3`

Variable names are displayed in typewriter font, obtained by using \mathtt{var}:

We square the input parameter `alpha` to obtain
:math:`\mathtt{alpha}^2`.


Logging
=======

Simple example of Python logging module functionality::

    import logging

    logger = logging.getLogger(__name__)
    logger.debug("Debug level message.")
    logger.warning("Warning level message.")


TODO list
=========


API Reference
=============

.. autosummary::

   skyllh.core.analysis
