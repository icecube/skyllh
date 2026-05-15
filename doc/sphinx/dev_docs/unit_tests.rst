.. _unit_tests:

**********
Unit tests
**********

When extending SkyLLH with new code and features, the extension needs to be
covered by unit tests. SkyLLH uses the ``unittest`` package of Python with
``pytest`` as the test runner.

Setup
=====

Install the development dependencies (which include ``pytest``) with:

.. code-block:: bash

    pip install ".[dev]"

Directory structure
===================

All tests live under the ``tests/`` directory, organized to mirror the
package structure:

.. code-block:: text

    tests/
    ├── core/          # Core framework tests
    ├── i3/            # IceCube-specific tests
    └── publicdata_ps/ # Public data point-source tests

Writing a test
==============

Test classes inherit from ``unittest.TestCase`` and follow the naming
convention ``ClassNameTestCase``. Test methods must start with ``test_``.

.. code-block:: python

    import unittest

    class MyNewClassTestCase(unittest.TestCase):
        def setUp(self):
            self.obj = MyNewClass()

        def test_something(self):
            self.assertEqual(self.obj.something(), expected_value)

Skipping tests conditionally
-----------------------------

Some tests depend on external resources (e.g. datasets protected by a
password). Use ``@unittest.skipIf`` or ``self.skipTest()`` to skip them
automatically when the required environment variable is absent:

.. code-block:: python

    import os
    import unittest

    class MyClassTestCase(unittest.TestCase):
        def setUp(self):
            if not os.environ.get('ICECUBE_PASSWORD'):
                self.skipTest('ICECUBE_PASSWORD not set')

Running tests
=============

Run the full test suite::

    pytest

Run tests in a specific sub-directory::

    pytest tests/core/

Run a single test file::

    pytest tests/core/test_parameters.py
