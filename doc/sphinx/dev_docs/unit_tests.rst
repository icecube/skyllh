.. _unit_tests:

**********
Unit tests
**********

When extending SkyLLH with new code and features, the extension needs to be
covered by unit tests. SkyLLH uses the ``unittest`` package of Python.

.. code-block:: python

    import unittest

    class SelfDrivingCarTest(TestCase):
        def setUp(self):
            self.car = SelfDrivingCar()

To run all test we can use following command::

    python -m unittest discover