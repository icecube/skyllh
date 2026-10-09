import multiprocessing as mp
import pickle
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np

from skyllh.analyses.i3.publicdata_ps.utils import FctSpline1D
from skyllh.core.multiproc import parallelize


def create_spline(scale):
    return FctSpline1D(scale * np.array([1.0, 3.0, 2.0, 5.0, 4.0]), np.linspace(0, 5, 6), norm=True)


def transfer_splines_between_processes():
    """Creates splines in child processes and checks that they are equal to a
    spline created in this process. Raises an AssertionError otherwise.
    """
    expected = create_spline(1.0)
    x = np.linspace(-1, 6, 71)
    splines = parallelize(func=create_spline, args_list=[((1.0,), {}), ((1.0,), {})], ncpu=2)
    assert len(splines) == 2
    for spline in splines:
        np.testing.assert_array_equal(spline(x), expected(x))
        assert spline.norm == expected.norm


class FctSpline1DPickleTestCase(unittest.TestCase):
    # scipy 1.18.0 cannot pickle PchipInterpolator instances, hence it is
    # excluded in pyproject.toml. These tests guard against such regressions.

    def setUp(self):
        self.spline = create_spline(1.0)
        self.x = np.linspace(-1, 6, 71)

    def assert_equal_splines(self, spline):
        np.testing.assert_array_equal(spline(self.x), self.spline(self.x))
        np.testing.assert_array_equal(spline(self.x, oor_value=-1), self.spline(self.x, oor_value=-1))
        self.assertEqual(spline.norm, self.spline.norm)
        self.assertEqual(spline.x_min, self.spline.x_min)
        self.assertEqual(spline.x_max, self.spline.x_max)

    def test_pickle(self):
        self.assert_equal_splines(pickle.loads(pickle.dumps(self.spline)))

    def test_forking_pickler(self):
        self.assert_equal_splines(pickle.loads(bytes(ForkingPickler.dumps(self.spline))))

    def test_transfer_between_processes(self):
        # If a child process cannot pickle its result, parallelize waits
        # forever. Hence, run it in a separate process and fail after a timeout.
        proc = mp.Process(target=transfer_splines_between_processes)
        proc.start()
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            self.fail('Transferring FctSpline1D between processes did not finish within 60 seconds.')
        self.assertEqual(proc.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
