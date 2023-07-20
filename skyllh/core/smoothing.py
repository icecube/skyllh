# -*- coding: utf-8 -*-

import abc
import numpy as np

import scipy.signal
import scipy.stats

from skyllh.core.py import issequenceof

# Define a constant that can be used when specifying a histogram axis as
# unsmooth, i.e. no smoothing should be applied along that axis.
UNSMOOTH_AXIS = np.ones(1)


class HistSmoothingMethod(
        object,
        metaclass=abc.ABCMeta,
):
    """Abstract base class for implementing a histogram smoothing method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def smooth(self, h):
        """This method is supposed to smooth the given histogram h.

        Parameters
        ----------
        h : N-dimensional ndarray
            The ndarray holding histogram bin values.

        Returns
        -------
        smoothed_h : N-dimensional ndarray
            The array holding the smoothed histogram bin values.
        """
        pass


class NoHistSmoothingMethod(
        HistSmoothingMethod,
):
    """This class implements a no-shoothing histogram method.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def smooth(self, h):
        """Does not perform any smoothing and just returns the input histogram.

        Parameters
        ----------
        h : N-dimensional ndarray
            The ndarray holding histogram bin values.

        Returns
        -------
        h : N-dimensional ndarray
            The input histogram array.
        """
        return h


class NeighboringBinHistSmoothingMethod(
        HistSmoothingMethod,
):
    """This class implements a smoothing algorithm that smoothes a histogram
    based on the neighboring bins.
    """
    def __init__(
            self,
            axis_kernel_arrays,
            **kwargs,
    ):
        """Constructs a new neighboring bin histogram smoothing method.

        Parameters
        ----------
        axis_kernel_arrays: sequence of 1D ndarrays
            The sequence of smoothing kernel arrays, one for each axis. If an
            axis should not get smoothed, the UNSMOOTH_AXIS constant should be
            used for that axis' smoothing kernel array.
        """
        super().__init__(**kwargs)

        if not issequenceof(axis_kernel_arrays, np.ndarray):
            raise TypeError(
                'The axis_kernel_arrays argument must be a sequence of '
                'numpy.ndarray instances!')

        self._ndim = len(axis_kernel_arrays)

        # Construct the smoothing kernel k used by the smooth method.
        # k is a N-dimensional ndarray. It defines which neighboring bin values
        # of the histogram will contribute how much to the central bin value.
        self._k = np.prod(
            np.meshgrid(*axis_kernel_arrays, indexing='ij'), axis=0)

    @property
    def ndim(self):
        """(read-only) The dimensionality of the histogram this smoothing
        instances is made for.
        """
        return self._ndim

    def smooth(self, h):
        """Smoothes the given histogram array h with the internal kernel array
        k. Both arrays must have the same dimensionality. The shape
        values of k must be smaller than or equal to the shape values of h.

        Parameters
        ----------
        h : N-dimensional ndarray
            The ndarray holding histogram bin values.

        Returns
        -------
        smoothed_h : N-dimensional ndarray.
        """
        if h.ndim != self._ndim:
            raise ValueError(
                'The ndarrays of argument h and k must have the same '
                f'dimensionality! Currently they are {h.ndim:d} and '
                f'{self._ndim:d}, respectively.')
        for d in range(h.ndim):
            if self._k.shape[d] > h.shape[d]:
                raise ValueError(
                    f'The shape value ({self._k.shape[d]:d}) of dimension '
                    f'{d:d} of ndarray k must be smaller than or equal to the '
                    f'shape value ({h.shape[d]:d}) of dimension {d:d} of '
                    'ndarray h!')

        norm = scipy.signal.convolve(np.ones_like(h), self._k, mode="same")
        smoothed_h = scipy.signal.convolve(h, self._k, mode="same") / norm

        return smoothed_h


class SmoothingFilter(
        object):
    """This class provides a base class for a histogram smoothing filter. It
    provides an axis kernel array that defines how many neighboring bins of a
    histogram bin should be used to smooth that histogram bin.
    """
    def __init__(
            self,
            axis_kernel_array,
            **kwargs):
        super().__init__(**kwargs)

        self.axis_kernel_array = axis_kernel_array

    @property
    def axis_kernel_array(self):
        """The kernel array for a histogram axis.
        """
        return self._axis_kernel_array

    @axis_kernel_array.setter
    def axis_kernel_array(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                'The axis_kernel_array property must be an instance of '
                'numpy.ndarray!')
        self._axis_kernel_array = arr


class BlockSmoothingFilter(
        SmoothingFilter,
):
    """This class defines the histogram smoothing filter for smoothing a
    histogram via a block kernel function. The half-width of that
    block is specified via the nbins argument.
    """
    def __init__(
            self,
            nbins,
            **kwargs
    ):
        """Creates a new BlockSmoothingFilter instance.

        Parameters
        ----------
        nbins : int
            The number of neighboring bins into one direction of a histogram
            bin, which should be used to smooth that histogram bin.
        """
        if not isinstance(nbins, int):
            raise TypeError(
                'The nbins argument must be of type int!')
        if nbins <= 0:
            raise ValueError(
                'The nbins argument must be greater zero!')

        arr = np.ones(2*nbins + 1, dtype=np.float64)

        super().__init__(arr, **kwargs)


class GaussianSmoothingFilter(
        SmoothingFilter,
):
    """This class defines the histogram smoothing filter for smoothing a
    histogram via a Gaussian kernel function. The width of that Gaussian is
    approximately one standard deviation, spread over nbins on each side of the
    central histogram bin.
    """
    def __init__(
            self,
            nbins,
            **kwargs,
    ):
        """Creates a new GaussianSmoothingFilter instance.

        Parameters
        ----------
        nbins : int
            The number of neighboring bins into one direction of a histogram
            bin, which should be used to smooth that histogram bin.
        """
        if not isinstance(nbins, int):
            raise TypeError(
                'The nbins argument must be of type int!')
        if nbins <= 0:
            raise ValueError(
                'The nbins argument must be greater zero!')

        val = 1.6635
        r = np.linspace(-val, val, 2*nbins + 1)
        arr = scipy.stats.norm.pdf(r)

        super().__init__(arr, **kwargs)
