# -*- coding: utf-8 -*-

"""The analysis module of skylab contains analysis related utility classes.
"""

import numpy as np

from skylab.core.py import issequenceof
from skylab.core.dataset import Dataset

class Analysis(object):
    """This is the base class for all analysis classes. It contains common
    properties required by all analyses, like the used datasets.
    """
    def __init__(self, datasets):
        """Constructor of the Analysis base class.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances which define the used data. Each
            dataset is treated as an independent set of data, which means for
            each dataset PDF objects need to be created.
        """
        self.dataset_list = datasets

    @property
    def dataset_list(self):
        """The list of Dataset instances.
        """
        return self._dataset_list
    @dataset_list.setter
    def dataset_list(self, seq):
        if(not issequenceof(seq, Dataset))
            raise TypeError('The dataset_list property must be a list of Dataset instances!')
        self._dataset_list = list(seq)


class TimeIntegratedSinglePointLikeSourceAnalysis(Analysis):
    """This analysis class implements a time-integrated point-like source
    analysis assuming a single source.
    """
    def __init__(self, datasets, source):
        """Creates a new time-integrated point-like source analysis assuming a
        single source.

        Parameters
        ----------
        datasets : sequence of Dataset
            The sequence of Dataset instances which define the used data. Each
            dataset is treated as an independent set of data, which means for
            each dataset PDF objects need to be created.
        source : PointLikeSource
            The point-like source model, defining the location and flux model
            of the source.
        """
        super(TimeIntegratedSinglePointLikeSourceAnalysis, self).__init__(
            datasets)

        self.source = source

    @property
    def source(self):
        """The PointLikeSource source model.
        """
        return self._source
    @source.setter
    def source(self, src):
        if(not isinstance(src, PointLikeSource)):
            raise TypeError('The source property must be an instance of PointLikeSource!')
        self._source = src
