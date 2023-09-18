# -*- coding: utf-8 -*-
# Author: Dr. Martin Wolf <mail@martin-wolf.org>

"""This module provides classes for defining source catalogs.
"""

from skyllh.core.py import (
    str_cast,
)
from skyllh.core.source_model import (
    SourceModelCollection,
)


class SourceCatalog(
        SourceModelCollection):
    """This class describes a catalog of sources. It is derived from
    SourceModelCollection. A catalog has a name.
    """
    def __init__(
            self,
            name,
            sources=None,
            source_type=None,
            **kwargs):
        """Creates a new source catalog.

        Parameters
        ----------
        name : str
            The name of the catalog.
        sources : sequence of source_type | None
            The sequence of sources this catalog should be initialized with.
        source_type : type | None
            The type of the source class. If set to None (default), the
            default type defined by SourceCollection will be used.
        """
        super().__init__(
            sources=sources,
            source_type=source_type,
            **kwargs)

        self.name = name

    @property
    def name(self):
        """The name of the catalog.
        """
        return self._name

    @name.setter
    def name(self, name):
        name = str_cast(
            name,
            'The name property must be cast-able to type str!')
        self._name = name

    def __str__(self):
        s = f'"{self.name}" {super().__str__()}'
        return s

    def as_SourceModelCollection(self):
        """Creates a SourceModelCollection object for this catalog and
        returns it.

        Returns
        -------
        source_model_collection : instance of SourceModelCollection
            The created instance of SourceModelCollection.
        """
        return SourceModelCollection(
            sources=self.sources,
            source_type=self.source_type)
