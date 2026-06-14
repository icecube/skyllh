"""This modules defines base types for some of the SkyLLH classes to avoid
circular imports when actively checking for types.
"""


class SourceHypoGroup_t:
    """This is the base type for the
    :class:`~skyllh.core.source_hypo_grouping.SourceHypoGroup` class. It exists
    to allow type checks without importing the actual class, avoiding circular
    imports.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Creates a new instance of SourceHypoGroup_t."""
        super().__init__(*args, **kwargs)
