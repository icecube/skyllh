# -*- coding: utf-8 -*-

"""This modules defines base types for some of the SkyLLH classes to avoid
circular imports when actively checking for types.
"""


class SourceHypoGroup_t(
    object,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
