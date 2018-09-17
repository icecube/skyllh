# -*- coding: utf-8 -*-

from skylab.core.source_hypothesis import SourceHypoGroupManager

class SignalInjector(object):
    """Base class for a signal injector.
    """
    def __init__(self, src_hypo_group_manager):
        """Constructs a new signal injector.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The SourceHypoGroupManager instance defining the source groups with
            their spectra.
        """
        super(SignalInjector, self).__init__()

        self.src_hypo_group_manager = src_hypo_group_manager

    @property
    def src_hypo_group_manager(self):
        """The SourceHypoGroupManager instance defining the source groups with
        their spectra.
        """
        return self._src_hypo_group_manager
    @src_hypo_group_manager.setter
    def src_hypo_group_manager(self, manager):
        if(not isinstance(manager, SourceHypoGroupManager)):
            raise TypeError('The src_hypo_group_manager property must be an '
                            'instance of SourceHypoGroupManager!')
        self._src_hypo_group_manager = manager
