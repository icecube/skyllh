# -*- coding: utf-8 -*-

from tqdm import tqdm

from skyllh.core import (
    session,
)
from skyllh.core.py import (
    int_cast,
)


class ProgressBar(
        object):
    """This class provides an hierarchical progress bar for SkyLLH.
    For rendering it uses the tqdm Python package.
    In case of multiple layers of progress bars, it creates only a single
    progress bar, which gets updated whenever the deeper level progress bars
    are updated.
    """
    def __init__(
            self,
            maxval,
            startval=0,
            parent=None,
            **kwargs):
        """Creates a new ProgressBar instance.

        Parameters
        ----------
        maxval : int
            The maximal value the progress can reach.
        startval : int
            The progress value to start with. Must be smaller than `maxval`.
        parent : instance of ProgressBar | False | None
            The parent instance of ProgressBar if this progress bar is a sub
            progress bar.
            If set to ``False``, this progress bar is deactivated and the
            property ``is_shown`` will return ``False``.
            If set to ``None``, this progress bar will be a primary progress
            bar.

        Additional keyword arguments
        ----------------------------
        Additional keyword arguments are passed to the constructor of the tqdm
        class.
        """
        super().__init__(
            **kwargs)

        self._is_deactivated = False
        if parent is False:
            self._is_deactivated = True
            parent = None

        self.maxval = maxval
        self.startval = startval
        self.parent = parent

        self._val = 0
        self._sub_pbar_list = []

        self._tqdm = None
        if (self._parent is None) and self.is_shown:
            self._tqdm = tqdm(
                total=maxval,
                initial=startval,
                leave=True,
                position=0,
                **kwargs)

    @property
    def maxval(self):
        """The maximal integer value the progress can reach.
        """
        return self._maxval

    @maxval.setter
    def maxval(self, v):
        v = int_cast(
            v,
            'The maxval property must be castable to an integer value!')
        self._maxval = v

    @property
    def startval(self):
        """The progress integer value to start with. It must be smaller than
        `maxval`.
        """
        return self._startval

    @startval.setter
    def startval(self, v):
        v = int_cast(
            v,
            'The startval property must be castable to an integer value!')

        if v >= self._maxval:
            raise ValueError(
                f'The startval value ({v}) must be smaller than the value of '
                f'the `maxval` property ({self._maxval})!')

        self._startval = v

    @property
    def val(self):
        """(read-only) The current value of the progess.
        """
        return self._val

    @property
    def is_shown(self):
        """(read-only) Flag if the progress bar is shown. This is ``True``
        if the program is run in an interactive session, ``False`` otherwise.
        """
        if self._is_deactivated is True:
            return False
        return session.is_interactive_session()

    @property
    def parent(self):
        """The parent ProgressBar instance of this progress bar, or ``None`` if
        no parent exist.
        """
        return self._parent

    @parent.setter
    def parent(self, pbar):
        if pbar is not None:
            if not isinstance(pbar, ProgressBar):
                raise TypeError(
                    'The parent property must be None, or an instance of '
                    'ProgressBar!')
        self._parent = pbar

    def add_sub_progress_bar(self, pbar):
        """Adds the given progress bar to the list of running sub progress bars
        of this progress bar.
        """
        if not isinstance(pbar, ProgressBar):
            raise TypeError(
                'The pbar argument must be an instance of ProgressBar!')
        self._sub_pbar_list.append(pbar)

    def remove_sub_progress_bars(self):
        """Removes all progress bar instances from the list of sub progress bars
        of this progress bar. It calles the ``remove_sub_progress_bars`` method
        of each sub progress bar.
        """
        for pbar in self._sub_pbar_list:
            pbar.remove_sub_progress_bars()

        self._sub_pbar_list = []

    def get_progressbar_list(self):
        """Retrieves the list of ProgressBar instances.

        Returns
        -------
        pbar_list : list of instance of ProgressBar
            The list of ProgressBar instances, which are part of this
            ProgressBar instance.
        """
        pbar_list = []
        for pbar in self._sub_pbar_list:
            pbar_list.extend(pbar.get_progressbar_list())

        pbar_list.append(self)

        return pbar_list

    def rerender(self):
        """Rerenders this progress bar on the display. It calls the ``update``
        method of the tqdm progess bar.
        """
        if not self.is_shown:
            return

        pbar_list = self.get_progressbar_list()

        maxval = 0
        val = 0
        for pbar in pbar_list:
            maxval += pbar.maxval
            val += pbar.val

        dval = val - self._tqdm.n
        self._tqdm.total = maxval
        self._tqdm.update(dval)

    def trigger_rerendering(self):
        """Triggers a rerendering / update of the most top progress bar.
        """
        if self._parent is not None:
            self._parent.trigger_rerendering()
            return

        # We are the most top parent progress bar. So we need to get rerendered.
        self.rerender()

    def start(self):
        """Sets the current progess value to ``startval`` and updates the
        progess bar to fulfill the start conditions.
        """
        self._val = self._startval

        if self._parent is not None:
            self._parent.add_sub_progress_bar(self)
        elif not self.is_shown:
            return self
        else:
            self._tqdm.initial = self._val
            self._tqdm.n = self._val
            self._tqdm.reset()

        self.trigger_rerendering()

        return self

    def finish(self):
        """Finishes this progress bar by setting the current progress value to
        its max value.
        If this progress bar is the top most progessbar, it will also close
        the tqdm instance, what will trigger a flush of the output buffer.
        """
        self._val = self._maxval

        self.trigger_rerendering()

        if (self._parent is None) and self.is_shown:
            self._tqdm.close()

        self.remove_sub_progress_bars()

    def increment(self, dval=1):
        """Updates the progress bar by incrementing the progress by the given
        integral amount.

        Parameters
        ----------
        dval : int
            The amount of progress to increment the progress bar with.
        """
        self.update(self._val + dval)

    def update(self, val):
        """Updates the progress value to the given value.

        Parameters
        ----------
        val : int
            The new current progress value.
        """
        self._val = val

        self.trigger_rerendering()
