# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys
import time

from skyllh.core import display
from skyllh.core.py import int_cast


class ProgressBar(object):
    """This class implements a progress bar on stdout.
    """
    def __init__(self, maxval, startval=0):
        """Creates a new ProgressBar instance.

        Parameters
        ----------
        maxval : int
            The maximal value the progress can reach.
        startval : int
            The progress value to start with. Must be smaller than `maxval`.
        """
        super(ProgressBar, self).__init__()

        self.maxval = maxval
        self.startval = startval

        self._start_time = None
        self._val = 0
        self._last_rendered_pbar_str = None

    @property
    def maxval(self):
        """The maximal integer value the progress can reach.
        """
        return self._maxval
    @maxval.setter
    def maxval(self, v):
        v = int_cast(v, 'The maxval property must be castable to an integer '
            'value!')
        self._maxval = v

    @property
    def startval(self):
        """The progress integer value to start with.
        """
        return self._startval
    @startval.setter
    def startval(self, v):
        v = int_cast(v, 'The startval property must be castable to an integer '
            ' value!')
        self._startval = v

    def _render_pbar_str(self):
        """Renders the progress bar string.

        Returns
        -------
        pbar_str : str
            The rendered progress bar string.
        """
        # Calculate the elapsed time (ELT) for the first 10 seconds or if we
        # are at the end of the progress. Otherwise calculate the estimated
        # remaining time (ERT).
        if(self._val > 0):
            curr_time = time.time()
            t_elapsed = curr_time - self._start_time
            if((t_elapsed < 10) or (self._val == self._maxval)):
                t_label = 'ELT'
                t = t_elapsed
            else:
                t_label = 'ERT'
                t_total = t_elapsed * self._maxval / self._val
                t = t_total - t_elapsed
        else:
            t_label = 'ELT'
            t = 0

        # Calculate hours, minutes, and seconds from `t` given in seconds.
        t_h = int(t / 3600)
        t -= t_h*3600
        t_m = int(t / 60)
        t -= t_m*60
        t_s = int(np.round(t, 0))

        frac = self._val / self._maxval
        barwidth = max(display.PAGE_WIDTH - 22, 10)
        nchar = int(np.round(frac * barwidth, 0))
        fmt = "[%-"+str(barwidth)+"s] %3d%% %s %dh:%02dm:%02ds"
        pbar_str = fmt%(
            '='*nchar,
            int(np.round(frac*100, 0)),
            t_label, t_h, t_m, t_s
        )

        return pbar_str

    def start(self):
        self._start_time = time.time()
        self._val = self._startval
        self._last_rendered_pbar_str = self._render_pbar_str()

        sys.stdout.write(self._last_rendered_pbar_str)
        sys.stdout.flush()

    def finish(self):
        self._val = self._maxval
        self._last_rendered_pbar_str = self._render_pbar_str()

        sys.stdout.write('\r'+self._last_rendered_pbar_str+"\n")
        sys.stdout.flush()

    def update(self, val):
        self._val = val

        pbar_str = self._render_pbar_str()
        if(pbar_str == self._last_rendered_pbar_str):
            return

        sys.stdout.write('\r'+pbar_str)
        sys.stdout.flush()
        self._last_rendered_pbar_str = pbar_str
