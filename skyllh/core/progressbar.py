# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys

from skyllh.core import display

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

        self._val = 0

    def _print_bar(self):
        frac = self._val / self.maxval
        barwidth = display.PAGE_WIDTH - 7
        nchar = int(np.round(frac * barwidth, 0))
        fmt = "[%-"+str(barwidth)+"s] %d%%"
        sys.stdout.write(fmt%('='*nchar, int(np.round(frac*100, 0))))

    def start(self):
        self._val = self.startval

        self._print_bar()
        sys.stdout.flush()

    def finish(self):
        self._val = self.maxval

        sys.stdout.write('\r')
        self._print_bar()
        sys.stdout.write("\n")
        sys.stdout.flush()

    def update(self, val):
        self._val = val

        sys.stdout.write('\r')
        self._print_bar()
        sys.stdout.flush()
