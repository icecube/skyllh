# -*- coding: utf-8 -*-

import os.path

from skyllh.core import session
from skyllh.core.config import CFG

# Automatically enable interactive mode, if the Python interpreter is in
# interactive mode.
if(session.is_python_interpreter_in_interactive_mode()):
    session.enable_interactive_session()
else:
    session.disable_interactive_session()


def wd_filename(filename):
    """Generates the fully qualified file name under the project's working
    directory of the given file.

    Parameters
    ----------
    filename : str
        The name of the file for which to generate the working directory path
        file name.

    Returns
    -------
    pathfilename : str
        The generated fully qualified path file name of ``filename`` with the
        project's working directory prefixed.
    """
    pathfilename = os.path.join(CFG['project']['working_directory'], filename)
    return pathfilename
