# -*- coding: utf-8 -*-

from skyllh.core import (
    session,
)

# Automatically enable interactive mode, if the Python interpreter is in
# interactive mode.
if session.is_python_interpreter_in_interactive_mode():
    session.enable_interactive_session()
else:
    session.disable_interactive_session()
