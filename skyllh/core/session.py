# -*- coding: utf-8 -*-

"""The session module provides global settings for session handling.
"""

# By default SkyLLH will not be in interactive session, i.e. will be in batch
# mode. Hence, progress bars will not be displayed not not screw up the output.
IS_INTERACTIVE_SESSION = False

def enable_interactive_session():
    """Enables interactive session mode.
    """
    global IS_INTERACTIVE_SESSION

    IS_INTERACTIVE_SESSION = True

def disable_interactive_session():
    """Disables interactive session mode.
    """
    global IS_INTERACTIVE_SESSION

    IS_INTERACTIVE_SESSION = False

def is_interactive_session():
    """Checks whether the current session is interactive (True) or not (False).
    """
    return IS_INTERACTIVE_SESSION
