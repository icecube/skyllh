# -*- coding: utf-8 -*-

"""The display module provides global settings for pretty command line
displaying.
"""

# Define the width (number of characters) of the display.
PAGE_WIDTH = 80

def add_leading_text_line_padding(padwidth, text):
    """Adds leading white spaces to all the lines of the given text.
    """
    return '\n'.join([ ' '*padwidth + line for line in text.split('\n') ])
