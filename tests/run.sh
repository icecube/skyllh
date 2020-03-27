#!/usr/bin/env bash
# This script must be executed from the skyllh main directory.
/usr/bin/env python -m unittest discover tests/core
/usr/bin/env python -m unittest discover tests/i3
/usr/bin/env python -m unittest discover tests/physics
