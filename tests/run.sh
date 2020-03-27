#!/usr/bin/env bash
# This script must be executed from the skyllh main directory.
rcode=0

/usr/bin/env python -m unittest discover tests/core || rcode=$?
/usr/bin/env python -m unittest discover tests/i3 || rcode=$?
/usr/bin/env python -m unittest discover tests/physics || rcode=$?

exit $rcode
