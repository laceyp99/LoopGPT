"""Temporary compatibility alias for conductor_core.routing."""

import sys

from conductor_core import routing as _routing

sys.modules[__name__] = _routing
