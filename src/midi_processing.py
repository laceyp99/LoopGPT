"""Temporary compatibility alias for conductor_core.midi."""

import sys

from conductor_core import midi as _midi

sys.modules[__name__] = _midi
