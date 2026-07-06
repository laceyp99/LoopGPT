"""Temporary compatibility alias for conductor_core.playback."""

import sys

from conductor_core import playback as _playback

sys.modules[__name__] = _playback
