"""Temporary compatibility alias for conductor_core.providers.google."""

import sys

from conductor_core.providers import google as _google

sys.modules[__name__] = _google
