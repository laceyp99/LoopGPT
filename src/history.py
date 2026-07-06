"""Temporary compatibility alias for conductor_core.storage."""

import sys

from conductor_core import storage as _storage

sys.modules[__name__] = _storage
