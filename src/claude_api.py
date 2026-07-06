"""Temporary compatibility alias for conductor_core.providers.anthropic."""

import sys

from conductor_core.providers import anthropic as _anthropic

sys.modules[__name__] = _anthropic
