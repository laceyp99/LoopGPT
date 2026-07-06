"""Temporary compatibility alias for conductor_core.providers.ollama."""

import sys

from conductor_core.providers import ollama as _ollama

sys.modules[__name__] = _ollama
