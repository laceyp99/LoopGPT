"""Temporary compatibility alias for conductor_core.providers.openai."""

import sys

from conductor_core.providers import openai as _openai

sys.modules[__name__] = _openai
