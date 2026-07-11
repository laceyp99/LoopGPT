"""Temporary compatibility alias for conductor_core.models."""

import sys

from conductor_core import models as _models

sys.modules[__name__] = _models
