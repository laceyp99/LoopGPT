"""Conductor Core public API."""

import logging

from conductor_core.config import (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    ProgressEvent,
    ProviderCredentials,
)
from conductor_core.engine import LoopGenerationEngine

# Library logging: Core emits records under the "conductor_core" namespace and
# never configures handlers itself. Consumers attach handlers (for example via
# logging.basicConfig or a handler on this logger) to surface Core logs.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "LoopGenerationEngine",
    "ProgressEvent",
    "ProviderCredentials",
]
