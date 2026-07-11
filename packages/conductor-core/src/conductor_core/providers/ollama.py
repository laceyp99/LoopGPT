"""Ollama provider adapter for Conductor Core."""

import logging
import os

from conductor_core import models as objects
from conductor_core import music as utils

try:
    import ollama
except ImportError:  # pragma: no cover - exercised only in minimal installs
    ollama = None

logger = logging.getLogger(__name__)

_ollama_status_cache = None


def _resolve_host(host_address: str | None = None) -> str:
    return host_address or os.getenv("OLLAMA_API_HOST_ADDRESS") or "http://localhost:11434"


def initialize_ollama_client(host_address: str | None = None):
    """Initialize and return an Ollama client."""
    if ollama is None:
        raise ImportError("Install conductor-core[ollama] to use Ollama models.")

    return ollama.Client(host=_resolve_host(host_address))


def get_ollama_status(force_refresh=False, host_address: str | None = None):
    """Get the current Ollama availability and discovered models."""
    global _ollama_status_cache

    host = _resolve_host(host_address)
    cache_key = (host,)
    if _ollama_status_cache is not None and not force_refresh:
        if _ollama_status_cache.get("cache_key") == cache_key:
            return {k: v for k, v in _ollama_status_cache.items() if k != "cache_key"}

    status = {
        "available": False,
        "models": [],
        "host": host,
        "error": None,
    }

    if ollama is None:
        status["error"] = "Install conductor-core[ollama] to use Ollama models."
        _ollama_status_cache = {**status, "cache_key": cache_key}
        return status

    try:
        client = initialize_ollama_client(host_address=host)
        status["models"] = [model.model for model in client.list().models]
        status["available"] = True
    except Exception as exc:
        status["error"] = str(exc)
        logger.warning("Ollama unavailable at %s: %s", host, exc)

    _ollama_status_cache = {**status, "cache_key": cache_key}
    return status


def get_model_list(force_refresh=False, host_address: str | None = None):
    """Get the available Ollama model names."""
    return get_ollama_status(
        force_refresh=force_refresh,
        host_address=host_address,
    )["models"]


def loop_gen(
    prompt,
    model,
    temp=0.0,
    host_address: str | None = None,
    system_prompt: str | None = None,
):
    """Generate a MIDI loop using the specified Ollama model and prompt."""
    try:
        client = initialize_ollama_client(host_address=host_address)
    except TypeError:
        client = initialize_ollama_client()
    loop_prompt = system_prompt or utils.get_loop_prompt()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    completion = client.chat(
        model=model,
        messages=messages,
        format=objects.Loop.model_json_schema(),
        options={"temperature": temp},
    )
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("Ollama response did not include generated content.")

    midi_loop = objects.Loop.model_validate_json(content)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    messages.append({"role": "assistant", "content": str(midi_loop)})
    return midi_loop, messages, 0
