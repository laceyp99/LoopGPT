"""Anthropic provider adapter for Conductor Core."""

import logging
import os
import sys

from conductor_core import models as objects
from conductor_core import music as utils

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Anthropic = None

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALWAYS_ON_ADAPTIVE_THINKING_MODELS = {"claude-fable-5", "claude-mythos-5"}
ANTHROPIC_CACHE_CONTROL_MIN_CHARS = 4096


def initialize_anthropic_client(api_key: str | None = None):
    """Initialize and return an Anthropic client."""
    if Anthropic is None:
        raise ImportError("Install conductor-core[anthropic] to use Anthropic models.")

    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        logger.error("ANTHROPIC_API_KEY is not set!")
    return Anthropic(api_key=resolved_api_key)


def calc_price(model, output):
    """Calculate the cost for a completion based on token usage."""
    model_info = utils.get_model_info()
    if model not in model_info["models"]["Anthropic"].keys():
        logger.warning("Model %s not found in model info.", model)
        return False

    input_cost = model_info["models"]["Anthropic"][model]["cost"]["input"] / 1000000
    output_cost = model_info["models"]["Anthropic"][model]["cost"]["output"] / 1000000
    cached_5min = model_info["models"]["Anthropic"][model]["cost"]["5m cache input"] / 1000000
    cache_hits = model_info["models"]["Anthropic"][model]["cost"]["cache hits/refreshes"] / 1000000

    return (
        output["input_tokens"] * input_cost
        + output["output_tokens"] * output_cost
        + output["cache_creation"] * cached_5min
        + output["cache_read"] * cache_hits
    )


def build_system_prompt_block(loop_prompt):
    """Build Anthropic's system text block, marking only likely-cacheable prompts."""
    block = {"type": "text", "text": loop_prompt}
    if len(loop_prompt) >= ANTHROPIC_CACHE_CONTROL_MIN_CHARS:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def process_streaming_response(completion):
    """Extract text, tool JSON, and token usage from a streaming response."""
    output = {
        "loop": "",
        "text": "",
        "thinking_content": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 0,
        "cache_read": 0,
    }

    for chunk in completion:
        if chunk.type == "message_start":
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                output["input_tokens"] += getattr(chunk.message.usage, "input_tokens", 0)
                output["output_tokens"] += getattr(chunk.message.usage, "output_tokens", 0)
                output["cache_creation"] += getattr(
                    chunk.message.usage,
                    "cache_creation_input_tokens",
                    0,
                )
                output["cache_read"] += getattr(
                    chunk.message.usage,
                    "cache_read_input_tokens",
                    0,
                )
        elif chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "thinking"):
                output["thinking_content"] += chunk.delta.thinking
            elif hasattr(chunk.delta, "text"):
                output["text"] += chunk.delta.text
            elif hasattr(chunk.delta, "partial_json"):
                output["loop"] += chunk.delta.partial_json
        elif chunk.type == "message_delta":
            if hasattr(chunk, "usage"):
                output["input_tokens"] += getattr(chunk.usage, "input_tokens", 0)
                output["output_tokens"] += getattr(chunk.usage, "output_tokens", 0)
                output["cache_creation"] += getattr(
                    chunk.usage,
                    "cache_creation_input_tokens",
                    0,
                )
                output["cache_read"] += getattr(
                    chunk.usage,
                    "cache_read_input_tokens",
                    0,
                )
        elif chunk.type == "message_stop":
            break
    return output


def loop_gen(
    prompt,
    model,
    temp=0.0,
    use_thinking=False,
    effort="low",
    api_key: str | None = None,
    system_prompt: str | None = None,
):
    """Generate a MIDI loop using the specified Anthropic model and prompt."""
    try:
        client = initialize_anthropic_client(api_key=api_key)
    except TypeError:
        client = initialize_anthropic_client()
    loop_prompt = system_prompt or utils.get_loop_prompt()
    tools = [
        {
            "name": "build_MIDI_loop",
            "description": "builds a music loop in MIDI format",
            "input_schema": objects.Loop.model_json_schema(),
        }
    ]

    model_info = utils.get_model_info()
    model_config = model_info["models"]["Anthropic"][model]
    always_on_adaptive_thinking = model in ALWAYS_ON_ADAPTIVE_THINKING_MODELS
    api_params = {
        "model": model,
        "max_tokens": model_config["max_tokens"],
        "system": [build_system_prompt_block(loop_prompt)],
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "tool", "name": "build_MIDI_loop"},
        "stream": True,
    }
    if not always_on_adaptive_thinking:
        api_params["temperature"] = temp

    if model_config.get("effort_options"):
        api_params["tool_choice"] = {"type": "auto"}
        if not always_on_adaptive_thinking:
            api_params["thinking"] = {"type": "adaptive"}
        api_params["output_config"] = {"effort": effort}
        if not always_on_adaptive_thinking:
            api_params["temperature"] = 1.0
    elif use_thinking and model_config.get("extended_thinking"):
        api_params["tool_choice"] = {"type": "auto"}
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": model_config["max_thinking_budget"],
        }
        api_params["temperature"] = 1.0
    elif use_thinking and not model_config.get("extended_thinking"):
        logger.warning("Extended thinking requested but not supported by model: %s", model)

    completion = client.messages.create(**api_params)
    output = process_streaming_response(completion)
    if not output["loop"]:
        raise ValueError(
            f"Model {model} did not call the build_MIDI_loop tool. "
            f"Response text: {output['text'][:200]}"
        )
    loop = objects.Loop.model_validate_json(output["loop"])

    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    if output["thinking_content"]:
        messages.append({"role": "assistant", "content": output["thinking_content"]})
    messages.append({"role": "assistant", "content": output["loop"]})

    return loop, messages, calc_price(model, output)
