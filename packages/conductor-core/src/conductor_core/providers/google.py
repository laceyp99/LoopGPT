"""Google Gemini provider adapter for Conductor Core."""

import logging
import os
import sys

from conductor_core import models as objects
from conductor_core import music as utils

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - exercised only in minimal installs
    genai = None
    types = None

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_gemini_client(api_key: str | None = None):
    """Initialize and return a Gemini client."""
    if genai is None:
        raise ImportError("Install conductor-core[google] to use Google models.")

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    return genai.Client(api_key=resolved_api_key)


def calc_cost(model, usage):
    """Calculate the cost for a Gemini completion based on token usage."""
    model_info = utils.get_model_info()
    model_cost = model_info["models"]["Google"][model]["cost"]
    cached = usage.cached_content_token_count or 0

    if isinstance(model_cost["input"], dict):
        if usage.prompt_token_count <= 200000:
            input_cost = model_cost["input"]["<=200k"] / 1000000
            output_cost = model_cost["output"]["<=200k"] / 1000000
            cache_cost = model_cost["cache"]["<=200k"] / 1000000
        else:
            input_cost = model_cost["input"][">200k"] / 1000000
            output_cost = model_cost["output"][">200k"] / 1000000
            cache_cost = model_cost["cache"][">200k"] / 1000000
    else:
        input_cost = model_cost["input"] / 1000000
        output_cost = model_cost["output"] / 1000000
        if "cache" in model_cost:
            cache_cost = model_cost["cache"]["text"] / 1000000
        else:
            cache_cost = 0

    new_input_tokens, cached = utils.split_reported_cache_tokens(
        usage.prompt_token_count,
        cached,
    )
    return (
        new_input_tokens * input_cost
        + usage.candidates_token_count * output_cost
        + cached * cache_cost
    )


def process_output(response):
    final_result = ""
    thinking_content = ""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ValueError("Google response did not include any candidates.")

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts:
        raise ValueError("Google response did not include generated content parts.")

    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            thinking_content += text
        else:
            final_result += text
    if not final_result:
        raise ValueError("Google response did not include final loop content.")
    return final_result, thinking_content


def loop_gen(
    prompt,
    model,
    temp=0.0,
    use_thinking=None,
    effort=None,
    api_key: str | None = None,
    system_prompt: str | None = None,
):
    """Generate a MIDI loop using the specified Gemini model and prompt."""
    try:
        client = initialize_gemini_client(api_key=api_key)
    except TypeError:
        client = initialize_gemini_client()
    loop_prompt = system_prompt or utils.get_loop_prompt()

    model_info = utils.get_model_info()
    config = {
        "system_instruction": loop_prompt,
        "temperature": temp,
        "response_mime_type": "application/json",
        "response_schema": objects.Loop_G,
    }
    model_config = model_info["models"]["Google"][model]
    model_with_thinking = model_config["extended_thinking"]
    effort_options = model_config.get("effort_options", [])

    if effort_options:
        if effort in effort_options:
            config.update(
                {
                    "thinking_config": types.ThinkingConfig(
                        thinking_level=effort,
                        include_thoughts=True,
                    )
                }
            )
        else:
            print("No effort level specified; using default thinking configuration.")
    elif model_with_thinking and use_thinking:
        config.update(
            {
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=model_config["max_thinking_budget"],
                    include_thoughts=True,
                )
            }
        )
    elif model_with_thinking and use_thinking == False:
        config.update(
            {
                "thinking_config": types.ThinkingConfig(
                    thinking_budget=model_config["min_thinking_budget"],
                    include_thoughts=True,
                )
            }
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    content, thinking_content = process_output(response)
    midi_loop: objects.Loop_G = response.parsed
    if midi_loop is None:
        raise ValueError("Google response did not include parsed loop content.")

    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content},
    ]
    if thinking_content:
        messages.insert(2, {"role": "assistant", "content": thinking_content})

    return midi_loop, messages, calc_cost(model, response.usage_metadata)
