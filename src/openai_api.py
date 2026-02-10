from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
)
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
import logging
import json
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join("Prompts", "loop gen.txt"), "r") as f:
    loop_prompt = f.read()
with open(os.path.join("Prompts", "prompt translation.txt"), "r") as f:
    pt_prompt = f.read()

# Load model list and pricing details from a JSON file
with open("model_list.json", "r") as f:
    model_info = json.load(f)


def initialize_openai_client():
    """
    Initializes and returns an OpenAI client using API key from the .env file.

    Returns:
        OpenAI: The OpenAI client instance.
    """
    load_dotenv(os.path.join("src", ".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("OPENAI_API_KEY is not set!")
    return OpenAI(api_key=api_key)


def calc_price(model, response):
    """
    Calculate the cost for a given response based on token usage.

    Args:
        model (str): The model identifier used for the request.
        response: The response object from the Responses API containing token usage.

    Returns:
        float: Calculated price for the API call.
    """
    usage = response.usage
    input_cost = model_info["models"]["OpenAI"][model]["cost"]["input"] / 1000000
    output_cost = model_info["models"]["OpenAI"][model]["cost"]["output"] / 1000000
    cached_input_cost = model_info["models"]["OpenAI"][model]["cost"]["cached input"] / 1000000

    # Determine cached tokens if available (Responses API structure)
    if (hasattr(usage, "input_tokens_details") and usage.input_tokens_details and hasattr(usage.input_tokens_details, "cached_tokens")):
        cached_tokens = usage.input_tokens_details.cached_tokens or 0
        new_input_tokens = usage.input_tokens - cached_tokens
    else:
        new_input_tokens = usage.input_tokens
        cached_tokens = 0

    total_price = (input_cost * new_input_tokens + output_cost * usage.output_tokens + cached_input_cost * cached_tokens)
    return total_price

def extract_reasoning(response):
    reasoning = ""
    for item in getattr(response, "output", []):
        if item.type == "reasoning":
            for s in item.summary:
                reasoning += s.text + "\n"
    return reasoning

def prompt_gen(prompt, model, temp=0.0, effort=None):
    """
    Generate text content using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate text.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        effort (str, optional): Reasoning effort level for reasoning models. Defaults to None.

    Returns:
        tuple: (content, messages, cost)
    """
    client = initialize_openai_client()
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]

    # Build request parameters
    request_params = {
        "model": model,
        "instructions": pt_prompt,
        "input": prompt,
        "store": False,
    }

    # Add temperature for non-reasoning models, reasoning effort for reasoning models
    model_config = model_info["models"]["OpenAI"][model]
    if model_config.get("extended_thinking") and effort:
        request_params["reasoning"] = {"effort": effort, "summary": "auto"}
    else:
        request_params["temperature"] = temp

    try:
        response = client.responses.create(**request_params)
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        raise ValueError("Invalid OpenAI API key")
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise

    reasoning = extract_reasoning(response)
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "assistant", "content": response.output_text})
    cost = calc_price(model, response)
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return response.output_text, messages, cost


def loop_gen(prompt, model, temp=0.0, effort=None):
    """
    Generate a MIDI loop using the specified model and prompt with structured output.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        effort (str, optional): Reasoning effort level for reasoning models. Defaults to None.

    Returns:
        tuple: (midi_loop, messages, cost)
    """
    client = initialize_openai_client()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]

    # Build request parameters
    request_params = {
        "model": model,
        "instructions": loop_prompt,
        "input": prompt,
        "text_format": objects.Loop,
        "store": False,
    }

    # Add temperature for non-reasoning models, reasoning effort for reasoning models
    model_config = model_info["models"]["OpenAI"][model]
    if model_config.get("extended_thinking") and effort:
        request_params["reasoning"] = {"effort": effort, "summary": "auto"}
    else:
        request_params["temperature"] = temp

    try:
        response = client.responses.parse(**request_params)
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        raise ValueError("Invalid OpenAI API key")
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise

    reasoning = extract_reasoning(response)
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "assistant", "content": str(response.output_parsed)})

    midi_loop = response.output_parsed
    cost = calc_price(model, response)
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost
