from google import genai
from google.genai import types
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
# import utils, objects
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRICE_PER_MILLION_TOKENS = 1000000
# Gemini implicit cache storage is reported as roughly 5-6 minutes.
IMPLICIT_CACHE_STORAGE_TTL_HOURS = 0.1


def _cache_storage_cost(cached_tokens, storage_hour_price):
    """Estimate Gemini implicit cache storage from cached tokens and assumed TTL."""
    if not cached_tokens or not storage_hour_price:
        return 0
    return (
        cached_tokens
        * (storage_hour_price / PRICE_PER_MILLION_TOKENS)
        * IMPLICIT_CACHE_STORAGE_TTL_HOURS
    )


def initialize_gemini_client():
    """Initializes and returns a Gemini client using API key from the .env file or the default key.
    
    Returns:
        genai.Client: The Gemini client instance.
    """
    user_key = os.getenv('GEMINI_API_KEY')
    if not user_key:
        # Load environment variables from .env file
        load_dotenv(os.path.join('src', '.env'))
        user_key = os.getenv("GEMINI_API_KEY")
    
    if not user_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file.")
    
    return genai.Client(api_key=user_key)

def calc_cost(model, usage):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        model (str): The model identifier used for the request.
        usage (types.UsageMetadata): The usage metadata from the API response.

    Returns:
        float: Calculated price for the API call.
    """
    model_info = utils.get_model_info()
    model_cost = model_info["models"]["Google"][model]["cost"]
    prompt_tokens = usage.prompt_token_count or 0
    cached = min(usage.cached_content_token_count or 0, prompt_tokens)

    # Models with tiered pricing switch rates once the prompt exceeds 200k tokens.
    if isinstance(model_cost["input"], dict):
        if prompt_tokens <= 200000:
            input_cost = model_cost["input"]["<=200k"] / PRICE_PER_MILLION_TOKENS
            output_cost = model_cost["output"]["<=200k"] / PRICE_PER_MILLION_TOKENS
            cache_cost = model_cost["cache"]["<=200k"] / PRICE_PER_MILLION_TOKENS
        else:
            input_cost = model_cost["input"][">200k"] / PRICE_PER_MILLION_TOKENS
            output_cost = model_cost["output"][">200k"] / PRICE_PER_MILLION_TOKENS
            cache_cost = model_cost["cache"][">200k"] / PRICE_PER_MILLION_TOKENS
        storage_cost = _cache_storage_cost(cached, model_cost["cache"]["storage hour"])
    else:
        # For other models, use the default cost structure
        input_cost = model_cost["input"] / PRICE_PER_MILLION_TOKENS
        output_cost = model_cost["output"] / PRICE_PER_MILLION_TOKENS
        # Check if cache cost is defined for the model
        if "cache" in model_cost:
            cache_cost = model_cost["cache"]["text"] / PRICE_PER_MILLION_TOKENS
            storage_cost = _cache_storage_cost(cached, model_cost["cache"]["storage hour"])
        else:
            cache_cost = 0
            storage_cost = 0
    # Gemini doesn't subtract cached tokens from prompt tokens, so we need to do that here
    new_input_tokens = max(prompt_tokens - cached, 0)
    # Calculate total cost
    return (new_input_tokens * input_cost) + (usage.candidates_token_count * output_cost) + (cached * cache_cost) + storage_cost

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


def loop_gen(prompt, model, temp=0.0, use_thinking=None, effort=None):
    """
    Generate a MIDI bar (chord progression/melody) using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        use_thinking (bool, optional): Whether to enable extended thinking. Defaults to None.
        effort (str, optional): Reasoning effort level (not used in Gemini). Defaults to None.

    Returns:
        tuple: (midi_loop, messages, cost)
    """
    client = initialize_gemini_client()
    loop_prompt = utils.get_loop_prompt()

    # Configure the generation parameters based on whether extended thinking is enabled
    model_info = utils.get_model_info()
    config = {
        'system_instruction': loop_prompt,
        'temperature': temp,
        'response_mime_type': 'application/json',
        'response_schema': objects.Loop_G,
    }
    model_config = model_info["models"]["Google"][model]
    model_with_thinking = model_config["extended_thinking"]
    effort_options = model_config.get("effort_options", [])

    if effort_options:
        if effort in effort_options:
            config.update({"thinking_config": types.ThinkingConfig(thinking_level=effort, include_thoughts=True)})
        else:
            print("No effort level specified; using default thinking configuration.")
    elif model_with_thinking and use_thinking:
        config.update({"thinking_config": types.ThinkingConfig(thinking_budget=model_config["max_thinking_budget"], include_thoughts=True)})
    elif model_with_thinking and use_thinking == False:
        config.update({"thinking_config": types.ThinkingConfig(thinking_budget=model_config["min_thinking_budget"], include_thoughts=True)})

    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )  
    content, thinking_content = process_output(response)
    midi_loop: objects.Loop_G = response.parsed
    if midi_loop is None:
        raise ValueError("Google response did not include parsed loop content.")
    # Format into a message history for training and debugging purposes
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content}
    ]
    if thinking_content:
        messages.insert(2, {"role": "assistant", "content": thinking_content})
    # Calculate the cost of the generation
    cost = calc_cost(model, response.usage_metadata)
    # Save the messages to a JSON file for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost
