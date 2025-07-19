"""
This file contains all the API calls to the Google Gemini models.
It includes endpoints for text-based prompt translation and MIDI-based generations using Gemini's Structured Outputs.
"""
from google import genai
from google.genai import types
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join('Prompts', 'loop gen.txt'), 'r') as f:
    loop_prompt = f.read()
with open(os.path.join('Prompts', 'prompt translation.txt'), 'r') as f:
    pt_prompt = f.read()

# List of available models from Google Gemini (as of 3/29/2025)
model_list = [
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite-preview-06-17', 
    'gemini-2.0-flash', 
    'gemini-2.0-flash-lite'
]

# COST OF MODELS
# NOTE: The Gemini API "free tier" is offered with lower rate limits.
# Google AI Studio usage is completely free in all available countries.
# The Gemini API "paid tier" comes with higher rate limits, additional features,
# and different data handling.

def initialize_gemini_client():
    """Initializes and returns a Gemini client using API key from the .env file or the default key.
    
    Returns:
        genai.Client: The Gemini client instance.
    """
    load_dotenv(os.path.join('src', '.env'))
    user_key = os.getenv('GEMINI_API_KEY')
    if user_key and user_key.strip():
        logger.info("Using GEMINI_API_KEY")
        return genai.Client(api_key=user_key)
    else:
        default_key = os.getenv('DEFAULT_GEMINI_API_KEY')
        logger.info("Using DEFAULT_GEMINI_API_KEY")
        return genai.Client(api_key=default_key)

def calc_cost(model, input_tokens, output_tokens, cached_tokens=0):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        model (str): The model identifier used for the request.
        input_tokens (int): The number of input tokens.
        output_tokens (int): The number of output tokens.
        cached_tokens (int, optional): The number of cached tokens. Defaults to 0.

    Returns:
        float: Calculated price for the API call.
    """
    if cached_tokens is None:
        cached_tokens = 0

    # Gemini 2.5 Pro
    if model == 'gemini-2.5-pro':
        if input_tokens <= 200000:
            input_cost = 1.25 / 1000000
            output_cost = 10.00 / 1000000
            cache_cost = 0.31 / 1000000
        else:
            input_cost = 2.50 / 1000000
            output_cost = 15.00 / 1000000
            cache_cost = 0.625 / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost) + (cached_tokens * cache_cost)

    # Gemini 2.5 Flash
    elif model == 'gemini-2.5-flash':
        input_cost = 0.30 / 1000000
        output_cost = 2.50 / 1000000
        cache_cost = 0.075 / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost) + (cached_tokens * cache_cost)

    # Gemini 2.5 Flash-Lite Preview
    elif model == 'gemini-2.5-flash-lite-preview-06-17':
        input_cost = 0.10 / 1000000
        output_cost = 0.40 / 1000000
        cache_cost = 0.025 / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost) + (cached_tokens * cache_cost)

    # Gemini 2.0 Flash
    elif model == 'gemini-2.0-flash':
        input_cost = 0.10 / 1000000
        output_cost = 0.40 / 1000000
        cache_cost = 0.025 / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost) + (cached_tokens * cache_cost)

    # Gemini 2.0 Flash-Lite
    elif model == 'gemini-2.0-flash-lite':
        input_cost = 0.075 / 1000000
        output_cost = 0.30 / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost)
    else:
        logger.warning(f"Cost calculation not implemented for model: {model}")
        return 0

def prompt_gen(prompt, model, temp=0.0):
    """
    Generate text content using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate text.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.

    Returns:
        tuple: (content, messages, cost)
    """
    client = initialize_gemini_client()
    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=pt_prompt,
            temperature=temp,
            response_mime_type='text/plain',
        )
    )
    content = response.text
    # Format into a message history for training and debugging purposes
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content}
    ]
    # Calculate the cost of the generation
    usage = response.usage_metadata
    cost = calc_cost(model, usage.prompt_token_count, usage.candidates_token_count, usage.cached_content_token_count)
    # Save the messages to a JSON file for debugging and training purposes
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return content, messages, cost

def loop_gen(prompt, model, temp=0.0):
    """
    Generate a MIDI bar (chord progression/melody) using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.

    Returns:
        tuple: (midi_loop, messages, cost)
    """
    client = initialize_gemini_client()
    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': objects.Loop_G,
            'temperature': temp,
            'system_instruction': loop_prompt,
        },
    )    
    # Extract the json string response
    assistant_response = response.text
    # Format into a message history for training and debugging purposes
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response}
    ]
    # Convert the response to the appropriate object
    midi_loop: objects.Loop_G = response.parsed
    # Calculate the cost of the generation
    usage = response.usage_metadata
    cost = calc_cost(model, usage.prompt_token_count, usage.candidates_token_count, usage.cached_content_token_count)
    # Save the messages to a JSON file for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost