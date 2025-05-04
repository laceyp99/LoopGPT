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
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-pro-exp-03-25', 
    'gemini-2.0-flash', 
    'gemini-2.0-flash-lite', 
    'gemini-1.5-flash', 
    'gemini-1.5-pro'
]

# COST OF MODELS
# NOTE: The Gemini API "free tier" is offered with lower rate limits.
# Google AI Studio usage is completely free in all available countries.
# The Gemini API "paid tier" comes with higher rate limits, additional features,
# and different data handling.
g_2_flash_input_cost        = 0.10 / 1000000
g_2_flash_output_cost       = 0.40 / 1000000

g_2_flash_lite_input_cost   = 0.075 / 1000000
g_2_flash_lite_output_cost  = 0.30 / 1000000

g_1_5_flash_input_cost      = 0.15 / 1000000
g_1_5_flash_output_cost     = 0.60 / 1000000
g_1_5_flash_cached_cost     = 0.0375 / 1000000

gemini_1_5_pro_input_cost   = 2.50 / 1000000
gemini_1_5_pro_output_cost  = 10.00 / 1000000
gemini_1_5_pro_cached_cost  = 0.625 / 1000000

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
        completion: The response object from the API containing token usage.
        model (str): The model identifier used for the request.

    Returns:
        float: Calculated price for the API call.
    """
    if cached_tokens is None:
        cached_tokens = 0

    if model == 'gemini-2.0-flash':
        return (input_tokens * g_2_flash_input_cost) + (output_tokens * g_2_flash_output_cost)
    elif model == 'gemini-2.0-flash-lite':
        return (input_tokens * g_2_flash_lite_input_cost) + (output_tokens * g_2_flash_lite_output_cost)
    elif model == 'gemini-1.5-flash':
        return ((input_tokens * g_1_5_flash_input_cost) + (output_tokens * g_1_5_flash_output_cost) + (cached_tokens * g_1_5_flash_cached_cost))
    elif model == 'gemini-1.5-pro':
        return ((input_tokens * gemini_1_5_pro_input_cost) + (output_tokens * gemini_1_5_pro_output_cost) + (cached_tokens * gemini_1_5_pro_cached_cost))
    elif model == 'gemini-2.5-pro-exp-03-25' or model == 'gemini-2.5-flash-preview-04-17':
        # No API cost attached yet for this model due to it being experimental
        return 0
    else:
        return None

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