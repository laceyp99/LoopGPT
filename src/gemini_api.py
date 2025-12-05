from google import genai
from google.genai import types
from dotenv import load_dotenv
# import src.utils as utils
# import src.objects as objects
import utils, objects
import logging
import os
import sys
import json

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join('Prompts', 'loop gen.txt'), 'r') as f:
    loop_prompt = f.read()
with open(os.path.join('Prompts', 'prompt translation.txt'), 'r') as f:
    pt_prompt = f.read()

# Load model list and pricing details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

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
    # Gemini 3 and 2.5 Pro has a different cost structure based on token usage
    if model in ['gemini-2.5-pro', 'gemini-3-pro-preview']:
        if usage.prompt_token_count <= 200000:
            input_cost = model_info["models"]["Google"][model]["cost"]["input"]["<=200k"] / 1000000
            output_cost = model_info["models"]["Google"][model]["cost"]["output"]["<=200k"] / 1000000
            cache_cost = model_info["models"]["Google"][model]["cost"]["cache"]["<=200k"] / 1000000
        else:
            input_cost = model_info["models"]["Google"][model]["cost"]["input"][">200k"] / 1000000
            output_cost = model_info["models"]["Google"][model]["cost"]["output"][">200k"] / 1000000
            cache_cost = model_info["models"]["Google"][model]["cost"]["cache"][">200k"] / 1000000
        # Implicit caching storage is reported 5-6 minutes, so we can estimate storage cost per api call
        storage_cost = model_info["models"]["Google"][model]["cost"]["cache"]["storage hour"] * 0.1
    else:
        # For other models, use the default cost structure
        input_cost = model_info["models"]["Google"][model]["cost"]["input"] / 1000000
        output_cost = model_info["models"]["Google"][model]["cost"]["output"] / 1000000
        # Check if cache cost is defined for the model
        if hasattr(model_info["models"]["Google"][model]["cost"], "cache"):
            cache_cost = model_info["models"]["Google"][model]["cost"]["cache"]["text"] / 1000000
            # Implicit caching storage is reported 5-6 minutes, so we can estimate storage cost per api call
            storage_cost = model_info["models"]["Google"][model]["cost"]["cache"]["storage hour"] * 0.1
        else:
            cache_cost = 0
            storage_cost = 0
    # Gemini doesn't subtract cached tokens from prompt tokens, so we need to do that here
    if usage.cached_content_token_count:
        new_input_tokens = usage.prompt_token_count - usage.cached_content_token_count
        cached = usage.cached_content_token_count
    else:
        new_input_tokens = usage.prompt_token_count
        cached = 0
    # Calculate total cost
    return (new_input_tokens * input_cost) + (usage.candidates_token_count * output_cost) + (cached * cache_cost) + storage_cost

def prompt_gen(prompt, model, temp=0.0, use_thinking=False):
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

    # Configure the generation parameters based on whether extended thinking is enabled
    if model_info["models"]["Google"][model]["extended_thinking"] and use_thinking:
        config = types.GenerateContentConfig(
            system_instruction=pt_prompt,
            temperature=temp,
            response_mime_type='text/plain',
            thinking_config=types.ThinkingConfig(thinking_budget=model_info["models"]["Google"][model]["max_thinking_budget"])
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=pt_prompt,
            temperature=temp,
            response_mime_type='text/plain',
            thinking_config=types.ThinkingConfig(thinking_budget=model_info["models"]["Google"][model]["min_thinking_budget"])
        )
    
    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    # Format into a message history for training and debugging purposes
    content = response.text
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content}
    ]
    # Calculate the cost of the generation
    cost = calc_cost(model, response.usage_metadata)
    # Save the messages to a JSON file for debugging and training purposes
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return content, messages, cost

def loop_gen(prompt, model, temp=0.0, use_thinking=False):
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

    # Configure the generation parameters based on whether extended thinking is enabled
    config = {
        'response_mime_type': 'application/json',
        'response_schema': objects.Loop_G,
        'temperature': temp,
        'system_instruction': loop_prompt,
    }
    if model_info["models"]["Google"][model]["extended_thinking"] and use_thinking:
        config['thinking_config'] = types.ThinkingConfig(thinking_budget=model_info["models"]["Google"][model]["max_thinking_budget"])
    else:
        config['thinking_config'] = types.ThinkingConfig(thinking_budget=model_info["models"]["Google"][model]["min_thinking_budget"])
    
    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )  
    midi_loop: objects.Loop_G = response.parsed  
    # Format into a message history for training and debugging purposes
    assistant_response = response.text
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response}
    ]
    # Calculate the cost of the generation
    cost = calc_cost(model, response.usage_metadata)
    # Save the messages to a JSON file for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost