"""
This file contains API calls to OpenAI language GPT models.
It includes endpoints for text-based prompt translation and MIDI-based generations using OpenAI's Structured Outputs.
"""
from openai import OpenAI
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join('Prompts', 'loop gen.txt'), 'r') as f:
    loop_prompt = f.read()
with open(os.path.join('Prompts', 'prompt translation.txt'), 'r') as f:
    pt_prompt = f.read()

# Load model list and pricing details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

def initialize_openai_client():
    """
    Initializes and returns an OpenAI client using API key from the .env file.
    
    Returns:
        OpenAI: The OpenAI client instance.
    """
    load_dotenv(os.path.join("src", ".env"))
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.strip():
        logger.error("OPENAI_API_KEY is not set!")
    return OpenAI(api_key=api_key)

def calc_price(model, completion):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        completion: The response object from the API containing token usage.
        model (str): The model identifier used for the request.

    Returns:
        float: Calculated price for the API call.
    """
    usage = completion.usage
    # Determine cached tokens if available
    cached_tokens = 0
    if hasattr(completion.usage, "prompt_tokens_details") and "cached_tokens" in completion.usage.prompt_tokens_details:
        cached_tokens = completion.usage.prompt_tokens_details["cached_tokens"]
    else:
        logger.info("No cached tokens found in usage details.")    
    # Calculate the total price based on the model and token usage
    input_cost = model_info["models"]["OpenAI"][model]["cost"]["input"] / 1000000
    output_cost = model_info["models"]["OpenAI"][model]["cost"]["output"] / 1000000
    cached_input_cost = model_info["models"]["OpenAI"][model]["cost"]["cached input"] / 1000000
    total_price = (input_cost * usage.prompt_tokens + output_cost * usage.completion_tokens + cached_input_cost * cached_tokens)
    
    # Return the total price based on the model and the completion
    return total_price

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
    # Initialize OpenAI client and build messages for the API call
    client = initialize_openai_client()
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the API call for chat completion
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    # Extract the generated content and calculate cost
    content = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    cost = calc_price(model, completion)
    # Save messages for debugging/training purposes
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
    # Initialize OpenAI client and build messages for the API call
    client = initialize_openai_client()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the structured output API call for MIDI data generation
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=objects.Loop,
        temperature=temp
    )
    # Extract the generated MIDI loop and calculate cost
    midi_loop = completion.choices[0].message.parsed
    messages.append({"role": "assistant", "content": str(midi_loop)})
    cost = calc_price(model, completion)
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost