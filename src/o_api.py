"""
This file contains API calls to OpenAI reasoning models.
It includes endpoints for text-based and MIDI-based generations using OpenAI's Structured Outputs.
"""
from openai import OpenAI
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
import logging
import json
import sys
import os

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
        completion: The response object from the API containing token usage and model details.

    Returns:
        float: The calculated price for the API call, or None if the model is unrecognized.
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


def prompt_gen(prompt, model):
    """
    Generate a text-based completion using the specified model and prompt.

    Args:
        prompt (str): The user prompt for generation.
        model (str): The model identifier to be used.

    Returns:
        tuple: (content, messages, cost) where content is the response text,
               messages is the conversation log, and cost is the calculated API cost.
    """
    # Initialize OpenAI client and build messages for the API call
    client = initialize_openai_client()
    messages = [
        {"role": "developer", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]
    # Generate the response using the OpenAI chat API
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # Extract the content from the response and append it to the messages
    content = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    # Calculate the cost of the API call
    cost = calc_price(model, completion)
    # Save messages for debugging and training purposes
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return content, messages, cost

def loop_gen(prompt, model):
    """
    Generate MIDI-based output using the specified model and prompt.
    
    Args:
        prompt (str): The user prompt for generating a MIDI bar.
        model (str): The model identifier to be used.

    Returns:
        tuple: (midi_loop, messages, cost) where midi_loop is the generated MIDI data,
               messages is the conversation log, and cost is the calculated API cost.
    """
    # Initialize OpenAI client and build messages for the API call
    client = initialize_openai_client()
    messages = [
        {"role": "developer", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    # Generate the response and parse it as a Loop object using structured outputs
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=objects.Loop,
    )
    # Extract the MIDI loop from the response and append it to the messages
    midi_loop = completion.choices[0].message.parsed
    messages.append({"role": "assistant", "content": f"{midi_loop}"})
    # Calculate the cost of the API call
    cost = calc_price(model, completion)
    # Save messages for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost