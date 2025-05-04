"""
This file contains API calls to OpenAI language GPT models.
It includes endpoints for text-based prompt translation and MIDI-based generations using OpenAI's Structured Outputs.
"""
from openai import OpenAI
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

# List of available models from OpenAI Structured Outputs (as of 3/29/2025)
model_list = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini']

# OPENAI API PRICING (per token, rates for 1M tokens)
# Pricing for GPT-4o model
input_token_price               = 2.50 / 1000000
cached_token_price              = 1.25 / 1000000
output_token_price              = 1.00 / 1000000

# Pricing for GPT-4o-mini model
mini_input_token_price          = 0.150 / 1000000
mini_cached_token_price         = 0.075 / 1000000
mini_output_token_price         = 0.600 / 1000000

# Pricing for GPT-4.1 models
gpt41_input_price               = 2.00 / 1000000
gpt41_cached_input_price        = 0.50 / 1000000
gpt41_output_price              = 8.00 / 1000000

gpt41_mini_input_price          = 0.40 / 1000000
gpt41_mini_cached_input_price   = 0.10 / 1000000
gpt41_mini_output_price         = 1.60 / 1000000

gpt41_nano_input_price          = 0.10 / 1000000
gpt41_nano_cached_input_price   = 0.025 / 1000000
gpt41_nano_output_price         = 0.40 / 1000000

def calc_price(completion, model):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        completion: The response object from the API containing token usage.
        model (str): The model identifier used for the request.

    Returns:
        float: Calculated price for the API call.
    """
    # Determine cached tokens if available
    cached_tokens = 0
    if hasattr(completion.usage, "prompt_tokens_details") and "cached_tokens" in completion.usage.prompt_tokens_details:
        cached_tokens = completion.usage.prompt_tokens_details["cached_tokens"]

    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    # Calculate the total price based on the model and token usage
    if model == "gpt-4o-mini":
        total_price = ((prompt_tokens - cached_tokens) * mini_input_token_price) + (cached_tokens * mini_cached_token_price) + (completion_tokens * mini_output_token_price)
    elif model == "gpt-4o":
        total_price = ((prompt_tokens - cached_tokens) * input_token_price) + (cached_tokens * cached_token_price) + (completion_tokens * output_token_price)
    elif model == "gpt-4.1":
        total_price = ((prompt_tokens - cached_tokens) * gpt41_input_price) + (cached_tokens * gpt41_cached_input_price) + (completion_tokens * gpt41_output_price)
    elif model == "gpt-4.1-mini":
        total_price = ((prompt_tokens - cached_tokens) * gpt41_mini_input_price) + (cached_tokens * gpt41_mini_cached_input_price) + (completion_tokens * gpt41_mini_output_price)
    elif model == "gpt-4.1-nano":
        total_price = ((prompt_tokens - cached_tokens) * gpt41_nano_input_price) + (cached_tokens * gpt41_nano_cached_input_price) + (completion_tokens * gpt41_nano_output_price)
    else:
        logger.warning("Pricing not defined for model '%s'. Falling back to zero cost.", model)
        total_price = 0
    
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
    cost = calc_price(completion, model)
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
    cost = calc_price(completion, model)
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost