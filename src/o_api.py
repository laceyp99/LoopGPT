"""
This file contains API calls to OpenAI reasoning models.
It includes endpoints for text-based and MIDI-based generations using OpenAI's Structured Outputs.
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

# List of available models from OpenAI Structured Outputs (as of 3/9/2025)
model_list = ['o1', 'o3', 'o3-mini', 'o4-mini', "gpt-5", "gpt-5-mini", "gpt-5-nano"]

# OPENAI API PRICING (per token, rates for 1M tokens)

# Pricing for the gpt-5 models
# gpt-5
gpt5_input_token_price              = 1.25	/ 1000000
gpt5_cached_input_token_price       = 0.125 / 1000000
gpt5_output_token_price             = 10.00 / 1000000
# gpt-5-mini
gpt5_mini_input_token_price         = 0.25	/ 1000000
gpt5_mini_cached_input_token_price  = 0.025 / 1000000
gpt5_mini_output_token_price        = 2.00  / 1000000
# gpt-5-nano
gpt5_nano_input_token_price         = 0.05	/ 1000000
gpt5_nano_cached_input_token_price  = 0.005 / 1000000
gpt5_nano_output_token_price        = 0.40  / 1000000

# Pricing for the o1 model
o1_input_token_price                = 15.00 / 1000000
o1_cached_input_token_price         = 7.500 / 1000000
o1_output_token_price               = 60.00 / 1000000

# Pricing for the o3 model
o3_input_token_price                = 2.00 / 1000000
o3_cached_input_token_price         = 0.50 / 1000000
o3_output_token_price               = 8.00 / 1000000

# Pricing for the o3 mini model
o3_mini_input_token_price           = 1.10 / 1000000
o3_mini_cached_input_token_price    = 0.55 / 1000000
o3_mini_output_token_price          = 4.40 / 1000000

# Pricing for the o4 mini model
o4_mini_input_token_price           = 1.10 / 1000000
o4_mini_cached_input_token_price    = 0.275 / 1000000
o4_mini_output_token_price          = 4.40 / 1000000

def calc_price(completion):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        completion: The response object from the API containing token usage and model details.

    Returns:
        float: The calculated price for the API call, or None if the model is unrecognized.
    """
    
    model = completion.model
    usage = completion.usage
    cached_tokens = usage.prompt_tokens_details.cached_tokens
    
    # Calculate the total price based on the model and token usage
    if "o1" in model:
        total_price = (o1_input_token_price * usage.prompt_tokens + o1_output_token_price * usage.completion_tokens + o1_cached_input_token_price * cached_tokens)
    elif "o3-mini" in model:
        total_price = (o3_mini_input_token_price * usage.prompt_tokens + o3_mini_output_token_price * usage.completion_tokens + o3_mini_cached_input_token_price * cached_tokens)
    elif "o3" in model:
        total_price = (o3_input_token_price * usage.prompt_tokens + o3_output_token_price * usage.completion_tokens + o3_cached_input_token_price * cached_tokens)
    elif "o4-mini" in model:
        total_price = (o4_mini_input_token_price * usage.prompt_tokens + o4_mini_output_token_price * usage.completion_tokens + o4_mini_cached_input_token_price * cached_tokens)
    elif "gpt-5" in model:
        total_price = (gpt5_input_token_price * usage.prompt_tokens + gpt5_output_token_price * usage.completion_tokens + gpt5_cached_input_token_price * cached_tokens)
    elif "gpt-5-mini" in model:
        total_price = (gpt5_mini_input_token_price * usage.prompt_tokens + gpt5_mini_output_token_price * usage.completion_tokens + gpt5_mini_cached_input_token_price * cached_tokens)
    elif "gpt-5-nano" in model:
        total_price = (gpt5_nano_input_token_price * usage.prompt_tokens + gpt5_nano_output_token_price * usage.completion_tokens + gpt5_nano_cached_input_token_price * cached_tokens)
    else:
        logger.warning("Pricing not defined for model '%s'.", model)
        total_price = 0
    
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
    cost = calc_price(completion)
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
    cost = calc_price(completion)
    # Save messages for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost