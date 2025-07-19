"""
This file contains API calls to Anthropic models.
It includes endpoints for text-based prompt translation and MIDI-based generations using Anthropic's Tool Use.
"""
from anthropic import Anthropic
from dotenv import load_dotenv
import src.objects as objects
import src.utils as utils
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

def initialize_anthropic_client():
    """
    Initializes and returns an Anthropic client using API key from the .env file.
    
    Returns:
        Anthropic: The Anthropic client instance.
    """
    load_dotenv(os.path.join("src", ".env"))
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or not api_key.strip():
        logger.error("ANTHROPIC_API_KEY is not set!")
    return Anthropic(api_key=api_key)

# List of available models from Anthropic 
model_list = [
    # "claude-3-7-sonnet-latest", Needs to use streaming https://github.com/anthropics/anthropic-sdk-python#long-requests
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-latest",
    "claude-3-haiku-20240307"
]
# Model pricing per million tokens (MTok)

# Claude Opus 4 Pricing
claude_opus_4_input_cost        = 15.00 / 1000000
claude_opus_4_output_cost       = 75.00 / 1000000
# Claude Sonnet 4 Pricing
claude_sonnet_4_input_cost      = 3.00 / 1000000
claude_sonnet_4_output_cost     = 15.00 / 1000000
# Claude 3.7 Sonnet Pricing
claude_37_sonnet_input_cost     = 3.00 / 1000000
claude_37_sonnet_output_cost    = 15.00 / 1000000
# Claude 3.5 Sonnet Pricing
claude_35_sonnet_input_cost     = 3.00 / 1000000
claude_35_sonnet_output_cost    = 15.00 / 1000000
# Claude 3.5 Haiku Pricing
claude_35_haiku_input_cost      = 0.80 / 1000000
claude_35_haiku_output_cost     = 4.00 / 1000000
# Claude 3 Haiku Pricing
claude_3_haiku_input_cost       = 0.25 / 1000000
claude_3_haiku_output_cost      = 1.25 / 1000000

def calc_price(response):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        response: The response object from the API containing token usage.

    Returns:
        float: Calculated price for the API call.
    """
    model = response.model
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    if model == "claude-opus-4-20250514":
        return (input_tokens * claude_opus_4_input_cost) + (output_tokens * claude_opus_4_output_cost)
    elif model == "claude-sonnet-4-20250514":
        return (input_tokens * claude_sonnet_4_input_cost) + (output_tokens * claude_sonnet_4_output_cost)
    elif model == "claude-3-7-sonnet-latest":
        return (input_tokens * claude_37_sonnet_input_cost) + (output_tokens * claude_37_sonnet_output_cost)
    elif model in ["claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620"]:
        return (input_tokens * claude_35_sonnet_input_cost) + (output_tokens * claude_35_sonnet_output_cost)
    elif model == "claude-3-5-haiku-latest":
        return (input_tokens * claude_35_haiku_input_cost) + (output_tokens * claude_35_haiku_output_cost)
    elif model == "claude-3-haiku-20240307":
        return (input_tokens * claude_3_haiku_input_cost) + (output_tokens * claude_3_haiku_output_cost)
    else:
        logger.warning(f"Cost calculation not implemented for model: {model}. Returning 0.")
        return 0

def calculate_output_tokens(model):
    """Calculates the maximum number of output tokens based on the model to be used for the API calls as a parameter.

    Args:
        model (str): The model name to be used for the API calls.

    Raises:
        ValueError: If the model name is not recognized.

    Returns:
        int: The maximum number of output tokens for the specified model.
    """
    if model == "claude-opus-4-20250514":
        return 32000
    elif model == "claude-sonnet-4-20250514" or model == "claude-3-7-sonnet-20250219":
        return 64000
    elif model == "claude-3-5-sonnet-latest" or model == "claude-3-5-sonnet-20240620" or model == "claude-3-5-haiku-latest":
        return 8192
    elif model == "claude-3-opus-latest" or model == "claude-3-haiku-20240307":
        return 4096
    else:
        raise ValueError("Invalid model selected.")

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
    # Initialize Anthropic client and make the API call
    client = initialize_anthropic_client()
    completion = client.messages.create(
        model=model,
        max_tokens=calculate_output_tokens(model),
        system=pt_prompt,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract the generated content and format it into a message history
    content = completion.content[0].text
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content}
    ]
    cost = calc_price(completion)
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
    # Initialize Anthropic client and make the API call
    client = initialize_anthropic_client()
    # Define the tool schema for the loop generation
    loop_schema = objects.Loop.model_json_schema()
    tools = [
        {
            "name": "build_MIDI_loop",
            "description": "builds a music loop in MIDI format",
            "input_schema": loop_schema
        }
    ]
    # Make the API call for structured output generation
    completion = client.messages.create(
        model=model,
        max_tokens=calculate_output_tokens(model),
        system=loop_prompt,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice={"type": "tool", "name": "build_MIDI_loop"},
    )
    # Extract the generated MIDI loop and convert it to a Loop object
    midi_loop = completion.content[0].input
    loop = objects.Loop.model_validate_json(json.dumps(midi_loop))
    # Create the messages history 
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": midi_loop}
    ]
    cost = calc_price(completion)
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return loop, messages, cost