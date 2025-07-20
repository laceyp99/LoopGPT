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
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
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

def supports_thinking(model):
    """
    Check if a model supports extended thinking functionality.
    
    Args:
        model (str): The model name to check.
        
    Returns:
        bool: True if the model supports extended thinking, False otherwise.
    """
    thinking_models = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514", 
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest"
    ]
    return model in thinking_models

def calc_price(model, input_tokens, output_tokens):
    """
    Calculate the cost for a given completion based on token usage.

    Args:
        model (str): The model used.
        input_tokens (int): The input token count from the start event.
        output_tokens (int): The total output token count.

    Returns:
        float: Calculated price for the API call.
    """
    if model == "claude-opus-4-20250514":
        return (input_tokens * claude_opus_4_input_cost) + (output_tokens * claude_opus_4_output_cost)
    elif model == "claude-sonnet-4-20250514":
        return (input_tokens * claude_sonnet_4_input_cost) + (output_tokens * claude_sonnet_4_output_cost)
    elif model in ["claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"]:
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
    elif model in ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest"]:
        return 64000
    elif model == "claude-3-5-sonnet-latest" or model == "claude-3-5-sonnet-20240620" or model == "claude-3-5-haiku-latest":
        return 8192
    elif model == "claude-3-opus-latest" or model == "claude-3-haiku-20240307":
        return 4096
    else:
        raise ValueError("Invalid model selected.")

def prompt_gen(prompt, model, temp=0.0, use_thinking=False, thinking_budget=10000):
    """
    Generate text content using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate text.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        use_thinking (bool, optional): Enable extended thinking for supported models. Defaults to False.
        thinking_budget (int, optional): Token budget for thinking when enabled. Defaults to 10000.

    Returns:
        tuple: (content, messages, cost)
    """
    # Initialize Anthropic client and make the API call
    client = initialize_anthropic_client()
    
    # Prepare API call parameters
    api_params = {
        "model": model,
        "max_tokens": calculate_output_tokens(model),
        "system": pt_prompt,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # Add thinking parameter if enabled and model supports it
    if use_thinking and supports_thinking(model):
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
        api_params["temperature"] = 1.0  # Set temperature to 1 for thinking
    elif use_thinking and not supports_thinking(model):
        logger.warning(f"Extended thinking requested but not supported by model: {model}")
    
    completion = client.messages.create(**api_params)
    
    input_tokens = 0
    output_tokens = 0
    content = "Prompt: "
    thinking_content = ""
    
    for chunk in completion:
        if chunk.type == 'message_start':
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                input_tokens = chunk.message.usage.input_tokens
                output_tokens += chunk.message.usage.output_tokens
        elif chunk.type == 'content_block_delta':
            if hasattr(chunk.delta, "thinking"):
                # Handle thinking content
                thinking_content += chunk.delta.thinking
            elif hasattr(chunk.delta, "text"):
                content += chunk.delta.text
        elif chunk.type == 'message_delta':
            if hasattr(chunk, "usage"):
                output_tokens += chunk.usage.output_tokens
    
    # Include thinking content in the response if present
    # if thinking_content:
    #     content = f"Thinking: {thinking_content}\n\n{content}"
    
    # Extract the generated content and format it into a message history
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": thinking_content},
        {"role": "assistant", "content": content}
    ]
    cost = calc_price(model, input_tokens, output_tokens)
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return content, messages, cost

def loop_gen(prompt, model, temp=0.0, use_thinking=False, thinking_budget=10000):
    """
    Generate a MIDI bar (chord progression/melody) using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        use_thinking (bool, optional): Enable extended thinking for supported models. Defaults to False.
        thinking_budget (int, optional): Token budget for thinking when enabled. Defaults to 10000.

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
    
    # Prepare API call parameters
    api_params = {
        "model": model,
        "max_tokens": calculate_output_tokens(model),
        "system": loop_prompt,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "tool", "name": "build_MIDI_loop"},
        "stream": True
    }
    
    # Add thinking parameter if enabled and model supports it
    # Note: Tool use with thinking only supports "auto" or "none" tool_choice
    if use_thinking and supports_thinking(model):
        # For tool use with thinking, we need to change tool_choice to auto
        api_params["tool_choice"] = {"type": "auto"}
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        api_params["temperature"] = 1.0  # Set temperature to 1 for thinking
    elif use_thinking and not supports_thinking(model):
        logger.warning(f"Extended thinking requested but not supported by model: {model}")
    
    # Make the API call for structured output generation
    completion = client.messages.create(**api_params)
    # Extract the generated MIDI loop and convert it to a Loop object
    input_tokens = 0
    output_tokens = 0
    midi_loop_chunks = []
    thinking_content = ""
    
    for chunk in completion:
        if chunk.type == 'message_start':
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                input_tokens = chunk.message.usage.input_tokens
                output_tokens += chunk.message.usage.output_tokens
        elif chunk.type == 'content_block_delta':
            if hasattr(chunk.delta, "thinking"):
                # Handle thinking content
                thinking_content += chunk.delta.thinking
            elif hasattr(chunk.delta, "partial_json"):
                midi_loop_chunks.append(chunk.delta.partial_json)
        elif chunk.type == 'message_delta':
            if hasattr(chunk, "usage"):
                output_tokens += chunk.usage.output_tokens
    
    midi_loop = ''.join(midi_loop_chunks)
    loop = objects.Loop.model_validate_json(midi_loop)
    
    # Create the messages history 
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": thinking_content},
        {"role": "assistant", "content": midi_loop}
    ]
    cost = calc_price(model, input_tokens, output_tokens)
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return loop, messages, cost