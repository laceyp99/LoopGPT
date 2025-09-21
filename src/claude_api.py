from anthropic import Anthropic
from dotenv import load_dotenv
import src.objects as objects
import src.utils as utils
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
    if model not in model_info["models"]["Anthropic"].keys():
        logger.warning(f"Model {model} not found in model info.")
        return False
    else:
        input_cost = model_info["models"]["Anthropic"][model]["cost"]["input"] / 1000000
        output_cost = model_info["models"]["Anthropic"][model]["cost"]["output"] / 1000000
        return (input_tokens * input_cost) + (output_tokens * output_cost)

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
        "max_tokens": model_info["models"]["Anthropic"][model]["max_tokens"],
        "system": pt_prompt,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # Add thinking parameter if enabled and model supports it
    if use_thinking and model_info["models"]["Anthropic"][model]["extended_thinking"]:
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
        api_params["temperature"] = 1.0  # Set temperature to 1 for thinking
    elif use_thinking and not model_info["models"]["Anthropic"][model]["extended_thinking"]:
        logger.warning(f"Extended thinking requested but not supported by model: {model}")
    
    # Make the API call for prompt translation generation
    completion = client.messages.create(**api_params)
    
    # Extract the generated prompt translation content
    input_tokens = 0
    output_tokens = 0
    content = "Prompt: "
    thinking_content = ""
    
    # Process the streaming response
    for chunk in completion:
        if chunk.type == 'message_start':
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                input_tokens = chunk.message.usage.input_tokens
                output_tokens += chunk.message.usage.output_tokens
        elif chunk.type == 'content_block_delta':
            if hasattr(chunk.delta, "thinking"):
                thinking_content += chunk.delta.thinking
            elif hasattr(chunk.delta, "text"):
                content += chunk.delta.text
        elif chunk.type == 'message_delta':
            if hasattr(chunk, "usage"):
                output_tokens += chunk.usage.output_tokens
    
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
        "max_tokens": model_info["models"]["Anthropic"][model]["max_tokens"],
        "system": loop_prompt,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "tool", "name": "build_MIDI_loop"},
        "stream": True
    }
    
    # Add thinking parameter if enabled and model supports it
    # Note: Tool use with thinking only supports "auto" or "none" tool_choice
    if use_thinking and model_info["models"]["Anthropic"][model]["extended_thinking"]:
        # For tool use with thinking, we need to change tool_choice to auto
        api_params["tool_choice"] = {"type": "auto"}
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        api_params["temperature"] = 1.0  # Set temperature to 1 for thinking
    elif use_thinking and not model_info["models"]["Anthropic"][model]["extended_thinking"]:
        logger.warning(f"Extended thinking requested but not supported by model: {model}")
    
    # Make the API call for structured output generation
    completion = client.messages.create(**api_params)
    
    # Extract the generated MIDI loop and convert it to a Loop object
    input_tokens = 0
    output_tokens = 0
    midi_loop_chunks = []
    thinking_content = ""
    
    # Process the streaming response
    for chunk in completion:
        if chunk.type == 'message_start':
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                input_tokens = chunk.message.usage.input_tokens
                output_tokens += chunk.message.usage.output_tokens
        elif chunk.type == 'content_block_delta':
            if hasattr(chunk.delta, "thinking"):
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