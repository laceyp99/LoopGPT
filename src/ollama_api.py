import ollama
import src.utils as utils
import src.objects as objects
import logging
import sys
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join('Prompts', 'loop gen.txt'), 'r') as f:
    loop_prompt = f.read()
with open(os.path.join('Prompts', 'prompt translation.txt'), 'r') as f:
    pt_prompt = f.read()

def initialize_ollama_client(host_address="http://localhost:11434"):
    """Initializes and returns an Ollama client.

    Args:
        host_address (str, optional): The host address for the Ollama API. Defaults to "http://localhost:11434", assuming the API is running locally.

    Returns:
        ollama.Client: The initialized Ollama client.
    """
    load_dotenv(os.path.join('src', '.env'))
    if os.getenv('OLLAMA_API_HOST_ADDRESS'):
        client = ollama.Client(
            host=os.getenv('OLLAMA_API_HOST_ADDRESS')
        )
    else:
        client = ollama.Client(
            host=host_address
        )
    return client

# Load model list
model_list = [model.model for model in initialize_ollama_client().list().models]

def prompt_gen(prompt, model, temp=0.0):
    """
    Generate text content using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate text.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.

    Returns:
        tuple: (content, messages, cost=0 for Ollama)
    """
    # Initialize Ollama client and build messages for the API call
    client = initialize_ollama_client()
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the API call for chat completion
    completion = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temp
        }
    )
    # Extract the generated content
    content = completion.message.content
    if completion.message.thinking:
        messages.append({"role": "assistant", "content": completion.message.thinking})
    messages.append({"role": "assistant", "content": content})
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="prompt_translation")
    return content, messages, 0

def loop_gen(prompt, model, temp=0.0):
    """
    Generate a MIDI bar (chord progression/melody) using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.

    Returns:
        tuple: (midi_loop, messages, cost=0 for Ollama)
    """
    # Initialize Ollama client and build messages for the API call
    client = initialize_ollama_client()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the structured output API call for MIDI data generation
    completion = client.chat(
        model=model,
        messages=messages,
        format=objects.Loop.model_json_schema(),
        options={
            "temperature": temp
        }
    )
    # Extract the generated MIDI loop
    if completion.message.content:
        midi_loop = objects.Loop.model_validate_json(completion.message.content)
    else:
        print("No content returned from the model")
        midi_loop = None
    if completion.message.thinking:
        messages.append({"role": "assistant", "content": completion.message.thinking})
    messages.append({"role": "assistant", "content": str(midi_loop)})
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, 0