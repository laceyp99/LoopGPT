from ollama import chat
from pydantic import BaseModel
import ollama
# import src.utils as utils
# import src.objects as objects
import objects, utils, midi_processing
import logging
import json
import sys
import os
from mido import MidiFile

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load prompt files
with open(os.path.join('Prompts', 'loop gen.txt'), 'r') as f:
    loop_prompt = f.read()
with open(os.path.join('Prompts', 'prompt translation.txt'), 'r') as f:
    pt_prompt = f.read()

ollama_host = 'http://localhost:11434'

def initialize_ollama_client(host_address):
    client = ollama.Client(
        host=host_address
    )
    return client

# Load model list
model_list = [model.model for model in initialize_ollama_client(ollama_host).list().models]
print(model_list)

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
    # Initialize Ollama client and build messages for the API call
    client = initialize_ollama_client(ollama_host)
    messages = [
        {"role": "system", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the API call for chat completion
    completion = chat(
        model=model,
        messages=messages,
        # temperature=temp
    )
    # Extract the generated content and calculate cost
    content = completion.message.content
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
        tuple: (midi_loop, messages, cost)
    """
    # Initialize Ollama client and build messages for the API call
    client = initialize_ollama_client(ollama_host)
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the structured output API call for MIDI data generation
    completion = chat(
        model=model,
        messages=messages,
        format=objects.Loop.model_json_schema(),
        # temperature=temp
    )
    # Extract the generated MIDI loop and calculate cost
    midi_loop = objects.Loop.model_validate_json(completion.message.content)
    messages.append({"role": "assistant", "content": str(midi_loop)})
    # Save messages for debugging/training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, 0

# if __name__ == "__main__":
#     prompt = "an arpeggiator in G Major using only eighth note lengths "
#     model = model_list[4]  # Select the fifth model from the list
#     midi_loop, messages, cost = loop_gen(prompt, model)
#     midi = MidiFile() 
#     midi_processing.loop_to_midi(midi, midi_loop, times_as_string=False)
#     output_path = "output.mid"
#     midi.save(output_path)