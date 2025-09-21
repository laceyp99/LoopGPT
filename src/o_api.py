import src.gpt_api as gpt_api
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
    client = gpt_api.initialize_openai_client()
    messages = [
        {"role": "developer", "content": pt_prompt},
        {"role": "user", "content": prompt},
    ]
    # Generate the response using the OpenAI chat API
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # Extract the generated content and calculate cost
    content = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    cost = gpt_api.calc_price(model, completion)
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
    client = gpt_api.initialize_openai_client()
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
    # Extract the generated content and calculate cost
    midi_loop = completion.choices[0].message.parsed
    messages.append({"role": "assistant", "content": f"{midi_loop}"})
    cost = gpt_api.calc_price(model, completion)
    # Save messages for debugging and training purposes
    utils.save_messages_to_json(messages, filename="loop")
    return midi_loop, messages, cost