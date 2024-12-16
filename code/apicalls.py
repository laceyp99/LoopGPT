'''
This file holds all the functions and variables that are involved in making API calls to OpenAI.

This includes the following functions:
* check_api_key(): Checks if the OpenAI API key is set in the environment variables.
* calc_price(completion): Calculates the total price for an API call based on the token usage.
* prompt_translation(prompt, melody=True): Translates a user prompt into a fully detailed musical description for MIDI generation.
* generate_chords(prompt, temp=0.0): Generates a 4 bar MIDI chord progression based on the key, mode, and keywords provided.
* generate_melody(messages, temp=0.0): Generates a 4 bar MIDI melody based on the chord progression provided.
* generate_accompaniment(melody, temp=0.0): Generates a 4 bar MIDI chord progression based on the melody provided.
'''

# IMPORTS
from dotenv import load_dotenv # import the api key from the .env file
from openai import OpenAI # calling the OpenAI chat completions API
from pydantic import ValidationError # error handling for the API call
import os # access the environment variables
import code.objects as objects # import the objects module for the Bar and MelodyBar classes
from code.decorators import handle_errors
from code.exceptions import APICallError
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Use INFO or DEBUG level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# LOAD THE PROMPTS FROM THE PROMPTS DIRECTORY
with open('Prompts/prompt translation.txt', 'r') as prompt_translation_file:
    prompt_translation_prompt = prompt_translation_file.read()

with open('Prompts/prompt translation with melody.txt', 'r') as prompt_translation_with_melody_file:
    prompt_translation_with_melody_prompt = prompt_translation_with_melody_file.read()

with open('Prompts/chord generation.txt', 'r') as chord_generation_file:
    chord_generation_prompt = chord_generation_file.read()

with open('Prompts/melody generation.txt', 'r') as melody_generation_file:
    melody_generation_prompt = melody_generation_file.read()

with open('Prompts/accompaniment chord generation.txt', 'r') as accompaniment_file:
    accompaniment_prompt = accompaniment_file.read()

# LOAD API KEY AND CREATE CLIENT
load_dotenv() # load the .env file
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def check_api_key():
    if not os.getenv('OPENAI_API_KEY'):
        sys.exit("Error: OPENAI_API_KEY environment variable not set. Please set your OpenAI API key.")

check_api_key()

# OPENAI API PRICING (as of 2024-11-17)
# GPT-4o API PRICING (per token)
input_token_price = 0.00250 / 1000
cached_token_price = 0.00125 / 1000
output_token_price = 0.01000 / 1000
# GPT-4o-mini API PRICING (per token)
mini_input_token_price = 0.000150 / 1000
mini_cached_token_price = 0.075 / 1000000
mini_output_token_price = 0.000600 / 1000

def calc_price(completion, mini=False):
    """ Calculates the total price for an API call based on the token usage.
    
    Args:
        completion (dict): The completion object returned by the API call.
        mini (bool): Determines wether to calculate the cost for gpt-4o or gpt-4o-mini
    
    Returns:
        float: The total price for the API call.
    """
    if mini: # if using gpt-4o-mini
        if completion.usage.prompt_tokens_details.keys() == "cached_tokens": # if there are cached tokens, calculate the price accordingly
            total_price = ((completion.usage.prompt_tokens - completion.usage.prompt_tokens_details["cached_tokens"]) * mini_input_token_price) + (completion.usage.prompt_tokens_details["cached_tokens"] * mini_cached_token_price) + (completion.usage.completion_tokens * mini_output_token_price)
        else:
            total_price = (completion.usage.prompt_tokens * mini_input_token_price) + (completion.usage.completion_tokens * mini_output_token_price)
    else: # if using gpt-4o
        if completion.usage.prompt_tokens_details.keys() == "cached_tokens": # if there are cached tokens, calculate the price accordingly
            total_price = ((completion.usage.prompt_tokens - completion.usage.prompt_tokens_details["cached_tokens"]) * input_token_price) + (completion.usage.prompt_tokens_details["cached_tokens"] * cached_token_price) + (completion.usage.completion_tokens * output_token_price)
        else:
            total_price = (completion.usage.prompt_tokens * input_token_price) + (completion.usage.completion_tokens * output_token_price)
    return total_price

@handle_errors
def prompt_translation(prompt, melody=True):
    """ Translates a user prompt into a fully detailed musical description for MIDI generation.
    
    Args:
        prompt (str): The user's input prompt that needs to be translated into a musical description.
        melody (bool): A flag indicating whether the prompt is melody-focused (default is True).
   
    Returns:
        str: A detailed musical description based on the given prompt.
        List of Dictionaries: The message list that is used to further the conversation.
        float: The total cost associated with the API call.
    """
    # Check if the prompt is melody-focused
    if melody:
        system_prompt = prompt_translation_with_melody_prompt
    else:
        system_prompt = prompt_translation_prompt
    
    # Initialize the message list with the system message
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    # Add the user prompt to the message list
    messages.append({"role": "user", "content": prompt})
    # Make the API call to generate the translation
    completion = client.chat.completions.create(
        model="gpt-4o-mini", # gpt-4o-mini because this is an easier natural language task and the model is cheaper to use
        messages=messages,
        temperature=0.0
    )
    # Check if the API call returned None
    if completion == None:
        raise APICallError("API call returned None.")
    # Extract the translation from the API completion
    translation = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": translation})
    # Calculate the cost of the API call
    cost = calc_price(completion, mini=True)
    # print(translation)
    return translation, messages, cost

@handle_errors
def generate_chords(prompt, temp=0.0):
    """ Generates a 4 bar MIDI chord progression based on the prompt provided.

    Args:
        prompt (str): The musical description prompt for generating the chord progression
        temp (float): The temperature value for the API call (default is 0.0).
    
    Returns:
        List of Bar Objects: A list of 4 Bar objects, each containing a list of Chord objects
        List of Dictionaries: The message list that is used to further the conversation and generate melody.
        float: The total cost associated with the API call.
    """
    # Initialize the message list with the system message and the formatted user message
    messages = [
        {"role": "system", "content": chord_generation_prompt},
        {"role": "user", "content": prompt},
    ]
    # Initialize the list of bars and total cost for the API calls
    bars = []
    cost = 0
    # Loop through for 4 bars of generation
    for i in range(4):
        logging.info(f"Generating bar {i+1} of 4")

        # Make the API call to generate a bar of MIDI data based on the message list
        completion = client.beta.chat.completions.parse(
            model= "gpt-4o-2024-08-06", # gpt-4o-mini doesn't perform well for this task
            messages=messages,
            response_format=objects.Bar,
            temperature=temp
        )
        # Check if the API call returned None
        if completion == None:
            raise APICallError("API call returned None.")
        # Extract the MIDI data from the API completion and append it to the list of bars
        midi_loop = completion.choices[0].message.parsed
        bars.append(midi_loop)
        # Append the bar of MIDI data to the message list to be used as context for the next bar's generation
        messages.append(
            {
                "role": "assistant", 
                "content": f"{midi_loop}"
            }
        )
        cost += calc_price(completion, mini=False) 
    return bars, messages, cost

@handle_errors
def generate_melody(messages, temp=0.0):
    """ Generates a 4 bar MIDI melody based on the chord progression provided.
    
    Args:
        messages (list): A list of messages containing the chord progression for the melody generation.
        temp (float): The temperature value for the API call (default is 0.0).
    
    Returns:
        List of MelodyBar Objects: A list of 4 MelodyBar objects, each containing a list of Note objects.
        List of Dictionaries: The message list that is used to log the conversation.
        float: The total cost associated with the API call.
    """
    # Add a user message to prompt the model to create a melody using the chords generated
    messages.append({"role": "user", "content": melody_generation_prompt})
    # Initialize the list of melody bars
    melody_bars = []
    cost = 0
    # Loop through for 4 bars of generation
    for i in range(4):
        logging.info(f"Generating bar {i+1} of 4")
        # Make the API call to generate a bar of MIDI data based on the message list
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", # gpt-4o-mini doesn't perform well for this task
            messages=messages,
            response_format=objects.MelodyBar,
            temperature=temp
        )
        # Check if the API call returned None
        if completion == None:
            raise APICallError("API call returned None.")
        # Extract the MIDI data from the API completion and append it to the list of melody bars
        midi_loop = completion.choices[0].message.parsed
        melody_bars.append(midi_loop)
        # Append the bar of MIDI data to the message list to be used as context for the next bar's generation
        messages.append(
            {
                "role": "assistant", 
                "content": f"{midi_loop}"
            }
        )
        cost += calc_price(completion, mini=False) 
    return melody_bars, messages, cost

@handle_errors
def generate_accompaniment(melody, temp=0.0):
    """ Generates a 4 bar MIDI chord progression based on the melody provided.

    Args:
        melody (List[MelodyNote]): The user's melody to generate the accompaniment for.
        temp (float): The temperature value for the API call (default is 0.0).
    
    Returns:
        List of Bar Objects: A list of 4 Bar objects, each containing a list of Chord objects
        List of Dictionaries: The message list that is used to log the conversation.
        float: The total cost associated with the API calls.
    """
    # Initialize the message list with the system message and the formatted user message
    messages = [
        {"role": "system", "content": accompaniment_prompt},
        {"role": "user", "content": f"{melody}"},
    ]
    # Initialize the list of bars
    bars = []
    cost = 0
    # Loop through for 4 bars of generation
    for i in range(4):
        logging.info(f"Generating bar {i+1} of 4")

        # Make the API call to generate a bar of MIDI data based on the message list
        completion = client.beta.chat.completions.parse(
            model= "gpt-4o-2024-08-06", # gpt-4o-mini doesn't perform well for this task
            messages=messages,
            response_format=objects.Bar,
            temperature=temp
        )
        if completion == None:
            raise APICallError("API call returned None.")
        # Extract the MIDI data from the API completion and append it to the list of bars
        midi_loop = completion.choices[0].message.parsed
        bars.append(midi_loop)
        # Append the bar of MIDI data to the message list to be used as context for the next bar's generation
        messages.append(
            {
                "role": "assistant", 
                "content": f"{midi_loop}"
            }
        )
        cost += calc_price(completion, mini=False) 
    return bars, messages, cost