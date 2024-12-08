'''
This file holds all the functions and variables that are involved in making API calls to OpenAI.

This includes the following functions:
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
        float: The total cost associated with the API call.
    """
    # Initialize the message list for the API call
    messages = [
        {
            "role": "system",
            "content": """
            You are a music theory expert and MIDI generator assistant. When a user provides a weak, vague, or incomplete prompt, your role is to creatively interpret their input and fill in the gaps by providing detailed and musically coherent suggestions. Your response must offer a rich musical description that gives clear direction for MIDI generation. If the user’s prompt lacks specific details, you must suggest chord progressions, voicing styles, and other relevant musical elements to ensure the resulting composition has clear structure and purpose.

            Additionally, keep in mind that the MIDI generation will only be 4 bars long and will be created with up to sixteenth note zoom quality. All suggestions must fit within these parameters, ensuring the composition remains concise yet musically rich within these constraints.

            Your response should address the following aspects:
            Genre or Style: Determine the genre or style and musical elements that align with that genre.
            Chord Progression: If not specified, propose a chord progression based on the implied mood or genre, mentioning the key, any modulation points, and harmonic movements. Also determine the rhythymic component to the progression and ensure the progression fits within the 4-bar format.
            Voicing Style: Suggest how the chords should be voiced (e.g., open or close-voiced, spread across octaves, etc.), and mention any voicing style that fits the implied genre or mood (e.g., jazz, classical, pop).
            Mood and Atmosphere: Provide an overall description of the mood, dynamics, and texture. Should the piece be serene, energetic, dramatic, or minimalistic? Suggest ways to convey the implied emotion musically, keeping in mind the limitations of the short 4-bar form.
            
            Always ensure that your musical suggestions are coherent, detailed, and creative, while respecting the 4-bar length and sixteenth note zoom quality. Fill in any missing information to provide clear musical direction.
            """
        }
    ]
    # if the melody bool is True, then the prompt needs a melody focused description that aligns with the chord progression as well
    if melody:
        messages[0]["content"] += """
        You are to also generate a description of a melody with the same level of detail as the chord progression. The melody should complement the chord progression and be structured to fit within the 4-bar format. Provide a detailed description of the melody's phrasing, range, intervals, and style, ensuring it aligns with the underlying harmony and enhances the overall musical flow. Mention any rhythmic elements, melodic motifs, or stylistic features that would enhance the melody's connection to the chords and the implied genre or mood. Keep the melody engaging and expressive, with a clear sense of direction and resolution by the end of the 4th bar.
        
        Melody Characteristics: Describe the melody, focusing on its phrasing, range, intervals, and style. Should it be lyrical or rhythmic, simple or complex? Cover both the rhythymic and melodic structure for this generation. Ensure the melody works within the 4-bar structure and up to sixteenth note detail.
        """
    # Add the user prompt to the message list
    messages.append({"role": "user", "content": prompt})
    # Make the API call to generate the translation
    completion = client.chat.completions.create(
        model="gpt-4o-mini", # gpt-4o-mini because this is an easier natural language task and the model is cheaper to use
        messages=messages,
        temperature=0.0
    )
    if completion == None:
        raise APICallError("API call returned None.")
    # Extract the translation from the API completion
    translation = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": translation})
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
        {"role": "system", "content": "You are an advanced MIDI music generator. I will provide a prompt, and you will create a 4-bar chord progression in a structured and musically creative way. The progression should evolve from bar to bar, rather than repeating the same pattern, to keep it interesting and dynamic. Ensure that each bar develops harmonically while maintaining cohesion across the 4 bars. Use rich, varied rhythms that avoid predictability, and ensure that the progression naturally leads from one bar to the next. The chords should be voiced beautifully, with smooth transitions, and always include the root note as the lowest note in each chord. Pay attention to voice leading for a balanced, professional sound. The progression should be in a consistent key, with a satisfying harmonic resolution by the 4th bar, allowing the progression to loop seamlessly."},
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
        if i < 3:
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the chord progression with generating the next bar. Remember that the progression is only 4 bars long."
                }
            )
        # print(midi_loop)
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
        float: The total cost associated with the API call.
    """
    # Add a user message to prompt the model to create a melody using the chords generated
    messages.append({"role": "user", "content": """
        Now, create a melody that complements the 4-bar chord progression you just generated. Build the melody bar by bar, maintaining a strong connection with the underlying harmony while adding rhythmic and melodic variation. The melody should explore the scale creatively, staying in key and providing contrast and interest through syncopation, note lengths, and dynamics. Incorporate a rhythmic theme that avoids static repetition or simple straight 4ths, ideally including at least one note with a duration longer than one beat.

        Timing Consideration: When determining the placement of each note, remember that the timing system ranges from 1 to 16 for each bar, with base 1 indexing. This means if you want a note to start on the second beat, the start_beat value should be 5 (not 4). Ensure that the melody’s rhythmic elements respect this timing structure, and use this system to create rhythmic interest.

        Ensure that the melody develops naturally from bar to bar, with a clear sense of direction. It should resolve gracefully at the end of the 4th bar while being able to loop back into the beginning smoothly. Keep the melody’s length and structure tightly aligned with the chord progression, enhancing the overall musical flow.
    """})
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
        if i < 3:
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the melody generation with the next bar. Remember that the progression is only 4 bars long."
                }
            )
        # print(midi_loop)
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
        float: The total cost associated with the API calls.
    """
    # Initialize the message list with the system message and the formatted user message
    messages = [
        {"role": "system", "content": "You are an advanced MIDI music generator. I will provide a melody, and you will create a 4-bar chord progression in a structured and musically creative way to accompany the melody. The progression should evolve from bar to bar, rather than repeating the same pattern, to keep it interesting and dynamic. Ensure that each bar develops harmonically while maintaining cohesion with relation to the melody across the 4 bars. Use rich, varied rhythms that avoid predictability, and ensure that the progression naturally leads from one bar to the next. The chords should be voiced beautifully, with smooth transitions, and always include the root note as the lowest note in each chord. Pay attention to voice leading for a balanced, professional sound. The progression should be in a consistent key, with a satisfying harmonic resolution by the 4th bar, allowing the progression to loop seamlessly."},
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
        if i < 3:
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the chord progression with generating the next bar. Remember that the progression is only 4 bars long. Try to keep the progression in line with the melody provided."
                }
            )
        # print(midi_loop)
        cost += calc_price(completion, mini=False) 
    return bars, cost