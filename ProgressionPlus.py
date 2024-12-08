'''
This file does the main processing for a chord progression generation with additional melody generation options. It takes in the user prompt and generates the MIDI file.
'''

# IMPORTS
from mido import MidiFile
import code.apicalls as apicalls
import code.midi_processing as midi_processing
import code.utils as utils 
import json , os

def main(prompt_dict, melody=True, visualize=False, play=False, filename="N/A"):
    """ Generate a chord progression and possibly a melody based on the user prompt.

    Args:
        prompt_dict (Dict): A dictionary containing the user's prompt for the generation.
        melody (bool, optional): A boolean that decides whether to generate the melody on top of the chord progression. Defaults to True.
        visualize (bool, optional): A boolean that will visualize the MIDI in a pop up window if set to True. Defaults to False.
        play (bool, optional): A boolean that will play the MIDI after generation if set to True. Defaults to False.
        filename (str, optional): The filename of the generated MidiFile. Defaults to "N/A".
    """
    # Define the filename if not provided
    if filename == "N/A":
        filename = f"ProgressionPlus/{prompt_dict["t"]}/{prompt_dict['key']} {prompt_dict['mode']} {prompt_dict['keywords']}.mid"
    
    json_filename = f"Training Examples/{prompt_dict["t"]}/{prompt_dict['key']} {prompt_dict['mode']} {prompt_dict['keywords']}.json"
    # Initialize the MIDI file
    midi = MidiFile()
    # Construct the full prompt
    full_prompt_json = json.dumps(prompt_dict)
    # Generate the fully fleshed out prompt
    prompt, prompt_messages, prompt_cost = apicalls.prompt_translation(full_prompt_json, melody=melody)    
    # Generate the chords and add them to the MIDI file
    cp_gen, cp_messages, cp_cost = apicalls.generate_chords(prompt, prompt_dict["t"])
    midi_processing.add_bars_to_midi(midi, cp_gen)
    
    # Combine all messages
    all_messages = prompt_messages + cp_messages

    # If melody is enabled, generate the melody and add it to the MIDI file
    if melody:
        mel_gen, mel_messages, mel_cost = apicalls.generate_melody(cp_messages)
        midi_processing.add_bars_to_midi(midi, mel_gen, melody=True)
        # Update the combined messages
        all_messages = all_messages[:3] + mel_messages
    
    # Save the MIDI file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    midi.save(filename)
    # Save the messages to a JSON file
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    utils.save_messages_to_json(all_messages, json_filename)
    # Play and visualize the MIDI file if user wants
    if visualize:
        utils.visualize_midi(filename)
    if play:
        utils.play_midi(filename)
        
    # Sum up the costs and print the total cost
    sum = prompt_cost + cp_cost + mel_cost
    print(f"Total Cost: ${sum}")

if __name__ == "__main__":
    # Prompt for the MIDI generation
    prompt_dict = {
        "key": "C",
        "mode": "major",
        "keywords": "a rhythmic sad pop song",
        "t": 0.3,
        # "note density (%)": 60,
        # "syncopation (%)": 100,
        # "velocity dynamics (%)": 25
    }
    # Run the main function with the user inputs
    main(
        prompt_dict=prompt_dict,
        melody=True,
        visualize=False,
        play=True,
        filename="N/A"
    )