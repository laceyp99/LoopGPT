'''
This file does the main processing for an accompaniment generation. Accompaniment is a chord progression generated based on the user's melody.
'''

# IMPORTS
from mido import MidiFile
import code.apicalls as apicalls
import code.midi_processing as midi_processing
import code.utils as utils
import os


def main(melody, visualize=False, play=False, filename="N/A", temp=0.0):
    """ Generate an accompaniment based on the user's melody.
    
    Args: 
        melody (MidiFile): The user's melody as a MidiFile to generate the accompaniment for.
        visualize (bool, optional): A boolean that will visualize the MIDI in a pop up window if set to True. Defaults to False.
        play (bool, optional): A boolean that will play the MIDI after generation if set to True. Defaults to False.
        filename (str, optional): The filename of the generated MidiFile. Defaults to "N/A".
        temp (float, optional): The temperature for the API call. Defaults to 0.0.
    """
    # Define the filename if not provided
    if filename == "N/A":
        filename = f"Accompaniment/Output.mid"
    
    json_filename = f"{filename}.json"
    
    # Decode the melody
    melody_objectified = midi_processing.objectify_midi(melody)
    # Generate the accompaniment and add it to the melody
    chords, messages, cost = apicalls.generate_accompaniment(melody_objectified, temp=temp)
    midi_processing.add_bars_to_midi(midi=melody, bars=chords, melody=False)
    melody.type = 1 # type 1 (synchronous): all tracks start at the same time
    
    # Save the MIDI file with the accompaniment
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    midi.save(filename)
    
    # Save the messages to a JSON file
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    utils.save_messages_to_json(messages=messages, midi_filename=json_filename[:-5]) 
    
    # Play and visualize the MIDI file if user wants
    if visualize:
        utils.visualize_midi(filename)
    if play:
        utils.play_midi(filename)
    
    # Print the total cost
    print(f"Total Cost: ${cost}")

if __name__ == "__main__":
    midi = MidiFile("path/to/your/melody")  # Replace with your MIDI file path
    # Run the main function with the user inputs
    main(
        melody=midi,
        visualize=False,
        play=True,
        filename="N/A",
        temp=0.2
    )