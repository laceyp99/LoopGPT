'''
This file holds all the utility functions that will be used to process MIDI information.

This includes the following functions:
* calculate_midi_number: Calculates the MIDI number for a given note.
* midi_number_to_name_and_octave: Converts a MIDI number to a note name and octave.
* midi_to_note_name: Converts a list of MIDI numbers to a list of note names.
* visualize_midi: Visualizes a MIDI file using prettyMIDI and matplotlib.
* play_midi: Plays a MIDI file using pygame.
* save_messages_to_json: Saves messages to a JSON file with the same name as the MIDI file.
'''

# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pygame, pretty_midi
import os
import json

# A dictionary that maps note names to their corresponding MIDI numbers
base_midi_numbers = {"C": 0, "B♯♯": 1, "B##": 1, "C♯": 1, "C#": 1,"D♭": 1,"Db": 1, "C♯♯": 2, "C##": 2, "D": 2, "D♯": 3, "D#": 3, "E♭": 3, "Eb": 3,"D♯♯": 4, "D##": 4, "E": 4, "Fb": 4, "F♭": 4, "E♯": 5, "E#": 5, "F": 5, "E♯♯": 6, "E##": 6, "F♯": 6, "F#": 6, "Gb": 6, "G♭": 6, "F♯♯": 7, "F##": 7, "G": 7, "G♯": 8, "G#": 8, "A♭": 8, "Ab": 8, "G♯♯": 9, "G##": 9, "A": 9, "A♯": 10, "A#": 10, "B♭": 10, "Bb": 10, "A♯♯": 11, "A##": 11, "B": 11, "Cb": 11, "C♭": 11, "B♯": 12, "B#": 12}

def scale(scale_letter, scale_mode):
    """Returns all the possible notes of a scale given the scale letter and mode.

    Args:
        scale_letter (str): The letter of the scale.
        scale_mode (str): The mode of the scale (either "major" or "minor").

    Returns:
        list[str]: A list of note names in the scale.
    """
    # Define the scale intervals for each mode
    scale_intervals = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10]
    }
    # Define the note names including all possible enharmonic spellings
    note_names = [
        ["B#", "C", "Dbb"],     # 0
        ["C#", "Db", "B##"],    # 1
        ["D", "C##", "Ebb"],    # 2
        ["D#", "Eb", "Fbb"],    # 3
        ["E", "Fb", "D##"],     # 4
        ["E#", "F", "Gbb"],     # 5
        ["F#", "Gb", "E##"],    # 6
        ["G", "F##", "Abb"],    # 7
        ["G#", "Ab"],           # 8
        ["A", "G##", "Bbb"],    # 9
        ["A#", "Bb", "Cbb"],    #10
        ["B", "Cb", "A##"]      #11
    ]
    # Find the starting index of the scale
    start_index = None
    for i, enharmonics in enumerate(note_names):
        if scale_letter in enharmonics:
            start_index = i
            break
    if start_index is None:
        raise ValueError(f"Invalid scale letter: {scale_letter}")
    
    scale = []
    # Get the scale intervals
    intervals = scale_intervals[scale_mode]
    # Generate the scale
    for interval in intervals:
        for note in note_names[(start_index + interval) % 12]:
            scale.append(note)
    return scale

def calculate_midi_number(note):
    """Calculates the MIDI number for a given note.

    Args:
        note (Note Object): The note object that holds the pitch and octave of the note.

    Returns:
        int: A MIDI number that corresponds to the note.
    """
    base_number = base_midi_numbers[note.pitch] 
    midi_number = base_number + ((note.octave + 1) * 12)
    return midi_number

def midi_number_to_name_and_octave(midi_number):
    """Converts a MIDI number to a note name and octave.

    Args:
        midi_number (int): The MIDI number to convert.

    Returns:
        note_name (str): The note name corresponding to the MIDI number.
        octave (int): The octave of the note corresponding to the MIDI number.
    """
    # Calculate the octave and note name based on the MIDI number
    octave = midi_number // 12 - 1
    note_names = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    return note_names[midi_number % 12], octave

def midi_to_note_name(midi_numbers):
    """Converts a list of MIDI numbers to a list of note names.

    Args:
        midi_numbers (list[int]): A list of MIDI numbers to convert.

    Returns:
        midi_names (list[str]): A list of note names corresponding to the MIDI numbers.
    """
    octave = midi_numbers // 12 - 1
    note_names = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    return [f'{note}{oct}' for note, oct in zip(note_names[midi_numbers % 12], octave)]

def visualize_midi(midi_file):
    """ Visualize a MIDI file using prettyMIDI to get the MIDI analysis data and matplotlib to create the display.

    Args:
        midi_file (str): The filename of the MIDI file to play and visualize.
    """
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    # Extract the piano roll
    piano_roll = midi_data.get_piano_roll(fs=100)
    
    # Find the lowest and highest active notes
    active_notes = np.where(piano_roll > 0)[0]
    lowest_note = np.min(active_notes)
    highest_note = np.max(active_notes)
    # Define padding (in number of MIDI notes)
    padding = 5  # Adjust this value as needed
    # Calculate the note range with padding
    min_note = max(0, lowest_note - padding)
    max_note = min(127, highest_note + padding)
    cropped_piano_roll = piano_roll[min_note:max_note + 1]

    # Assume a tempo of 120 BPM
    tempo = 120
    # Get the time per beat and per bar
    seconds_per_beat = 60.0 / tempo
    seconds_per_bar = seconds_per_beat * 4
    
    # Calculate the positions of bars and beats
    total_frames = cropped_piano_roll.shape[1]
    time_per_frame = midi_data.get_end_time() / total_frames
    bar_positions = np.arange(0, total_frames, seconds_per_bar / time_per_frame)
    beat_positions = np.arange(0, total_frames, seconds_per_beat / time_per_frame)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(cropped_piano_roll, aspect='auto', cmap='gray_r', origin='lower', interpolation='nearest')

    # Add grid lines at the top and bottom of each note block
    note_range = np.arange(min_note, max_note + 1)
    for i in range(len(note_range)):
        plt.hlines(i + 0.5, xmin=0, xmax=cropped_piano_roll.shape[1], color='gray', linestyle='--', linewidth=0.5)
        plt.hlines(i - 0.5, xmin=0, xmax=cropped_piano_roll.shape[1], color='gray', linestyle='--', linewidth=0.5)

    # Set y-axis ticks to note names
    plt.yticks(ticks=np.arange(len(note_range)), labels=midi_to_note_name(note_range))

    # Set x-axis ticks to bars and beats
    plt.xticks(ticks=bar_positions, labels=[f'Bar {i+1}' for i in range(len(bar_positions))])
    for beat_position in beat_positions:
        plt.axvline(x=beat_position, color='gray', linestyle='--', linewidth=0.5)

    # Set labels and title
    plt.xlabel('Bars and Beats')
    plt.ylabel('MIDI Note')
    plt.title(midi_file)
    plt.colorbar(label='Velocity')

    # Display the plot
    plt.show()

def play_midi(midi_file):
    """ Play a MIDI file using pygame.

    Args:
        midi_file (MidiFile): The MIDI file to play.
    """
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(midi_file)
    # Play the MIDI file
    pygame.mixer.music.play()


def save_messages_to_json(messages, midi_filename):
    """Saves messages to a JSON file with the same name as the MIDI file.

    Args:
        messages (list of dictionaries): A list of messages to save to the JSON file.
        midi_filename (str): The filename of the MIDI file to save the messages for.
    """
    # Construct the JSON filename similar to the MIDI filename
    base_filename = os.path.splitext(os.path.basename(midi_filename))[0]
    json_filename = os.path.join("Training Examples", f"{base_filename}.json")
    # Save the messages to the JSON file with indent=4
    with open(json_filename, 'w') as json_file:
        json.dump(messages, json_file, indent=4)