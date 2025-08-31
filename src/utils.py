"""
This file contains utility functions for handling the creation of MIDI objects, conversion to MIDI files, and visualization of MIDI data.
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import json
import mido
import io

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
    cleaned_pitch = note.pitch.strip()
    for char in cleaned_pitch:
        if char.isdigit():
            cleaned_pitch = cleaned_pitch.replace(char, "")
    cleaned_pitch = cleaned_pitch.replace("♯", "#").replace("𝄪", "##")
    base_number = base_midi_numbers[cleaned_pitch]
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

def save_messages_to_json(messages, filename):
    """Saves messages to a JSON file with the same name as the MIDI file.

    Args:
        messages (list of dictionaries): A list of messages to save to the JSON file.
        midi_filename (str): The filename of the MIDI file to save the messages for.
    """
    # Construct the JSON filename similar to the MIDI filename
    base_filename = f"{filename}.json"
    # Save the messages to the JSON file with indentation for readability
    with open(base_filename, 'w') as json_file:
        json.dump(messages, json_file, indent=4)

def convert_sixteenth(sixteenth_g):
    """
    Converts a SixteenthNote_G instance to its corresponding integer value.
    
    Args:
        sixteenth_g (SixteenthNote_G): A SixteenthNote_G enum value.
        
    Returns:
        int: The integer corresponding to the sixteenth note (1-16).
    """
    mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16
    }
    return mapping[sixteenth_g.value.lower()]

def visualize_midi_beats(input_midi):
    """Visualizes a MIDI file as a piano roll plot using matplotlib.

    Args:
        input_midi (str or MidiFile): The MIDI file to visualize. Can be a filename or a mido.MidiFile object.

    Raises:
        ValueError: If the input is neither a filename nor a MidiFile object.

    Returns:
        buffer: The buffer containing the image data of the visualization.
    """
    # Load MIDI if input is a filename
    if isinstance(input_midi, str):
        mid = mido.MidiFile(input_midi)
    elif isinstance(input_midi, mido.MidiFile):
        mid = input_midi
    else:
        raise ValueError("Input must be a filename or a MidiFile object")
        
    ticks_per_beat = mid.ticks_per_beat

    merged = mido.merge_tracks(mid.tracks)
    notes = []  # Each note: (pitch, start_tick, end_tick)
    time_ticks = 0
    active_notes = {}  # Dictionary to keep track of note on times
    # Iterate through the merged messages
    for msg in merged:
        time_ticks += msg.time
        # Handle note_on messages
        if msg.type == 'note_on':
            if msg.velocity > 0:
                active_notes.setdefault(msg.note, []).append(time_ticks)
            else:  # note_on with velocity 0 is equivalent to note_off
                if active_notes.get(msg.note):
                    start = active_notes[msg.note].pop(0)
                    notes.append((msg.note, start, time_ticks))
        # Handle note_off messages
        elif msg.type == 'note_off':
            if active_notes.get(msg.note):
                start = active_notes[msg.note].pop(0)
                notes.append((msg.note, start, time_ticks))
    
    # Convert note timings to beats
    notes_beats = [(pitch, start/ticks_per_beat, end/ticks_per_beat) for pitch, start, end in notes]
    # Convert beats to sixteenth notes (4 sixteenths per beat)
    notes_sixteenths = [(pitch, start * 4, end * 4) for pitch, start, end in notes_beats]
    
    if not notes_sixteenths:
        print("No notes found.")
        return

    # Determine the range of MIDI pitches in the file
    pitches = [pitch for pitch, _, _ in notes_sixteenths]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    
    # Set up the matplotlib figure with night mode style
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Plot each note as an orange rectangle using sixteenth note units
    for pitch, start, end in notes_sixteenths:
        duration = end - start
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1,
                                 facecolor='darkorange', edgecolor='none')
        ax.add_patch(rect)
    
    # Set x-axis to show sixteenth notes
    ax.set_xlabel("16th Note", color='white')
    
    # Set y-axis limits to the actual range of pitches and label with note names as before
    ax.set_ylim(min_pitch - 0.5, max_pitch + 0.5)
    yticks = list(range(min_pitch, max_pitch + 1))
    ylabels = [f"{name}{octave}" for name, octave in (midi_number_to_name_and_octave(p) for p in yticks)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color='white')
    ax.set_ylabel("Note", color='white')
    
    # Set x-axis limits (0 to max sixteenth note count)
    max_sixteenth = max(end for _, _, end in notes_sixteenths)
    ax.set_xlim(0, max_sixteenth)
    
    # Set x-axis ticks for every 4 sixteenth notes
    sixteenth_ticks = np.arange(0, int(max_sixteenth) + 1, 4, dtype=int)
    ax.set_xticks(sixteenth_ticks)
    ax.set_xticklabels(sixteenth_ticks, color='white')
    
    plt.title("Piano Roll", color='white')
    
    # Save the figure to a BytesIO buffer instead of a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    plt.close(fig)
    return buffer.getvalue()