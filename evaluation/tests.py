from mido import MidiFile

major_intervals = [0, 2, 4, 5, 7, 9, 11]
minor_intervals = [0, 2, 3, 5, 7, 8, 10]
harmonic_minor_intervals = [0, 2, 3, 5, 7, 8, 11]
melodic_minor_intervals = [0, 2, 3, 5, 7, 9, 11]

scale_intervals = {
    "major": major_intervals,
    "minor": minor_intervals,
    "harmonic minor": harmonic_minor_intervals,
    "melodic minor": melodic_minor_intervals
}

# Define note names with their potential enharmonic spellings.
note_names = [
    ["B#", "C"],     # 0
    ["C#", "Db"],    # 1
    ["D"],           # 2
    ["D#", "Eb"],    # 3
    ["E", "Fb"],     # 4
    ["E#", "F"],     # 5
    ["F#", "Gb"],    # 6
    ["G"],           # 7
    ["G#", "Ab"],    # 8
    ["A"],           # 9
    ["A#", "Bb"],    #10
    ["B", "Cb"]      #11
]

note_lengths = {
    "sixteenth": 0.25,
    "eighth": 0.5,
    "quarter": 1,
    "half": 2,
    "whole": 4
}

def scale_test(midi, root, scale):
    """
    Test whether all note events in a MIDI file belong to the specified scale.

    Args:
        midi (MidiFile): The MIDI file to check.
        root (str): The root note (e.g., "C", "D#", etc.).
        scale (str): The scale mode ("Major" or "minor").

    Returns:
        bool: True if all note events are within the allowed pitch classes of the scale; False otherwise.

    Raises:
        ValueError: If the provided root note or scale mode is invalid.
    """
    # Map each note name to its pitch class.
    note_to_pc = {}
    for pc, names in enumerate(note_names):
        for name in names:
            note_to_pc[name.upper()] = pc

    # Validate root and scale.
    if root.upper() not in note_to_pc:
        raise ValueError(f"Invalid root note: {root}")
    if scale.lower() not in scale_intervals:
        raise ValueError(f"Invalid scale mode: {scale.lower()}")

    # Determine the acceptable pitch classes for the given scale.
    root_pc = note_to_pc[root.upper()]
    acceptable_pcs = [(root_pc + interval) % 12 for interval in scale_intervals[scale.lower()]]
    # print(f"Root Note: {root}, Scale Mode: {scale}, Acceptable Pitch Classes: {acceptable_pcs}")

    correct = 0
    incorrect = 0
    total = 0
    correct_pitches = set()
    incorrect_pitches = set()
    
    # Iterate through all messages in the MIDI file.
    for msg in midi:
        if msg.type == "note_on" and msg.velocity > 0:
            # print(f"Checking note: {msg.note}, Pitch Class: {msg.note % 12}")
            total += 1
            if (msg.note % 12) in acceptable_pcs:
                correct += 1
                correct_pitches.add(msg.note % 12)
            else:
                incorrect += 1
                incorrect_pitches.add(msg.note % 12)
    
    results = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "pitches": {
            "correct": list(correct_pitches),
            "incorrect": list(incorrect_pitches)
        }
    }
    return results
                

def duration_test(midi, duration):
    """Test whether all note events in a MIDI file have the specified duration.

    Args:
        midi (MidiFile): The MIDI file to check.
        duration (str): The expected duration of each note event.

    Returns:
        bool: True if all note events have the specified duration; False otherwise.
    """
    if duration not in note_lengths:
        raise ValueError(f"Invalid duration: {duration}")

    duration_ticks = note_lengths[duration] 
    ticks_per_beat = midi.ticks_per_beat
    expected_ticks = duration_ticks * ticks_per_beat
    # print(f"Expected duration in ticks: {expected_ticks}")
    
    total = 0
    correct = 0
    incorrect = 0
    incorrect_lengths = {}
    
    for track in midi.tracks:
        active_notes = {}
        current_time_ticks = 0
        for msg in track:
            current_time_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = current_time_ticks
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    total += 1
                    start_time = active_notes.pop(msg.note)
                    note_duration = current_time_ticks - start_time
                    # print(f"Note {msg.note} duration: {note_duration} ticks")
                    
                    if note_duration != expected_ticks:
                        incorrect += 1
                        ratio = note_duration / ticks_per_beat
                        if ratio not in incorrect_lengths.keys():
                            incorrect_lengths[ratio] = 1
                        else: 
                            incorrect_lengths[ratio] += 1
                    else:
                        correct += 1
    results = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "lengths": incorrect_lengths
    }
    return results

def run_midi_tests(midi_data, root, scale, duration):
    """ Run a series of tests on the generated MIDI data to validate its structure and musicality.

    Args:
        midi_data (MidiFile): The MIDI data to test.
        root (str): The musical root note.
        scale (str): The musical scale.
        duration (str): The note duration.
    
    Returns:
        dict: A dictionary containing the results of the tests, including whether each test passed.
    """
    key_results = scale_test(midi_data, root, scale)
    duration_results = duration_test(midi_data, duration)
    return {
        "key_results": key_results,
        "duration_results": duration_results,
        "overall_pass": key_results["incorrect"] == 0 and duration_results["incorrect"] == 0
    }

if __name__ == "__main__":
    # Single test example
    test_path = "path/to/your/midi.mid"
    midi = MidiFile(test_path)
    print(f"Scale Test: {scale_test(midi, 'C', 'major')}")
    print(f"Duration Test: {duration_test(midi, 'sixteenth')}")