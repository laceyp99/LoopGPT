from mido import MidiFile
import sys
import os

# Imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import SCALE_INTERVALS, DURATION_BEATS, note_name_to_pitch_class

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
    # Validate root and scale.
    try:
        root_pc = note_name_to_pitch_class(root)
    except ValueError:
        raise ValueError(f"Invalid root note: {root}")
    if scale.lower() not in SCALE_INTERVALS:
        raise ValueError(f"Invalid scale mode: {scale.lower()}")

    # Determine the acceptable pitch classes for the given scale.
    acceptable_pcs = [(root_pc + interval) % 12 for interval in SCALE_INTERVALS[scale.lower()]]
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
    if duration not in DURATION_BEATS:
        raise ValueError(f"Invalid duration: {duration}")

    duration_ticks = DURATION_BEATS[duration] 
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

def is_monophonic(midi):
    """ Tests whether the MIDI file is either monophonic fully throughout or has polyphonic moments. 
    Great quick test for simple checks on one line melody prompts.

    Args:
        midi (MidiFile): The MIDI file to check.
    
    Returns:
        dict: Contains 'is_monophonic' (bool) and 'max_polyphony' (int) indicating the maximum
              number of simultaneous notes at any point.
    """
    max_polyphony = 0
    
    for track in midi.tracks:
        active_notes = {}  # note -> count (handles same note played multiple times)
        current_polyphony = 0
        
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = active_notes.get(msg.note, 0) + 1
                current_polyphony = sum(active_notes.values())
                max_polyphony = max(max_polyphony, current_polyphony)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    active_notes[msg.note] -= 1
                    if active_notes[msg.note] <= 0:
                        del active_notes[msg.note]
    
    return {
        "is_monophonic": max_polyphony <= 1,
        "max_polyphony": max_polyphony
    }

def polyphonic_profile(midi):
    """Tests and measures the time the MIDI file exhibits different levels of polyphony (including monophony).

    Args:
        midi (MidiFile): The MIDI file to check.
    
    Returns:
        dict: Contains 'polyphony_distribution' mapping polyphony levels to time in beats, 
        'max_polyphony' (int), 
        'total_duration' in beats, 
        'polyphony_percentages',
        and 'ticks_per_beat' used for conversion.
    """
    ticks_per_beat = midi.ticks_per_beat
    polyphony_distribution_ticks = {}
    max_polyphony = 0
    total_duration_ticks = 0
    
    for track in midi.tracks:
        active_notes = {}  # note -> count
        current_polyphony = 0
        
        for msg in track:
            # Accumulate time at current polyphony level before processing the event
            if msg.time > 0:
                polyphony_distribution_ticks[current_polyphony] = polyphony_distribution_ticks.get(current_polyphony, 0) + msg.time
                total_duration_ticks += msg.time
            
            # Update active notes count
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = active_notes.get(msg.note, 0) + 1
                current_polyphony = sum(active_notes.values())
                max_polyphony = max(max_polyphony, current_polyphony)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    active_notes[msg.note] -= 1
                    if active_notes[msg.note] <= 0:
                        del active_notes[msg.note]
                    current_polyphony = sum(active_notes.values())
    
    # Convert ticks to beats and calculate percentages
    polyphony_distribution = {}
    polyphony_percentages = {}
    total_duration = total_duration_ticks / ticks_per_beat
    
    for level, ticks in polyphony_distribution_ticks.items():
        polyphony_distribution[level] = round(ticks / ticks_per_beat, 4)
        if total_duration_ticks > 0:
            polyphony_percentages[level] = round((ticks / total_duration_ticks) * 100, 2)
    
    return {
        "polyphony_distribution": polyphony_distribution,
        "polyphony_percentages": polyphony_percentages,
        "max_polyphony": max_polyphony,
        "total_duration": round(total_duration, 4),
        "ticks_per_beat": ticks_per_beat
    }


def arpeggio_test(midi, root, scale, duration):
    """Tests whether the MIDI file contains arpeggiated patterns, which are characterized by a monophonic sequence of notes that outline a chord.

    Args:
        midi (MidiFile): The MIDI file to check.
        root (str): The musical root note.
        scale (str): The musical scale.
        duration (str): The note duration.
    Returns:
        boolean: True if arpeggiated patterns are detected, False otherwise.
    """
    return scale_test(midi, root, scale)["incorrect"] == 0 and duration_test(midi, duration)["incorrect"] == 0 and is_monophonic(midi)["is_monophonic"]

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
    return {
        "key_results": scale_test(midi_data, root, scale),
        "duration_results": duration_test(midi_data, duration),
        "polyphony_results": is_monophonic(midi_data),
        "polyphonic_profile": polyphonic_profile(midi_data),
        "arpeggio_results": arpeggio_test(midi_data, root, scale, duration)
    }

if __name__ == "__main__":
    # Single test example
    test_path = "/path/to/output.mid"
    midi = MidiFile(test_path)
    print(f"Test Results: {run_midi_tests(midi, 'C', 'major', 'quarter')}")