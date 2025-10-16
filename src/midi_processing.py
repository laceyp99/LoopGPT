"""
This module provides functions to convert between MIDI files and our loop object representation.
It supports converting a loop into MIDI (with proper absolute and delta timing) and parsing a MIDI file into our structured loop objects.
"""

import logging
import sys
from mido import MidiTrack, Message, MidiFile, merge_tracks, MetaMessage
import src.objects as objects
import src.utils as utils
# import objects, utils
import math

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def loop_to_midi(midi, loop, times_as_string=True):
    """
    Converts a loop object into MIDI format.
    
    This function schedules all note on/off events with absolute timing, sorts them,
    then computes the delta times required by MIDI format before appending them to a new track.
    
    Args:
        midi (MidiFile): The MidiFile instance to which the track will be added.
        loop: The loop object containing 4 bars (Bar_1, Bar_2, Bar_3, Bar_4).
        times_as_string (bool): If True, assume note time values are string enums that need conversion
                                via `utils.convert_sixteenth`; otherwise, they are integers.
                                
    Returns:
        None. The function modifies the midi.tracks in place.
    """
    if loop is None:
        raise ValueError("The loop object is None. Ensure it is properly initialized.")
    
    bars = [getattr(loop, f"Bar_{i}", None) for i in range(1, 5)]
    if any(bar is None for bar in bars):
        raise ValueError("One or more bars in the loop object are None. Ensure all bars are initialized.")

    # Create a new track to hold the MIDI messages of the Loop
    track = MidiTrack()
    # Calculated ticks per sixteenth note, based on provided ticks per beat (quarter note).
    ticks_per_beat = int(midi.ticks_per_beat / 4)
    bar_length = 16 * ticks_per_beat 
    
    # Initialize a list to hold all note events (on and off).
    events = []
    bars = [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    # Iterate through each bar and its notes to schedule events.
    for bar_index, bar in enumerate(bars):
        bar_offset = bar_index * bar_length
        # Iterate through each note in the bar.
        for note in bar.notes:
            # Get the note's time information from either style loop object.
            if times_as_string:
                start_beat = utils.convert_sixteenth(note.time.start_beat)
                duration = utils.convert_sixteenth(note.time.duration)
            else:
                start_beat = note.time.start_beat
                duration = note.time.duration
            
            # Convert the note's time information to absolute ticks.
            note_start_tick = bar_offset + (start_beat - 1) * ticks_per_beat
            note_duration_ticks = duration * ticks_per_beat
            note_end_tick = note_start_tick + note_duration_ticks
            
            # Append note on and note off events.
            events.append((note_start_tick, 'note_on', note))
            events.append((note_end_tick, 'note_off', note))
    
    # Sort events by time; for equal timestamps, note_off is processed before note_on.
    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 'note_off' else 1))
    
    # Convert absolute times to delta times for the MIDI messages.
    prev_tick = 0
    # Iterate through the sorted events and create MIDI messages.
    for event_time, event_type, note in events:
        if not 0 <= note.velocity <= 127:
            logger.info(f"[MIDI] Handling out-of-range velocity {note.velocity}")
            velocity = max(0, min(127, note.velocity))
        else:
            velocity = note.velocity
        
        note_num = utils.calculate_midi_number(note)
        if not 0 <= note_num <= 127:
            logger.info(f"[MIDI] Handling out-of-range note {note_num}")
            note_num = max(0, min(127, note_num))
            velocity = 0
        
        # Calculate the delta time from the previous event.
        delta_time = event_time - prev_tick
        
        # Create the MIDI message using the delta time.
        msg = Message(event_type, note=note_num, velocity=velocity, time=delta_time)
        # Append the message to the track and update previous tick.
        track.append(msg)
        prev_tick = event_time
    
    final_bar_ticks = 4 * 16 * ticks_per_beat
    if prev_tick < final_bar_ticks:
        remaining = final_bar_ticks - prev_tick
        track.append(MetaMessage('end_of_track', time=remaining))
    else:
        track.append(MetaMessage('end_of_track', time=0))

    # Append the track to the MIDI file.
    midi.tracks.append(track)

def midi_to_loop(midi_filename, times_as_string=True):
    """
    Converts a MIDI file into a loop object.

    The function merges all tracks, computes absolute timestamps, and then assigns notes to the 
    appropriate bar based on their timing (assuming 16 sixteenth notes per bar).

    Args:
        midi_filename (str): The path to the MIDI file.
        times_as_string (bool): If True, convert timing information to string enums; else leave as integers.

    Returns:
        The constructed loop object (either Loop_G if times_as_string is True, or Loop otherwise).
    """
    # Load the MIDI file and set some basic time parameters.
    midi = MidiFile(midi_filename)
    ticks_per_beat = midi.ticks_per_beat
    ticks_per_16th = int(ticks_per_beat / 4)
    
    # Merge all tracks and compute absolute time for each message.
    merged = merge_tracks(midi.tracks)
    absolute_time = 0
    active_notes = {}  # MIDI note number -> list of (start_tick, velocity)
    note_events = []   # Collection of tuples: (note_number, start_tick, end_tick, velocity)
    
    # Iterate through the merged messages to process note events.
    for msg in merged:
        absolute_time += msg.time
        # Handle note_on messages.
        if msg.type == 'note_on':
            # If velocity is greater than 0, it's a note_on event.
            if msg.velocity > 0:
                active_notes.setdefault(msg.note, []).append((absolute_time, msg.velocity))
            else:
                # A note_on with velocity 0 signifies note_off.
                # Check if the note is active and remove it from the active notes.
                if msg.note in active_notes and active_notes[msg.note]:
                    start_tick, velocity = active_notes[msg.note].pop(0)
                    note_events.append((msg.note, start_tick, absolute_time, velocity))
        # Handle note_off messages.
        elif msg.type == 'note_off':
            # Check if the note is active and remove it from the active notes.
            if msg.note in active_notes and active_notes[msg.note]:
                start_tick, velocity = active_notes[msg.note].pop(0)
                note_events.append((msg.note, start_tick, absolute_time, velocity))
    
    # Create empty bars for the first four bars.
    bars = {0: [], 1: [], 2: [], 3: []}
    
    # Mapping sixteenth note positions from integer to string.
    int_to_str = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen"}
    
    # Process each note event and assign it to the appropriate bar.
    for note_number, start_tick, end_tick, velocity in note_events:
        # Determine the sixteenth-note position (1-based indexing).
        start_sixteenth = (start_tick // ticks_per_16th) + 1
        duration_sixteenth = max(1, math.ceil((end_tick - start_tick) / ticks_per_16th))
        
        # Determine the bar (each bar has 16 sixteenth notes).
        bar_index = (start_sixteenth - 1) // 16
        if bar_index < 0 or bar_index > 3:
            continue  # Skip notes outside the first four bars.
        
        # Compute the relative start within the bar (1-16).
        relative_start = ((start_sixteenth - 1) % 16) + 1
        
        # Convert the MIDI note number to pitch name and octave.
        pitch_name, octave = utils.midi_number_to_name_and_octave(note_number)
        
        # Create the note objects with time information.
        if times_as_string:
            time_info = objects.TimeInformation_G(
                start_beat=objects.SixteenthNote_G(utils.convert_sixteenth(relative_start)),
                duration=objects.SixteenthNote_G(utils.convert_sixteenth(duration_sixteenth))
            )
            note_obj = objects.Note_G(
                pitch=pitch_name,
                octave=octave,
                velocity=velocity,
                time=time_info
            )
        else:
            time_info = objects.TimeInformation(
                start_beat=relative_start,
                duration=duration_sixteenth
            )
            note_obj = objects.Note(
                pitch=pitch_name,
                octave=octave,
                velocity=velocity,
                time=time_info
            )
        bars[bar_index].append(note_obj)
    
    # Build the loop object using the processed bars.
    if times_as_string:
        loop = objects.Loop_G(
            Bar_1=objects.Bar_G(num=1, notes=[n.model_dump() for n in bars[0]]),
            Bar_2=objects.Bar_G(num=2, notes=[n.model_dump() for n in bars[1]]),
            Bar_3=objects.Bar_G(num=3, notes=[n.model_dump() for n in bars[2]]),
            Bar_4=objects.Bar_G(num=4, notes=[n.model_dump() for n in bars[3]])
        )
    else:
        loop = objects.Loop(
            Bar_1=objects.Bar(num=1, notes=bars[0]),
            Bar_2=objects.Bar(num=2, notes=bars[1]),
            Bar_3=objects.Bar(num=3, notes=bars[2]),
            Bar_4=objects.Bar(num=4, notes=bars[3])
        )
    
    # Return the constructed loop object.
    return loop