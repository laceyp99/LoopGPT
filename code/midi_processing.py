'''
This file contains all the functions that are involved in processing MIDI files.

This includes the following functions:
* add_bars_to_midi: Converts a list of bars or melody bars to MIDI information and adds them to a MIDI file.
* objectify_midi: Converts a MIDI file to a list of melody objects.
'''

# IMPORTS
from mido import MidiTrack, Message
import code.utils as utils
import code.objects as objects

def add_bars_to_midi(midi, bars, melody=False):
    """Converts the bars to MIDI information and adds either the chords or melody to a midi file as another track in the MIDIFile.

    Args:
        midi (MidiFile): The midi file to which the bars will be added.
        bars (list[Bar] or list[MelodyBar]): The list of bars or melody bars to be converted to MIDI.
        melody (bool, optional): Determines whether the bars variable is a list of MelodyBars or Bars. Defaults to False (Bars).
        
    Returns:
        None
    """
    # Create a new track for the chords or melody
    track = MidiTrack()
    # Set the ticks per beat to midi.ticks_per_beat / 4 to get a sixteenth note resolution 
    ticks_per_beat = int(midi.ticks_per_beat / 4)
    # If the melody boolean is False, then the bars variable is a list of Bars with a chord progression
    if melody == False:
        # For each bar in the list of bars
        for bar in bars:
            # Iterate through each chord in the bar
            # Every time you start another bar, you need to reset the past_event_time
            past_event_time = 0
            # For each chord in the bar
            for chord in bar.chords:            
                # Calculate the note_on and note_off times
                # MIDI time is in ticks and is relative to the previous event
                note_on_time = (chord.time.start_beat - 1) * ticks_per_beat - past_event_time
                note_off_time = (chord.time.duration) * ticks_per_beat

                # Ensure the note on time is not negative
                if note_on_time < 0:
                    note_on_time = 0
                
                # Add note_on events for each note in the chord
                # Currently, each note in the chord is played at the same time
                for i, note in enumerate(chord.voicing):
                    if i == 0:
                        # First note on message with the calculated time
                        track.append(Message('note_on', note=utils.calculate_midi_number(note), velocity=note.velocity, time=note_on_time))
                    else:
                        # Subsequent note on messages with time = 0
                        track.append(Message('note_on', note=utils.calculate_midi_number(note), velocity=note.velocity, time=0))
                
                # Add note_off events for each note in the chord
                for i, note in enumerate(chord.voicing):
                    if i == 0:
                        # First note off message with time = ticks_per_bar
                        track.append(Message('note_off', note=utils.calculate_midi_number(note), velocity=note.velocity, time=note_off_time))
                    else:
                        # Subsequent note off messages with time = 0
                        track.append(Message('note_off', note=utils.calculate_midi_number(note), velocity=note.velocity, time=0))

                # Update the past event time
                past_event_time += note_on_time + note_off_time 
    # If the melody boolean is True, then the bars variable is a list of MelodyBars with a melody
    else:
        # Iterate through each melody bar and add the notes to the melody track
        for melody_bar in bars:
            # Initialize the past event time
            past_event_time = 0
            for note in melody_bar.melody:
                # Calculate the note on and note off times in ticks
                note_on_time = (note.time.start_beat - 1) * ticks_per_beat - past_event_time
                note_off_time = note.time.duration * ticks_per_beat
                
                # Ensure the note on time is not negative
                if note_on_time < 0:
                    note_on_time = 0
                
                # Create note on and note off messages and add them to the melody track
                track.append(Message('note_on', note=utils.calculate_midi_number(note), velocity=note.velocity, time=note_on_time))
                track.append(Message('note_off', note=utils.calculate_midi_number(note), velocity=note.velocity, time=note_off_time))
                
                # Update the past event time
                past_event_time += note_on_time + note_off_time
    # Append the track to the midi file
    midi.tracks.append(track)
    
def objectify_midi(midi):
    """Converts a midi file to a list of melody objects.

    Args:
        midi (MidiFile): The midi file to be converted to a list of melody objects.

    Returns:
        list[MelodyNote]: A list of melody note objects.
    """
    # Initialize the list of midi objects
    midi_objects = []
    # Get ticks per beat from the midi file / 4 for sixteenth note resolution
    ticks_per_beat = midi.ticks_per_beat / 4
    # Initialize a dictionary to keep track of active notes
    active_notes = {}
    # Iterate through each track in the midi file
    for track in midi.tracks:
        # Initialize absolute time
        absolute_time = 0
        # Iterate through each message in the track
        for msg in track:
            # Add the delta time to the absolute time
            absolute_time += msg.time
            # If the message is 'note_on' with velocity > 0
            if msg.type == 'note_on' and msg.velocity > 0:
                # Store the start time in beats
                start_time = absolute_time / ticks_per_beat
                active_notes[msg.note] = start_time
            # If the message is 'note_off' or 'note_on' with velocity == 0
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    # Get the start and end times
                    start_time = active_notes[msg.note]
                    end_time = absolute_time / ticks_per_beat
                    duration = end_time - start_time
                    # Get note name and octave
                    note_name, note_octave = utils.midi_number_to_name_and_octave(msg.note)
                    note_velocity = msg.velocity
                    # Calculate start beat within bar (1 to 16)
                    start_beat = int((start_time % 16) + 1)
                    # Create the midi event object and append it to the list
                    midi_objects.append(objects.MelodyNote(
                        pitch=note_name,
                        octave=note_octave,
                        velocity=note_velocity,
                        time=objects.TimeInformation(start_beat=start_beat, duration=duration)))
                    # Remove the note from active_notes
                    del active_notes[msg.note]
                else:
                    print("OBJECTIFY MIDI ERROR: Did not find note in active notes")
    return midi_objects