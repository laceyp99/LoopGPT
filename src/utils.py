"""
This file contains utility functions for handling the creation of MIDI objects, conversion to MIDI files, and visualization of MIDI data.
"""

import plotly.graph_objects as go
import numpy as np
import json
import mido

# Flat list of chromatic note names (pitch class 0-11, sharps only)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic note name variants per pitch class (for scale spelling, validation, etc.)
ENHARMONIC_NOTE_NAMES = [
    ["B#", "C", "Dbb"],   # 0
    ["C#", "Db", "B##"],  # 1
    ["D", "C##", "Ebb"],  # 2
    ["D#", "Eb", "Fbb"],  # 3
    ["E", "Fb", "D##"],   # 4
    ["E#", "F", "Gbb"],   # 5
    ["F#", "Gb", "E##"],  # 6
    ["G", "F##", "Abb"],  # 7
    ["G#", "Ab"],          # 8
    ["A", "G##", "Bbb"],  # 9
    ["A#", "Bb", "Cbb"],  # 10
    ["B", "Cb", "A##"],   # 11
]
# A dictionary that maps note names to their corresponding MIDI numbers
base_midi_numbers = {
    "C": 0,
    "B♯♯": 1, "B##": 1, "C♯": 1, "C#": 1,
    "D♭": 1, "Db": 1,
    "C♯♯": 2, "C##": 2, "D": 2,
    "D♯": 3, "D#": 3, "E♭": 3, "Eb": 3,
    "D♯♯": 4, "D##": 4, "E": 4, "Fb": 4, "F♭": 4,
    "E♯": 5, "E#": 5, "F": 5,
    "E♯♯": 6, "E##": 6, "F♯": 6, "F#": 6, "Gb": 6, "G♭": 6,
    "F♯♯": 7, "F##": 7, "G": 7,
    "G♯": 8, "G#": 8, "A♭": 8, "Ab": 8,
    "G♯♯": 9, "G##": 9, "A": 9,
    "A♯": 10, "A#": 10, "B♭": 10, "Bb": 10,
    "A♯♯": 11, "A##": 11, "B": 11, "Cb": 11, "C♭": 11,
    "B♯": 12, "B#": 12
}

# Scale intervals (semitones from root) for each mode
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor": [0, 2, 3, 5, 7, 9, 11],
}

# Canonical duration definitions: name -> beats, sixteenths, display string, and aliases
DURATION_MAP = {
    "sixteenth": {"beats": 0.25, "sixteenths": 1,  "display": "1/16",  "aliases": ["16th"]},
    "eighth":    {"beats": 0.5,  "sixteenths": 2,  "display": "1/8",   "aliases": ["8th"]},
    "quarter":   {"beats": 1.0,  "sixteenths": 4,  "display": "1/4",   "aliases": []},
    "half":      {"beats": 2.0,  "sixteenths": 8,  "display": "1/2",   "aliases": []},
    "whole":     {"beats": 4.0,  "sixteenths": 16, "display": "1 bar", "aliases": []},
}

# Derived lookups from DURATION_MAP
DURATION_BEATS = {name: d["beats"] for name, d in DURATION_MAP.items()}
DURATION_SIXTEENTHS_TO_DISPLAY = {d["sixteenths"]: d["display"] for d in DURATION_MAP.values()}
DURATION_KEYWORDS = {name: name for name in DURATION_MAP}
for name, d in DURATION_MAP.items():
    for alias in d["aliases"]:
        DURATION_KEYWORDS[alias] = name
DURATION_BEATS_TO_NAME = {d["beats"]: name.title() for name, d in DURATION_MAP.items()}

# Interval names (semitones 0-11 relative to root)
INTERVAL_NAMES = [
    "Root", "m2", "M2", "m3", "M3", "P4",
    "Tritone", "P5", "m6", "M6", "m7", "M7",
]

PLOTLY_BG = "#1a1a2e"
PLOTLY_BG_ALT = "#252540"       # Slightly lighter (e.g. black key lanes)
PLOTLY_GRID = "#2a2a4a"
PLOTLY_GRID_STRONG = "#4a4a6a"  # Bar boundaries
PLOTLY_TEXT = "#e0e0e0"
PLOTLY_ACCENT = "#0f3460"
PLOTLY_CARD_BG = "#16213e"

def pitch_class_to_note(pc):
    """Convert a pitch class integer (0-11) to a note name.

    Args:
        pc (int): Pitch class (0 = C, 1 = C#, ..., 11 = B).

    Returns:
        str: Note name string (sharp spelling).
    """
    return NOTE_NAMES[pc % 12]


def note_name_to_pitch_class(name):
    """Convert a note name to its pitch class using base_midi_numbers.

    Supports ASCII and unicode sharps/flats, double sharps, etc.

    Args:
        name (str): Note name (e.g. "C", "F#", "Eb", "G##").

    Returns:
        int: Pitch class (0-11).

    Raises:
        ValueError: If the note name is not recognized.
    """
    pc = base_midi_numbers.get(name)
    if pc is None:
        raise ValueError(f"Unrecognized note name: {name}")
    return pc % 12


def pitch_class_to_interval(pc, root_pc):
    """Convert a pitch class to an interval name relative to a root.

    Args:
        pc (int): Pitch class of the note.
        root_pc (int): Pitch class of the root note.

    Returns:
        str: Interval name (e.g. "m3", "P5").
    """
    semitones = (pc - root_pc) % 12
    return INTERVAL_NAMES[semitones]


def velocity_to_color(velocity):
    """Map MIDI velocity (0-127) to a color gradient from dark purple to bright coral.

    Args:
        velocity (int): MIDI velocity value (0-127).

    Returns:
        str: CSS rgb color string.
    """
    t = velocity / 127.0
    r = int(45 + t * (255 - 45))
    g = int(27 + t * (107 - 27))
    b = int(78 + t * (91 - 78))
    return f"rgb({r},{g},{b})"


def is_black_key(midi_pitch):
    """Check if a MIDI pitch corresponds to a black key on a piano.

    Args:
        midi_pitch (int): MIDI pitch number.

    Returns:
        bool: True if the pitch is a black key (C#, D#, F#, G#, A#).
    """
    return (midi_pitch % 12) in [1, 3, 6, 8, 10]


def format_duration_sixteenths(sixteenths):
    """Format a duration in sixteenth notes to a human-readable string.

    Uses the canonical DURATION_MAP for standard values, falls back to
    fractional notation for non-standard durations.

    Args:
        sixteenths (float): Duration in sixteenth note units.

    Returns:
        str: Formatted duration string (e.g. "1/4", "1 bar", "3/16").
    """
    sixteenths_int = int(sixteenths)
    display = DURATION_SIXTEENTHS_TO_DISPLAY.get(sixteenths_int)
    if display:
        return display
    return f"{sixteenths_int}/16"


def beats_to_duration_name(beats):
    """Convert a beat ratio to a human-readable duration name.

    Args:
        beats (float): Duration in beats (e.g. 0.25, 0.5, 1.0, 2.0, 4.0).

    Returns:
        str: Duration name (e.g. "Sixteenth", "Quarter") or "{beats} beats" for
            non-standard values.
    """
    name = DURATION_BEATS_TO_NAME.get(beats)
    if name:
        return name
    return f"{beats} beats"


def apply_plotly_theme(fig):
    """Apply the shared LoopGPT dark theme to a Plotly figure.

    Args:
        fig (go.Figure): Plotly figure to style.

    Returns:
        go.Figure: The styled figure (modified in place).
    """
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_TEXT, family="Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        margin=dict(l=60, r=30, t=50, b=60),
    )
    return fig

def scale(scale_letter, scale_mode):
    """Returns all the possible notes of a scale given the scale letter and mode.

    Args:
        scale_letter (str): The letter of the scale.
        scale_mode (str): The mode of the scale (either "major" or "minor").

    Returns:
        list[str]: A list of note names in the scale.

    Raises:
        ValueError: If the scale letter or mode is invalid.
    """
    # Find the starting pitch class of the scale letter
    start_index = None
    for i, enharmonics in enumerate(ENHARMONIC_NOTE_NAMES):
        if scale_letter in enharmonics:
            start_index = i
            break
    if start_index is None:
        raise ValueError(f"Invalid scale letter: {scale_letter}")

    if scale_mode not in SCALE_INTERVALS:
        raise ValueError(f"Invalid scale mode: {scale_mode}")

    result = []
    for interval in SCALE_INTERVALS[scale_mode]:
        for note in ENHARMONIC_NOTE_NAMES[(start_index + interval) % 12]:
            result.append(note)
    return result


def calculate_midi_number(note):
    """Calculates the MIDI number for a given note.

    Args:
        note (Note Object): The note object that holds the pitch and octave of the note.

    Returns:
        int: A MIDI number that corresponds to the note.
    """
    cleaned_pitch = note.pitch.strip().replace("♯", "#").replace("𝄪", "##")
    for char in cleaned_pitch:
        if not char.isalpha() and char != "#":
            cleaned_pitch = cleaned_pitch.replace(char, "")
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
    octave = midi_number // 12 - 1
    return pitch_class_to_note(midi_number), octave


def midi_to_note_name(midi_numbers):
    """Converts a list of MIDI numbers to a list of note names.

    Args:
        midi_numbers (list[int]): A list of MIDI numbers to convert.

    Returns:
        midi_names (list[str]): A list of note names corresponding to the MIDI numbers.
    """
    octave = midi_numbers // 12 - 1
    return [f"{pitch_class_to_note(n)}{oct}" for n, oct in zip(midi_numbers, octave)]


def save_messages_to_json(messages, filename):
    """Saves messages to a JSON file with the same name as the MIDI file.

    Args:
        messages (list of dictionaries): A list of messages to save to the JSON file.
        midi_filename (str): The filename of the MIDI file to save the messages for.
    """
    # Construct the JSON filename similar to the MIDI filename
    base_filename = f"{filename}.json"
    # Save the messages to the JSON file with indentation for readability
    with open(base_filename, "w") as json_file:
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
        "sixteen": 16,
    }
    return mapping[sixteenth_g.value.lower()]


def visualize_midi_plotly(input_midi):
    """Visualizes a MIDI file as an Ableton-style piano roll using Plotly.

    Args:
        input_midi (str or MidiFile): The MIDI file to visualize. Can be a filename or a mido.MidiFile object.

    Raises:
        ValueError: If the input is neither a filename nor a MidiFile object.

    Returns:
        go.Figure: A Plotly figure object for the piano roll visualization.
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
    notes = []  # Each note: (pitch, start_sixteenth, end_sixteenth, velocity)
    time_ticks = 0
    active_notes = {}  # Dictionary to keep track of note on times and velocities

    # Iterate through the merged messages
    for msg in merged:
        time_ticks += msg.time
        # Handle note_on messages
        if msg.type == "note_on":
            if msg.velocity > 0:
                active_notes.setdefault(msg.note, []).append((time_ticks, msg.velocity))
            else:  # note_on with velocity 0 is equivalent to note_off
                if active_notes.get(msg.note):
                    start, velocity = active_notes[msg.note].pop(0)
                    notes.append((msg.note, start, time_ticks, velocity))
        # Handle note_off messages
        elif msg.type == "note_off":
            if active_notes.get(msg.note):
                start, velocity = active_notes[msg.note].pop(0)
                notes.append((msg.note, start, time_ticks, velocity))

    # Convert note timings to sixteenth notes
    notes_sixteenths = [
        (pitch, start / ticks_per_beat * 4, end / ticks_per_beat * 4, velocity)
        for pitch, start, end, velocity in notes
    ]

    if not notes_sixteenths:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No notes found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Determine the range of MIDI pitches in the file
    pitches = [pitch for pitch, _, _, _ in notes_sixteenths]
    min_pitch = min(pitches)
    max_pitch = max(pitches)

    # Expand range slightly for visual padding
    min_pitch = max(0, min_pitch - 1)
    max_pitch = min(127, max_pitch + 1)

    # Color scheme (using shared theme constants)
    bg_color = PLOTLY_BG
    black_key_color = PLOTLY_BG_ALT
    grid_color_bar = PLOTLY_GRID_STRONG
    grid_color_beat = PLOTLY_GRID
    text_color = PLOTLY_TEXT

    # Create figure
    fig = go.Figure()

    # Add background lane shading for black keys
    for pitch in range(min_pitch, max_pitch + 1):
        if is_black_key(pitch):
            fig.add_shape(
                type="rect",
                x0=0,
                x1=64,
                y0=pitch - 0.5,
                y1=pitch + 0.5,
                fillcolor=black_key_color,
                line=dict(width=0),
                layer="below",
            )

    # Add beat grid lines (every 4 sixteenths) - thinner
    for beat in range(1, 16):  # beats 1-15 (excluding bar boundaries)
        if beat % 4 != 0:  # Skip bar boundaries, handled separately
            x_pos = beat * 4
            fig.add_shape(
                type="line",
                x0=x_pos,
                x1=x_pos,
                y0=min_pitch - 0.5,
                y1=max_pitch + 0.5,
                line=dict(color=grid_color_beat, width=1),
                layer="below",
            )

    # Add bar grid lines (every 16 sixteenths) - thicker
    for bar in range(0, 5):  # 0, 16, 32, 48, 64
        x_pos = bar * 16
        fig.add_shape(
            type="line",
            x0=x_pos,
            x1=x_pos,
            y0=min_pitch - 0.5,
            y1=max_pitch + 0.5,
            line=dict(color=grid_color_bar, width=2),
            layer="below",
        )

    # Add bar number annotations at the top
    for bar_num in range(1, 5):
        x_center = (bar_num - 1) * 16 + 8  # Center of each bar
        fig.add_annotation(
            x=x_center,
            y=max_pitch + 1,
            text=f"Bar {bar_num}",
            showarrow=False,
            font=dict(color=text_color, size=12),
            yanchor="bottom",
        )

    # Add notes as rectangles with velocity-based coloring
    hover_x = []
    hover_y = []
    hover_text = []

    for pitch, start, end, velocity in notes_sixteenths:
        color = velocity_to_color(velocity)

        # Add note rectangle with slight vertical padding
        padding = 0.1
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=pitch - 0.5 + padding,
            y1=pitch + 0.5 - padding,
            fillcolor=color,
            line=dict(color="rgba(255,255,255,0.3)", width=1),
            layer="above",
        )

        # Prepare hover data - place hover point at center of note
        note_center_x = (start + end) / 2
        hover_x.append(note_center_x)
        hover_y.append(pitch)

        # Calculate musical position
        bar = int(start // 16) + 1
        beat_in_bar = int((start % 16) // 4) + 1
        sixteenth_in_beat = int(start % 4) + 1
        duration_sixteenths = end - start

        # Format duration using shared utility
        duration_str = format_duration_sixteenths(duration_sixteenths)

        note_name, octave = midi_number_to_name_and_octave(pitch)
        hover_text.append(
            f"Note: {note_name}{octave}<br>"
            f"Velocity: {velocity}<br>"
            f"Position: Bar {bar}, Beat {beat_in_bar}.{sixteenth_in_beat}<br>"
            f"Duration: {duration_str}"
        )

    # Add invisible scatter for hover functionality
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=10, opacity=0),
            hoverinfo="text",
            hovertext=hover_text,
            hoverlabel=dict(bgcolor="#2a2a4a", font_size=12, font_color=text_color),
        )
    )

    # Generate y-axis labels (note names)
    y_tickvals = list(range(min_pitch, max_pitch + 1))
    y_ticktext = [
        f"{name}{octave}"
        for name, octave in (midi_number_to_name_and_octave(p) for p in y_tickvals)
    ]

    # Update layout for Ableton-style appearance
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(
            range=[0, 64],
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=[0, 16, 32, 48, 64],
            ticktext=["", "", "", "", ""],
            tickfont=dict(color=text_color),
            title=dict(text="", font=dict(color=text_color)),
            fixedrange=False,  # Allow zoom
        ),
        yaxis=dict(
            range=[min_pitch - 0.5, max_pitch + 1.5],
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickfont=dict(color=text_color, size=10),
            title=dict(text="", font=dict(color=text_color)),
        ),
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        height=400,
        hoverdistance=20,
        hovermode="closest",
    )

    # Configure modebar to show only useful tools
    fig.update_layout(
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=text_color, activecolor="#FF6B5B")
    )

    return fig