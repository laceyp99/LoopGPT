import json
from importlib import resources

from conductor_core import models as objects

# Flat list of chromatic note names (pitch class 0-11, sharps only)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic note name variants per pitch class (for scale spelling, validation, etc.)
ENHARMONIC_NOTE_NAMES = [
    ["B#", "C", "Dbb"],  # 0
    ["C#", "Db", "B##"],  # 1
    ["D", "C##", "Ebb"],  # 2
    ["D#", "Eb", "Fbb"],  # 3
    ["E", "Fb", "D##"],  # 4
    ["E#", "F", "Gbb"],  # 5
    ["F#", "Gb", "E##"],  # 6
    ["G", "F##", "Abb"],  # 7
    ["G#", "Ab"],  # 8
    ["A", "G##", "Bbb"],  # 9
    ["A#", "Bb", "Cbb"],  # 10
    ["B", "Cb", "A##"],  # 11
]
# A dictionary that maps note names to their corresponding MIDI numbers
base_midi_numbers = {
    "C": 0,
    "Dbb": 0,
    "B♯♯": 1,
    "B##": 1,
    "C♯": 1,
    "C#": 1,
    "D♭": 1,
    "Db": 1,
    "C♯♯": 2,
    "C##": 2,
    "D": 2,
    "Ebb": 2,
    "D♯": 3,
    "D#": 3,
    "E♭": 3,
    "Eb": 3,
    "Fbb": 3,
    "D♯♯": 4,
    "D##": 4,
    "E": 4,
    "Fb": 4,
    "F♭": 4,
    "E♯": 5,
    "E#": 5,
    "F": 5,
    "Gbb": 5,
    "E♯♯": 6,
    "E##": 6,
    "F♯": 6,
    "F#": 6,
    "Gb": 6,
    "G♭": 6,
    "F♯♯": 7,
    "F##": 7,
    "G": 7,
    "Abb": 7,
    "G♯": 8,
    "G#": 8,
    "A♭": 8,
    "Ab": 8,
    "G♯♯": 9,
    "G##": 9,
    "A": 9,
    "Bbb": 9,
    "A♯": 10,
    "A#": 10,
    "B♭": 10,
    "Bb": 10,
    "Cbb": 10,
    "A♯♯": 11,
    "A##": 11,
    "B": 11,
    "Cb": 11,
    "C♭": 11,
    "B♯": 12,
    "B#": 12,
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
    "sixteenth": {"beats": 0.25, "sixteenths": 1, "display": "1/16", "aliases": ["16th"]},
    "eighth": {"beats": 0.5, "sixteenths": 2, "display": "1/8", "aliases": ["8th"]},
    "quarter": {"beats": 1.0, "sixteenths": 4, "display": "1/4", "aliases": []},
    "half": {"beats": 2.0, "sixteenths": 8, "display": "1/2", "aliases": []},
    "whole": {"beats": 4.0, "sixteenths": 16, "display": "1 bar", "aliases": []},
}

# Derived lookups from DURATION_MAP
DURATION_BEATS = {name: d["beats"] for name, d in DURATION_MAP.items()}
DURATION_SIXTEENTHS_TO_DISPLAY = {
    d["sixteenths"]: d["display"] for d in DURATION_MAP.values()
}
DURATION_KEYWORDS = {name: name for name in DURATION_MAP}
for name, d in DURATION_MAP.items():
    for alias in d["aliases"]:
        DURATION_KEYWORDS[alias] = name
DURATION_BEATS_TO_NAME = {d["beats"]: name.title() for name, d in DURATION_MAP.items()}

# Interval names (semitones 0-11 relative to root)
INTERVAL_NAMES = [
    "Root",
    "m2",
    "M2",
    "m3",
    "M3",
    "P4",
    "Tritone",
    "P5",
    "m6",
    "M6",
    "m7",
    "M7",
]

_model_info_cache = None


def get_model_info():
    """Load packaged model metadata once and reuse it."""
    global _model_info_cache

    if _model_info_cache is None:
        model_info_resource = resources.files("conductor_core.resources").joinpath(
            "model_list.json"
        )
        with model_info_resource.open("r", encoding="utf-8") as model_file:
            _model_info_cache = json.load(model_file)

    return _model_info_cache


def get_loop_prompt():
    """Load the packaged default loop generation prompt."""
    prompt_resource = resources.files("conductor_core.resources").joinpath(
        "prompts",
        "loop_gen.txt",
    )
    with prompt_resource.open("r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def split_reported_cache_tokens(total_tokens, cached_tokens):
    """Return uncached and cached token counts from provider-reported usage.

    Cache savings should come from actual provider usage fields, not estimated
    cache behavior. Malformed negative values are ignored, and cached tokens are
    capped to the reported total so input-token costs cannot go negative.
    """
    total = max(total_tokens or 0, 0)
    cached = min(max(cached_tokens or 0, 0), total)
    return total - cached, cached


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
    cleaned_pitch = (
        note.pitch.strip().replace("♯", "#").replace("♭", "b").replace("𝄪", "##").replace("x", "##").replace("𝄫", "bb")
    )
    if cleaned_pitch not in base_midi_numbers:
        raise ValueError(f"Unrecognized note name: {note.pitch}")
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
    return [f"{pitch_class_to_note(n)}{n // 12 - 1}" for n in midi_numbers]


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
    return objects.SIXTEENTH_NOTE_G_TO_INT[sixteenth_g.value.lower()]


def int_to_sixteenth_g(sixteenth):
    """Convert an integer sixteenth-note position into a SixteenthNote_G enum."""
    return objects.SixteenthNote_G.from_int(sixteenth)
