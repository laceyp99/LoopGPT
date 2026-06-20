import pytest

from src import utils
from src.objects import SixteenthNote_G


@pytest.mark.parametrize(
    ("pitch_class", "expected_note"),
    [
        (0, "C"),
        (12, "C"),
        (-1, "B"),
        (25, "C#"),
    ],
)
def test_pitch_class_to_note_wraps_chromatic_values(pitch_class, expected_note):
    assert utils.pitch_class_to_note(pitch_class) == expected_note


@pytest.mark.parametrize(
    ("name", "expected_pitch_class"),
    [
        ("C", 0),
        ("C#", 1),
        ("Db", 1),
        ("G♭", 6),
        ("Cb", 11),
        ("B#", 0),
    ],
)
def test_note_name_to_pitch_class_supports_enharmonics(name, expected_pitch_class):
    assert utils.note_name_to_pitch_class(name) == expected_pitch_class


def test_note_name_to_pitch_class_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unrecognized note name"):
        utils.note_name_to_pitch_class("H")


@pytest.mark.parametrize(
    ("pitch_class", "root_pitch_class", "expected_interval"),
    [
        (0, 0, "Root"),
        (4, 0, "M3"),
        (11, 4, "P5"),
        (0, 11, "m2"),
    ],
)
def test_pitch_class_to_interval_uses_wrapped_distance(
    pitch_class,
    root_pitch_class,
    expected_interval,
):
    assert utils.pitch_class_to_interval(pitch_class, root_pitch_class) == expected_interval


@pytest.mark.parametrize(
    ("velocity", "expected_color"),
    [
        (0, "rgb(45,27,78)"),
        (127, "rgb(255,107,91)"),
    ],
)
def test_velocity_to_color_maps_expected_endpoint_colors(velocity, expected_color):
    assert utils.velocity_to_color(velocity) == expected_color


@pytest.mark.parametrize(
    ("midi_pitch", "expected"),
    [
        (60, False),
        (61, True),
        (66, True),
        (67, False),
    ],
)
def test_is_black_key_identifies_black_and_white_keys(midi_pitch, expected):
    assert utils.is_black_key(midi_pitch) is expected


@pytest.mark.parametrize(
    ("sixteenths", "expected"),
    [
        (1, "1/16"),
        (4, "1/4"),
        (16, "1 bar"),
        (3, "3/16"),
    ],
)
def test_format_duration_sixteenths_formats_standard_and_fallback_values(sixteenths, expected):
    assert utils.format_duration_sixteenths(sixteenths) == expected


@pytest.mark.parametrize(
    ("beats", "expected"),
    [
        (0.25, "Sixteenth"),
        (1.0, "Quarter"),
        (4.0, "Whole"),
        (1.5, "1.5 beats"),
    ],
)
def test_beats_to_duration_name_formats_standard_and_nonstandard_lengths(beats, expected):
    assert utils.beats_to_duration_name(beats) == expected


def test_scale_returns_expected_note_family_for_c_major():
    scale_notes = utils.scale("C", "major")

    for note_name in ["C", "D", "E", "F", "G", "A", "B"]:
        assert note_name in scale_notes


def test_scale_rejects_invalid_root_and_mode():
    with pytest.raises(ValueError, match="Invalid scale letter"):
        utils.scale("H", "major")

    with pytest.raises(ValueError, match="Invalid scale mode"):
        utils.scale("C", "dorian")


@pytest.mark.parametrize(
    ("pitch", "octave", "expected_midi_number"),
    [
        ("C", 4, 60),
        ("F♯", 3, 54),
        ("A♭", 4, 68),
        ("C♯", 4, 61),
    ],
)
def test_calculate_midi_number_normalizes_pitch_names(
    note_factory,
    pitch,
    octave,
    expected_midi_number,
):
    note = note_factory(pitch=pitch, octave=octave)

    assert utils.calculate_midi_number(note) == expected_midi_number


@pytest.mark.parametrize(
    ("midi_number", "expected_name", "expected_octave"),
    [
        (60, "C", 4),
        (61, "C#", 4),
        (73, "C#", 5),
    ],
)
def test_midi_number_to_name_and_octave_returns_canonical_name_and_octave(
    midi_number,
    expected_name,
    expected_octave,
):
    note_name, octave = utils.midi_number_to_name_and_octave(midi_number)

    assert note_name == expected_name
    assert octave == expected_octave


def test_midi_to_note_name_accepts_plain_python_lists():
    assert utils.midi_to_note_name([60, 61, 73]) == ["C4", "C#4", "C#5"]


def test_sixteenth_converters_round_trip_enum_values():
    sixteenth_note = utils.int_to_sixteenth_g(16)

    assert sixteenth_note is SixteenthNote_G.SIXTEEN
    assert utils.convert_sixteenth(sixteenth_note) == 16
