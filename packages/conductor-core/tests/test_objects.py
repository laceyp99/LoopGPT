import pytest
from pydantic import ValidationError

from conductor_core.models import Loop, SixteenthNote, SixteenthNote_G, TimeInformation, TimeInformation_G


def test_sixteenth_note_g_from_int_returns_expected_enum_member():
    assert SixteenthNote_G.from_int(16) is SixteenthNote_G.SIXTEEN


def test_sixteenth_note_g_from_int_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="Invalid sixteenth-note value"):
        SixteenthNote_G.from_int(17)


def test_time_information_coerces_integer_inputs_to_enum_members():
    time_info = TimeInformation(start_beat=1, duration=16)

    assert time_info.start_beat is SixteenthNote.ONE
    assert time_info.duration is SixteenthNote.SIXTEEN


def test_time_information_g_rejects_unknown_string_values():
    with pytest.raises(ValidationError):
        TimeInformation_G(start_beat="zero", duration="sixteen")


def test_loop_validates_nested_bar_and_note_data_from_dicts():
    loop = Loop(
        Bar_1={"num": 1, "notes": [{"pitch": "C", "octave": 4, "velocity": 96, "time": {"start_beat": 1, "duration": 16}}]},
        Bar_2={"num": 2, "notes": []},
        Bar_3={"num": 3, "notes": []},
        Bar_4={"num": 4, "notes": []},
    )

    assert loop.Bar_1.notes[0].time.start_beat is SixteenthNote.ONE
    assert loop.Bar_1.notes[0].time.duration is SixteenthNote.SIXTEEN
