from pathlib import Path

import pytest
from mido import MidiFile

from conductor_core.midi import loop_to_midi
from conductor_core.models import Bar, Bar_G, Loop, Loop_G, Note, Note_G, TimeInformation, TimeInformation_G


@pytest.fixture
def note_factory():
    def factory(
        *,
        pitch="C",
        octave=4,
        velocity=96,
        start_beat=1,
        duration=16,
    ):
        return Note(
            pitch=pitch,
            octave=octave,
            velocity=velocity,
            time=TimeInformation(start_beat=start_beat, duration=duration),
        )

    return factory


@pytest.fixture
def note_g_factory():
    def factory(
        *,
        pitch="C",
        octave=4,
        velocity=96,
        start_beat="one",
        duration="sixteen",
    ):
        return Note_G(
            pitch=pitch,
            octave=octave,
            velocity=velocity,
            time=TimeInformation_G(start_beat=start_beat, duration=duration),
        )

    return factory


@pytest.fixture
def loop_factory(note_factory):
    def factory(*, bars=None):
        if bars is None:
            bars = [
                [note_factory(pitch="C", start_beat=1, duration=16)],
                [note_factory(pitch="E", start_beat=1, duration=16)],
                [note_factory(pitch="G", start_beat=1, duration=16)],
                [note_factory(pitch="B", start_beat=1, duration=16)],
            ]

        return Loop(
            Bar_1=Bar(num=1, notes=bars[0]),
            Bar_2=Bar(num=2, notes=bars[1]),
            Bar_3=Bar(num=3, notes=bars[2]),
            Bar_4=Bar(num=4, notes=bars[3]),
        )

    return factory


@pytest.fixture
def loop_g_factory(note_g_factory):
    def factory(*, bars=None):
        if bars is None:
            bars = [
                [note_g_factory(pitch="C", start_beat="one", duration="sixteen")],
                [note_g_factory(pitch="E", start_beat="one", duration="sixteen")],
                [note_g_factory(pitch="G", start_beat="one", duration="sixteen")],
                [note_g_factory(pitch="B", start_beat="one", duration="sixteen")],
            ]

        return Loop_G(
            Bar_1=Bar_G(num=1, notes=bars[0]),
            Bar_2=Bar_G(num=2, notes=bars[1]),
            Bar_3=Bar_G(num=3, notes=bars[2]),
            Bar_4=Bar_G(num=4, notes=bars[3]),
        )

    return factory


@pytest.fixture
def sample_loop(loop_factory):
    return loop_factory()


@pytest.fixture
def sample_loop_g(loop_g_factory):
    return loop_g_factory()


@pytest.fixture
def midi_builder(tmp_path):
    def factory(loop, *, times_as_string=False, ticks_per_beat=480, filename="test_loop.mid"):
        midi = MidiFile(ticks_per_beat=ticks_per_beat)
        loop_to_midi(midi, loop, times_as_string=times_as_string)

        midi_path = Path(tmp_path) / filename
        midi.save(midi_path)
        return midi_path

    return factory
