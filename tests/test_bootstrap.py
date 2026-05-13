from mido import MidiFile


def test_sample_loop_fixture_builds_valid_loop(sample_loop):
    assert sample_loop.Bar_1.notes[0].pitch == "C"
    assert sample_loop.Bar_1.notes[0].time.start_beat == 1
    assert sample_loop.Bar_1.notes[0].time.duration == 16


def test_midi_builder_writes_midi_file(sample_loop, midi_builder):
    midi_path = midi_builder(sample_loop)

    midi = MidiFile(midi_path)

    assert midi_path.exists()
    assert len(midi.tracks) == 1