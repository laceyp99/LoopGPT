from pathlib import Path

from conductor_core import playback as audio


def _write_file(path: Path, content: bytes = b"data") -> Path:
    path.write_bytes(content)
    return path


def test_list_soundfonts_returns_sorted_sf2_files(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "zeta.sf2")
    _write_file(tmp_path / "Alpha.sf2")
    _write_file(tmp_path / "notes.txt")

    soundfonts = audio.list_soundfonts()

    assert soundfonts == ["Alpha.sf2", "zeta.sf2"]


def test_get_default_soundfont_prefers_known_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    _write_file(tmp_path / "custom.sf2")
    preferred = _write_file(tmp_path / "FM-Piano1 20190916.sf2")

    soundfont_path = audio.get_default_soundfont()

    assert soundfont_path == str(preferred)


def test_resolve_soundfont_returns_requested_file_from_soundfont_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    expected = _write_file(tmp_path / "custom.sf2")

    soundfont_path = audio.resolve_soundfont("custom.sf2")

    assert soundfont_path == str(expected)


def test_resolve_soundfont_ignores_cwd_soundfonts(monkeypatch, tmp_path):
    core_soundfonts = tmp_path / "core"
    cwd = tmp_path / "cwd"
    cwd_soundfonts = cwd / "soundfonts"
    core_soundfonts.mkdir()
    cwd_soundfonts.mkdir(parents=True)
    _write_file(cwd_soundfonts / "cwd-only.sf2")
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(core_soundfonts))
    monkeypatch.chdir(cwd)

    assert audio.resolve_soundfont("cwd-only.sf2") is None


def test_is_playback_available_reports_missing_requested_soundfont(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: True)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: True)

    available, error = audio.is_playback_available("missing.sf2")

    assert available is False
    assert error == f"Requested SoundFont 'missing.sf2' was not found in '{tmp_path}'."


def test_get_playback_status_message_prioritizes_missing_dependencies(monkeypatch):
    monkeypatch.setattr(
        audio, "is_playback_available", lambda soundfont_name=None: (False, "dependency error")
    )
    monkeypatch.setattr(audio, "is_fluidsynth_available", lambda: False)
    monkeypatch.setattr(audio, "is_ffmpeg_available", lambda: False)
    monkeypatch.setattr(audio, "find_soundfont", lambda soundfont_name=None: None)

    status_message = audio.get_playback_status_message("missing.sf2")

    assert "Install FluidSynth" in status_message
    assert "Install FFmpeg" in status_message
    assert "Add the requested SoundFont" not in status_message


def test_midi_to_mp3_uses_requested_soundfont(monkeypatch, tmp_path):
    midi_path = _write_file(tmp_path / "loop.mid")
    soundfont_path = _write_file(tmp_path / "custom.sf2")
    output_path = tmp_path / "loop.mp3"
    captured = {}

    monkeypatch.setattr(audio, "SOUNDFONT_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "is_playback_available", lambda soundfont_name=None: (True, None))

    class FakeFluidSynth:
        def __init__(self, selected_soundfont):
            captured["soundfont_path"] = selected_soundfont

        def midi_to_audio(self, input_midi_path, temp_wav_path):
            captured["input_midi_path"] = input_midi_path
            Path(temp_wav_path).write_bytes(b"wav")

    class FakeAudioSegment:
        @staticmethod
        def from_wav(temp_wav_path):
            captured["temp_wav_path"] = temp_wav_path

            class FakeExport:
                def export(self, target_output_path, format, bitrate):
                    captured["output_path"] = target_output_path
                    captured["format"] = format
                    captured["bitrate"] = bitrate
                    Path(target_output_path).write_bytes(b"mp3")

            return FakeExport()

    monkeypatch.setattr(audio, "FluidSynth", FakeFluidSynth)
    monkeypatch.setattr(audio, "AudioSegment", FakeAudioSegment)

    rendered_output = audio.midi_to_mp3(
        str(midi_path),
        output_path=str(output_path),
        soundfont_name="custom.sf2",
    )

    assert rendered_output == str(output_path)
    assert captured["soundfont_path"] == str(soundfont_path)
    assert captured["input_midi_path"] == str(midi_path)
    assert captured["output_path"] == str(output_path)
    assert output_path.read_bytes() == b"mp3"
