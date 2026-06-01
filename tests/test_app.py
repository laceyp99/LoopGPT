from pathlib import Path
from types import SimpleNamespace

import app


def _write_binary_file(path: Path, content: bytes = b"data") -> Path:
    path.write_bytes(content)
    return path


def test_get_selected_soundfont_prefers_requested_choice(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_soundfont_choices",
        lambda: ["FM-Piano1 20190916.sf2", "custom.sf2"],
    )
    monkeypatch.setattr(
        app,
        "get_default_soundfont",
        lambda: str(Path("soundfonts") / "FM-Piano1 20190916.sf2"),
    )

    selected_soundfont = app.get_selected_soundfont("custom.sf2")

    assert selected_soundfont == "custom.sf2"


def test_rerender_current_audio_skips_existing_matching_soundfont(monkeypatch, tmp_path):
    midi_path = _write_binary_file(tmp_path / "loop.mid")
    audio_path = _write_binary_file(tmp_path / "loop.mp3")

    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")

    def fail_render(*args, **kwargs):
        raise AssertionError("midi_to_mp3 should not be called when the audio is already current")

    monkeypatch.setattr(app, "midi_to_mp3", fail_render)

    rerendered_audio_path, status, saved_soundfont, current_audio_path = app.rerender_current_audio(
        str(midi_path),
        "custom.sf2",
        "custom.sf2",
        "gen_1",
        str(audio_path),
    )

    assert rerendered_audio_path == str(audio_path)
    assert status == "Audio already rendered with custom.sf2."
    assert saved_soundfont == "custom.sf2"
    assert current_audio_path == str(audio_path)


def test_rerender_current_audio_updates_saved_generation(monkeypatch, tmp_path):
    midi_path = _write_binary_file(tmp_path / "loop.mid")
    rendered_audio = _write_binary_file(tmp_path / "rendered.mp3", b"rendered")

    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")
    monkeypatch.setattr(
        app,
        "midi_to_mp3",
        lambda midi_path, output_path=None, soundfont_name=None: str(rendered_audio),
    )
    monkeypatch.setattr(
        app,
        "update_generation_audio",
        lambda gen_id, audio_path, soundfont=None: SimpleNamespace(
            audio_path=str(tmp_path / "saved-loop.mp3"),
            soundfont=soundfont,
        ),
    )

    rerendered_audio_path, status, saved_soundfont, current_audio_path = app.rerender_current_audio(
        str(midi_path),
        "custom.sf2",
        "old.sf2",
        "gen_1",
        None,
    )

    assert rerendered_audio_path == str(tmp_path / "saved-loop.mp3")
    assert status == "Rendered audio with custom.sf2."
    assert saved_soundfont == "custom.sf2"
    assert current_audio_path == str(tmp_path / "saved-loop.mp3")


def test_refresh_soundfont_controls_updates_dropdown_choices(monkeypatch):
    midi_path = _write_binary_file(Path("active.mid"))

    monkeypatch.setattr(app, "get_soundfont_choices", lambda: ["FM-Piano1 20190916.sf2", "new.sf2"])
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    try:
        dropdown_update, rerender_update, status_message = app.refresh_soundfont_controls(
            "new.sf2",
            str(midi_path),
        )

        assert dropdown_update["choices"] == ["FM-Piano1 20190916.sf2", "new.sf2"]
        assert dropdown_update["value"] == "new.sf2"
        assert rerender_update["interactive"] is True
        assert status_message == "Found 2 SoundFonts. Selected new.sf2."
    finally:
        midi_path.unlink(missing_ok=True)


def test_refresh_soundfont_controls_prefers_dependency_status_message(monkeypatch):
    monkeypatch.setattr(app, "get_soundfont_choices", lambda: ["FM-Piano1 20190916.sf2", "new.sf2"])
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(
        app,
        "is_playback_available",
        lambda soundfont_name=None: (False, "FluidSynth is not installed or not in PATH"),
    )
    monkeypatch.setattr(
        app,
        "get_playback_status_message",
        lambda soundfont_name=None: "Audio playback is not available. Setup required:\n  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases",
    )
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    dropdown_update, rerender_update, status_message = app.refresh_soundfont_controls("new.sf2", None)

    assert dropdown_update["choices"] == ["FM-Piano1 20190916.sf2", "new.sf2"]
    assert dropdown_update["value"] == "new.sf2"
    assert rerender_update["interactive"] is False
    assert status_message == (
        "Audio playback is not available. Setup required:\n"
        "  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases"
    )


def test_get_rerender_button_update_requires_active_midi(monkeypatch):
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    rerender_update = app.get_rerender_button_update("new.sf2", None)

    assert rerender_update["interactive"] is False


def test_delete_history_item_disables_rerender_for_deleted_loaded_generation(monkeypatch, tmp_path):
    midi_path = _write_binary_file(tmp_path / "loop.mid")
    audio_path = _write_binary_file(tmp_path / "loop.mp3")

    monkeypatch.setattr(app, "delete_generation", lambda gen_id: True)
    monkeypatch.setattr(app, "get_history_choices", lambda: ["gen_2"])
    monkeypatch.setattr(app, "render_history_html", lambda: "<div>history</div>")
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    (
        dropdown_update,
        status_message,
        history_html,
        cleared_midi_path,
        cleared_audio_path,
        cleared_visualization,
        current_generation_id,
        current_saved_soundfont,
        current_audio_path,
        rerender_update,
    ) = app.delete_history_item(
        "gen_1",
        current_generation_id="gen_1",
        soundfont_choice="new.sf2",
        midi_path=str(midi_path),
        current_saved_soundfont="old.sf2",
        current_audio_path=str(audio_path),
    )

    assert dropdown_update == {"choices": ["gen_2"], "value": None}
    assert status_message == "Deleted generation"
    assert history_html == "<div>history</div>"
    assert cleared_midi_path is None
    assert cleared_audio_path is None
    assert cleared_visualization is None
    assert current_generation_id is None
    assert current_saved_soundfont is None
    assert current_audio_path is None
    assert rerender_update["interactive"] is False


def test_refresh_soundfont_controls_stays_disabled_after_active_delete(monkeypatch):
    monkeypatch.setattr(app, "get_soundfont_choices", lambda: ["FM-Piano1 20190916.sf2", "new.sf2"])
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    _, rerender_update, _ = app.refresh_soundfont_controls("new.sf2", None)

    assert rerender_update["interactive"] is False