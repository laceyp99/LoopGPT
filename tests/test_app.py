import json
from pathlib import Path
from types import SimpleNamespace

import app
from src import history


def _write_binary_file(path: Path, content: bytes = b"data") -> Path:
    path.write_bytes(content)
    return path


def test_run_loop_persists_artifacts_in_generation_workspace(monkeypatch, tmp_path, sample_loop):
    generations_dir = tmp_path / "generations"
    monkeypatch.setattr(history, "GENERATIONS_DIR", str(generations_dir))
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(app.runs, "generate_midi", lambda **kwargs: (sample_loop, [{"role": "user", "content": "prompt"}], 0.25))
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")
    monkeypatch.setattr(
        app,
        "get_model_info",
        lambda: {
            "models": {
                "Google": {},
                "OpenAI": {"gpt-test": {}},
            }
        },
    )

    def fake_midi_to_mp3(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(app, "midi_to_mp3", fake_midi_to_mp3)
    monkeypatch.setattr(app, "visualize_midi_plotly", lambda midi: "viz")

    outputs = list(
        app.run_loop(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            temp=0.3,
            model_choice="gpt-test",
            use_thinking=False,
            effort="low",
            soundfont_choice="custom.sf2",
            openai_key="",
            gemini_key="",
            claude_key="",
        )
    )

    final_output = outputs[-1]
    gen_dir = generations_dir / "gen_fixed_id"

    assert final_output[0] == str(gen_dir / "loop.mid")
    assert final_output[1] == str(gen_dir / "loop.mp3")
    assert final_output[2] == "viz"
    assert final_output[3] == ""
    assert final_output[5] == "fixed_id"
    assert final_output[6] == "custom.sf2"
    assert final_output[7] == str(gen_dir / "loop.mp3")
    assert (gen_dir / "loop.mid").exists()
    assert (gen_dir / "loop.mp3").read_bytes() == b"audio"
    assert json.loads((gen_dir / "messages.json").read_text(encoding="utf-8")) == [
        {"role": "user", "content": "prompt"}
    ]

    metadata = history.get_generation("fixed_id")
    assert metadata is not None
    assert metadata.midi_path == str(gen_dir / "loop.mid")
    assert metadata.audio_path == str(gen_dir / "loop.mp3")
    assert metadata.messages_path == str(gen_dir / "messages.json")


def test_run_loop_cleans_workspace_when_generator_closes_before_finalization(
    monkeypatch, tmp_path, sample_loop
):
    generations_dir = tmp_path / "generations"
    monkeypatch.setattr(history, "GENERATIONS_DIR", str(generations_dir))
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        app.runs,
        "generate_midi",
        lambda **kwargs: (sample_loop, [{"role": "user", "content": "prompt"}], 0.25),
    )
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")
    monkeypatch.setattr(
        app,
        "get_model_info",
        lambda: {
            "models": {
                "Google": {},
                "OpenAI": {"gpt-test": {}},
            }
        },
    )

    def fail_midi_to_mp3(*args, **kwargs):
        raise AssertionError("midi_to_mp3 should not run before this stop-waiting point")

    monkeypatch.setattr(app, "midi_to_mp3", fail_midi_to_mp3)
    monkeypatch.setattr(app, "visualize_midi_plotly", lambda midi: "viz")

    generator = app.run_loop(
        key="C",
        scale="Major",
        description="warm rhodes loop",
        temp=0.3,
        model_choice="gpt-test",
        use_thinking=False,
        effort="low",
        soundfont_choice="custom.sf2",
        openai_key="",
        gemini_key="",
        claude_key="",
    )

    assert next(generator)[3] == "Working on it..."
    assert next(generator)[3] == "Processing MIDI..."
    assert next(generator)[3] == "Rendering Audio..."

    generator.close()

    assert not (generations_dir / "gen_fixed_id").exists()


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


def test_default_model_exists_in_model_metadata():
    model_info = app.get_model_info()

    assert app.DEFAULT_PROVIDER in model_info["models"]
    assert app.DEFAULT_MODEL in model_info["models"][app.DEFAULT_PROVIDER]


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


def test_load_history_item_warns_when_saved_soundfont_is_missing(monkeypatch, tmp_path):
    midi_path = _write_binary_file(tmp_path / "loop.mid")
    audio_path = _write_binary_file(tmp_path / "loop.mp3")

    monkeypatch.setattr(app, "get_soundfont_choices", lambda: ["FM-Piano1 20190916.sf2", "new.sf2"])
    monkeypatch.setattr(
        app,
        "get_default_soundfont",
        lambda: str(Path("soundfonts") / "FM-Piano1 20190916.sf2"),
    )
    monkeypatch.setattr(
        app,
        "get_generation",
        lambda gen_id: SimpleNamespace(
            midi_path=str(midi_path),
            audio_path=str(audio_path),
            soundfont="missing.sf2",
            id=gen_id,
        ),
    )
    monkeypatch.setattr(app, "MidiFile", lambda path: object())
    monkeypatch.setattr(app, "visualize_midi_plotly", lambda midi: "viz")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    (
        loaded_midi_path,
        loaded_audio_path,
        dropdown_update,
        visualization,
        error_message,
        generation_id,
        saved_soundfont,
        current_audio_path,
        rerender_update,
    ) = app.load_history_item("gen_1")

    assert loaded_midi_path == str(midi_path)
    assert loaded_audio_path == str(audio_path)
    assert dropdown_update["value"] == "FM-Piano1 20190916.sf2"
    assert visualization == "viz"
    assert error_message == "Previously used SoundFont: missing.sf2 (missing)"
    assert generation_id == "gen_1"
    assert saved_soundfont == "missing.sf2"
    assert current_audio_path == str(audio_path)
    assert rerender_update["interactive"] is True


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


def test_render_history_html_displays_zero_cost(monkeypatch):
    monkeypatch.setattr(
        app,
        "load_history",
        lambda: [
            SimpleNamespace(
                id="20260101_120000",
                timestamp=__import__("datetime").datetime(2026, 1, 1, 12, 0),
                prompt="local model loop",
                key="C",
                scale="Major",
                model="llama3",
                cost=0,
            )
        ],
    )

    html = app.render_history_html()

    assert "Cost: $0.0000" in html
    assert "Cost: N/A" not in html


def test_render_history_html_displays_missing_cost_as_na(monkeypatch):
    monkeypatch.setattr(
        app,
        "load_history",
        lambda: [
            SimpleNamespace(
                id="20260101_120000",
                timestamp=__import__("datetime").datetime(2026, 1, 1, 12, 0),
                prompt="cloud model loop",
                key="C",
                scale="Major",
                model="gpt-5-mini",
                cost=None,
            )
        ],
    )

    html = app.render_history_html()

    assert "Cost: N/A" in html


def test_refresh_soundfont_controls_stays_disabled_after_active_delete(monkeypatch):
    monkeypatch.setattr(app, "get_soundfont_choices", lambda: ["FM-Piano1 20190916.sf2", "new.sf2"])
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "new.sf2")
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont_name=None: (True, None))
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    _, rerender_update, _ = app.refresh_soundfont_controls("new.sf2", None)

    assert rerender_update["interactive"] is False
