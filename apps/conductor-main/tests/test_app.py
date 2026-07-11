from pathlib import Path
from types import SimpleNamespace

from conductor_main import app


def _write_binary_file(path: Path, content: bytes = b"data") -> Path:
    path.write_bytes(content)
    return path


def test_run_loop_passes_ui_configuration_to_core(monkeypatch, tmp_path):
    captured = {}
    midi_path = tmp_path / "loop.mid"
    midi_path.write_bytes(b"midi")
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")
    monkeypatch.setattr(app, "load_app_prompt_override", lambda: "override prompt")
    monkeypatch.setattr(app, "MidiFile", lambda path: "midi")
    monkeypatch.setattr(app, "visualize_midi_plotly", lambda midi: "viz")

    class FakeEngine:
        def __init__(self, config):
            captured["config"] = config

        def generate(self, request, progress_callback=None):
            captured["request"] = request
            if progress_callback:
                progress_callback(
                    SimpleNamespace(stage="provider_call", message="Generating MIDI...")
                )
            return SimpleNamespace(
                midi_path=str(midi_path),
                audio_path=None,
                cost=0.25,
                generation_id="fixed_id",
                metadata=SimpleNamespace(soundfont=None),
            )

    monkeypatch.setattr(app, "LoopGenerationEngine", FakeEngine)

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
            openai_key=" openai-key ",
            gemini_key=" gemini-key ",
            claude_key=" claude-key ",
        )
    )

    final_output = outputs[-1]

    assert final_output[0] == str(midi_path)
    assert final_output[2] == "viz"
    assert captured["config"].prompt_override == "override prompt"
    assert captured["config"].provider_credentials.openai_api_key == "openai-key"
    assert captured["config"].provider_credentials.google_api_key == "gemini-key"
    assert captured["config"].provider_credentials.anthropic_api_key == "claude-key"
    assert captured["config"].default_soundfont_path == "custom.sf2"
    assert captured["request"].render_audio is True
    assert captured["request"].soundfont_path == "custom.sf2"
    assert captured["request"].description == "warm rhodes loop"


def test_run_loop_reports_core_generation_errors(monkeypatch):
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(app, "get_selected_soundfont", lambda choice=None: "custom.sf2")

    class FailingEngine:
        def __init__(self, config):
            pass

        def generate(self, request, progress_callback=None):
            raise ValueError("provider failed")

    monkeypatch.setattr(app, "LoopGenerationEngine", FailingEngine)

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

    assert outputs[-1][3] == "provider failed"
    assert outputs[-1][4] == {"visible": False}


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


def test_prompt_override_uses_the_app_data_directory(monkeypatch, tmp_path):
    override_path = tmp_path / "Prompts" / "loop gen.txt"
    monkeypatch.setattr(app, "PROMPT_OVERRIDE_PATH", override_path)

    assert app.load_app_prompt_override() is None
    assert app.save_prompts("standalone override").startswith("Prompts saved successfully")
    assert override_path.read_text(encoding="utf-8") == "standalone override"
    assert app.load_app_prompt_override() == "standalone override"


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
        lambda soundfont_name=None: (
            "Audio playback is not available. Setup required:\n  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases"
        ),
    )
    monkeypatch.setattr(app.gr, "update", lambda **kwargs: kwargs)

    dropdown_update, rerender_update, status_message = app.refresh_soundfont_controls(
        "new.sf2", None
    )

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


def test_main_allows_gradio_to_serve_generation_history(monkeypatch, tmp_path):
    artifact_root = tmp_path / "app-data" / "generations"
    launched_with = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched_with.update(kwargs)

    monkeypatch.setattr(app, "HISTORY_STORE", SimpleNamespace(artifact_root=str(artifact_root)))
    monkeypatch.setattr(app, "get_selected_soundfont", lambda: None)
    monkeypatch.setattr(app, "is_playback_available", lambda soundfont=None: (True, None))
    monkeypatch.setattr(app, "create_demo", lambda playback_status=None: FakeDemo())

    app.main()

    assert launched_with == {"allowed_paths": [str(artifact_root.resolve())]}
