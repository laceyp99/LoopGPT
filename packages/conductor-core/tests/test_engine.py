import json
from pathlib import Path

import pytest

from conductor_core import (
    EngineConfig,
    GenerationRequest,
    LoopGenerationEngine,
    ProviderCredentials,
)
from conductor_core import engine as engine_module


def test_engine_generates_persisted_artifacts_with_mocked_provider(
    monkeypatch,
    tmp_path,
    sample_loop,
):
    captured = {}
    progress_events = []

    def fake_generate_midi(**kwargs):
        captured.update(kwargs)
        return sample_loop, [{"role": "user", "content": "prompt"}], 0.25

    def fake_midi_to_mp3(midi_path, output_path=None, soundfont_name=None):
        Path(output_path).write_bytes(b"audio")
        captured["audio"] = {
            "midi_path": midi_path,
            "output_path": output_path,
            "soundfont_name": soundfont_name,
        }
        return output_path

    monkeypatch.setattr(engine_module.routing, "generate_midi", fake_generate_midi)
    monkeypatch.setattr(engine_module.playback, "midi_to_mp3", fake_midi_to_mp3)
    monkeypatch.setattr(engine_module.music, "get_loop_prompt", lambda: "default prompt")

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(
            artifact_root=tmp_path / "generations",
            provider_credentials=ProviderCredentials(openai_api_key="openai-key"),
            prompt_override="config prompt",
        )
    )
    result = engine.generate(
        GenerationRequest(
            key="C",
            scale="Major",
            description="warm rhodes loop",
            model="gpt-4o-mini",
            temperature=0.3,
            render_audio=True,
            soundfont_path="custom.sf2",
        ),
        progress_callback=progress_events.append,
    )

    generation_dir = tmp_path / "generations" / f"gen_{result.generation_id}"

    assert result.midi_path == str(generation_dir / "loop.mid")
    assert result.audio_path == str(generation_dir / "loop.mp3")
    assert result.metadata.soundfont == "custom.sf2"
    assert Path(result.midi_path).exists()
    assert Path(result.audio_path).read_bytes() == b"audio"
    assert json.loads((generation_dir / "messages.json").read_text(encoding="utf-8")) == [
        {"role": "user", "content": "prompt"}
    ]
    assert captured["model_choice"] == "gpt-4o-mini"
    assert captured["prompt"] == "C Major warm rhodes loop."
    assert captured["provider_credentials"].openai_api_key == "openai-key"
    assert captured["system_prompt"] == "config prompt"
    assert captured["audio"]["soundfont_name"] == "custom.sf2"
    assert [event.stage for event in progress_events] == [
        "provider_call",
        "midi",
        "audio",
    ]


def test_engine_cleans_unfinalized_workspace_when_processing_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        engine_module.routing,
        "generate_midi",
        lambda **kwargs: (None, [], 0),
    )

    engine = LoopGenerationEngine(
        EngineConfig.from_defaults(artifact_root=tmp_path / "generations")
    )

    with pytest.raises(ValueError, match="loop object is None"):
        engine.generate(
            GenerationRequest(
                key="C",
                scale="Major",
                description="broken loop",
                model="gpt-4o-mini",
            )
        )

    generations_root = tmp_path / "generations"
    assert not generations_root.exists() or list(generations_root.iterdir()) == []
