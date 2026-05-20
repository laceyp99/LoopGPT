import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src import history


@pytest.fixture
def isolated_history_dir(monkeypatch, tmp_path):
    generations_dir = tmp_path / "generations"
    monkeypatch.setattr(history, "GENERATIONS_DIR", str(generations_dir))
    return generations_dir


def _write_binary_file(path: Path, content: bytes = b"test-data"):
    path.write_bytes(content)
    return path


def _write_generation_metadata(
    base_dir: Path,
    *,
    gen_id: str,
    timestamp: datetime,
    prompt: str = "prompt",
    key: str = "C",
    scale: str = "major",
    model: str = "model",
    provider: str = "OpenAI",
    temperature: float = 0.0,
    cost: float | None = None,
):
    gen_dir = base_dir / f"gen_{gen_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    metadata = history.GenerationMetadata(
        id=gen_id,
        timestamp=timestamp,
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        midi_path=str(gen_dir / "loop.mid"),
        audio_path=str(gen_dir / "loop.mp3"),
    )
    (gen_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return gen_dir


def test_save_generation_copies_files_and_persists_metadata(isolated_history_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    midi_source = _write_binary_file(tmp_path / "source.mid")
    audio_source = _write_binary_file(tmp_path / "source.mp3")

    gen_id = history.save_generation(
        midi_path=str(midi_source),
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        cost=1.5,
        audio_path=str(audio_source),
    )

    gen_dir = isolated_history_dir / "gen_fixed_id"
    metadata = history.get_generation(gen_id)

    assert gen_id == "fixed_id"
    assert gen_dir.exists()
    assert (gen_dir / "loop.mid").read_bytes() == midi_source.read_bytes()
    assert (gen_dir / "loop.mp3").read_bytes() == audio_source.read_bytes()
    assert metadata is not None
    assert metadata.prompt == "warm rhodes loop"
    assert metadata.key == "D"
    assert metadata.scale == "minor"
    assert metadata.model == "gpt-4o-mini"
    assert metadata.provider == "OpenAI"
    assert metadata.temperature == 0.3
    assert metadata.cost == 1.5


def test_load_history_sorts_newest_first(isolated_history_dir):
    now = datetime.now()
    _write_generation_metadata(isolated_history_dir, gen_id="older", timestamp=now - timedelta(days=1))
    _write_generation_metadata(isolated_history_dir, gen_id="newer", timestamp=now)

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["newer", "older"]


def test_load_history_skips_missing_and_malformed_metadata(isolated_history_dir):
    valid_dir = _write_generation_metadata(
        isolated_history_dir,
        gen_id="valid",
        timestamp=datetime.now(),
    )
    missing_dir = isolated_history_dir / "gen_missing"
    missing_dir.mkdir(parents=True)
    bad_dir = isolated_history_dir / "gen_bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text("{not-json}", encoding="utf-8")

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["valid"]
    assert valid_dir.exists()
    assert missing_dir.exists()
    assert bad_dir.exists()


def test_get_generation_returns_none_for_missing_or_invalid_metadata(isolated_history_dir):
    missing_result = history.get_generation("missing")

    bad_dir = isolated_history_dir / "gen_broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text(json.dumps({"id": "broken"}), encoding="utf-8")

    broken_result = history.get_generation("broken")

    assert missing_result is None
    assert broken_result is None


def test_delete_generation_removes_directory_and_handles_missing_id(isolated_history_dir):
    _write_generation_metadata(isolated_history_dir, gen_id="delete_me", timestamp=datetime.now())

    assert history.delete_generation("delete_me") is True
    assert not (isolated_history_dir / "gen_delete_me").exists()
    assert history.delete_generation("delete_me") is False


def test_enforce_limit_removes_oldest_generations(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "MAX_GENERATIONS", 2)
    now = datetime.now()
    _write_generation_metadata(isolated_history_dir, gen_id="oldest", timestamp=now - timedelta(days=2))
    _write_generation_metadata(isolated_history_dir, gen_id="middle", timestamp=now - timedelta(days=1))
    _write_generation_metadata(isolated_history_dir, gen_id="newest", timestamp=now)

    history._enforce_limit()

    remaining = {path.name for path in isolated_history_dir.iterdir()}

    assert remaining == {"gen_middle", "gen_newest"}


def test_history_count_and_clear_history_reflect_saved_generations(isolated_history_dir):
    _write_generation_metadata(isolated_history_dir, gen_id="one", timestamp=datetime.now() - timedelta(minutes=1))
    _write_generation_metadata(isolated_history_dir, gen_id="two", timestamp=datetime.now())

    assert history.get_history_count() == 2
    assert history.clear_history() == 2
    assert history.get_history_count() == 0


def test_get_provider_for_model_returns_matching_provider_or_ollama_default():
    model_info = {
        "models": {
            "OpenAI": {"gpt-5-mini": {}},
            "Google": {"gemini-3.5-flash": {}},
        }
    }

    assert history.get_provider_for_model("gpt-5-mini", model_info) == "OpenAI"
    assert history.get_provider_for_model("gemini-3.5-flash", model_info) == "Google"
    assert history.get_provider_for_model("llama3.2:1b", model_info) == "Ollama"