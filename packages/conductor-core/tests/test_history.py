import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conductor_core import storage as history


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
    soundfont: str | None = None,
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
        soundfont=soundfont,
    )
    (gen_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return gen_dir


def test_create_generation_workspace_allocates_canonical_paths(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    workspace = history.create_generation_workspace()

    gen_dir = isolated_history_dir / "gen_fixed_id"
    assert workspace.id == "fixed_id"
    assert workspace.directory == str(gen_dir)
    assert workspace.midi_path == str(gen_dir / "loop.mid")
    assert workspace.audio_path == str(gen_dir / "loop.mp3")
    assert workspace.messages_path == str(gen_dir / "messages.json")
    assert workspace.metadata_path == str(gen_dir / "metadata.json")
    assert gen_dir.exists()


def test_finalize_generation_persists_metadata_for_direct_written_artifacts(
    isolated_history_dir, monkeypatch
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), b"audio")
    Path(workspace.messages_path).write_text("[]", encoding="utf-8")

    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        cost=1.5,
        soundfont="FM-Piano1 20190916.sf2",
    )

    gen_dir = isolated_history_dir / "gen_fixed_id"
    loaded_metadata = history.get_generation("fixed_id")

    assert gen_dir.exists()
    assert (gen_dir / "loop.mid").read_bytes() == b"midi"
    assert (gen_dir / "loop.mp3").read_bytes() == b"audio"
    assert (gen_dir / "messages.json").read_text(encoding="utf-8") == "[]"
    assert metadata.id == "fixed_id"
    assert metadata.prompt == "warm rhodes loop"
    assert metadata.key == "D"
    assert metadata.scale == "minor"
    assert metadata.model == "gpt-4o-mini"
    assert metadata.provider == "OpenAI"
    assert metadata.temperature == 0.3
    assert metadata.cost == 1.5
    assert metadata.midi_path == str(gen_dir / "loop.mid")
    assert metadata.audio_path == str(gen_dir / "loop.mp3")
    assert metadata.messages_path == str(gen_dir / "messages.json")
    assert metadata.soundfont == "FM-Piano1 20190916.sf2"
    assert loaded_metadata == metadata


def test_finalize_generation_requires_direct_written_midi(isolated_history_dir, monkeypatch):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = history.create_generation_workspace()

    with pytest.raises(FileNotFoundError):
        history.finalize_generation(
            workspace=workspace,
            prompt="warm rhodes loop",
            key="D",
            scale="minor",
            model="gpt-4o-mini",
            provider="OpenAI",
            temperature=0.3,
        )

    assert not (isolated_history_dir / "gen_fixed_id" / "metadata.json").exists()


def test_cleanup_generation_workspace_removes_only_unfinalized_directories(
    isolated_history_dir, monkeypatch
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")
    workspace = history.create_generation_workspace()

    assert history.cleanup_generation_workspace(workspace) is True
    assert not (isolated_history_dir / "gen_fixed_id").exists()

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
    )

    assert history.cleanup_generation_workspace(workspace) is False
    assert (isolated_history_dir / "gen_fixed_id").exists()


def test_update_generation_audio_copies_audio_and_updates_soundfont(
    isolated_history_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(history, "_generate_id", lambda: "fixed_id")

    original_audio = _write_binary_file(tmp_path / "source.mp3", b"old-audio")
    updated_audio = _write_binary_file(tmp_path / "updated.mp3", b"new-audio")

    workspace = history.create_generation_workspace()
    _write_binary_file(Path(workspace.midi_path), b"midi")
    _write_binary_file(Path(workspace.audio_path), original_audio.read_bytes())
    metadata = history.finalize_generation(
        workspace=workspace,
        prompt="warm rhodes loop",
        key="D",
        scale="minor",
        model="gpt-4o-mini",
        provider="OpenAI",
        temperature=0.3,
        cost=1.5,
        soundfont="old.sf2",
    )

    updated = history.update_generation_audio(
        metadata.id,
        str(updated_audio),
        soundfont="new.sf2",
    )
    metadata = history.get_generation(metadata.id)
    gen_dir = isolated_history_dir / "gen_fixed_id"

    assert updated is not None
    assert updated.audio_path == str(gen_dir / "loop.mp3")
    assert updated.soundfont == "new.sf2"
    assert (gen_dir / "loop.mp3").read_bytes() == b"new-audio"
    assert metadata is not None
    assert metadata.soundfont == "new.sf2"


def test_load_history_allows_older_entries_without_soundfont(isolated_history_dir):
    now = datetime.now()
    _write_generation_metadata(
        isolated_history_dir,
        gen_id="older",
        timestamp=now - timedelta(days=1),
        soundfont=None,
    )
    _write_generation_metadata(
        isolated_history_dir,
        gen_id="newer",
        timestamp=now,
        soundfont="FM-Piano1 20190916.sf2",
    )

    loaded = history.load_history()

    assert [entry.id for entry in loaded] == ["newer", "older"]
    assert loaded[0].soundfont == "FM-Piano1 20190916.sf2"
    assert loaded[1].soundfont is None


def test_load_history_sorts_newest_first(isolated_history_dir):
    now = datetime.now()
    _write_generation_metadata(
        isolated_history_dir, gen_id="older", timestamp=now - timedelta(days=1)
    )
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
    _write_generation_metadata(
        isolated_history_dir, gen_id="oldest", timestamp=now - timedelta(days=2)
    )
    _write_generation_metadata(
        isolated_history_dir, gen_id="middle", timestamp=now - timedelta(days=1)
    )
    _write_generation_metadata(isolated_history_dir, gen_id="newest", timestamp=now)

    history._enforce_limit()

    remaining = {path.name for path in isolated_history_dir.iterdir()}

    assert remaining == {"gen_middle", "gen_newest"}


def test_history_count_and_clear_history_reflect_saved_generations(isolated_history_dir):
    _write_generation_metadata(
        isolated_history_dir, gen_id="one", timestamp=datetime.now() - timedelta(minutes=1)
    )
    _write_generation_metadata(isolated_history_dir, gen_id="two", timestamp=datetime.now())

    assert history.get_history_count() == 2
    assert history.clear_history() == 2
    assert history.get_history_count() == 0


def test_filesystem_artifact_store_uses_instance_root_without_global_mutation(
    tmp_path, monkeypatch
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_store = history.FilesystemArtifactStore(first_root)
    second_store = history.FilesystemArtifactStore(second_root)
    ids = iter(["one", "two"])
    monkeypatch.setattr(history, "_generate_id", lambda: next(ids))

    first_workspace = first_store.create_generation_workspace()
    second_workspace = second_store.create_generation_workspace()
    _write_binary_file(Path(first_workspace.midi_path), b"first-midi")
    _write_binary_file(Path(second_workspace.midi_path), b"second-midi")

    first_metadata = first_store.finalize_generation(
        workspace=first_workspace,
        prompt="first prompt",
        key="C",
        scale="major",
        model="model-a",
        provider="OpenAI",
        temperature=0.0,
    )
    second_metadata = second_store.finalize_generation(
        workspace=second_workspace,
        prompt="second prompt",
        key="D",
        scale="minor",
        model="model-b",
        provider="Google",
        temperature=0.1,
    )

    assert first_metadata.id == "one"
    assert second_metadata.id == "two"
    assert [entry.id for entry in first_store.load_history()] == ["one"]
    assert [entry.id for entry in second_store.load_history()] == ["two"]
    assert history.GENERATIONS_DIR == "generations"


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
