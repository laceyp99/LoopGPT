"""Generation history management for LoopGPT.

This module provides functionality to save, load, and manage
the history of MIDI loop generations. Each generation is stored
with its MIDI file, audio file (if available), and metadata.

Storage structure:
    generations/
        gen_<timestamp>/
            loop.mid        # Generated MIDI file
            loop.mp3        # Rendered audio (if available)
            messages.json   # Provider message history (if available)
            metadata.json   # Generation parameters and info
"""

from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import shutil
import json
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Directory for storing generations when using module-level compatibility helpers.
GENERATIONS_DIR = "generations"

# Maximum number of generations to keep
MAX_GENERATIONS = 20


class FilesystemArtifactStore:
    """Filesystem generation store bound to a caller-provided artifact root."""

    def __init__(self, artifact_root: str | Path = GENERATIONS_DIR):
        self.artifact_root = str(artifact_root)

    def _call_with_root(self, func, *args, **kwargs):
        global GENERATIONS_DIR

        previous_root = GENERATIONS_DIR
        GENERATIONS_DIR = self.artifact_root
        try:
            return func(*args, **kwargs)
        finally:
            GENERATIONS_DIR = previous_root

    def create_generation_workspace(self) -> "GenerationWorkspace":
        return self._call_with_root(create_generation_workspace)

    def finalize_generation(self, *args, **kwargs) -> "GenerationMetadata":
        return self._call_with_root(finalize_generation, *args, **kwargs)

    def cleanup_generation_workspace(self, workspace: "GenerationWorkspace") -> bool:
        return self._call_with_root(cleanup_generation_workspace, workspace)

    def update_generation_audio(
        self,
        gen_id: str,
        audio_path: Optional[str],
        soundfont: Optional[str] = None,
    ) -> Optional["GenerationMetadata"]:
        return self._call_with_root(
            update_generation_audio,
            gen_id,
            audio_path,
            soundfont=soundfont,
        )

    def load_history(self) -> list["GenerationMetadata"]:
        return self._call_with_root(load_history)

    def get_generation(self, gen_id: str) -> Optional["GenerationMetadata"]:
        return self._call_with_root(get_generation, gen_id)

    def delete_generation(self, gen_id: str) -> bool:
        return self._call_with_root(delete_generation, gen_id)


class GenerationMetadata(BaseModel):
    """Metadata for a single generation.

    Attributes:
        id: Unique identifier (timestamp-based).
        timestamp: When the generation was created.
        prompt: User's description/prompt.
        key: Musical key (C, D, etc.).
        scale: Major or minor.
        model: Model name used for generation.
        provider: API provider (OpenAI, Anthropic, Google, Ollama).
        temperature: Temperature setting used.
        cost: API cost if available.
        midi_path: Path to the MIDI file.
        audio_path: Path to the audio file (None if synthesis failed).
        messages_path: Path to the provider message history file.
        soundfont: SoundFont filename used to render the audio file.
    """

    id: str
    timestamp: datetime
    prompt: str
    key: str
    scale: str
    model: str
    provider: str
    temperature: float
    cost: Optional[float] = None
    midi_path: str
    audio_path: Optional[str] = None
    messages_path: Optional[str] = None
    soundfont: Optional[str] = None


class GenerationWorkspace(BaseModel):
    """Canonical artifact paths for a generation in progress."""

    id: str
    directory: str
    midi_path: str
    audio_path: str
    messages_path: str
    metadata_path: str


def _ensure_generations_dir() -> None:
    """Create the generations directory if it doesn't exist."""
    if not os.path.exists(GENERATIONS_DIR):
        os.makedirs(GENERATIONS_DIR)
        logger.info(f"Created generations directory: {GENERATIONS_DIR}")


def _generate_id() -> str:
    """Generate a unique ID for a generation based on timestamp.

    Returns:
        str: A unique identifier string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _get_generation_dir(gen_id: str) -> str:
    """Get the directory path for a specific generation.

    Args:
        gen_id: The generation ID.

    Returns:
        str: Path to the generation's directory.
    """
    return os.path.join(GENERATIONS_DIR, f"gen_{gen_id}")


def _build_workspace(gen_id: str) -> GenerationWorkspace:
    """Build the canonical path set for a generation ID."""
    gen_dir = _get_generation_dir(gen_id)
    return GenerationWorkspace(
        id=gen_id,
        directory=gen_dir,
        midi_path=os.path.join(gen_dir, "loop.mid"),
        audio_path=os.path.join(gen_dir, "loop.mp3"),
        messages_path=os.path.join(gen_dir, "messages.json"),
        metadata_path=os.path.join(gen_dir, "metadata.json"),
    )


def get_provider_for_model(model: str, model_info: dict) -> str:
    """Determine the provider for a given model name.

    Args:
        model: The model name.
        model_info: The model information dictionary from model_list.json.

    Returns:
        str: The provider name.
    """
    for provider, models in model_info.get("models", {}).items():
        if model in models:
            return provider
    return "Ollama"  # Default to Ollama for local models


def create_generation_workspace() -> GenerationWorkspace:
    """Create a generation directory and return its canonical artifact paths.

    Returns:
        GenerationWorkspace: The newly allocated generation workspace.

    Raises:
        RuntimeError: If a unique generation ID cannot be allocated.
    """
    _ensure_generations_dir()

    for _ in range(100):
        gen_id = _generate_id()
        workspace = _build_workspace(gen_id)
        try:
            os.makedirs(workspace.directory)
            logger.info(f"Created generation workspace: {workspace.directory}")
            return workspace
        except FileExistsError:
            continue

    raise RuntimeError("Unable to allocate a unique generation workspace")


def finalize_generation(
    workspace: GenerationWorkspace,
    prompt: str,
    key: str,
    scale: str,
    model: str,
    provider: str,
    temperature: float,
    cost: Optional[float] = None,
    soundfont: Optional[str] = None,
) -> GenerationMetadata:
    """Finalize a generation after its artifacts have been written.

    Args:
        workspace: Generation workspace with canonical artifact paths.
        prompt: User's description/prompt.
        key: Musical key.
        scale: Major or minor.
        model: Model name used.
        provider: API provider.
        temperature: Temperature setting.
        cost: API cost (optional).
        soundfont: SoundFont filename used to render the audio file (optional).

    Returns:
        GenerationMetadata: The persisted metadata.
    """
    if not os.path.exists(workspace.midi_path):
        raise FileNotFoundError(f"Missing MIDI file for generation: {workspace.midi_path}")

    audio_path = workspace.audio_path if os.path.exists(workspace.audio_path) else None
    messages_path = workspace.messages_path if os.path.exists(workspace.messages_path) else None

    # Create metadata
    metadata = GenerationMetadata(
        id=workspace.id,
        timestamp=datetime.now(),
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        midi_path=workspace.midi_path,
        audio_path=audio_path,
        messages_path=messages_path,
        soundfont=soundfont if audio_path else None,
    )

    # Save metadata
    with open(workspace.metadata_path, "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    logger.info(f"Finalized generation {workspace.id} in history")

    # Enforce generation limit
    _enforce_limit()

    return metadata


def cleanup_generation_workspace(workspace: GenerationWorkspace) -> bool:
    """Remove an unfinalized generation workspace.

    Finalized generation directories are left intact so cleanup can be called
    from broad exception handlers without deleting visible history entries.

    Args:
        workspace: Generation workspace to remove.

    Returns:
        bool: True if the workspace was removed, False otherwise.
    """
    if os.path.exists(workspace.metadata_path):
        return False

    if not os.path.isdir(workspace.directory):
        return False

    shutil.rmtree(workspace.directory)
    logger.info(f"Removed unfinalized generation workspace: {workspace.directory}")
    return True


def update_generation_audio(
    gen_id: str,
    audio_path: Optional[str],
    soundfont: Optional[str] = None,
) -> Optional[GenerationMetadata]:
    """Update the stored audio path and SoundFont for an existing generation.

    Args:
        gen_id: The generation ID.
        audio_path: Path to the rendered audio file.
        soundfont: SoundFont filename used to render the audio file.

    Returns:
        GenerationMetadata or None if the generation does not exist.
    """
    metadata = get_generation(gen_id)
    if metadata is None:
        return None

    gen_dir = _get_generation_dir(gen_id)
    metadata_path = os.path.join(gen_dir, "metadata.json")

    dest_audio_path = metadata.audio_path
    if audio_path and os.path.exists(audio_path):
        dest_audio_path = os.path.join(gen_dir, "loop.mp3")
        if os.path.abspath(audio_path) != os.path.abspath(dest_audio_path):
            shutil.copy2(audio_path, dest_audio_path)
        else:
            dest_audio_path = audio_path

    metadata.audio_path = dest_audio_path
    metadata.soundfont = soundfont

    with open(metadata_path, "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    return metadata


def load_history() -> list[GenerationMetadata]:
    """Load all generations from history.

    Returns:
        list: List of GenerationMetadata objects, sorted by timestamp (newest first).
    """
    _ensure_generations_dir()

    generations = []

    for item in os.listdir(GENERATIONS_DIR):
        if not item.startswith("gen_"):
            continue

        gen_dir = os.path.join(GENERATIONS_DIR, item)
        if not os.path.isdir(gen_dir):
            continue

        metadata_path = os.path.join(gen_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Missing metadata for generation: {item}")
            continue

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            metadata = GenerationMetadata(**data)
            generations.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load generation {item}: {e}")
            continue

    # Sort by timestamp, newest first
    generations.sort(key=lambda g: g.timestamp, reverse=True)

    return generations


def get_generation(gen_id: str) -> Optional[GenerationMetadata]:
    """Get a specific generation by ID.

    Args:
        gen_id: The generation ID.

    Returns:
        GenerationMetadata or None if not found.
    """
    gen_dir = _get_generation_dir(gen_id)
    metadata_path = os.path.join(gen_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return GenerationMetadata(**data)
    except Exception as e:
        logger.error(f"Failed to load generation {gen_id}: {e}")
        return None


def delete_generation(gen_id: str) -> bool:
    """Delete a generation from history.

    Args:
        gen_id: The generation ID to delete.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    gen_dir = _get_generation_dir(gen_id)

    if not os.path.exists(gen_dir):
        logger.warning(f"Generation not found: {gen_id}")
        return False

    try:
        shutil.rmtree(gen_dir)
        logger.info(f"Deleted generation: {gen_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete generation {gen_id}: {e}")
        return False


def _enforce_limit() -> None:
    """Delete oldest generations if over the limit."""
    generations = load_history()

    if len(generations) <= MAX_GENERATIONS:
        return

    # Delete oldest generations (they're at the end since list is sorted newest first)
    generations_to_delete = generations[MAX_GENERATIONS:]

    for gen in generations_to_delete:
        logger.info(f"Removing old generation {gen.id} to enforce limit")
        delete_generation(gen.id)


def get_history_count() -> int:
    """Get the number of generations in history.

    Returns:
        int: Number of saved generations.
    """
    return len(load_history())


def clear_history() -> int:
    """Clear all generations from history.

    Returns:
        int: Number of generations deleted.
    """
    generations = load_history()
    count = 0

    for gen in generations:
        if delete_generation(gen.id):
            count += 1

    return count
