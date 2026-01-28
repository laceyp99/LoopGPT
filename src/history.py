"""Generation history management for LoopGPT.

This module provides functionality to save, load, and manage
the history of MIDI loop generations. Each generation is stored
with its MIDI file, audio file (if available), and metadata.

Storage structure:
    generations/
        gen_<timestamp>/
            loop.mid        # Generated MIDI file
            loop.mp3        # Rendered audio (if available)
            metadata.json   # Generation parameters and info
"""

from pydantic import BaseModel
from datetime import datetime
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

# Directory for storing generations
GENERATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "generations")

# Maximum number of generations to keep
MAX_GENERATIONS = 20


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


def save_generation(
    midi_path: str,
    prompt: str,
    key: str,
    scale: str,
    model: str,
    provider: str,
    temperature: float,
    cost: Optional[float] = None,
    audio_path: Optional[str] = None,
) -> str:
    """Save a generation to history.

    Copies the MIDI and audio files to a timestamped directory
    and saves metadata.

    Args:
        midi_path: Path to the generated MIDI file.
        prompt: User's description/prompt.
        key: Musical key.
        scale: Major or minor.
        model: Model name used.
        provider: API provider.
        temperature: Temperature setting.
        cost: API cost (optional).
        audio_path: Path to rendered audio file (optional).

    Returns:
        str: The generation ID.
    """
    _ensure_generations_dir()

    # Generate unique ID and create directory
    gen_id = _generate_id()
    gen_dir = _get_generation_dir(gen_id)
    os.makedirs(gen_dir)

    # Copy MIDI file
    dest_midi_path = os.path.join(gen_dir, "loop.mid")
    shutil.copy2(midi_path, dest_midi_path)

    # Copy audio file if available
    dest_audio_path = None
    if audio_path and os.path.exists(audio_path):
        dest_audio_path = os.path.join(gen_dir, "loop.mp3")
        shutil.copy2(audio_path, dest_audio_path)

    # Create metadata
    metadata = GenerationMetadata(
        id=gen_id,
        timestamp=datetime.now(),
        prompt=prompt,
        key=key,
        scale=scale,
        model=model,
        provider=provider,
        temperature=temperature,
        cost=cost,
        midi_path=dest_midi_path,
        audio_path=dest_audio_path,
    )

    # Save metadata
    metadata_path = os.path.join(gen_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    logger.info(f"Saved generation {gen_id} to history")

    # Enforce generation limit
    _enforce_limit()

    return gen_id


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
