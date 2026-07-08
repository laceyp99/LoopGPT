"""Audio synthesis utilities for MIDI playback.

This module provides functionality to convert MIDI files to MP3 audio
using FluidSynth for synthesis and pydub for MP3 encoding.

Requires:
    - FluidSynth system library installed
    - FFmpeg installed (for MP3 encoding)
    - At least one SoundFont file in the soundfonts directory
"""

from importlib import resources
import tempfile
import logging
import shutil
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FluidSynth = None
AudioSegment = None

SOUNDFONT_DIR = str(resources.files("conductor_core.resources").joinpath("soundfonts"))
# Preferred SoundFont filenames searched in order.
DEFAULT_SOUNDFONT_CANDIDATES = [
    "FM-Piano1 20190916.sf2",
    "SalamanderGrandPiano.sf2",
    "salamander-grand-piano.sf2",
    "piano.sf2",
    "GeneralUser.sf2",
    "FluidR3_GM.sf2",
]


def _soundfont_search_dirs() -> list[str]:
    """Return SoundFont search directories in priority order."""
    return [SOUNDFONT_DIR]


def _find_soundfont_file(soundfont_name: str) -> str | None:
    """Find a SoundFont by filename across configured search directories."""
    requested_name = os.path.basename(soundfont_name)
    for soundfont_dir in _soundfont_search_dirs():
        candidate_path = os.path.join(soundfont_dir, requested_name)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def list_soundfonts() -> list[str]:
    """List the available SoundFont filenames.

    Returns:
        list[str]: Sorted `.sf2` filenames in the soundfonts directory.
    """
    names = set()
    for soundfont_dir in _soundfont_search_dirs():
        if not os.path.exists(soundfont_dir):
            continue
        names.update(
            file for file in os.listdir(soundfont_dir) if file.lower().endswith(".sf2")
        )
    return sorted(names, key=str.lower)


def get_default_soundfont() -> str | None:
    """Resolve the default SoundFont path.

    Returns:
        str | None: Path to the preferred available SoundFont, or None.
    """
    available_soundfonts = list_soundfonts()
    if not available_soundfonts:
        return None

    available_lookup = {name.lower(): name for name in available_soundfonts}

    for soundfont_name in DEFAULT_SOUNDFONT_CANDIDATES:
        matched_name = available_lookup.get(soundfont_name.lower())
        if matched_name:
            return _find_soundfont_file(matched_name)

    fallback_soundfont = available_soundfonts[0]
    logger.info(f"Falling back to SoundFont: {fallback_soundfont}")
    return _find_soundfont_file(fallback_soundfont)


def resolve_soundfont(soundfont_name: str | None = None) -> str | None:
    """Resolve an explicit or default SoundFont path.

    Args:
        soundfont_name (str | None): Optional SoundFont filename or path.

    Returns:
        str | None: Path to a SoundFont file if found, None otherwise.
    """
    if soundfont_name:
        if os.path.exists(soundfont_name):
            return soundfont_name

        requested_path = _find_soundfont_file(soundfont_name)
        if requested_path:
            return requested_path

        return None

    return get_default_soundfont()


def find_soundfont(soundfont_name: str | None = None) -> str | None:
    """Backwards-compatible wrapper for SoundFont resolution.

    Args:
        soundfont_name (str | None): Optional SoundFont filename or path.

    Returns:
        str | None: Path to a SoundFont file if found, None otherwise.
    """
    return resolve_soundfont(soundfont_name)


def is_fluidsynth_available() -> bool:
    """Check if FluidSynth is installed and available.

    Returns:
        bool: True if FluidSynth is available, False otherwise.
    """
    return shutil.which("fluidsynth") is not None


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available.

    Returns:
        bool: True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def is_playback_available(soundfont_name: str | None = None) -> tuple[bool, str | None]:
    """Check if audio playback is available.

    Verifies that all required components are present:
    - FluidSynth installed
    - FFmpeg installed
    - SoundFont file exists

    Returns:
        tuple: (is_available, error_message)
            - is_available (bool): True if playback is fully available
            - error_message (str | None): Description of what's missing, or None if all good
    """
    issues = []

    if not is_fluidsynth_available():
        issues.append("FluidSynth is not installed or not in PATH")

    if not is_ffmpeg_available():
        issues.append("FFmpeg is not installed or not in PATH")

    resolved_soundfont = find_soundfont(soundfont_name)
    if resolved_soundfont is None:
        if soundfont_name:
            issues.append(
                f"Requested SoundFont '{os.path.basename(soundfont_name)}' was not found in '{SOUNDFONT_DIR}'."
            )
        else:
            issues.append(
                f"No SoundFont file found in '{SOUNDFONT_DIR}'. Add a .sf2 file to enable audio playback."
            )

    if issues:
        return False, "; ".join(issues)

    return True, None


def midi_to_mp3(
    midi_path: str,
    output_path: str | None = None,
    soundfont_name: str | None = None,
) -> str | None:
    """Convert a MIDI file to MP3 audio using FluidSynth.

    Args:
        midi_path (str): Path to the input MIDI file.
        output_path (str, optional): Path for the output MP3 file.
            If not provided, uses the same name as the MIDI file with .mp3 extension.
        soundfont_name (str, optional): SoundFont filename or path to use.

    Returns:
        str | None: Path to the generated MP3 file, or None if conversion failed.

    Raises:
        FileNotFoundError: If the MIDI file doesn't exist.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Check if playback is available
    available, error = is_playback_available(soundfont_name)
    if not available:
        logger.warning(f"Audio playback not available: {error}")
        return None

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(midi_path)[0]
        output_path = f"{base_name}.mp3"

    # Find the soundfont
    soundfont_path = find_soundfont(soundfont_name)
    if soundfont_path is None:
        logger.error("No SoundFont file available")
        return None

    try:
        global AudioSegment, FluidSynth

        if FluidSynth is None:
            from midi2audio import FluidSynth as _FluidSynth

            FluidSynth = _FluidSynth
        if AudioSegment is None:
            from pydub import AudioSegment as _AudioSegment

            AudioSegment = _AudioSegment

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        # Use FluidSynth to render MIDI to WAV
        logger.info(f"Rendering MIDI to WAV using SoundFont: {soundfont_path}")
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_path, temp_wav_path)

        # Convert WAV to MP3 using pydub
        logger.info(f"Converting WAV to MP3: {output_path}")
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(output_path, format="mp3", bitrate="192k")

        logger.info(f"Successfully created MP3: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to convert MIDI to MP3: {e}")
        return None

    finally:
        # Clean up temporary WAV file
        if "temp_wav_path" in locals() and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


def get_playback_status_message(soundfont_name: str | None = None) -> str:
    """Get a user-friendly status message about playback availability.

    Returns:
        str: A message describing the playback status and any setup required.
    """
    available, error = is_playback_available(soundfont_name)

    if available:
        soundfont = find_soundfont(soundfont_name)
        sf_name = os.path.basename(soundfont) if soundfont else "Unknown"
        return f"Audio playback ready (using {sf_name})"

    dependency_instructions = []

    if not is_fluidsynth_available():
        dependency_instructions.append(
            "  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases"
        )

    if not is_ffmpeg_available():
        dependency_instructions.append(
            "  - Install FFmpeg: https://ffmpeg.org/download.html"
        )

    if dependency_instructions:
        return "\n".join([
            "Audio playback is not available. Setup required:",
            *dependency_instructions,
        ])

    instructions = ["Audio playback is not available. Setup required:"]

    resolved_soundfont = find_soundfont(soundfont_name)
    if resolved_soundfont is None:
        if soundfont_name:
            instructions.append(
                f"  - Pass an existing SoundFont path or package '{os.path.basename(soundfont_name)}' with Core"
            )
        else:
            instructions.append("  - Package a `.sf2` SoundFont file with Core")

    return "\n".join(instructions)
