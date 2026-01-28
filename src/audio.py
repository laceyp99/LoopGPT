"""Audio synthesis utilities for MIDI playback.

This module provides functionality to convert MIDI files to MP3 audio
using FluidSynth for synthesis and pydub for MP3 encoding.

Requires:
    - FluidSynth system library installed
    - FFmpeg installed (for MP3 encoding)
    - A SoundFont file (Salamander Grand Piano recommended)
"""

from midi2audio import FluidSynth
from pydub import AudioSegment
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

# Path to the SoundFont file - Salamander Grand Piano
SOUNDFONT_DIR = os.path.join(os.path.dirname(__file__), "..", "soundfonts")
SOUNDFONT_FILENAME = "SalamanderGrandPiano.sf2"
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, SOUNDFONT_FILENAME)

# Alternative soundfont names to search for
ALTERNATIVE_SOUNDFONTS = [
    "SalamanderGrandPiano.sf2",
    "salamander-grand-piano.sf2",
    "piano.sf2",
    "GeneralUser.sf2",
    "FluidR3_GM.sf2",
]


def find_soundfont() -> str | None:
    """Search for an available SoundFont file.

    Returns:
        str: Path to a SoundFont file if found, None otherwise.
    """
    # Check primary path first
    if os.path.exists(SOUNDFONT_PATH):
        return SOUNDFONT_PATH

    # Search for alternatives in the soundfonts directory
    if os.path.exists(SOUNDFONT_DIR):
        for sf_name in ALTERNATIVE_SOUNDFONTS:
            sf_path = os.path.join(SOUNDFONT_DIR, sf_name)
            if os.path.exists(sf_path):
                logger.info(f"Found alternative SoundFont: {sf_name}")
                return sf_path

        # Check for any .sf2 file in the directory
        for file in os.listdir(SOUNDFONT_DIR):
            if file.lower().endswith(".sf2"):
                sf_path = os.path.join(SOUNDFONT_DIR, file)
                logger.info(f"Found SoundFont: {file}")
                return sf_path

    return None


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


def is_playback_available() -> tuple[bool, str | None]:
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

    if find_soundfont() is None:
        issues.append(
            f"No SoundFont file found in '{SOUNDFONT_DIR}'. "
            f"Please download Salamander Grand Piano and place the .sf2 file there."
        )

    if issues:
        return False, "; ".join(issues)

    return True, None


def midi_to_mp3(midi_path: str, output_path: str | None = None) -> str | None:
    """Convert a MIDI file to MP3 audio using FluidSynth.

    Args:
        midi_path (str): Path to the input MIDI file.
        output_path (str, optional): Path for the output MP3 file.
            If not provided, uses the same name as the MIDI file with .mp3 extension.

    Returns:
        str | None: Path to the generated MP3 file, or None if conversion failed.

    Raises:
        FileNotFoundError: If the MIDI file doesn't exist.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Check if playback is available
    available, error = is_playback_available()
    if not available:
        logger.warning(f"Audio playback not available: {error}")
        return None

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(midi_path)[0]
        output_path = f"{base_name}.mp3"

    # Find the soundfont
    soundfont_path = find_soundfont()
    if soundfont_path is None:
        logger.error("No SoundFont file available")
        return None

    try:
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


def get_playback_status_message() -> str:
    """Get a user-friendly status message about playback availability.

    Returns:
        str: A message describing the playback status and any setup required.
    """
    available, error = is_playback_available()

    if available:
        soundfont = find_soundfont()
        sf_name = os.path.basename(soundfont) if soundfont else "Unknown"
        return f"Audio playback ready (using {sf_name})"

    # Build setup instructions
    instructions = ["Audio playback is not available. Setup required:"]

    if not is_fluidsynth_available():
        instructions.append(
            "  - Install FluidSynth: https://github.com/FluidSynth/fluidsynth/releases"
        )

    if not is_ffmpeg_available():
        instructions.append("  - Install FFmpeg: https://ffmpeg.org/download.html")

    if find_soundfont() is None:
        instructions.append(
            "  - Download Salamander Grand Piano SoundFont and place in 'soundfonts/' folder"
        )
        instructions.append(
            "    https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
        )

    return "\n".join(instructions)
