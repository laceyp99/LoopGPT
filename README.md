# LoopGPT: AI-Powered Music Generation

LoopGPT is a Python application that generates MIDI music using OpenAI's GPT models. It can create chord progressions, melodies, and accompaniments for existing melodies.

## Features

- Generate 4-bar chord progressions with customizable parameters
- Create melodic lines that complement the chord progressions 
- Generate accompaniment for existing MIDI melodies
- Visualize MIDI output using piano roll display
- Play generated music through MIDI playback
- Support for sixteenth note resolution
- Cost tracking for API usage

## Installation

1. Clone this repository
2. Install required packages:
```sh
pip install mido pygame matplotlib pretty_midi python-dotenv openai pydantic
```
3. Insert your OpenAI API key to the `.env` file in the project structure:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Generate a Chord Progression with Melody
Edit `ProgressionPlus.py`:
```python
if __name__ == "__main__":
    # Prompt for the MIDI generation
    prompt_dict = {
        "key": "C",
        "mode": "major",
        "keywords": "a rhythmic sad pop song",
        "t": 0.3,
        # "note density (%)": 60,
        # "syncopation (%)": 50,
        # "velocity dynamics (%)": 25
    }
    # Run the main function with the user inputs
    main(
        prompt_dict=prompt_dict,
        melody=True,
        visualize=False,
        play=True,
        filename="N/A"
    )
```
Run:
```
python ProgressionPlus.py
```

### Generate Accompaniment for a Melody
Edit `Accompaniment.py`:
```python
if __name__ == "__main__":
    midi = MidiFile("path/to/your/melody.mid")  # Replace with your MIDI file path
    # Run the main function with the user inputs
    main(
        melody=midi,
        visualize=False,
        play=True,
        filename="N/A",
        temp=0.2
    )
```
Run:
```
python Accompaniment.py
```

## Project Structure

- `ProgressionPlus.py`: Main entry point for generating chord progressions and melodies
- `Accompaniment.py`: Main entry point for generating accompaniments
- `code/`
  - `__init__.py`: ***Needs to be in the directory.*** Can be empty, just needs to exist
  - `.env`: Configuration file for OpenAI API key
  - `apicalls.py`: OpenAI API interaction and music generation logic
  - `midi_processing.py`: MIDI file manipulation utilities
  - `objects.py`: Data models for musical elements
  - `utils.py`: Helper functions for MIDI processing

## Limitations

- The `ProgressionPlus.py` script generates MIDI files at a fixed tempo of ***120 BPM***.
- The application currently supports only ***4-bar*** segments in ***4/4*** time signature.
- Generated music may require post-processing for professional use.
- The quality of generated music depends on the prompt and model parameters.
- Requires an active internet connection for API calls to OpenAI.
- High API usage may incur significant costs.
- Error handling and user feedback mechanisms are minimal.
- The application does not include a graphical user interface (GUI).