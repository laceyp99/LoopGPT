# 🎵 LoopGPT: AI-Powered Music Generation 🎵

This project is a music generation tool that enables users to create chord progressions, melodies, and accompaniments using an intuitive Gradio web interface. By leveraging OpenAI’s API, the application generates MIDI compositions based on user-defined parameters, making it a great resource for musicians, producers, and AI music enthusiasts.

## ✨ Features

- Interact with a sleek interface for an inviting user experience
- Generate 4-bar chord progressions with customizable parameters
- Create melodic lines that complement the chord progressions 
- Generate accompaniment for existing MIDI melodies
- Visualize MIDI output using piano roll display
- Support for sixteenth note resolution
- Save full API message history to JSON files for training examples
- Cost tracking for API usage

## 🚀 Quick Start
### Installation
Clone this repository:
```sh
git clone https://github.com/laceyp99/LoopGPT.git
cd LoopGPT
```
Install required dependencies:
```sh
pip install mido pygame matplotlib pretty_midi python-dotenv openai pydantic gradio
```
Set up your OpenAI API key in a .env file:
```sh
OPENAI_API_KEY="your_api_key_here"
```
### Run the Gradio UI
To start the web interface locally, run:
```sh
python app.py
```
Then visit [localhost](http://127.0.0.1:7860/) to access the UI.

## 🎼 Usage

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
```sh
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
```sh
python Accompaniment.py
```

## 📂 Project Structure

- `app.py`: The Gradio UI layout with connections to the *ProgressionPlus* and *Accompaniment* scripts
- `ProgressionPlus.py`: Main entry point for generating chord progressions and melodies
- `Accompaniment.py`: Main entry point for generating accompaniments
- `code/`
  - `__init__.py`: ***Needs to be in the directory.*** Can be empty, just needs to exist
  - `.env`: Configuration file for OpenAI API key
  - `apicalls.py`: OpenAI API interaction and music generation logic
  - `midi_processing.py`: MIDI file manipulation utilities
  - `objects.py`: Data models for musical elements
  - `utils.py`: Helper functions for MIDI processing
  - `decorators.py`: Decorators for error handling and logging
  - `exceptions.py`: Custom exception classes
- `Prompts/`
  - `accompaniment chord generation.txt`: The system prompt for the accompaniment chord generation API calls.
  - `chord generation.txt`: The system prompt for the chord generation API calls.
  - `melody generation.txt`: The system prompt for the melody generation API calls.
  - `prompt translation with melody.txt`: The system prompt for the prompt translation with melody API calls.
  - `prompt translation.txt`: The system prompt for the prompt translation API calls.
- `Training Examples/`: Directory where the JSON files containing the API message history are saved
- `ProgressionPlus/`: Default directory where generated MIDI files are saved for the ProgressionPlus.py generation.
- `Accompaniment/`: Default directory where generated MIDI files are saved for the Accompaniment.py generation.

## ⚠️ Limitations

- The `ProgressionPlus.py` script generates MIDI files at a fixed tempo of ***120 BPM***.
- The application currently supports only ***4-bar*** segments in ***4/4*** time signature.
- Choosing `gpt-4o-mini` to generate MIDI may impact the musical form and quality.
- Generated music may require post-processing for professional use.
- The quality of generated music depends on the prompt and model parameters.
- Requires an active internet connection for API calls to OpenAI.
- High API usage may incur significant costs.
