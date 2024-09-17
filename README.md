# LoopGPT: MIDI Loop Generator
This Jupyter Notebook generates a n-bar MIDI loop consisting of a chord progression and melody. With the **num_bars** variable, users can choose the n-value for the generation. I suggest at maximum 8 due to the quality of responses. The loop is created using OpenAI's GPT-4o model, which generates the music structure, and the resulting MIDI files are manipulated using Python libraries like mido and pretty_midi.

## Features
* Prompts OpenAI's GPT-4o model to generate musical notes for a MIDI loop.
* Produces a combination of chord progressions and melodies.
* Exports the generated loop as a MIDI file for further playback and analysis.

## Requirements
### Python Packages
The following Python packages are required to run the notebook:
* **dotenv**: To manage environment variables for API keys.
* **pydantic**: For object creation and data validation.
* **openai**: To interact with the OpenAI API.
* **mido**: For creating and handling MIDI files.
* **pretty_midi**: To parse and process MIDI data.
* **matplotlib**: For visualization (optional).
* **numpy**: For numerical operations.
* **pygame**: To handle MIDI playback.

## API
OpenAI's API key is required to access GPT-4o for music generation. The key is loaded using a **.env** file.

## Setup Instructions
1. **Install Dependencies:** Install the required Python packages by running:
  ```
  pip install python-dotenv pydantic openai mido pretty_midi matplotlib numpy pygame
  ```

2. **Set Up OpenAI API Key:**
* Create a **.env** file in the root directory of your project.
* Add your OpenAI API key to the **.env** file as follows:
  ```
  SANDBOX_API_KEY=your_openai_api_key
  ```
3. Run the Notebook: Open the notebook in Jupyter and execute the cells to generate a n-bar MIDI loop.

## Usage
1. Music Data Structure:
* The notebook defines **Note**, **Chord**, and **TimeInformation** classes to represent musical notes and their timing.
* The GPT-4o model generates the music data based on this structure.

2. MIDI Generation:
* The generated notes are converted into MIDI format using **mido** and **pretty_midi**.
* The notebook provides functionality to export the MIDI file and visualize the melody.

3. Playback:
* Optionally, you can use **pygame** to playback the generated MIDI file within the notebook environment.

## Output
The notebook generates a n-bar MIDI file consisting of:
* **Chord Progression:** A series of chords that form the harmonic structure.
* **Melody:** A sequence of notes that play over the chord progression. (OPTIONAL)

## Customization
* You can modify the note generation logic and parameters to create different types of MIDI loops.
* The notebook is designed to be extensible and flexible for various musical experiments.
